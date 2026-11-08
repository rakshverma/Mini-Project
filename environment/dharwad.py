"""Dharwad, Karnataka - Saptapur / Channabasaveshwara Layout / KCD Circle, from OpenStreetMap.

Real data (downloaded once from the Overpass API and cached next to this file):
  * streets   : OSM highways -> directed graph (one-way streets & roundabouts respected),
                half-width from road class;  drivable area = union of street capsules
  * buildings : OSM footprints -> lidar features (the corner nearest the street)
                and building density -> GPS quality (multipath / urban canyon)
Simulated on top of the map:
  * HD map    : features along the arterial roads (primary / secondary) are pre-surveyed
  * parked cars along residential lanes, moving traffic (keep-left) on the arterials
  * the ego car: kinematic bicycle, noisy actuation, lidar + GPS
"""
import json
import os
import subprocess
import urllib.parse
import urllib.request
import numpy as np
from slam import Bicycle, wrap
from pomdp import phi

NAME = "Dharwad - Saptapur & KCD Circle"
BBOX = (15.4490, 74.9900, 15.4560, 74.9990)                   # south, west, north, east
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dharwad_osm.json")
HALF_WIDTH = {"primary": 6.0, "primary_link": 4.0, "secondary": 5.0, "secondary_link": 4.0,
              "tertiary": 4.5, "unclassified": 3.5, "residential": 3.5, "living_street": 3.0}
CLOSURE_XY, CLOSURE_AT = ((-263.0, -204.0), (-210.0, -171.0)), 0.25  # Saptapur Rd stretch closed in --closure
                                                                       # runs; barrier 15 m past the junction
START_LATLON, GOAL_LATLON = (15.4505, 74.9917), (15.4539, 74.9920)   # Saptapur Rd junction -> NW colony junction


def download():
    """Fetch drivable streets + building footprints inside BBOX (cached)."""
    if os.path.exists(CACHE):
        return json.load(open(CACHE))["elements"]
    s, w, n, e = BBOX
    hw = "|".join(HALF_WIDTH)
    q = (f'[out:json][timeout:60];(way["highway"~"^({hw})$"]({s},{w},{n},{e});'
         f'way["building"]({s},{w},{n},{e}););out body;>;out skel qt;')
    url, body = "https://overpass-api.de/api/interpreter", urllib.parse.urlencode({"data": q})
    try:
        req = urllib.request.Request(url, data=body.encode(), headers={"User-Agent": "slam-pomdp-student-project"})
        raw = urllib.request.urlopen(req, timeout=90).read()
    except OSError:                               # old LibreSSL builds of Python fail the TLS handshake
        raw = subprocess.run(["curl", "-s", "-m", "90", "-A", "slam-pomdp-student-project", "--data", body, url],
                             capture_output=True, check=True).stdout
    data = json.loads(raw)
    json.dump(data, open(CACHE, "w"))
    return data["elements"]


class Dharwad:
    model = Bicycle
    dt = 0.2                                      # [s]
    max_range, max_features = 25.0, 10            # lidar range [m], features tracked per scan
    car_radius, other_radius = 1.0, 1.0           # [m] disc footprints
    a_lat_max = 5.0                               # [m/s^2] friction / comfort limit
    R = np.diag([0.25, np.deg2rad(1.0)]) ** 2     # lidar noise (range, bearing)
    hd_map_std = 0.10                             # [m]
    sigma_parked, sigma_moving = 0.2, 0.5         # [m] perception / prediction noise of other cars

    @staticmethod
    def Q(u):
        """Actuation noise covariance: speed error 4 % of speed, steering error 1 deg."""
        return np.diag([0.04 * abs(u[0]) + 1e-3, np.deg2rad(1.0)]) ** 2

    def __init__(self, seed=0, closure=False, obstruction=False):
        self.rng = np.random.default_rng(seed)
        s, w, n, e = BBOX
        self.lat0, self.lon0 = (s + n) / 2, (w + e) / 2
        self.xmax, self.ymax = self.to_xy(n, e)
        els = download()
        ll = {el["id"]: (el["lat"], el["lon"]) for el in els if el["type"] == "node"}
        self._build_roads(els, ll)
        self._build_buildings(els, ll)
        self._make_traffic()
        self.start_node, self.goal_node = (self.nearest_node(self.to_xy(*START_LATLON)),
                                           self.nearest_node(self.to_xy(*GOAL_LATLON)))
        self.goal = self.xy[self.goal_node]
        self.x = np.r_[self.xy[self.start_node], 0.0]
        # road closure (construction barrier across the street): unknown until the lidar sees it
        self.closed, self.closure_known = set(), False
        # obstruction: stalled vehicles across the street at the same place. Not a mapped closure --
        # they are just obstacles, each known to the planners only once the lidar has seen it.
        self.hazards, self.hazard_seen = np.zeros((0, 2)), np.zeros(0, bool)
        if obstruction:
            a, b = (self.nearest_node(np.array(p)) for p in CLOSURE_XY)
            t = (self.xy[b] - self.xy[a]) / np.linalg.norm(self.xy[b] - self.xy[a])
            c = self.xy[a] + CLOSURE_AT * (self.xy[b] - self.xy[a])
            self.hazards = np.array([c + k * np.array([-t[1], t[0]]) for k in (-4.0, 0.0, 4.0)])
            self.hazard_seen = np.zeros(len(self.hazards), bool)
        if closure:
            a, b = (self.nearest_node(np.array(p)) for p in CLOSURE_XY)
            self.closed = {(a, b), (b, a)} & set(self.edge_len)
            self.barrier = self.xy[a] + CLOSURE_AT * (self.xy[b] - self.xy[a])
            self.barrier_r = self.edge_hw[next(iter(self.closed))]

    # ------------------------------------------------------------------ map construction
    def to_xy(self, lat, lon):
        """Local tangent-plane (equirectangular) projection, metres east / north of the bbox centre."""
        R = 6371000.0
        return np.array([R * np.cos(np.radians(self.lat0)) * np.radians(lon - self.lon0),
                         R * np.radians(lat - self.lat0)])

    def inside(self, p, margin=0.0):
        return abs(p[0]) <= self.xmax - margin and abs(p[1]) <= self.ymax - margin

    def _build_roads(self, els, ll):
        self.xy, self.adj, self.edge_len, self.edge_hw, self.edge_lane = {}, {}, {}, {}, {}
        segs, self.ways = [], []
        for el in els:
            tags = el.get("tags", {})
            if el["type"] != "way" or tags.get("highway") not in HALF_WIDTH:
                continue
            hw = HALF_WIDTH[tags["highway"]]
            oneway = tags.get("oneway") == "yes" or tags.get("junction") == "roundabout"
            lane = 0.0 if oneway or hw < 5 else hw / 2              # keep left on two-way arterials
            ids = [i for i in el["nodes"] if i in ll]
            for i in ids:
                self.xy.setdefault(i, self.to_xy(*ll[i]))
            run = []
            for a, b in zip(ids[:-1], ids[1:]):
                if not (self.inside(self.xy[a]) and self.inside(self.xy[b])):
                    if len(run) > 1:
                        self.ways.append((np.array(run), hw, tags))
                    run = []
                    continue
                run = run or [self.xy[a]]
                run.append(self.xy[b])
                L = float(np.linalg.norm(self.xy[b] - self.xy[a]))
                segs.append((self.xy[a], self.xy[b], hw))
                for u, v in ([(a, b)] if oneway else [(a, b), (b, a)]):
                    self.adj.setdefault(u, []).append((v, L))
                    self.adj.setdefault(v, [])
                    self.edge_len[(u, v)], self.edge_hw[(u, v)], self.edge_lane[(u, v)] = L, hw, lane
            if len(run) > 1:
                self.ways.append((np.array(run), hw, tags))
        self.xy = {k: v for k, v in self.xy.items() if k in self.adj}
        self.A = np.array([a for a, b, h in segs])
        self.B = np.array([b for a, b, h in segs])
        self.HW = np.array([h for a, b, h in segs])

    def _build_buildings(self, els, ll):
        """Footprints; one lidar feature per building (corner nearest a street, if within 20 m);
        features within 20 m of an arterial are in the HD map."""
        self.buildings, feats, hd = [], [], []
        for el in els:
            if el["type"] != "way" or "building" not in el.get("tags", {}):
                continue
            poly = np.array([self.to_xy(*ll[i]) for i in el["nodes"] if i in ll])
            if len(poly) < 3 or not self.inside(poly.mean(0)):
                continue
            self.buildings.append(poly)
            d = np.array([self.road_distance(c) for c in poly[:-1]])
            if d.min() < 20.0:
                c = poly[np.argmin(d)]
                feats.append(c)
                hd.append(self.road_distance(c, arterial=True) < 20.0)
        self.landmarks = np.array(feats)
        self.hd_map = {i: self.landmarks[i] for i in np.flatnonzero(hd)}
        self.centroids = np.array([b.mean(0) for b in self.buildings])

    def _make_traffic(self):
        """Parked cars on residential lanes (kerb side) and moving cars on the arterials (keep left)."""
        parked = []
        for a, b, hw in zip(self.A, self.B, self.HW):
            L = np.linalg.norm(b - a)
            if hw > 4.0 or L < 15:
                continue
            t = (b - a) / L
            n = np.array([-t[1], t[0]])
            for s in np.arange(8.0, L - 8.0, 30.0):
                if self.rng.random() < 0.25:
                    parked.append(a + s * t + self.rng.choice([-1, 1]) * (hw - 1.0) * n)
        self.parked = np.array(parked).reshape(-1, 2)
        self.movers, s0 = [], []                      # (polyline, cum. length, lane offset, speed)
        for pts, hw, tags in self.ways:
            if hw < 4.5:
                continue
            oneway = tags.get("oneway") == "yes" or tags.get("junction") == "roundabout"
            for p in ([pts] if oneway else [pts, pts[::-1]]):
                cum = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
                if cum[-1] < 40:
                    continue
                for _ in range(max(1, int(cum[-1] // 120))):
                    self.movers.append((p, cum, 0.0 if oneway else hw / 2, self.rng.uniform(5.0, 8.0)))
                    s0.append(self.rng.uniform(0, cum[-1]))
        self.mover_s, self.t = np.array(s0), 0.0      # arc position of each moving car, sim clock
        # pack all lanes into flat arrays so every car is advanced in one vectorised step
        n = np.array([len(m[0]) for m in self.movers])
        self.m_off = np.r_[0, np.cumsum(n)[:-1]]                       # first point of each lane
        self.m_last = self.m_off + n - 2                               # last segment of each lane
        self.m_len = np.array([m[1][-1] for m in self.movers])
        self.m_base = np.r_[0, np.cumsum(self.m_len + 1.0)[:-1]]        # lanes laid end to end
        self.m_pts = np.vstack([m[0] for m in self.movers])
        self.m_cum = np.concatenate([m[1] + b for m, b in zip(self.movers, self.m_base)])
        self.m_lane = np.array([m[2] for m in self.movers])
        self.m_v = np.array([m[3] for m in self.movers])
        self.m_go = np.ones(len(self.movers))                         # 0 while a car is braking for us
        self._mcache = {}

    # ------------------------------------------------------------------ geometry & constraints
    def road_distance(self, p, arterial=False):
        A, B = (self.A, self.B) if not arterial else (self.A[self.HW >= 5], self.B[self.HW >= 5])
        AB = B - A
        t = np.clip(np.einsum("ij,ij->i", p - A, AB) / np.einsum("ij,ij->i", AB, AB), 0, 1)
        return np.min(np.hypot(*(A + t[:, None] * AB - p).T))

    def road_margin(self, p):
        """m = max_e (hw_e - r_car - d_e(p)) (> 0 on the road) and the outward normal of the binding street."""
        AB = self.B - self.A
        t = np.clip(np.einsum("ij,ij->i", p - self.A, AB) / np.einsum("ij,ij->i", AB, AB), 0, 1)
        off = p - (self.A + t[:, None] * AB)
        d = np.hypot(*off.T)
        e = np.argmax(self.HW - self.car_radius - d)
        n = off[e] / d[e] if d[e] > 1e-6 else np.array([-AB[e, 1], AB[e, 0]]) / np.linalg.norm(AB[e])
        return self.HW[e] - self.car_radius - d[e], n

    def movers_at(self, t):
        """Positions and headings of moving cars at time t: exact for t = now, constant-velocity
        prediction with each car's CURRENT measured speed for t > now (what the planner uses)."""
        key = round(t - self.t, 4)
        if key not in self._mcache:                                    # cleared every sim step
            g = self.m_base + (self.mover_s + self.m_v * self.m_go * max(key, 0.0)) % self.m_len
            i = np.minimum(np.searchsorted(self.m_cum, g, side="right") - 1, self.m_last)
            seg = self.m_pts[i + 1] - self.m_pts[i]
            d = seg / np.maximum(self.m_cum[i + 1] - self.m_cum[i], 1e-9)[:, None]
            P = self.m_pts[i] + (g - self.m_cum[i])[:, None] * d + self.m_lane[:, None] * np.stack([-d[:, 1], d[:, 0]], 1)
            self._mcache[key] = (P, np.arctan2(d[:, 1], d[:, 0]))
        return self._mcache[key]

    def risk(self, mu, Pxy=None, t=None):
        """Chance constraint  max( Pr(off-road), max_j Pr(hit car j) ),  each a 1-D Gaussian tail:
             Pr(off-road) = Phi(-m / sqrt(n^T Pxy n))
             Pr(hit j)    = Phi((r_car + r_j - ||p - c_j||) / sqrt(n_j^T Pxy n_j + sigma_j^2))
        Pxy=None -> certainty-equivalent (sigma -> 0).  t=None -> static obstacles only."""
        p = mu[:2]
        m, n = self.road_margin(p)
        worst = phi(-m / (np.sqrt(n @ Pxy @ n) if Pxy is not None else 1e-3))
        obs = [(self.parked, self.sigma_parked), (self.hazards[self.hazard_seen], self.sigma_parked)]
        if t is not None:
            obs.append((self.movers_at(t)[0], self.sigma_moving))
        if self.closure_known:
            d = self.barrier - p
            dist = np.linalg.norm(d)
            s = np.sqrt(d @ Pxy @ d) / dist if Pxy is not None else 1e-3
            worst = max(worst, phi((self.car_radius + self.barrier_r - dist) / max(s, 1e-3)))
        for C, sig in obs:
            d = C - p
            dist = np.hypot(*d.T)
            for dj, rj in zip(d[dist < 15], dist[dist < 15]):
                nj = dj / rj
                s = np.sqrt(nj @ Pxy @ nj + sig ** 2) if Pxy is not None else 1e-3
                worst = max(worst, phi((self.car_radius + self.other_radius - rj) / s))
        return worst

    def gps_sigma(self, p):
        """GPS quality from real building density within 35 m (multipath / sky blockage):
        open -> 1.5 m,  built-up -> 4 m,  dense -> no fix."""
        k = np.count_nonzero(np.hypot(*(self.centroids - p).T) < 35.0)
        return 1.5 if k < 5 else (4.0 if k < 9 else None)

    def nearest_node(self, p):
        ids = list(self.xy)
        return ids[int(np.argmin([np.linalg.norm(self.xy[i] - p) for i in ids]))]

    # ------------------------------------------------------------------ ground-truth simulation
    def step(self, u):
        """Ego car moves with actuation noise; traffic advances but brakes for the ego car
        when it is less than 8 m ahead in the same lane."""
        self.x = self.model.f(self.x, np.asarray(u) + self.rng.multivariate_normal(np.zeros(2), self.Q(u)), self.dt)
        P, H = self.movers_at(self.t)
        rel = self.x[:2] - P
        ahead = np.einsum("ij,ij->i", rel, np.stack([np.cos(H), np.sin(H)], 1))
        side = np.abs(np.einsum("ij,ij->i", rel, np.stack([-np.sin(H), np.cos(H)], 1)))
        go = ~((ahead > 0) & (ahead < 8.0) & (side < 2.5))
        self.m_go = go.astype(float)
        self.mover_s = (self.mover_s + go * self.m_v * self.dt) % self.m_len
        self.t += self.dt
        if len(self.hazards):                                          # lidar sees stalled vehicles
            self.hazard_seen |= np.hypot(*(self.hazards - self.x[:2]).T) < self.max_range
        if self.closed and not self.closure_known and \
                np.linalg.norm(self.barrier - self.x[:2]) < self.max_range:   # lidar sees the barrier
            self.closure_known = True
        self._mcache = {}

    def observe(self):
        """Up to max_features nearest building corners within range: z ~ N(h(x, m), R)."""
        d = self.landmarks - self.x[:2]
        r = np.hypot(*d.T)
        out = []
        for i in np.argsort(r)[:self.max_features]:
            if r[i] < self.max_range:
                z = np.array([r[i], np.arctan2(d[i, 1], d[i, 0]) - self.x[2]])
                z += self.rng.multivariate_normal(np.zeros(2), self.R)
                out.append((int(i), np.array([z[0], wrap(z[1])])))
        return out

    def gps(self):
        sigma = self.gps_sigma(self.x[:2])
        return None if sigma is None else (self.x[:2] + self.rng.normal(0, sigma, 2), sigma)

    def collided(self, t):
        if self.road_margin(self.x[:2])[0] < 0:
            return "off-road"
        if self.closed and np.linalg.norm(self.barrier - self.x[:2]) < self.barrier_r + self.car_radius:
            return "hit barrier"
        cars = np.vstack([self.parked, self.hazards, self.movers_at(t)[0]])
        if len(cars) and np.min(np.hypot(*(cars - self.x[:2]).T)) < self.car_radius + self.other_radius:
            return "hit car"
        return None
