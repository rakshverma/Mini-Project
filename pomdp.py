"""POMDP planning in belief space, in two layers.

POMDP  <S, A, O, T, Z, R, gamma>
  S : car pose + landmark map (hidden)        A : (speed, steering) macro-actions
  T : bicycle model + actuation noise         Z : lidar range-bearing + GPS
  A : closed-loop macro-actions a = (speed v, steering offset): pure-pursuit lane tracking
      re-evaluated every tick, held for `repeat` ticks
  b : Gaussian belief N(mu, P) from EKF-SLAM  (slam.py)

Layer 1 - ROUTE (Belief Roadmap, Prentice & Roy 2009)
  k alternative street routes from the OSM graph; each is scored by rolling the EKF belief
  along it:   J(route) = length + w_risk * sum_t Pr(collision | b_t)

Layer 1b - D* LITE (Koenig & Likhachev 2002) on belief-dependent edge costs
  c(u, v) = length + w_risk * sum_t p_t + c_fail * clip(2 max_t p_t - 1, 0, 1),  p_t = Pr(collision | b_t), from an EKF roll-out
  that starts from the car's CURRENT pose covariance. As EKF-SLAM's uncertainty changes, the
  costs of nearby streets change and D* Lite repairs the shortest path incrementally.

Layer 2 - LOCAL (online forward search on the belief MDP)
  Q(b, a) = R(b, a) + gamma * max_a' Q(tau(b, a, z*), a')
  z* = h(mu) is the Most-Likely Observation (Platt et al. 2010): the mean is unchanged but the
  covariance shrinks exactly as the EKF would, so the planner can value information.
  R(b, a) = w_g * progress  - w_u * sqrt(tr P_xy)  - w_c * Pr(collision | b)
            - w_t * |cross-track error|  - w_s * |dv|  + R_goal * [goal reached]
  Pr(collision) is the environment's chance constraint (off-road, parked & moving cars);
  aware=False ignores P  ->  certainty-equivalent baseline.
"""
import heapq
from math import erf, sqrt
import numpy as np
import slam


def phi(x):
    """Standard normal CDF."""
    return 0.5 * (1 + erf(x / sqrt(2)))


def mlo_update(env, mu, P, index):
    """Most-likely-observation update: GPS (if a fix is expected) + the nearest visible mapped landmarks."""
    sigma = env.gps_sigma(mu[:2])
    if sigma is not None:
        mu, P = slam.update_gps(mu, P, mu[:2], sigma)
    if index:
        ks = np.array(list(index.values()))
        d = np.hypot(*(np.stack([mu[ks], mu[ks + 1]], 1) - mu[:2]).T)
        for k in ks[np.argsort(d)][:env.max_features][np.sort(d)[:env.max_features] < env.max_range]:
            z_hat, _ = slam.measure(mu, k)
            mu, P = slam.update(mu, P, k, z_hat, env.R)
    return mu, P


def control(env, route, mu, a):
    """Macro-action a = (v, offset) -> command u = (v, delta) from the current belief mean.
    Pure pursuit toward the lane point l_d ahead:  delta = atan(2 L sin(alpha) / l_d) + offset,
    limited by the steering lock and the friction circle  v^2 tan|delta| / L <= a_lat."""
    v, off = a
    if v < 0:                                                  # reverse manoeuvre: fixed steering angle
        return np.array([v, off])
    ld = float(np.clip(1.0 * v + 2.5, 3.5, 15.0))               # look-ahead grows with speed
    d = route.lookahead(mu[:2], ld) - mu[:2]
    alpha = slam.wrap(np.arctan2(d[1], d[0]) - mu[2])
    L = env.model.L
    dmax = min(0.6, np.arctan(env.a_lat_max * L / max(v, 1e-3) ** 2))
    return np.array([v, np.clip(np.arctan(2 * L * np.sin(alpha) / ld) + off, -dmax, dmax)])


# =================================================================== layer 1: route
class Route:
    """Lane-centre polyline. Each segment is shifted left by its lane offset (India drives on the left)."""

    def __init__(self, nodes, pts, hw, lane):
        pts, lane = np.asarray(pts, float), np.asarray(lane, float)
        t = np.diff(pts, axis=0)
        t = t / np.maximum(np.hypot(*t.T), 1e-9)[:, None]
        n = np.stack([-t[:, 1], t[:, 0]], 1)
        # continuous lane-centre line: neighbouring offset segments are joined where they intersect
        # (mitre join), so the line has no sideways jump where the lane offset changes
        P = [pts[0] + lane[0] * n[0]]
        for k in range(1, len(pts) - 1):
            a, b = pts[k] + lane[k - 1] * n[k - 1], pts[k] + lane[k] * n[k]
            M = np.column_stack([t[k - 1], -t[k]])
            q = (a + b) / 2
            if abs(np.linalg.det(M)) > 1e-3:
                q_m = a + np.linalg.solve(M, b - a)[0] * t[k - 1]
                if np.linalg.norm(q_m - pts[k]) < 2 * max(lane[k - 1], lane[k]) + 1:   # no long spikes
                    q = q_m
            P.append(q)
        P.append(pts[-1] + lane[-1] * n[-1])
        P = np.array(P)
        seg = np.diff(P, axis=0)
        self.seg_len = np.maximum(np.hypot(*seg.T), 1e-9)
        self.A, self.B, self.T, self.hw = P[:-1], P[1:], seg / self.seg_len[:, None], np.asarray(hw)
        self.nodes, self.centre = nodes, pts
        self.s = np.concatenate([[0], np.cumsum(self.seg_len)])
        self.length = self.s[-1]

    def point(self, s):
        s = np.clip(s, 0, self.length)
        i = min(np.searchsorted(self.s, s, side="right") - 1, len(self.seg_len) - 1)
        return self.A[i] + (s - self.s[i]) * self.T[i], i

    def _closest(self, p):
        t = np.clip(np.einsum("ij,ij->i", p - self.A, self.T), 0, self.seg_len)
        d = np.hypot(*(self.A + t[:, None] * self.T - p).T)
        i = np.argmin(d)
        return d[i], self.s[i] + t[i]

    def distance(self, p):
        return self._closest(p)[0]

    def lookahead(self, p, dist):
        return self.point(self._closest(p)[1] + dist)[0]

    def heading(self):
        return np.arctan2(self.T[0, 1], self.T[0, 0])

    def turn_ahead(self, p, dist):
        """Largest heading change between the current segment and those starting within `dist` ahead."""
        s = self._closest(p)[1]
        i = min(np.searchsorted(self.s, s, side="right") - 1, len(self.seg_len) - 1)
        j = np.flatnonzero((self.s[:-1] > s) & (self.s[:-1] <= s + dist))
        head = np.arctan2(self.T[:, 1], self.T[:, 0])
        return float(np.max(np.abs(slam.wrap(head[j] - head[i])), initial=0.0))


def dijkstra(adj, src, dst, weight):
    dist, prev, pq = {src: 0.0}, {}, [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            break
        if d > dist[u]:
            continue
        for v, length in adj[u]:
            nd = d + weight.get((u, v), length)
            if nd < dist.get(v, np.inf):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    if dst not in prev and dst != src:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return path[::-1]


def make_route(env, path, start_xy=None):
    pts = [env.xy[n] for n in path]
    hw = [env.edge_hw[e] for e in zip(path[:-1], path[1:])]
    lane = [env.edge_lane[e] for e in zip(path[:-1], path[1:])]
    if start_xy is not None:                                   # re-route from the car's position
        pts, hw, lane = [start_xy] + pts, hw[:1] + hw, [0.0] + lane
    r = Route(path, pts, hw, lane)
    r.node_s = r.s[1:] if start_xy is not None else r.s        # arc length of each graph node
    return r


def candidate_routes(env, src, dst, k=6, penalty=1.4, max_ratio=1.8, start_xy=None, blocked=()):
    """Alternative routes by iterative edge penalisation: after each Dijkstra, edges used are
    made `penalty` x more expensive, which pushes the next search onto different streets."""
    weight, routes, seen = {e: np.inf for e in blocked}, [], set()
    for _ in range(k):
        path = dijkstra(env.adj, src, dst, weight)
        if path is None:
            break
        for e in zip(path[:-1], path[1:]):
            weight[e] = weight.get(e, env.edge_len[e]) * penalty
        if tuple(path) not in seen:
            seen.add(tuple(path))
            routes.append(make_route(env, path, start_xy))
    shortest = min(r.length for r in routes)
    return sorted([r for r in routes if r.length <= max_ratio * shortest], key=lambda r: r.length)


def route_risk(env, route, mu0, P0, index, v=6.0):
    """Belief roll-out along the route with MLO (GPS + HD-map landmarks only):
    sum_t Pr(collision | b_t) = expected number of chance-constraint violations."""
    near = [lid for lid in index if lid in env.hd_map and
            np.min(np.hypot(*(route.centre - env.landmarks[lid]).T)) < env.max_range + 30]
    mu, P, sub = slam.marginal(mu0, P0, near, index)
    risk, th = 0.0, route.heading()
    for s in np.arange(0, route.length, v * env.dt):
        p, i = route.point(s)
        th_new = np.arctan2(route.T[i, 1], route.T[i, 0])
        delta = np.arctan(env.model.L * slam.wrap(th_new - th) / (v * env.dt))
        mu, P = slam.predict(mu, P, np.array([v, delta]), env.Q, env.dt, env.model)
        mu[:3], th = [p[0], p[1], th_new], th_new                  # controller tracks the lane
        mu, P = mlo_update(env, mu, P, sub)
        risk += env.risk(mu, P[:2, :2], None)
    return risk


def plan_route(env, filt, src, belief=True, w_risk=300.0, start_xy=None, blocked=()):
    """belief=False -> shortest route;  belief=True -> argmin length + w_risk * expected violations.
    Edges in `blocked` (known road closures) are never used."""
    routes = candidate_routes(env, src, env.goal_node, start_xy=start_xy, blocked=blocked)
    if not belief:
        return routes[0], [(r, None) for r in routes]
    scored = [(r, route_risk(env, r, filt.mu, filt.P, filt.index)) for r in routes]
    return min(scored, key=lambda rk: rk[0].length + w_risk * rk[1])[0], scored


# =================================================================== layer 1b: D* Lite
def edge_cost(env, u, v, P0, w_risk=300.0, c_fail=5000.0, speed=6.0):
    """length + w_risk * sum_t p_t + c_fail * clip(2 max_t p_t - 1, 0, 1)  along street u->v, with p_t the
    chance-constraint probability while rolling the EKF (MLO: GPS + HD-map landmarks) forward from pose
    covariance P0.  The sum prices how risky the street is; the second term is zero unless somewhere on
    the street the car is more likely to hit something than not (p_t > 0.5), i.e. the street is
    effectively blocked, which then makes it prohibitive.  Always >= length (admissible heuristic)."""
    a, b = env.xy[u], env.xy[v]
    L = env.edge_len[(u, v)]
    th = np.arctan2(*(b - a)[::-1])
    hd = [lid for lid, m in env.hd_map.items()
          if np.linalg.norm(m - (a + b) / 2) < env.max_range + L / 2]
    mu = np.r_[a, th, np.ravel([env.hd_map[lid] for lid in hd])]
    P = np.eye(len(mu)) * env.hd_map_std ** 2
    P[:3, :3] = P0
    index = {lid: 3 + 2 * i for i, lid in enumerate(hd)}
    risk, p_max = 0.0, 0.0
    for s in np.arange(speed * env.dt, L, speed * env.dt):
        mu, P = slam.predict(mu, P, np.array([speed, 0.0]), env.Q, env.dt, env.model)
        mu[:3] = np.r_[a + s * (b - a) / L, th]
        mu, P = mlo_update(env, mu, P, index)
        p = env.risk(mu, P[:2, :2], None)
        risk, p_max = risk + p, max(p_max, p)
    return L + w_risk * risk + c_fail * min(max(2.0 * p_max - 1.0, 0.0), 1.0)


class DStarLite:
    """D* Lite: searches backwards from the goal, so the start (the car) can move and edge costs
    can change; only the vertices whose cost-to-goal is affected are re-expanded.
      g(s)   : current cost-to-goal estimate      rhs(s) = min_{s' in succ(s)} c(s, s') + g(s')
      key(s) = [min(g, rhs) + h(start, s) + k_m,  min(g, rhs)]"""

    def __init__(self, succ, pred, cost, h, start, goal):
        self.succ, self.pred, self.c, self.h = succ, pred, cost, h
        self.start, self.goal, self.last, self.km = start, goal, start, 0.0
        self.g, self.rhs = {}, {goal: 0.0}
        self.U, self.inU = [], {}
        self._push(goal)
        self.compute()

    def G(self, s):
        return self.g.get(s, np.inf)

    def RHS(self, s):
        return self.rhs.get(s, np.inf)

    def key(self, s):
        m = min(self.G(s), self.RHS(s))
        return (m + self.h(self.start, s) + self.km, m)

    def _push(self, s):
        k = self.key(s)
        self.inU[s] = k
        heapq.heappush(self.U, (k, s))

    def _top(self):
        while self.U and self.inU.get(self.U[0][1]) != self.U[0][0]:   # drop stale heap entries
            heapq.heappop(self.U)
        return self.U[0][0] if self.U else (np.inf, np.inf)

    def update_vertex(self, u):
        if u != self.goal:
            self.rhs[u] = min((self.c[(u, v)] + self.G(v) for v in self.succ[u]), default=np.inf)
        self.inU.pop(u, None)
        if self.G(u) != self.RHS(u):
            self._push(u)

    def compute(self):
        while self._top() < self.key(self.start) or self.RHS(self.start) != self.G(self.start):
            k_old, u = heapq.heappop(self.U)
            del self.inU[u]
            k_new = self.key(u)
            if k_old < k_new:                                  # key out of date -> re-insert
                self._push(u)
            elif self.G(u) > self.RHS(u):                      # over-consistent -> settle
                self.g[u] = self.RHS(u)
                for p in self.pred[u]:
                    self.update_vertex(p)
            else:                                              # under-consistent -> reset
                self.g[u] = np.inf
                for p in self.pred[u] + [u]:
                    self.update_vertex(p)
            if not self.U:
                break

    def replan(self, start, changed):
        """Move the start, apply changed edge costs {(u, v): c}, repair the search."""
        self.km += self.h(self.last, start)
        self.last = self.start = start
        for (u, v), c in changed.items():
            self.c[(u, v)] = c
            self.update_vertex(u)
        self.compute()

    def path(self, s):
        out = [s]
        while s != self.goal and len(out) < 10000:
            s = min(self.succ[s], key=lambda v: self.c[(s, v)] + self.G(v))
            out.append(s)
        return out

    def path_cost(self, path):
        return sum(self.c[e] for e in zip(path[:-1], path[1:]))


class DStarRouter:
    """EKF-SLAM-aware D* Lite: initial costs from a nominal pose covariance, then every decision the
    streets near the car are re-costed from the car's ACTUAL covariance and the path is repaired."""
    P_NOMINAL = np.diag([0.3, 0.3, 0.02]) ** 2

    def __init__(self, env, radius=60.0, hysteresis=5.0):
        self.env, self.radius, self.hyst = env, radius, hysteresis
        cost = {e: edge_cost(env, *e, self.P_NOMINAL) for e in env.edge_len}
        pred = {n: [] for n in env.adj}
        for u, vs in env.adj.items():
            for v, _ in vs:
                pred[v].append(u)
        succ = {u: [v for v, _ in vs] for u, vs in env.adj.items()}
        h = lambda a, b: float(np.linalg.norm(env.xy[a] - env.xy[b]))
        self.ds = DStarLite(succ, pred, cost, h, env.start_node, env.goal_node)
        self.repairs = 0

    BLOCKED = 1e9                                              # cost of a closed street

    def update(self, filt, start, current=None, blocked=(), came_from=None):
        """Re-cost streets within `radius` of the car from its current pose covariance (closed streets,
        and the U-turn back to `came_from`, get cost BLOCKED); returns the repaired path, or None if the
        current path is still (nearly) optimal."""
        mu, P0 = filt.mu[:2], filt.P[:3, :3]
        uturn = {(start, came_from)} & set(self.env.edge_len) if came_from not in (None, start) else set()
        ban = set(blocked) | uturn
        changed = {e: self.BLOCKED for e in ban if self.ds.c[e] != self.BLOCKED}
        for e in getattr(self, "_banned", set()) - ban:              # lift last decision's U-turn ban
            changed[e] = edge_cost(self.env, *e, P0)
        self._banned = ban
        for (u, v) in self.env.edge_len:
            if (u, v) in ban:
                continue
            if min(np.linalg.norm(self.env.xy[u] - mu), np.linalg.norm(self.env.xy[v] - mu)) < self.radius:
                c = edge_cost(self.env, u, v, P0)
                if abs(c - self.ds.c[(u, v)]) > 0.5:
                    changed[(u, v)] = c
        self.ds.replan(start, changed)
        best = self.ds.path(start)
        if current is None or (best != current and
                               self.ds.path_cost(best) < self.ds.path_cost(current) - self.hyst):
            self.repairs += current is not None
            return best
        return None


# =================================================================== layer 2: local POMDP
class BeliefPlanner:
    def __init__(self, env, actions, aware=True, depth=2, repeat=5, gamma=0.95,
                 w_g=1.0, w_u=2.0, w_c=300.0, w_t=2.0, w_s=0.5, r_goal=50.0, goal_tol=4.0):
        self.env, self.actions, self.aware = env, actions, aware
        self.depth, self.repeat, self.gamma = depth, repeat, gamma
        self.w_g, self.w_u, self.w_c, self.w_t = w_g, (w_u if aware else 0.0), w_c, w_t
        self.w_s, self.r_goal, self.goal_tol = w_s, r_goal, goal_tol
        self.v_prev = 0.0
        self.reach = max(a[0] for a in actions) * env.dt * repeat * depth + env.max_range

    def simulate(self, mu, P, a, index, t):
        """tau(b, a, z*) over one macro-action; also returns the worst Pr(collision) along it."""
        env, risk = self.env, 0.0
        for i in range(self.repeat):
            mu, P = slam.predict(mu, P, control(env, self.route, mu, a), env.Q, env.dt, env.model)
            if self.aware:
                mu, P = mlo_update(env, mu, P, index)
            risk = max(risk, env.risk(mu, P[:2, :2] if self.aware else None, t + (i + 1) * env.dt))
        return mu, P, risk

    def value(self, mu, P, index, t, depth, v_prev):
        """V(b) = max_a [ R(b, a) + gamma V(b') ]  and the maximising action."""
        best_q, best_a = -np.inf, None
        for a in self.actions:
            mu2, P2, risk = self.simulate(mu, P, a, index, t)
            r = self.w_g * (np.linalg.norm(mu[:2] - self.target) - np.linalg.norm(mu2[:2] - self.target))
            r -= self.w_u * np.sqrt(np.trace(P2[:2, :2])) + self.w_c * risk
            r -= self.w_t * self.route.distance(mu2[:2]) + self.w_s * abs(a[0] - v_prev)
            done = np.linalg.norm(mu2[:2] - self.env.goal) < self.goal_tol
            q = r + self.r_goal if done else r
            if not done and depth > 1:
                q += self.gamma * self.value(mu2, P2, index, t + self.repeat * self.env.dt, depth - 1, a[0])[0]
            if q > best_q:
                best_q, best_a = q, a
        return best_q, best_a

    def plan(self, filt, t, route, target):
        """Marginalise onto pose + landmarks within reach (exact for Gaussians), then search."""
        self.route, self.target = route, target
        near = [lid for lid, k in filt.index.items()
                if np.linalg.norm(filt.mu[k:k + 2] - filt.mu[:2]) < self.reach]
        mu, P, sub = slam.marginal(filt.mu, filt.P, near, filt.index)
        a = self.value(mu, P, sub, t, self.depth, self.v_prev)[1]
        self.v_prev = a[0]
        return a
