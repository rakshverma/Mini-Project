"""Autonomous car on real OpenStreetMap streets of Dharwad:  EKF-SLAM (slam.py) + two-layer POMDP (pomdp.py).

  python simulate.py                 # run the three stacks once      -> results/dharwad_run.png
  python simulate.py --gif           # + animation of the full stack  -> results/dharwad_full.gif
  python simulate.py --compare 20    # Monte-Carlo over 20 seeds, all CPU cores -> results/dharwad_static_*.md
  python simulate.py --closure       # same, with a road closure the car discovers en route
  python simulate.py --obstruction   # stalled vehicles block the road: only D* Lite re-costs and detours

Stacks (all use the same EKF-SLAM)
  full   : PROPOSED  - belief-aware route + POMDP local planner (EKF-SLAM + POMDP)
  dstar  : EXTENSION - route layer replaced by D* Lite on belief costs, repaired every second
  local  : ABLATION  - shortest route + POMDP local planner
  naive  : BASELINE  - shortest route + certainty-equivalent planner (trusts the SLAM mean, no POMDP)
"""
import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Patch, Polygon
import slam
import pomdp
from environment.dharwad import Dharwad, NAME

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ACTIONS = [np.array([0.0, 0.0])] + [np.array(a) for a in       # (speed, steering offset); (0, 0) = stop / yield
                                    itertools.product([1.5, 3.0, 6.0, 9.0], [-0.15, 0.0, 0.15])]
REVERSE = [np.array([-1.5, d]) for d in (-0.5, 0.0, 0.5)]        # recovery: reverse with a fixed steering angle
STACKS = {"full": ("brm", True), "dstar": ("dstar", True),                      # (route layer, POMDP aware)
          "local": ("shortest", True), "naive": ("shortest", False)}
LOOKAHEAD, REROUTE_DIST, MAX_TIME = 12.0, 6.0, 240.0


def speed_limit(route, mu):
    """Curvature-based speed planning  v <= sqrt(a_lat * R):  junction turns have R ~ 5-10 m.
    The turn still to be made = max(turns ahead on the route, car heading vs. route direction)."""
    i = route.point(route._closest(mu[:2])[1])[1]
    heading_err = abs(slam.wrap(np.arctan2(route.T[i, 1], route.T[i, 0]) - mu[2]))
    turn = np.degrees(max(route.turn_ahead(mu[:2], 30.0), heading_err))
    v = 1.5 if turn > 60 else (3.0 if turn > 45 else (6.0 if turn > 20 else 9.0))      # crawl through sharp turns
    off = route.distance(mu[:2])                                   # back to the lane centre before speeding up
    return min(v, 1.5 if off > 2.0 else (3.0 if off > 1.0 else 9.0))


# ============================================================== closed loop
def node_ahead(env, p, th):
    """Nearest street node in front of the car (start of a re-route)."""
    h = np.array([np.cos(th), np.sin(th)])
    best = min(((np.linalg.norm(q - p), n) for n, q in env.xy.items()
                if (q - p) @ h > 3.0 and env.adj[n]), default=(None, env.nearest_node(p)))
    return best[1]


def run(seed, stack, frames=None, closure=False, obstruction=False):
    """plan route -> loop { local POMDP -> act -> sense -> EKF-SLAM } until goal / crash / timeout.
    Also logs the wall-clock time of every module (ms) and per-tick accuracy."""
    env = Dharwad(seed, closure, obstruction)
    mode, aware = STACKS[stack]
    belief_route = mode == "brm"
    filt = slam.EKFSLAM(env.x, env.Q, env.R, env.dt, env.hd_map, env.hd_map_std, env.model)
    tm = dict(route_init=[], route=[], pomdp=[], slam=[], control=[])
    clock = time.perf_counter
    t0 = clock()
    if mode == "dstar":
        router = pomdp.DStarRouter(env)
        route, scored = pomdp.make_route(env, router.ds.path(env.start_node)), []
    else:
        route, scored = pomdp.plan_route(env, filt, env.start_node, belief_route)
    tm["route_init"].append(1e3 * (clock() - t0))
    env.x[2] = filt.mu[2] = route.heading()              # car waits facing its first street
    planner = pomdp.BeliefPlanner(env, ACTIONS, aware)
    log = dict(true=[env.x.copy()], est=[filt.mu[:3].copy()], sig=[0.0], speed=[0.0], gps=[True],
               routes=[route], scored=scored, status="timeout", track=[0.0], timing=tm, t_detect=None)
    t, last_node, t_max = 0.0, None, MAX_TIME * (1.75 if closure or obstruction else 1.0)  # detours
    stuck, visited = 0.0, [env.start_node]                        # seconds stopped in a row, nodes passed
    while t < t_max and log["status"] == "timeout":
        blocked = env.closed if env.closure_known else set()
        if blocked and log["t_detect"] is None:
            log["t_detect"] = t
        off_route = route.distance(filt.mu[:2]) > REROUTE_DIST
        uses_closed = bool(blocked & set(zip(route.nodes[:-1], route.nodes[1:])))
        s_car = route._closest(filt.mu[:2])[1]
        passed = [n for n, s in zip(route.nodes, route.node_s) if s <= s_car + 1.0]
        if passed and passed[-1] != last_node:
            visited.append(passed[-1])
        last_node = visited[-1] if visited else None
        t0 = clock()
        if mode == "dstar":                              # D* Lite repair from the next node ahead
            ahead = [n for n, s in zip(route.nodes, route.node_s) if s > s_car + 1.0]
            start = node_ahead(env, filt.mu[:2], filt.mu[2]) if off_route or not ahead else ahead[0]
            came = next((n for n in reversed(visited) if n != start), None)   # no U-turn back to it
            path = router.update(filt, start, None if off_route or not ahead else ahead, blocked, came)
            if path is not None:
                route = pomdp.make_route(env, path, start_xy=filt.mu[:2].copy())
                log["routes"].append(route)
            tm["route"].append(1e3 * (clock() - t0))
        elif off_route or uses_closed:                   # drifted off the route / route closed -> re-route
            ahead = [n for n, s in zip(route.nodes, route.node_s) if s > s_car + 1.0]
            start = ahead[0] if uses_closed and not off_route and ahead else \
                node_ahead(env, filt.mu[:2], filt.mu[2])
            came = next((n for n in reversed(visited) if n != start), None)
            ban = set(blocked) | ({(start, came)} if came is not None else set())
            route, _ = pomdp.plan_route(env, filt, start, belief_route, start_xy=filt.mu[:2].copy(),
                                        blocked=ban)
            log["routes"].append(route)
            tm["route"].append(1e3 * (clock() - t0))
        v_max = speed_limit(route, filt.mu)
        planner.actions = [a for a in ACTIONS if a[0] <= v_max] + (REVERSE if stuck >= 3.0 else [])
        t0 = clock()
        a = planner.plan(filt, t, route, route.lookahead(filt.mu[:2], LOOKAHEAD))
        tm["pomdp"].append(1e3 * (clock() - t0))
        stuck = stuck + planner.repeat * env.dt if a[0] == 0 else 0.0
        for _ in range(planner.repeat):
            t0 = clock()
            u = pomdp.control(env, route, filt.mu, a)          # closed-loop macro-action
            tm["control"].append(1e3 * (clock() - t0))
            env.step(u)
            t += env.dt
            obs, gps = env.observe(), env.gps()
            t0 = clock()
            filt.step(u, obs, gps)
            tm["slam"].append(1e3 * (clock() - t0))
            log["true"].append(env.x.copy()); log["est"].append(filt.mu[:3].copy())
            log["sig"].append(filt.pose_sigma); log["speed"].append(u[0])
            log["gps"].append(env.gps_sigma(env.x[:2])); log["track"].append(route.distance(env.x[:2]))
            hit = env.collided(t)
            if hit:
                log["status"] = hit
                break
        if log["status"] == "timeout" and np.linalg.norm(filt.mu[:2] - env.goal) < planner.goal_tol:
            log["status"] = "reached"
        if frames is not None:
            learned = np.array([filt.mu[k:k + 2] for lid, k in filt.index.items() if lid not in env.hd_map])
            frames.append(dict(t=t, n=len(log["true"]), mu=filt.mu[:2].copy(), Pxy=filt.P[:2, :2].copy(),
                               movers=env.movers_at(t), seen=[lid for lid, _ in obs], gps=gps, action=a,
                               learned=learned.reshape(-1, 2)))
    log.update(time=t, final_err=float(np.linalg.norm(env.x[:2] - env.goal)), env=env, filt=filt)
    return log


# ============================================================== drawing
def car_patch(x, y, th, color, L=4.4, W=1.8):
    c, s = np.cos(th), np.sin(th)
    corners = np.array([[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2]])
    return Polygon(np.array([x, y]) + corners @ np.array([[c, s], [-s, c]]), color=color, zorder=9)


def ellipse(mean, cov, k=3.0, **kw):
    vals, vecs = np.linalg.eigh(cov)
    return Ellipse(mean, *(2 * k * np.sqrt(np.maximum(vals[::-1], 0))),
                   angle=np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])), fill=False, **kw)


# colours used everywhere, so the legend matches every plot
C = dict(ego="#1f5fff", est="orange", route="cyan", reroute="magenta", unc="red", traffic="#14b8c4",
         parked="#c23bd4", corner="#7a4e2d", hd="gold", learned="red", lidar="#2ca02c",
         gps4="#f6c26b", nogps="#e8705f", building="#bdb5a6", road="#55595e")
STACK_TITLE = {"full": "PROPOSED: EKF-SLAM + POMDP",
               "dstar": "EXTENSION: + D* Lite route layer",
               "local": "ABLATION: shortest route + POMDP",
               "naive": "BASELINE: EKF-SLAM, no POMDP"}


def legend_handles(lidar=False):
    """One plain-language legend for every map."""
    h = [Patch(fc=C["building"], ec="#9c9384", label="Building (OpenStreetMap)"),
         Patch(fc=C["road"], label="Street (OpenStreetMap)"),
         Patch(fc=C["gps4"], alpha=0.6, label="Weak GPS, about 4 m error (many buildings around)"),
         Patch(fc=C["nogps"], alpha=0.6, label="No GPS signal (dense buildings)"),
         Line2D([], [], ls="", marker="o", ms=5, c=C["corner"],
                label="Building corner the lidar can see.\nNot on any map: SLAM must learn it"),
         Line2D([], [], ls="", marker="+", ms=8, mew=2, c=C["learned"], label="Building corner as learned by SLAM"),
         Line2D([], [], ls="", marker="o", ms=7, c=C["hd"], mec="k",
                label="Surveyed landmark (pre-loaded HD map)"),
         Line2D([], [], ls="", marker="o", ms=7, c=C["parked"], label="Parked car"),
         Patch(fc=C["traffic"], label="Moving traffic (keeps left)"),
         Patch(fc="#e8590c", label="Stalled vehicles (obstruction runs; seen only within 25 m)"),
         Line2D([], [], ls="", marker="s", ms=9, c="lime", mec="k", label="Start"),
         Line2D([], [], ls="", marker="*", ms=14, c="red", mec="k", label="Goal"),
         Line2D([], [], c=C["route"], lw=5, alpha=0.5, label="Route chosen by the planner"),
         Line2D([], [], c=C["reroute"], lw=3, alpha=0.5, label="New route after drifting off"),
         Line2D([], [], c=C["ego"], lw=2, label="Where the car really drove"),
         Line2D([], [], c=C["est"], lw=1.5, ls="--", label="Where the car thinks it is (EKF-SLAM)"),
         Line2D([], [], ls="", marker="o", ms=13, mfc="none", mec=C["unc"], mew=1.5,
                label="Car's position uncertainty (3-sigma)"),
         Line2D([], [], ls="", marker="o", ms=13, mfc="none", mec=C["ego"], mew=2, label="Our car")]
    if lidar:
        h.append(Line2D([], [], c=C["lidar"], lw=1, label="Lidar sees this landmark now"))
    return h


def draw_map(ax, env, view):
    """Static layer: GPS quality, buildings, streets, landmarks, parked cars."""
    (x0, x1), (y0, y1) = view
    xs, ys = np.arange(x0, x1, 6.0), np.arange(y0, y1, 6.0)
    g = np.array([[{1.5: 0, 4.0: 1, None: 2}[env.gps_sigma(np.array([x, y]))] for x in xs] for y in ys])
    ax.set_facecolor("#eef0e6")
    ax.contourf(xs, ys, g, levels=[0.5, 1.5, 2.5], colors=[C["gps4"], C["nogps"]], alpha=0.35, zorder=0)
    ax.add_collection(PatchCollection([Polygon(b) for b in env.buildings], fc=C["building"], ec="#9c9384",
                                      lw=0.4, zorder=1))
    roads = []
    for a, b, hw in zip(env.A, env.B, env.HW):
        t = (b - a) / np.linalg.norm(b - a)
        n = np.array([-t[1], t[0]]) * hw
        roads += [Polygon([a + n, b + n, b - n, a - n]), Circle(a, hw), Circle(b, hw)]
    ax.add_collection(PatchCollection(roads, fc=C["road"], ec="none", zorder=2))
    for pts, hw, tags in env.ways:
        if tags.get("name") and hw >= 5:
            mid = pts[len(pts) // 2]
            if x0 < mid[0] < x1 and y0 < mid[1] < y1:
                ax.text(*mid, tags["name"], fontsize=7, color="#1d3c78", zorder=11, weight="bold")
    corners = np.array([m for i, m in enumerate(env.landmarks) if i not in env.hd_map])
    ax.scatter(*corners.T, s=8, c=C["corner"], zorder=3)
    ax.scatter(*np.array(list(env.hd_map.values())).T, s=22, c=C["hd"], ec="k", lw=0.5, zorder=3)
    ax.add_collection(PatchCollection([Circle(p, 1.0) for p in env.parked], fc=C["parked"], zorder=4))
    for p in env.hazards:                                             # stalled vehicles (obstruction runs)
        ax.add_patch(car_patch(*p, 0.6, "#e8590c"))
    ax.plot(*env.xy[env.start_node], "s", c="lime", mec="k", ms=10, zorder=10)
    ax.plot(*env.goal, "*", c="red", mec="k", ms=18, zorder=10)
    ax.set(xlim=view[0], ylim=view[1], aspect="equal", xlabel="metres east", ylabel="metres north")


def view_of(logs, pad=40.0):
    pts = np.vstack([r.centre for L in logs for r in L["routes"]] + [np.array(L["true"])[:, :2] for L in logs])
    return (pts[:, 0].min() - pad, pts[:, 0].max() + pad), (pts[:, 1].min() - pad, pts[:, 1].max() + pad)


def draw_car(ax, x, mu, Pxy):
    """Our car (true pose), a ring so it is visible at map scale, and its 3-sigma belief ellipse."""
    return [ax.add_patch(car_patch(*x, C["ego"])),
            *ax.plot(*x[:2], "o", ms=16, mfc="none", mec=C["ego"], mew=2, zorder=10),
            ax.add_patch(ellipse(mu, Pxy, color=C["unc"], lw=1.5, zorder=10))]


def draw_run(ax, L, view, title):
    env, filt = L["env"], L["filt"]
    draw_map(ax, env, view)
    for i, r in enumerate(L["routes"]):
        ax.plot(*r.A.T, c=C["route"] if i == 0 else C["reroute"], lw=5 if i == 0 else 3, alpha=0.45, zorder=5)
    learned = np.array([filt.mu[k:k + 2] for lid, k in filt.index.items() if lid not in env.hd_map])
    if len(learned):
        ax.plot(*learned.T, "+", c=C["learned"], ms=6, mew=1.5, zorder=6)
    tr, es = np.array(L["true"]), np.array(L["est"])
    ax.plot(*tr[:, :2].T, c=C["ego"], lw=1.8, zorder=7)
    ax.plot(*es[:, :2].T, "--", c=C["est"], lw=1.3, zorder=7)
    draw_car(ax, tr[-1], filt.mu[:2], filt.P[:2, :2])
    for p, h in zip(*env.movers_at(L["time"])):
        ax.add_patch(car_patch(*p, h, C["traffic"]))
    ax.set_title(title, fontsize=10)


def single(seed, gif, closure=False, obstruction=False):
    logs = {}
    for stack in STACKS:
        t0 = time.time()
        L = logs[stack] = run(seed, stack, closure=closure, obstruction=obstruction)
        print(f"{stack:6s} route {L['routes'][0].length:5.0f} m  re-routes={len(L['routes']) - 1}  {L['status']:8s} "
              f"t={L['time']:5.1f}s  true dist to goal={L['final_err']:6.1f} m   ({time.time() - t0:.0f}s cpu)")
    view = view_of(list(logs.values()))
    fig = plt.figure(figsize=(24, 15))
    fig.suptitle(f"{NAME}  (real OpenStreetMap streets)  -  autonomous car using lidar + GPS -> EKF-SLAM -> POMDP,"
                 f" seed {seed}", fontsize=13)
    gs = fig.add_gridspec(4, 12, height_ratios=[1, 1, 0.42, 0.75])
    for i, (stack, L) in enumerate(logs.items()):
        draw_run(fig.add_subplot(gs[0:2, 3 * i:3 * i + 3]), L, view,
                 f"{STACK_TITLE[stack]}\n{L['status']} at t = {L['time']:.0f} s, {L['final_err']:.0f} m from goal")
    lg = fig.add_subplot(gs[2, :]); lg.axis("off")
    lg.legend(handles=legend_handles(), loc="center", ncol=4, fontsize=9, frameon=True, title="Legend")
    for j, (key, ttl) in enumerate([("err", "Position error: true vs. SLAM estimate [m]"),
                                    ("sig", "How unsure the car is: 1-sigma position [m]"),
                                    ("speed", "Speed chosen by the planner [m/s]")]):
        ax = fig.add_subplot(gs[3, 4 * j:4 * j + 4])
        for stack, L in logs.items():
            tt = np.arange(len(L["true"])) * Dharwad.dt
            y = (np.linalg.norm(np.array(L["true"])[:, :2] - np.array(L["est"])[:, :2], axis=1)
                 if key == "err" else L[key])
            ax.plot(tt, y, label=stack)
        ax.set(title=ttl, xlabel="time [s]"); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    out = os.path.join(RESULTS, "dharwad_closure_run.png" if closure else
                       ("dharwad_obstruction_run.png" if obstruction else "dharwad_run.png"))
    plt.savefig(out, dpi=100); print("saved", out)
    if gif:
        make_gif(seed)


def make_gif(seed):
    frames = []
    L = run(seed, "full", frames)
    env, tr = L["env"], np.array(L["true"])
    fig = plt.figure(figsize=(11.5, 9))
    ax = fig.add_axes([0.07, 0.06, 0.5, 0.84])
    draw_map(ax, env, view_of([L]))
    for r in L["routes"]:
        ax.plot(*r.A.T, c=C["route"], lw=5, alpha=0.45, zorder=5)
    fig.legend(handles=legend_handles(lidar=True), loc="center left", bbox_to_anchor=(0.58, 0.5),
               fontsize=8.5, title="Legend")
    title = fig.text(0.5, 0.985, "", ha="center", va="top", fontsize=10.5)
    dyn = []

    def frame(k):
        for a in dyn:
            a.remove()
        dyn.clear()
        f = frames[k]
        x = tr[f["n"] - 1]
        for lid in f["seen"]:                                    # lidar returns of the last scan
            dyn.extend(ax.plot(*np.c_[x[:2], env.landmarks[lid]], c=C["lidar"], lw=1.4, zorder=8))
        if len(f["learned"]):
            dyn.extend(ax.plot(*f["learned"].T, "+", c=C["learned"], ms=6, mew=1.5, zorder=6))
        dyn.extend(ax.plot(*tr[:f["n"], :2].T, c=C["ego"], lw=1.8, zorder=7))
        dyn.extend(draw_car(ax, x, f["mu"], f["Pxy"]))
        dyn.extend(ax.add_patch(car_patch(*p, h, C["traffic"])) for p, h in zip(*f["movers"]))
        gps = "no signal" if f["gps"] is None else f"{f['gps'][1]:.1f} m error"
        v, off = f["action"]
        n_seen = len(f["seen"])
        title.set_text(f"{NAME} - EKF-SLAM + POMDP     t = {f['t']:5.1f} s\n"
                       f"SENSE  lidar sees {n_seen} landmark{'' if n_seen == 1 else 's'},  GPS: {gps}     "
                       f"BELIEVE  uncertainty {np.sqrt(np.trace(f['Pxy'])):.2f} m     DECIDE  POMDP action: "
                       f"{'STOP / give way' if v == 0 else f'{v:.0f} m/s, steer offset {off:+.2f} rad'}")
        return dyn
    out = os.path.join(RESULTS, "dharwad_full.gif")
    FuncAnimation(fig, frame, frames=len(frames)).save(out, writer=PillowWriter(fps=8))
    print("saved", out)


def _summary(job):
    """One Monte-Carlo run, reduced to what the tables need (cheap to send between processes)."""
    seed, stack, closure, obstruction = job
    L = run(seed, stack, closure=closure, obstruction=obstruction)
    tr, es = np.array(L["true"]), np.array(L["est"])
    return dict(stack=stack, status=L["status"], time=L["time"], length=L["routes"][0].length,
                repairs=len(L["routes"]) - 1, timing=L["timing"],
                pos_err=np.linalg.norm(tr[:, :2] - es[:, :2], axis=1),
                head_err=np.degrees(np.abs(slam.wrap(tr[:, 2] - es[:, 2]))),
                track_err=np.array(L["track"]), sigma=np.array(L["sig"]))


def _stats(x):
    x = np.asarray(x, float)
    return [x.mean(), x.std(), x.min(), x.max(), np.percentile(x, 95)] if len(x) else [np.nan] * 5


def compare(n, closure=False, obstruction=False):
    """All runs are independent, so they are spread over every CPU core."""
    tag = "closure" if closure else ("obstruction" if obstruction else "static")
    jobs, res = [(s, stack, closure, obstruction) for stack in STACKS for s in range(n)], {}
    while len(res) < len(jobs):                       # a crashed worker breaks the pool: retry what is left
        todo = [j for j in jobs if j not in res]
        try:
            with ProcessPoolExecutor(max_workers=6) as ex:
                for j, r in zip(todo, ex.map(_summary, todo)):
                    res[j] = r
        except BrokenProcessPool:
            print(f"worker crashed; retrying {len(jobs) - len(res)} runs", flush=True)
    res = [res[j] for j in jobs]
    rows = ["| stack | success | off-road | hit car / barrier | timeout | mean time (success) | mean route | route repairs |",
            "|---|---|---|---|---|---|---|---|"]
    for stack in STACKS:
        R = [r for r in res if r["stack"] == stack]
        st = [r["status"] for r in R]
        ok = [r["time"] for r in R if r["status"] == "reached"]
        hit = st.count("hit car") + st.count("hit barrier")
        rows.append(f"| {stack} | {st.count('reached') / n:.0%} | {st.count('off-road') / n:.0%} | {hit / n:.0%} | "
                    f"{st.count('timeout') / n:.0%} | {np.mean(ok) if ok else float('nan'):.0f} s | "
                    f"{np.mean([r['length'] for r in R]):.0f} m | {np.mean([r['repairs'] for r in R]):.1f} |")
        print(rows[-1])
    # module timing (ms) and accuracy of every stack, for the report
    bench = [f"# {NAME} - {tag} scenario, {n} seeds per stack\n", "\n".join(rows), ""]
    for stack in STACKS:
        R = [r for r in res if r["stack"] == stack]
        bench += [f"\n## {stack}: module wall-clock time per call [ms]  (mean, std, min, max, p95)\n"]
        for k in ("slam", "control", "pomdp", "route", "route_init"):
            vals = np.concatenate([r["timing"][k] for r in R if r["timing"][k]] or [[]])
            bench.append(f"- {k:10s} calls={len(vals):6d}  " + "  ".join(f"{v:9.3f}" for v in _stats(vals)))
        bench += [f"\n## {stack}: accuracy over all ticks  (mean, std, min, max, p95)\n"]
        for k, unit in (("pos_err", "m"), ("head_err", "deg"), ("track_err", "m"), ("sigma", "m")):
            bench.append(f"- {k:9s} [{unit:3s}] " + "  ".join(f"{v:8.3f}" for v in
                                                              _stats(np.concatenate([r[k] for r in R]))))
    open(os.path.join(RESULTS, f"dharwad_{tag}_benchmark.md"), "w").write("\n".join(bench) + "\n")
    out = os.path.join(RESULTS, f"dharwad_{tag}_compare.md")
    open(out, "w").write(f"# {NAME} - {tag} scenario, Monte-Carlo over {n} seeds\n\n" + "\n".join(rows) + "\n")
    print("saved", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gif", action="store_true")
    p.add_argument("--compare", type=int, default=0, metavar="N")
    p.add_argument("--closure", action="store_true", help="road closure on Saptapur Rd, found by lidar en route")
    p.add_argument("--obstruction", action="store_true", help="stalled vehicles block Saptapur Rd (not a closure)")
    a = p.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    compare(a.compare, a.closure, a.obstruction) if a.compare else single(a.seed, a.gif, a.closure, a.obstruction)
