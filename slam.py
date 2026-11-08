"""SLAM: vehicle motion model + EKF-SLAM belief  b = N(mu, P)  over  s = [x, y, th, m1x, m1y, ...].

All filter steps are pure functions (mu, P in -> mu, P out) so the POMDP planner can reuse
them to *simulate* future beliefs without touching the real filter.
"""
import numpy as np


def wrap(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class Bicycle:
    """Kinematic bicycle (car), rear-axle reference, wheelbase L.  u = [v, delta]:
         x' = x + v dt cos th,   y' = y + v dt sin th,   th' = th + (v / L) tan(delta) dt"""
    L = 2.7

    @classmethod
    def f(cls, x, u, dt):
        v, d = u
        return np.array([x[0] + v * dt * np.cos(x[2]), x[1] + v * dt * np.sin(x[2]),
                         wrap(x[2] + v / cls.L * np.tan(d) * dt)])

    @classmethod
    def jac(cls, x, u, dt):
        """G = df/dx (3x3),  V = df/du (3x2)."""
        v, d = u
        th = x[2]
        G = np.array([[1, 0, -v * dt * np.sin(th)], [0, 1, v * dt * np.cos(th)], [0, 0, 1]])
        V = np.array([[dt * np.cos(th), 0],
                      [dt * np.sin(th), 0],
                      [dt * np.tan(d) / cls.L, v * dt / (cls.L * np.cos(d) ** 2)]])
        return G, V


# ------------------------------------------------------------------ EKF steps
def predict(mu, P, u, Q, dt, model=Bicycle):
    """P_xx <- G P_xx G^T + V Q V^T,   P_xm <- G P_xm   (landmarks are static).
    Q is the actuation-noise covariance, or a function Q(u) of the command."""
    Q = Q(u) if callable(Q) else Q
    G, V = model.jac(mu[:3], u, dt)
    mu, P = mu.copy(), P.copy()
    mu[:3] = model.f(mu[:3], u, dt)
    P[:3, :3] = G @ P[:3, :3] @ G.T + V @ Q @ V.T
    P[:3, 3:] = G @ P[:3, 3:]
    P[3:, :3] = P[:3, 3:].T
    return mu, P


def measure(mu, k):
    """Range-bearing  z_hat = h(mu) = [sqrt(q), atan2(dy, dx) - th]  to the landmark at index k, and H = dh/ds."""
    d = mu[k:k + 2] - mu[:2]
    q = d @ d
    r = np.sqrt(q)
    z_hat = np.array([r, wrap(np.arctan2(d[1], d[0]) - mu[2])])
    H = np.zeros((2, len(mu)))
    H[:, :3] = [[-d[0] / r, -d[1] / r, 0], [d[1] / q, -d[0] / q, -1]]
    H[:, k:k + 2] = [[d[0] / r, d[1] / r], [-d[1] / q, d[0] / q]]
    return z_hat, H


def correct(mu, P, nu, H, R):
    """S = H P H^T + R,  K = P H^T S^-1,  mu += K nu,  P -= K S K^T   (O(n^2), symmetrised)."""
    S = H @ P @ H.T + R
    PHt = P @ H.T
    K = PHt @ np.linalg.inv(S)
    mu = mu + K @ nu
    mu[2] = wrap(mu[2])
    P = P - K @ PHt.T
    return mu, 0.5 * (P + P.T)


def update(mu, P, k, z, R):
    """Lidar update with landmark k;  innovation nu = z - h(mu) (bearing wrapped)."""
    z_hat, H = measure(mu, k)
    nu = z - z_hat
    nu[1] = wrap(nu[1])
    return correct(mu, P, nu, H, R)


def update_gps(mu, P, z, sigma):
    """GPS fix  z = [x, y] + N(0, sigma^2 I)  ->  linear  H = [I_2  0]."""
    H = np.zeros((2, len(mu)))
    H[:, :2] = np.eye(2)
    return correct(mu, P, z - mu[:2], H, np.eye(2) * sigma ** 2)


def add_landmark(mu, P, z, R):
    """New landmark  m = p + r [cos(th+b), sin(th+b)]:   P_mm = Gx P_xx Gx^T + Gz R Gz^T,  P_m* = Gx P_x*."""
    r, b = z
    a = mu[2] + b
    m = mu[:2] + r * np.array([np.cos(a), np.sin(a)])
    Gx = np.array([[1, 0, -r * np.sin(a)], [0, 1, r * np.cos(a)]])
    Gz = np.array([[np.cos(a), -r * np.sin(a)], [np.sin(a), r * np.cos(a)]])
    P_mx = Gx @ P[:3, :]
    P_mm = Gx @ P[:3, :3] @ Gx.T + Gz @ R @ Gz.T
    return np.concatenate([mu, m]), np.block([[P, P_mx.T], [P_mx, P_mm]])


def marginal(mu, P, keep_ids, index):
    """Exact Gaussian marginal on pose + chosen landmarks (keep the sub-vector / sub-block).
    Returns (mu_sub, P_sub, index_sub)."""
    keep, sub = [0, 1, 2], {}
    for lid in keep_ids:
        k = index[lid]
        sub[lid] = len(keep)
        keep += [k, k + 1]
    return mu[keep], P[np.ix_(keep, keep)], sub


# ------------------------------------------------------------------ stateful filter
class EKFSLAM:
    """The car's online filter.  index[id] = position of landmark id inside mu."""

    def __init__(self, x0, Q, R, dt, prior_map=None, prior_std=0.1, model=Bicycle):
        self.Q, self.R, self.dt, self.model, self.index = Q, R, dt, model, {}
        n = 3 + 2 * len(prior_map or {})
        self.mu, self.P = np.zeros(n), np.eye(n) * prior_std ** 2
        self.mu[:3], self.P[:3, :3] = x0, np.eye(3) * 1e-6
        for j, (lid, m) in enumerate((prior_map or {}).items()):       # HD-map landmarks
            self.index[lid] = 3 + 2 * j
            self.mu[3 + 2 * j:5 + 2 * j] = m

    def step(self, u, observations, gps=None):
        self.mu, self.P = predict(self.mu, self.P, u, self.Q, self.dt, self.model)
        if gps is not None:
            self.mu, self.P = update_gps(self.mu, self.P, *gps)
        for lid, z in observations:
            if lid in self.index:
                self.mu, self.P = update(self.mu, self.P, self.index[lid], z, self.R)
            else:
                self.index[lid] = len(self.mu)
                self.mu, self.P = add_landmark(self.mu, self.P, z, self.R)

    @property
    def pose_sigma(self):
        return float(np.sqrt(np.trace(self.P[:2, :2])))
