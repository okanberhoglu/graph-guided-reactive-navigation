import numpy as np

from model.robot_config import CarConfig

def dd_step(x, u, dt):
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)

    single = x.ndim == 1

    if single:
        x = x[None, :]
        u = u[None, :]

    px = x[:, 0]
    py = x[:, 1]
    th = x[:, 2]

    v = u[:, 0]
    w = u[:, 1]

    eps = 1e-6
    small = np.abs(w) < eps

    w_safe = np.where(small, 1.0, w)

    th_arc = th + w * dt

    px_arc = (px + (v / w_safe) * (np.sin(th_arc) - np.sin(th)))

    py_arc = (py - (v / w_safe) * (np.cos(th_arc) - np.cos(th)))

    px_straight = px + v * dt * np.cos(th)
    py_straight = py + v * dt * np.sin(th)

    px_next = np.where(small, px_straight, px_arc)

    py_next = np.where(small, py_straight, py_arc)

    th_next = np.where(small, th, th_arc)

    out = np.stack((px_next, py_next, th_next), axis=1)

    if single:
        return out[0]

    return out

def ackerman_dd_step(x, u, dt):
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)

    single = x.ndim == 1

    if single:
        x = x[None, :]
        u = u[None, :]

    px = x[:, 0]
    py = x[:, 1]
    th = x[:, 2]
    v = x[:, 3]

    delta = u[:, 0]
    speed = u[:, 1]
    acc = u[:, 2]

    # acceleration towards the target speed, clamped when a limit is given
    a = (speed - v) / dt
    a = np.where(acc > 0.0, np.clip(a, -acc, acc), a)
    v_next = v + a * dt

    # signed distance along the path, exact for constant a
    s = 0.5 * (v + v_next) * dt

    lf = CarConfig.lf
    lr = CarConfig.lr

    beta = np.arctan(lr / (lf + lr) * np.tan(delta))
    kappa = np.sin(beta) / lr
    d_th = kappa * s

    # direction of the velocity, theta + beta
    course = th + beta

    eps = 1e-6
    small = np.abs(kappa) < eps

    kappa_safe = np.where(small, 1.0, kappa)

    px_arc = px + (np.sin(course + d_th) - np.sin(course)) / kappa_safe
    py_arc = py - (np.cos(course + d_th) - np.cos(course)) / kappa_safe

    px_straight = px + s * np.cos(course)
    py_straight = py + s * np.sin(course)

    px_next = np.where(small, px_straight, px_arc)
    py_next = np.where(small, py_straight, py_arc)
    th_next = th + d_th

    out = np.stack((px_next, py_next, th_next, v_next), axis=1)

    if single:
        return out[0]

    return out
