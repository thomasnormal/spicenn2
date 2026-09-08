"""Energy delivered by metered voltage sources (SI units throughout)."""
import numpy as np


def integrate_power(time, power, start, stop):
    """Integrate piecewise-linear power, with exact clipping at zero crossings.

    power has shape (samples, sources). Positive means delivered to the DUT.
    Returned energy is per source. Returned energy is never used as a credit
    against gross delivered energy.
    """
    time = np.asarray(time, dtype=float)
    power = np.asarray(power, dtype=float)
    if (time.ndim != 1 or len(time) < 2 or power.ndim != 2
            or power.shape[0] != len(time) or not np.isfinite(time).all()
            or not np.isfinite(power).all() or np.any(np.diff(time) <= 0)):
        raise ValueError("invalid power trace")
    if not time[0] <= start < stop <= time[-1]:
        raise ValueError("energy window is outside the trace")
    middle = (time > start) & (time < stop)
    t = np.r_[start, time[middle], stop]
    p = np.vstack(([np.interp(start, time, col) for col in power.T],
                   power[middle],
                   [np.interp(stop, time, col) for col in power.T]))
    a, b = p[:-1], p[1:]
    dt = np.diff(t)[:, None]
    positive = dt * (np.maximum(a, 0) + np.maximum(b, 0)) / 2
    crossing = (a * b) < 0
    denominator = np.abs(a) + np.abs(b)
    fraction = np.divide(np.maximum(a, b), denominator,
                         out=np.zeros_like(a), where=denominator != 0)
    positive = np.where(crossing, dt * np.maximum(a, b) * fraction / 2, positive)
    net = (dt * (a + b) / 2).sum(axis=0)
    delivered = positive.sum(axis=0)
    return {"delivered_j": delivered, "returned_j": delivered - net, "net_j": net}
