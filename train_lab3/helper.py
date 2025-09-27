import numpy as np

from scipy.interpolate import (
    PchipInterpolator,
    CubicSpline,
    UnivariateSpline,
    Akima1DInterpolator,
)
def generate_random_target_speeds(terminate_time, num_points=4):
    time_points = np.sort(np.random.uniform(0, terminate_time, size=num_points))
    speed_points = np.random.uniform(0, 50, size=num_points)
    speed_points[0] = 0.0  # start from 0
    return {"times": time_points.tolist(), "speeds": speed_points.tolist()}


def interp_1d(x: float, xp, fp, interp_method, interp_s) -> float:
    """Interpolation helper with multiple methods.
    Methods:
        - 'linear': piecewise linear (np.interp)
        - 'nearest': nearest neighbor
        - 'previous' / 'zoh' / 'step': zero-order hold (previous sample)
        - 'next': next-step hold (right continuous)
        - 'pchip': monotone cubic (PCHIP, no overshoot)
        - 'cubic': natural cubic spline
        - 'univariate': smoothed spline (UnivariateSpline), uses self.interp_s
        - 'akima': Akima1DInterpolator (robust to local oscillations)
    Boundary behavior: clamp to first/last fp.
    """
    xp_arr = np.asarray(xp, dtype=float)
    fp_arr = np.asarray(fp, dtype=float)

    method = str(interp_method).lower()
    # Clamp outside domain
    if x <= xp_arr[0]:
        return float(fp_arr[0])
    if x >= xp_arr[-1]:
        return float(fp_arr[-1])
    if method == "linear":
        return float(np.interp(x, xp_arr, fp_arr, left=fp_arr[0], right=fp_arr[-1]))
    elif method == "nearest":
        idx = int(np.argmin(np.abs(xp_arr - x)))
        return float(fp_arr[idx])
    elif method in ("previous", "zoh", "step"):
        idx = int(np.searchsorted(xp_arr, x, side="right") - 1)
        idx = int(np.clip(idx, 0, len(xp_arr) - 1))
        return float(fp_arr[idx])
    elif method == "next":
        idx = int(np.searchsorted(xp_arr, x, side="left"))
        idx = int(np.clip(idx, 0, len(xp_arr) - 1))
        return float(fp_arr[idx])
    elif method in ("pchip", "monotone"):
        f = PchipInterpolator(xp_arr, fp_arr, extrapolate=True)
        return float(f(x))
    elif method in ("cubic", "cubic_spline", "cspline"):
        f = CubicSpline(xp_arr, fp_arr, bc_type="natural", extrapolate=True)
        return float(f(x))
    elif method in ("univariate", "spline", "smooth"):
        f = UnivariateSpline(xp_arr, fp_arr, s=interp_s)
        return float(f(x))
    elif method == "akima":
        f = Akima1DInterpolator(xp_arr, fp_arr)
        return float(f(x))
    else:
        # Fallback to linear
        return float(np.interp(x, xp_arr, fp_arr, left=fp_arr[0], right=fp_arr[-1]))
