import numpy as np
from numba import njit

class NumbaInterpolator:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __call__(self, s):
        # Handle both scalar and array inputs
        s_array = np.atleast_1d(s)
        result = np.array([linear_interp_numba(si, self.x, self.y) for si in s_array])
        # Return scalar if input was scalar
        return result[0] if np.isscalar(s) else result
    
@njit
def linear_interp_numba(x, xp, fp):
    """
    Perform linear interpolation for a single value x.
    
    Parameters:
    - x: float, the point to interpolate.
    - xp: 1D array of floats, the x-coordinates of the data points, must be increasing.
    - fp: 1D array of floats, the y-coordinates of the data points.
    
    Returns:
    - float, the interpolated value.
    """
    n = len(xp)
    
    if x <= xp[0]:
        return fp[0]
    elif x >= xp[n-1]:
        return fp[n-1]
    
    # Binary search to find the right interval
    left = 0
    right = n - 1
    while left <= right:
        mid = (left + right) // 2
        if xp[mid] <= x < xp[mid + 1]:
            # Perform linear interpolation
            t = (x - xp[mid]) / (xp[mid + 1] - xp[mid])
            return fp[mid] * (1 - t) + fp[mid + 1] * t
        elif x < xp[mid]:
            right = mid - 1
        else:
            left = mid + 1
    
    return fp[-1]  # Fallback