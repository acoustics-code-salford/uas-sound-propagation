import math
import numpy as np
import scipy.special as sp
from .utils import cart_to_sph


class AmbiSpatialise():
    def __init__(self, N):
        self.N = N

    def spatialise(self, x, flightpath_array):
        # calculate samplewise array of SH weights
        y_mn = self._Y_array(*cart_to_sph(flightpath_array)[:2, :])
        # apply to signal
        return (x * y_mn).T

    def _Y_array(self, theta, phi):
        Y_mn = np.zeros([(self.N + 1) ** 2, len(theta)])
        for i in range((self.N + 1) ** 2):
            n = math.isqrt(i)
            m = i - (n ** 2) - n
            Y_mn[i, :] = self._Y(m, n, theta, phi).reshape(1, -1)
        return Y_mn

    def _Y(self, m, n, theta, phi):
        return (
                ((-1) ** m) *  # condon-shortley compensation
                self._sn3d(m, n) *
                sp.assoc_legendre_p(abs(m), n, np.sin(phi)) *
                (
                    np.cos(m * theta)
                    if m >= 0 else np.sin((-m) * theta)
                )
        )

    def _sn3d(self, m, n):
        return (
            np.sqrt(
                (2 - (lambda m: 1 if m == 0 else 0)(m)) * (
                        sp.factorial(n - abs(m)) /
                        sp.factorial(n + abs(m))
                )
            )
        )
