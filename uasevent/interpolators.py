import numpy as np
import scipy.signal as sig
import abc


class FractionalInterpolator():
    def __init__(self, x, delta):
        self.delta = delta
        self.x = x

    @property
    def x(self):
        return self.x

    @x.setter
    def x(self, x):
        self._x = x
        self._x_delayed = self._calc_delayed(x)

    @abc.abstractmethod
    def _calc_delayed(self, x):
        return NotImplemented

    def __getitem__(self, item):
        return self._x_delayed[item]


class LinearInterpolator(FractionalInterpolator):
    def _calc_delayed(self, x):
        # roll input for x + 1
        x_rolled = np.roll(x, 1)
        # zero first sample to remove end of array value rolled here
        x_rolled[0] = 0

        x_delayed = x + (self.delta * (x_rolled - x))
        return x_delayed


class AllpassInterpolator(FractionalInterpolator):
    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, delta):
        self.a = (1 - delta) / (1 + delta)
        self._delta = delta

    def _calc_delayed(self, x):
        y = np.zeros(len(x))
        for i in range(len(x)-1):
            y[i + 1] = self.a * (x[i + 1] - y[i]) + x[i]

        return y


class SincInterpolator(FractionalInterpolator):
    def __init__(self, x, delta, N=35):
        self.N = N
        self.h = self.sinc_filter(N, delta)
        super().__init__(x, delta)

    def sinc_filter(self, N, delta):
        n = np.arange(N)
        # calculate sinc
        h = np.sinc(n - ((N - 1) / 2) - delta)
        # window the sinc
        h *= np.blackman(N - delta)
        # normalise for unity gain
        h /= np.sum(h)
        return h

    def _calc_delayed(self, x):
        return sig.fftconvolve(x, self.h, 'same')


def interpolate(
        x, n, s,
        InterpAlgo=SincInterpolator,
        interp_win_len=64,
        **kwargs
):
    interp_win = np.arange(n-interp_win_len//2, n+interp_win_len//2)
    x_frame = x[interp_win]

    delayed_x_frame = InterpAlgo(x_frame, s, **kwargs)
    return delayed_x_frame[interp_win_len//2]


# TODO: write some unit tests dependent on these interpolators to make
# sure I'm not going to break them unintentionally
