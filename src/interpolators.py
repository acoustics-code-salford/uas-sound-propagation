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


# TODO: incorporate these new interpolator objects into main model
# e.g. in delay loop and for constant delay of reflection output
# output from the object at index n set at delta = frac will be n.frac
# TODO: write some unit tests dependent on these interpolators to make
# sure I'm not going to break them unintentionally

def sinc_fractional_delay_filter(frac, N=35):
    n = np.arange(N)
    # calculate sinc
    h = np.sinc(n - ((N - 1) / 2) - frac)
    # window the sinc
    h *= np.blackman(N - frac)
    # normalise for unity gain
    h /= np.sum(h)

    return h


def linear(x, n, s):
    return x[n] + (s * (x[n+1] - x[n]))


def lagrange(x, n, s):
    return (
        (x[n + 1] * ((s * (1 + s)) / 2))
        + (x[n] * (1 + s) * (1 - s))
        + x[n - 1] * ((-s) * ((1 - s) / 2))
    )


class FirstOrderAllpass():
    def __init__(self, frac):
        self.frac = frac
        # self.a = (1 - frac) / (1 + frac)
        self.y_prev = 0.0

    
    @property
    def frac(self):
        return self._frac

    @frac.setter
    def frac(self, frac):
        self.a = (1 - frac) / (1 + frac)
        self._frac = frac


    def interp(self, x, n):
        y = self.a * (x[n] - self.y_prev) + x[n + 1]
        self.y_prev = y
        return y
    
# # allpass
# frac = 0.3
# a = (1 - frac) / (1 + frac)
# y = np.zeros(len(x))
# for i in range(len(x)-1):
#     y[i + 1] = a * (x[i + 1] - y[i]) + x[i]


# # linear
# fracwholeting = x + (s * (np.roll(x, 1) - x))