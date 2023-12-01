import numpy as np
import scipy.signal as sig
import abc


class FractionalInterpolator():
    '''
    Defines framework for fractional delay interpolators.

    ...

    Attributes
    ----------
    x : np.ndarray
        signal on which to apply the fractional delay
    delta : float
        fraction of a sample by which to delay input signal
    '''
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
    '''
    Linear interpolation method for fractional delay.

    ...

    Attributes
    ----------
    x : np.ndarray
        signal on which to apply the fractional delay
    delta : float
        fraction of a sample by which to delay input signal
    '''
    def _calc_delayed(self, x):
        # roll input for x + 1
        x_rolled = np.roll(x, 1)
        # zero first sample to remove end of array value rolled here
        x_rolled[0] = 0

        x_delayed = x + (self.delta * (x_rolled - x))
        return x_delayed


class AllpassInterpolator(FractionalInterpolator):
    '''
    Allpass filter method for fractional delay.

    ...

    Attributes
    ----------
    x : np.ndarray
        signal on which to apply the fractional delay
    delta : float
        fraction of a sample by which to delay input signal
    '''
    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, delta):
        delta += 1
        self.a = (1 - delta) / (1 + delta)
        self._delta = delta

    def _calc_delayed(self, x):
        y = np.zeros(len(x))
        for i in range(0, len(x) - 1):
            y[i] = self.a * (x[i] - y[i - 1]) + x[i - 1]

        return y

    def __getitem__(self, item):
        return self._x_delayed[item + 1]


class SincInterpolator(FractionalInterpolator):
    '''
    Sinc interpolation method for fractional delay.

    ...

    Attributes
    ----------
    x : np.ndarray
        signal on which to apply the fractional delay
    delta : float
        fraction of a sample by which to delay input signal
    '''
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
        return np.expand_dims(h, 1)

    def _calc_delayed(self, x):
        if x.ndim == 1:
            x = np.expand_dims(x, 1)
        y = sig.fftconvolve(x, self.h, 'same')
        return np.squeeze(y)


def interpolate(
        x, n, s,
        InterpAlgo=SincInterpolator,
        interp_win_len=64,
        **kwargs
):
    '''
    Helper function returning a single interpolated sample of an input signal.

        Parameters:
            x (np.ndarray): signal on which to apply the fractional delay
            n (int): whole number index to sample
            s (float): fraction by which to interpolate between samples
        Returns:
            value (float): value of interpolated sample

    Attributes
    ----------
    x : np.ndarray
        signal on which to apply the fractional delay
    delta : float
        fraction of a sample by which to delay input signal
    '''
    interp_win = np.arange(n-interp_win_len//2, n+interp_win_len//2)
    x_frame = x[interp_win]

    delayed_x_frame = InterpAlgo(x_frame, s, **kwargs)
    value = delayed_x_frame[interp_win_len//2]
    return value
