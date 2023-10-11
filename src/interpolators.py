import numpy as np


def sinc_fractional_delay_filter(frac, filt_length=35):
    n = np.arange(filt_length)
    # calculate sinc
    h = np.sinc(n - ((filt_length - 1) / 2) - frac)
    # window the sinc
    h *= np.blackman(filt_length)
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
