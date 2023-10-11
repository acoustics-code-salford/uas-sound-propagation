import numpy as np


def linear(x, n, s):
    return x[n] + (s * (x[n+1] - x[n]))


def lagrange(x, n, s):
    return (
        (x[n + 1] * ((s * (1 + s)) / 2)) 
        + (x[n] * (1 + s) * (1 - s))
        + x[n - 1] * ((-s) * ((1 - s) / 2))
    )
