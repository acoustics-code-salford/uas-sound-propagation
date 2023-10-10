import numpy as np


def linear(x, delays, fs):

    # calculate 'backward' delay times relative to receiver
    time, D_time = np.array(
        [
            [i, i - delays[t]]
            for t, i in enumerate(
                range(int(delays[0])+1, len(delays)-1)
            )
        ]
    ).T

    time = time.astype(int)
    whole = D_time.astype(int)
    frac = np.mod(D_time, 1)
    out = np.zeros(len(x) + fs)  # leave extra second at end

    for t, n, s in zip(time, whole, frac):
        # one-multiply linear interpolation (J. O. Smith)
        out[t] = x[n] + (s * (x[n+1] - x[n]))

    return out


def lagrange(x, delays, fs):
    time, D_time = np.array(
        [
            [i, i - delays[t]]
            for t, i in enumerate(
                range(int(delays[0])+1, len(delays)-1)
            )
        ]
    ).T

    time = time.astype(int)
    whole = D_time.astype(int)
    frac = np.mod(D_time, 1)
    out = np.zeros(len(x) + fs)

    for t, n, s in zip(time, whole, frac):
        out[t] = (
            (x[n + 1] * ((s * (1 + s)) / 2)) + (x[n] * (1 + s) * (1 - s))
            + x[n - 1] * ((-s) * ((1 - s) / 2))
        )
    return out
