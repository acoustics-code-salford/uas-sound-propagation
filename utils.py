import numpy as np


def test_sine(t, f_0=440, A=0.75, fs=48000):
    t_samples = np.linspace(0, t, fs*t)
    sig = np.zeros_like(t_samples)

    for i, sample in enumerate(t_samples):
        sig[i] = np.sin(2*np.pi*f_0*sample) * A

    return sig
