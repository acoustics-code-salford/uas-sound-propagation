import numpy as np


def test_sine(t, f_0=440, A=0.75, fs=48000):
    t_samples = np.linspace(0, t, fs*t)
    sig = np.zeros_like(t_samples)

    for i, sample in enumerate(t_samples):
        sig[i] = np.sin(2*np.pi*f_0*sample) * A

    return sig


def load_params(csv_file):
    str_params = np.loadtxt(
        csv_file,
        delimiter=',',
        skiprows=1,
        usecols=np.arange(1, 4),
        dtype='str'
    ).reshape(-1, 3)  # reshape in case of single-event flight

    return [[np.array(cell.strip().split(' '), dtype=int)
             for cell in row]
             for row in str_params]


def nearest_whole_fraction(pos):
    n = np.round(pos).astype(int)
    s = (
        - (- pos % 1)
        if np.round(pos).astype(int) > pos
        else pos % 1
    )
    return n, s
