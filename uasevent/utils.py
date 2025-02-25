import json
import numpy as np


def test_sine(t, f_0=440, A=0.75, fs=48_000):
    '''
    Generate a sine wave signal of length t.

    Parameters:
    -----------
    `t`: Length of sine signal in seconds

    `f_0`: Frequency [Hz] (default 440)

    `A`: Amplitude (default 0.75)

    `fs`: Sampling frequency [Hz] (default 48_000)

    Returns
    -------
    `sig`: Array containing sinusoidal signal.
    '''
    t_samples = np.linspace(0, t, fs*t)
    sig = np.zeros_like(t_samples)

    for i, sample in enumerate(t_samples):
        sig[i] = np.sin(2*np.pi*f_0*sample) * A

    return sig


def nearest_whole_fraction(pos):
    '''Return nearest integer and fraction to input sample position.'''
    n = np.round(pos).astype(int)
    s = (
        - (- pos % 1)
        if np.round(pos).astype(int) > pos
        else pos % 1
    )
    return n, s


def cart_to_sph(xyz, return_r=True):
    '''
    Transform between cartesian and polar co-ordinates.

    Parameters
    ----------
    `xyz`: Array of cartesian co-ordinates.

    `return_r`: Bool toggling output of radius. If `False`, only azimuth and
    elevation will be output.

    Returns
    -------
    `sph_coords`: Array of spherical co-ordinates.
    '''
    r = np.linalg.norm(xyz, axis=0)
    x, y, z = xyz

    theta = np.arctan2(y, x) % (2 * np.pi)
    phi = np.arccos(z / r)

    sph_coords = np.array([theta, phi, r]) \
        if return_r else np.array([theta, phi])

    return sph_coords


def vector_t(pathdict, fs=48_000):
    '''
    Calculate source position at each sample time along specified trajectory.

    Parameters
    ----------
    `pathdict`: Specifies start and end positions and speeds of flight segment.

    `fs`: Sampling frequency [Hz] (default 48_000)

    Returns
    -------
    `xyz`: Array describing position of source at each sample time based on
    the specified flight segment.
    '''

    start = np.array(pathdict['start'])
    end = np.array(pathdict['end'])
    speeds = np.array(pathdict['speeds'])

    v_0, v_T = speeds
    distance = np.linalg.norm(start - end)
    # heading of source
    vector = ((end - start) / distance)
    # acceleration
    a = ((v_T**2) - (v_0**2)) / (2 * distance)
    # number of time steps in samples for operation
    n_output_samples = fs * ((v_T-v_0) / a) if a != 0 else \
        (distance / v_0) * fs

    # array of positions at each time step
    x_t = np.array([
        [
            (v_0 * (t / fs))
            + ((a * (t / fs)**2) / 2)
            for t in range(int(n_output_samples))
        ]
    ]).T

    # map to axes
    xyz = (start + vector * x_t).T
    return xyz


def rectify(x):
    '''Rectifies signal (negative segments mirrored into positive).'''
    return (np.abs(x) + x) / 2


def load_mapping(name, mapping_file='mappings/mappings.json'):
    '''Loads specification of loudspeaker position to required array format.'''
    with open(mapping_file, 'r') as file:
        mapping = json.load(file)[name]
        channel_numbers = [int(key) for key in mapping.keys()]
        theta = np.radians(
            [float(x['azimuth']) for x in mapping.values()]
        )
        phi = np.radians(
            [float(x['elevation']) for x in mapping.values()]
        )
        distances = np.array(
            [float(x['distance']) for x in mapping.values()]
        )
        return [channel_numbers, theta, phi, distances]


def sph_to_cart(sph_co_ords):
    '''Converts between spherical and cartesian co-ordinate systems.'''

    # allow for lack of r value (i.e. for unit sphere)
    if sph_co_ords.shape[1] < 3:
        theta, phi = sph_co_ords[:, 0], sph_co_ords[:, 1]
        r = 1

    else:
        theta, phi, r = sph_co_ords[:, 0], sph_co_ords[:, 1], sph_co_ords[:, 2]

    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)

    return np.array([x, y, z]).T
