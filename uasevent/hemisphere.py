import glob
import numpy as np
from scipy import io, signal
from . import pyoctaveband


def atmospheric_absorption(freqs, temp=20, humidity=80, pressure=101.325):
    '''Atmospheric absorption curves calculated as per ISO 9613-1'''
    # calculate temperatre variables
    kelvin = 273.15
    T_ref = kelvin + 20
    T_kel = kelvin + temp
    T_rel = T_kel / T_ref
    T_01 = kelvin + 0.01

    # calculate pressure variables
    p_ref = 101.325
    p_rel = pressure / p_ref

    # calculate humidity as molar concentration of water vapour
    C = -6.8346 * (T_01 / T_kel) ** 1.261 + 4.6151
    p_sat_by_p_ref = 10 ** C
    h = humidity * p_sat_by_p_ref * p_rel

    # calcuate relaxataion frequencies of atmospheric gases
    f_rO = p_rel * (
        24 + 4.04e4 * h * (0.02 + h) / (0.391 + h)
    )

    f_rN = p_rel / np.sqrt(T_rel) * (
        9 + 280 * h * np.exp(-4.17 * (T_rel ** (-1/3) - 1))
    )

    # calculate alpha
    xc = 1.84e-11 / p_rel * np.sqrt(T_rel)
    xo = 1.275e-2 * np.exp(-2239.1 / T_kel) * (
        f_rO + (freqs**2 / f_rO)) ** (-1)
    xn = 0.1068 * np.exp(-3352 / T_kel) * (
        f_rN + (freqs**2 / f_rN)) ** (-1)

    alpha = freqs**2 * (xc + T_rel**(-5/2) * (xo + xn))
    return alpha


def get_data_from_mat(
        mat_filename,
        n_mics=9,
        metrics=['LAFp'],
        ground_plate_compensation=True
):
    """
    Load and process microphone data from a MATLAB .mat file.

    Parameters:
        mat_filename (str):
            Path to the .mat file containing microphone recordings
            and metrics.
        n_mics (int, optional):
            Number of microphones to load data from. Defaults to 9.
        metrics (list of str, optional):
            List of metric names (e.g., 'LAFp') to extract from the
            file. Defaults to ['LAFp'].
        ground_plate_compensation (bool, optional):
            If True, apply ground plate compensation by halving raw
            microphone data and subtracting 6 dB from each metric.
            Defaults to True.

    Returns:
        tuple: A tuple containing:
            - raw_mic_data (np.ndarray): Array of concatenated raw
              microphone signals (time x channels).
            - metric_data (dict): Dictionary mapping each metric name
              to its corresponding data array (time x channels).
            - fs (int): Sampling rate of the recordings.
    """
    # loading of multiple files
    if isinstance(mat_filename, list):
        fm = [
            get_data_from_mat(file, metrics=metrics) for file in mat_filename]
        data_raw = np.array([f[0] for f in fm])
        data_metrics = [f[1] for f in fm]
        fs = int(fm[0][2])
        return data_raw, data_metrics, fs

    mat_raw = io.loadmat(mat_filename)

    fs = mat_raw['Sample_rate']
    raw_mic_data = np.concatenate(
        [mat_raw[f'Data1_Mic_{n+1}'] for n in range(n_mics)], axis=1)

    if ground_plate_compensation:
        raw_mic_data /= 2

    metric_data = {}
    for metric in metrics:
        metric_data[metric] = np.concatenate(
            [mat_raw[f'Data1_Mic_{n+1}_{metric}']
             for n in range(n_mics)], axis=1
        )

        if ground_plate_compensation:
            metric_data[metric] -= 6

    return raw_mic_data, metric_data, fs


def multi_psd(raw_data, fs, n_fft, win='hann'):

    n_reps = raw_data.shape[0]
    n_channels = raw_data.shape[1]

    psd_reps = np.empty([n_reps, n_channels, n_fft//2 + 1])

    for i, rep in enumerate(raw_data):
        for j, channel in enumerate(rep):
            frequencies, psd_reps[i, j] = signal.welch(
                channel, fs, window=win, noverlap=n_fft // 4,
                nperseg=n_fft, scaling='density')

    return psd_reps, np.array(frequencies)


def hover_hemisphere(
        data_folder,
        mic_positions=None,
        mic_angles=None,
        drone_position=np.array([0, 0, 10]),
        fs=50000,
        p_ref=20e-6,
        depropagation_radius=1.0,
):

    # sort out mic positions and angles
    if mic_angles is None and mic_positions is None:
        raise ValueError('mic_angles or mic_positions must be provided')
    elif mic_angles is not None and mic_positions is not None:
        raise ValueError(
            'please provide either mic_angles or mic_positions, not both')
    elif mic_angles is not None:
        # calculate mic positions from angles
        mic_positions = np.array([
            [0, -np.tan(np.deg2rad(angle)) * drone_position[-1], 0]
            for angle in mic_angles
        ])
    elif mic_positions is not None and mic_angles is None:
        # calculate mic angles from positions
        mic_angles = np.round(
            np.arctan(mic_positions[:, :-1] /
                      drone_position[-1]) * 180 / np.pi, 1)

    n_mics = len(mic_angles)

    third_octave_bands = [
        25, 31, 40, 50, 63, 80, 100, 125, 160, 200, 250, 400, 500, 630, 800,
        1_000, 1_250, 1_600, 2_000, 2_500, 3_150, 4_000, 5_000, 6_300, 8_000,
        10_000, 12_500, 16_000, 20_000
    ]

    files = glob.glob(f'{data_folder}/*.mat')
    raw_data = np.array([get_data_from_mat(file)[0].T for file in files])

    # calculate psd for all raw data
    psd_events, psd_freqs = multi_psd(raw_data, fs, n_fft=2**16)
    psd_db = 10*np.log10(psd_events / p_ref**2)

    drone_mic_distances = np.linalg.norm(
        mic_positions - drone_position, axis=1)

    deprop_distances = drone_mic_distances - depropagation_radius
    atmos = atmospheric_absorption(psd_freqs, temp=14.5)

    # depropagation of atmospheric absorption
    # calculate absorption for each frequency and distance to each microphone
    attenuation_per_freq = np.array(
        [atmos * dist for dist in deprop_distances])

    # apply to psd, calculate mean across all measurements
    psd_atmos_attenuated = psd_db + attenuation_per_freq
    psd_atmos_attenuated = psd_atmos_attenuated.mean(0)

    # depropagation of attenuation due to spherical spread
    psd_atmos_attenuated -= (
        20*np.log10(depropagation_radius / drone_mic_distances).reshape(-1, 1))

    # select only positive values
    psd_atmos_attenuated = psd_atmos_attenuated * (psd_atmos_attenuated > 1)

    # calculate frequency and angle-dependent levels at 1 metre
    band_spl = np.empty([len(third_octave_bands) - 2, n_mics])
    for i in range(1, len(third_octave_bands) - 1):
        low_bin = third_octave_bands[i-1]
        high_bin = third_octave_bands[i+1]

        selected_freqs = ((psd_freqs > low_bin) & (psd_freqs < high_bin))
        psd_freq_limited = psd_atmos_attenuated[:, selected_freqs]
        band_spl[i - 1] = 10*np.log10(
            np.sum(10**(psd_freq_limited/10), axis=1))

    return band_spl, mic_angles, third_octave_bands[1:-1]


def flyover_hemisphere(data_raw, data_level, mic_positions,
                       altitude=25,
                       flight_speed=5,
                       fs=50000,
                       n_mics=9,
                       ref_mic=4,
                       depropagation_radius=1.0,
                       time_slice=16,
                       threshold=10,
                       n_segments=11):

    n_samples = fs * time_slice

    # apply median filter to each row of the data (different recordings)
    level_median_filt = np.apply_along_axis(
        lambda x: signal.medfilt(
            x, kernel_size=5), axis=1, arr=data_level[..., ref_mic])
    index_max = np.argmax(level_median_filt, axis=1)

    # align data to central max value on each recording repeat
    segment_raw = np.zeros((len(index_max), n_samples, n_mics))
    segment_level = np.zeros((len(index_max), n_samples, n_mics))
    for rec, index in enumerate(index_max):
        segment_raw[rec] = data_raw[
            rec, index - n_samples//2:index + n_samples//2]
        segment_level[rec] = data_level[
            rec, index - n_samples//2:index + n_samples//2]

    peak_zones = segment_level[..., ref_mic] >= np.max(
        segment_level[..., ref_mic]) - threshold
    thresholded_raw = []
    thresholded_level = []
    for rec, peak_zone in enumerate(peak_zones):
        thresholded_raw.append(segment_raw[rec, peak_zone])
        thresholded_level.append(segment_level[rec, peak_zone])

    # this is selecting the first recording,
    # perhaps it should be the mean of all recordings
    raw_vector = thresholded_raw[0]
    level_vector = thresholded_level[0][:, ref_mic]
    # time_vector = np.linspace(0, len(level_vector), len(level_vector)) / fs
    max_level_index = np.argmax(level_vector)

    # number of samples per time segment
    len_time_seg = len(level_vector) // n_segments
    seg_midpoints = np.arange(len_time_seg//2, len(level_vector), len_time_seg)

    # calculates x distance between midpoint of each segment and
    # max level index, this is the max level index for the central microphone
    # we assume that this is when the UAV is directly above
    d_x = - (flight_speed * (max_level_index - seg_midpoints) / fs)

    chunks = np.lib.stride_tricks.sliding_window_view(
        raw_vector, len_time_seg-1, axis=0)[::len_time_seg-1]

    chunk_spls = []
    for chunk in chunks:
        band_spls = []
        for channel in chunk:
            spl, freqs = pyoctaveband.octavefilter(
                channel, fs, fraction=3, limits=[20, 20000])
            band_spls.append(spl)
        chunk_spls.append(band_spls)
    chunk_spls = np.array(chunk_spls)
    freqs = np.array(freqs)

    drone_positions = np.array(
        [d_x, np.zeros(len(d_x)), np.ones(len(d_x)) * altitude]).T

    drone_mic_distances = np.array([np.linalg.norm(
        mic_positions - pos, axis=1) for pos in drone_positions])
    deprop_distances = drone_mic_distances - depropagation_radius

    atmos = atmospheric_absorption(freqs, temp=14.5)
    attenuation_per_freq = np.array(
        [[atmos * dist for dist in drone_location]
         for drone_location in deprop_distances])
    freqdep_deattenuation = 10*np.log10(
        depropagation_radius / drone_mic_distances).reshape(
            -1, 9, 1) - attenuation_per_freq

    chunk_spls_deattenuated = chunk_spls - freqdep_deattenuation
    chunk_overall_spls_deattenuated = 10*np.log10(
        np.sum(10**(chunk_spls_deattenuated/10), axis=2))

    theta_angles = np.round(
        np.arctan(mic_positions[:, 1] / altitude) * 180 / np.pi, 1)
    phi_angles = np.round(
        np.arctan(drone_positions[:, 0] / altitude) * 180 / np.pi, 1)

    return chunk_spls_deattenuated, chunk_overall_spls_deattenuated, \
        theta_angles, phi_angles, freqs


# data_folder = 'EE_T1_25_F15_N_S_135949_uw/'
# files = glob.glob(f'{data_folder}/*.mat')
# data_raw, data_metrics, fs = get_data_from_mat(files)
# data_lafp = np.array([d['LAFp'] for d in data_metrics])

# mic_positions = np.array([
#     [0, -43.3, 0],
#     [0, -25.0, 0],
#     [0, -14.4, 0],
#     [0, -6.7, 0],
#     [0, 0, 0],
#     [0, 6.7, 0],
#     [0, 14.4, 0],
#     [0, 25.0, 0],
#     [0, 43.3, 0]
# ])

# hemisphere_freqdep, hemisphere_overall, thetas, phis, freqs = \
#     flyover_hemisphere(data_raw,
#                        data_lafp,
#                        mic_positions,
#                        altitude=25,
#                        flight_speed=15,
#                        fs=fs)
#
## calculate difference between hover hemisphere (directly overhead
## recording used for source signal) and other propagation angles
# db_diffs = hemisphere_freqdep - hemisphere_freqdep[5, 4]
