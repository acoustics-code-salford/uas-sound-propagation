import os
import json
import pyfar
import warnings
import numpy as np
import soundfile as sf
import multiprocessing

from scipy import signal
from scipy.ndimage import gaussian_filter
from pyfar.dsp.filter import fractional_octave_bands
from scipy.interpolate import RegularGridInterpolator
from . import interpolators, utils


class UASEventRenderer():
    '''
    Main propagation model for rendering of drone flight events.

    '''
    def __init__(
            self,
            flight_spec,
            fs=48_000,
            ground_material='grass',
            atmos_absorp=True,
            receiver_height=1.5):
        '''
        Initialises all necessary attributes for the UASEventRenderer object.
        '''
        self.atmos_absorp = atmos_absorp
        self.fs = fs
        '''Sampling frequency in Hz (default 48_000)'''
        self.receiver_height = receiver_height
        '''Height of receiver position, metres (default 1.5)'''
        self.ground_material = ground_material
        '''Material for ground reflection'''
        self.flight_parameters = json.load(open(flight_spec))
        '''JSON file with segmentwise description of flight path'''
        self.output = None
        '''Initialise var to contain rendered signal'''

    @property
    def receiver_height(self):
        return self._receiver_height

    @receiver_height.setter
    def receiver_height(self, height):
        self._receiver_height = height
        if hasattr(self, 'flight_parameters'):
            self._setup_paths()

    @property
    def ground_material(self):
        return self._ground_material

    @ground_material.setter
    def ground_material(self, material):
        self._ground_material = material
        if hasattr(self, 'flight_parameters'):
            self._setup_paths()

    @property
    def flight_parameters(self):
        return self._flight_parameters

    @flight_parameters.setter
    def flight_parameters(self, params):
        self._flight_parameters = params
        self._setup_paths()

    def _setup_paths(self):
        # set up direct and reflected paths
        self.direct_path = PropagationPath(
            FlightPath(self._flight_parameters),
            'direct', self.receiver_height, self.fs,
            atmos_absorp=self.atmos_absorp
        )

        self.ground_reflection = PropagationPath(
            FlightPath(self._flight_parameters),
            'reflection', self.receiver_height, self.fs,
            reflection_surface=self.ground_material,
            atmos_absorp=self.atmos_absorp
        )

        self._norm_scaling = np.max(abs(self.direct_path._inv_sqr_attn))
        self.direct_path._inv_sqr_attn /= self._norm_scaling
        self.ground_reflection._inv_sqr_attn /= self._norm_scaling

    def render(self, x, directivity_dir=None):
        '''
        Renders output signal based on input parameters.

        Parameters
        ----------
        `x`
            Signal to use as UAS source. Assumed to be a stationary recording.

        Returns
        -------
        `output`
            Signal containing direct and reflected paths reaching receiver.
        '''

        if x.ndim > 1:
            raise ValueError("Input source signal is not monaural.")

        # apply each propagation path to input signal (parallel)
        pathlist = [self.direct_path, self.ground_reflection]
        manager = multiprocessing.Manager()
        self.return_dict = manager.dict()
        jobs = []
        for i, path in enumerate(pathlist):
            p = multiprocessing.Process(
                target=self.worker,
                args=(path, x, directivity_dir, str(i)))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        direct = self.return_dict['0']
        reflection = self.return_dict['1']

        # in samples
        offset = (self.ground_reflection._init_delay
                  - self.direct_path._init_delay)

        # calculate whole and fractional number of samples to delay reflection
        whole_offset, frac_offset = utils.nearest_whole_fraction(offset)

        # calculate fractional delay of reflection)
        reflection = np.array([
            i for i in interpolators.SincInterpolator(reflection, frac_offset)
        ])

        # add back channel dimension if mono output
        if direct.shape[0] == 1:
            reflection = np.expand_dims(reflection, 0)

        # add zeros to start of reflection to account for whole sample delay
        if whole_offset:
            reflection_zeros = np.zeros_like(reflection)
            reflection_zeros[whole_offset:] += reflection[:-whole_offset]
            reflection = reflection_zeros
        self._d = direct.T
        self._r = reflection.T

    def write_output(self, filename, output='combined'):

        filename = os.path.splitext(filename)[0]
        if self.output is None:
            raise ValueError('No output signal to write out')

        if output == 'combined':
            self._write_outfiles(filename, self._d + self._r)
        elif output == 'direct':
            self._write_outfiles(f'{filename}_direct',
                                 self._d, path_type='direct')
        elif output == 'reflection':
            self._write_outfiles(f'{filename}_reflection',
                                 self._r, path_type='reflection')
        else:
            raise ValueError('Invalid output type')

    def _write_outfiles(self, filename, data,
                        start_t=0.0,
                        path_type='direct',
                        coord_fmt='unity'):

        sf.write(f'{filename}.wav', data, self.fs, 'PCM_24')
        np.savetxt(
            f'{filename}.csv', self._flightpath(
                receiver_height=self.receiver_height,
                path_type=path_type,
                coord_fmt=coord_fmt).T,
            delimiter=',', fmt='%.2f',
            header=f't={start_t}, fmt={coord_fmt}, path={path_type}'
        )

    def worker(self, prop_path, x, directivity_dir, key):
        '''
        Basic framework for parallel processes
        '''
        path_output = prop_path.process(x, directivity_dir)
        self.return_dict[key] = path_output


class PropagationPath():
    '''
    Defines a single propagation path from source to receiver.

    '''
    def __init__(
            self,
            flightpath,
            path_type,
            receiver_height=1.5,
            fs=48_000,
            reflection_surface=None,
            atmos_absorp=True,
            c=343.0,
            frame_len=512
    ):
        '''
        Initialises PropagationPath object.
        '''
        self.fs = fs
        '''Sampling frequency in Hz (default `48_000`)'''
        self.reflection_surface = reflection_surface
        '''String to select absorption coefficients for ground reflection.
        Options `"grass"`, `"soil"`, `"asphalt"`.'''
        self.frame_len = frame_len
        '''Number of samples for frames used for time-varying atmospheric
        absorption and ground reflection filtering (default `512`).'''
        self._hop_len = frame_len // 2
        '''Number of samples to hop between frames.'''
        self.flightpath = flightpath
        '''FlightPath object for array calculations'''
        self.path_array = self.flightpath(
                receiver_height=receiver_height,
                path_type=path_type,
                fs=self.fs
            )
        '''Array describing position of source at every sample point.'''
        
        self.atmos_absorp = atmos_absorp

        # calculate delays and amplitude curve
        _, _, r = utils.cart_to_sph(self.path_array)
        delays = (r / c) * self.fs
        self._init_delay = delays[0]
        self._delta_delays = np.diff(delays)
        self._inv_sqr_attn = 1 / r**2

        # calculate angles per frame for filters
        # self._sph_per_frame = utils.cart_to_sph(
        #     self.flightpath(fs=self.fs / self._hop_len)
        # ).T

    def _apply_doppler(self, x):
        # init output array and initial read position
        out = np.zeros(len(self._delta_delays) + 1)
        read_pointer = 0

        for i in range(len(out)):
            # find whole and fractional part
            n, s, = utils.nearest_whole_fraction(read_pointer)
            # negative s for correct indexing over n
            out[i] = interpolators.interpolate(x, n, -s)

            # update read pointer position
            read_pointer += (1 - self._delta_delays[i-1])

        return out

    def _apply_attenuation(self, x):
        return x * self._inv_sqr_attn

    def _filter(self, x, directivity_directory=None):
        # neat trick to get windowed frames
        x_windowed = np.lib.stride_tricks.sliding_window_view(
            x, self.frame_len)[::self._hop_len] * np.hanning(self.frame_len)

        # list of filter stages
        filters = []

        # add atmospheric absorption if enabled
        if self.atmos_absorp:
            filters.append(AtmosphericAbsorptionFilter(fs=self.fs))

        # # add directivity if hemisphere data is specified
        # if directivity_directory is not None:
        #     filters.append(DirectivityFilter(directivity_directory, fs=self.fs))

        # add ground filter if surface is set (reflected path)
        if self.reflection_surface is not None:
            filters.append(GroundReflectionFilter(
                self.reflection_surface, fs=self.fs))

        x_out = np.zeros(len(x))

        for i, (position, frame) in enumerate(
                zip(self.flightpath(fs=self.fs / self._hop_len).T, x_windowed)):

            for filter in filters:
                frame = filter.filter(frame, position)

            frame_index = i * self._hop_len
            x_out[frame_index:frame_index + self.frame_len] += frame

        if directivity_directory is not None:
            directivity = DirectivityFilter(
                directivity_directory, fs=self.fs)
            x_out = directivity.filter(x_out, self.flightpath(fs=self.fs))

        return x_out

    def process(self, x, directivity_directory=None):
        '''
        Processes input signal to add effects of propagation along single
        specified path. Incorporates amplitude envelopes, doppler effect,
        and filtering for atmospheric absorption and ground reflection.

        Parameters
        ----------
        `x`
            Signal to use as UAS source. Assumed to be a stationary recording.

        Returns
        -------
        `output`
            Array containing signal reaching receiver along specified path.
        '''
        path_len = len(self.path_array[0]) + 2
        if len(x) <= path_len:
            n_reps = int(np.ceil((path_len) / len(x)))
            warnings.warn(f'Input signal shorter than path, '
                          f'auto-repeating input x {n_reps}...')
            x = np.tile(x, n_reps)

        output = self._apply_doppler(x)
        output = self._filter(output, directivity_directory)
        output = self._apply_attenuation(output)
        return output


class GroundReflectionFilter():
    '''
    Implements material and incident angle-dependent lowpass filter
    to simulate ground reflection.

    '''
    def __init__(self,
                 material='asphalt',
                 freqs=np.geomspace(20, 24000),
                 Z_0=413.26,
                 fs=48_000,
                 n_taps=21):
        '''
        Initialises GroundReflectionFilter.

        '''

        self.freqs = freqs
        '''array of frequencies used to evaluate frequency response'''
        self.Z_0 = Z_0
        '''characteristic acoustic impedance of air [rayl/m^2]'''
        self.material = material
        '''string to select absorption coefficients for ground reflection'''
        self.fs = fs
        '''Sampling frequency in Hz'''
        self.n_taps = n_taps
        '''Number of taps for FIR filter'''

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, material):
        match material:
            case 'grass':
                self._sigma = 300  # kPa s m^-2
            case 'soil':
                self._sigma = 5000
            case 'asphalt':
                self._sigma = 20_000
        self._material = material

    def _Z(self):
        R = 1 + 9.08 * (self.freqs / self._sigma) ** -0.75
        X = -11.9 * (self.freqs / self._sigma) ** -0.73
        return (R + 1j*X) * self.Z_0

    def _R(self, angle):
        return np.real(
            (self._Z() * np.cos(angle) - self.Z_0)
            / (self._Z() * np.cos(angle) + self.Z_0)
        )

    def filter(self, x, position):
        '''
        Filters input signal based on angle-of-incidence on ground plane.

        Parameters
        ----------
        `x`:
            Frame of source signal.

        `position`:
            Array describing mean position of source during frame.

        Returns
        -------
        `lowpass_sig`:
            Filtered signal.
        '''
        position = utils.cart_to_sph(position)
        _, phi, _ = position
        phi = np.pi - phi
        h = signal.firls(self.n_taps, self.freqs,
                         utils.rectify(self._R(phi)),
                         fs=self.fs)
        lowpass_sig = signal.fftconvolve(x, h, 'same')
        return lowpass_sig


class AtmosphericAbsorptionFilter():
    '''
    Implements distance-dependent lowpass filter to simulate
    atmospheric absorption.
    '''
    def __init__(self,
                 freqs=np.geomspace(20, 24_000),
                 temp=20.0,
                 humidity=80.0,
                 pressure=101.325,
                 n_taps=21,
                 fs=48_000):
        '''
        Initialises GroundReflectionFilter.

        '''

        self._attenuation = self._alpha(freqs, temp, humidity, pressure)
        self.n_taps = n_taps
        '''Number of taps for FIR filter'''
        self.freqs = freqs
        '''array of frequencies used to evaluate frequency response'''
        self.fs = fs
        '''Sampling frequency [Hz]'''

    def _alpha(self, freqs, temp=20, humidity=80, pressure=101.325):
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
        return 1 - alpha

    def filter(self, x, position):
        position = utils.cart_to_sph(position)
        _, _, r = position
        h = signal.firls(self.n_taps, self.freqs,
                         self._attenuation**r,
                         fs=self.fs)
        return signal.fftconvolve(x, h, 'same')


class DirectivityFilter():

    def __init__(self,
        data_directory,
        fs=48_000):

        self.fs=fs

        self._thetas = np.loadtxt(
            f'{data_directory}/thetas.csv', delimiter=',')
        self._phis = np.loadtxt(
            f'{data_directory}/phis.csv', delimiter=',')

        # load raw data
        directionality_db = np.load(f'{data_directory}/db_diffs.npy')
        # smooth attenuation grids to avoid audible discontinuities in output
        smooth_atten = gaussian_filter(directionality_db, sigma=1)

        # interpolation object to return gain values
        self.grid_interpolator = RegularGridInterpolator(
            (self._phis, self._thetas), smooth_atten, bounds_error=False)

    def clamped_interp(self, point):
        theta, phi = point
        clamped_phi = np.clip(phi, self._phis[0], self._phis[-1])
        clamped_theta = np.clip(theta, self._thetas[0], self._thetas[-1])
        return self.grid_interpolator((clamped_phi, clamped_theta))

    def filter(self, x, flightpath):
        # filter into third-octave bands
        xfilt = pyfar.dsp.filter.fractional_octave_bands(
            pyfar.Signal(x, self.fs), 3, frequency_range=(19, 20e3))

        # need to consider how to change this to reorient in the direction of
        # travel for the drone (e.g. phi is front/back angle **relative to the
        # forward orientation of the drone**, theta is side to side)
        theta = np.rad2deg(np.arctan(flightpath[0] / flightpath[2]))
        phi = np.rad2deg(np.arctan(flightpath[1] / flightpath[2]))

        frequency_weighted_sig = \
            xfilt.time.squeeze() * self.clamped_interp((theta, phi)).T
        return frequency_weighted_sig.sum(0)


class FlightPath():
    def __init__(self, flight_spec):

        self.flight_spec = flight_spec

    def __call__(self,
                 fs=50,
                 receiver_height=0.0,
                 path_type='direct',
                 coord_fmt='default'):

        flightpath = self._calc_flightpath(
            flight_spec=self.flight_spec, fs=fs)

        if path_type == 'direct':
            flightpath[2] = flightpath[2] - receiver_height
        elif path_type == 'reflection':
            flightpath[2] = - flightpath[2] - receiver_height
        else:
            raise ValueError(
                'Path type must be either "direct" or "reflection"'
            )

        if coord_fmt == 'unity':
            return flightpath[[0, 2, 1]]
        elif coord_fmt == 'default':
            return flightpath
        else:
            raise ValueError(
                'Coordinate format must be either "unity" or "default"'
            )

    def _calc_flightpath(self, flight_spec, fs):
        flightpath = np.empty([3, 0])
        for _, p in flight_spec.items():
            flightpath = np.append(
                flightpath, self.vector_t(p, fs), axis=1
            )
        return flightpath

    def vector_t(self, flight_spec, fs):
        start = np.array(flight_spec['start'])
        end = np.array(flight_spec['end'])
        speeds = np.array(flight_spec['speeds'])

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
