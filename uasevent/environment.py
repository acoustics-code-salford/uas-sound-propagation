import os
import json
import numpy as np
import soundfile as sf
import multiprocessing
from toolz import pipe
from scipy import signal
from . import interpolators, utils


class UASEventRenderer():
    '''
    Main propagation model for rendering of drone flight events.

    '''
    def __init__(
            self,
            flight_spec,
            ground_material='grass',
            fs=48_000,
            receiver_height=1.5):
        '''
        Initialises all necessary attributes for the UASEventRenderer object.
        '''
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

    def render(self, x):
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
        return_list = manager.list()
        jobs = []
        for path in pathlist:
            p = multiprocessing.Process(
                target=self.worker, 
                args=(path, x, return_list))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        direct = return_list[0]
        reflection = return_list[1]

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
        self._flightpath = FlightPath(params)
        self._setup_paths()
        self._flight_parameters = params

    def _setup_paths(self):
        # set up direct and reflected paths
        self.direct_path = PropagationPath(
            self._flightpath(
                receiver_height=self.receiver_height,
                path_type='direct',
                fs=self.fs
            ),
            self.fs
        )

        self.ground_reflection = PropagationPath(
            self._flightpath(
                receiver_height=self.receiver_height,
                path_type='reflection',
                fs=self.fs
            ),
            self.fs,
            self.ground_material
        )

        self._norm_scaling = np.max(abs(self.direct_path._inv_sqr_attn))
        self.direct_path._inv_sqr_attn /= self._norm_scaling
        self.ground_reflection._inv_sqr_attn /= self._norm_scaling
    
    def worker(self, object, x, return_list):
        '''
        Basic framework for parallel processes
        '''
        path_output = object.process(x)
        return_list.append(path_output)


class PropagationPath():
    '''
    Defines a single propagation path from source to receiver.

    '''
    def __init__(
            self,
            flightpath,
            fs=48_000,
            reflection_surface=None,
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
        '''Array describing position of source at every sample point.'''

        # calculate delays and amplitude curve
        _, _, r = utils.cart_to_sph(flightpath)
        delays = (r / c) * self.fs
        self._init_delay = delays[0]
        self._delta_delays = np.diff(delays)
        self._inv_sqr_attn = 1 / r**2

        # calculate angles per frame for filters
        self._sph_per_frame = utils.cart_to_sph(
            np.lib.stride_tricks.sliding_window_view(
                flightpath, self.frame_len, 1)[:, ::self._hop_len].mean(2)
        ).T

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

    def _filter(self, x):
        # neat trick to get windowed frames
        x_windowed = np.lib.stride_tricks.sliding_window_view(
            x, self.frame_len)[::self._hop_len] * np.hanning(self.frame_len)

        # list of filter stages
        filters = []

        # add atmospheric absorption
        filters.append(AtmosphericAbsorptionFilter())

        # add ground filter if surface is set (reflected path)
        if self.reflection_surface is not None:
            filters.append(GroundReflectionFilter(self.reflection_surface))

        x_out = np.zeros(len(x))

        for i, (position, frame) in enumerate(
                zip(self._sph_per_frame, x_windowed)):

            for filter in filters:
                frame = filter.filter(frame, position)

            frame_index = i * self._hop_len
            x_out[frame_index:frame_index + self.frame_len] += frame

        return x_out

    def process(self, x):
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
        if len(x) < len(self._delta_delays + 1):
            raise ValueError('Input signal shorter than path to be rendered')

        output = pipe(
            x,
            self._apply_doppler,
            self._filter,
            self._apply_attenuation
        )
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
                 freqs=np.geomspace(20, 24000),
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
        _, _, r = position
        h = signal.firls(self.n_taps, self.freqs,
                         self._attenuation**r,
                         fs=self.fs)
        return signal.fftconvolve(x, h, 'same')


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
