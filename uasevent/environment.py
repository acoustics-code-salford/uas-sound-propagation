import numpy as np
from scipy import signal
from . import interpolators, utils


class UASEventRenderer():
    '''
    Main propagation model for rendering of drone flight events.

    '''
    def __init__(
            self,
            flight_parameters,
            ground_material='grass',
            fs=48_000,
            receiver_height=1.5,
            loudspeaker_mapping='Octagon + Cube'):
        '''
        Initialises all necessary attributes for the UASEventRenderer object.
        '''
        self.loudspeaker_mapping = loudspeaker_mapping
        '''Layout of loudspeaker array for rendering'''
        self.fs = fs
        '''Sampling frequency in Hz (default 48_000)'''
        self.receiver_height = receiver_height
        '''Height of receiver position, metres (default 1.5)'''
        self.ground_material = ground_material
        '''Material for ground reflection'''
        self.flight_parameters = flight_parameters
        '''Segment-wise description of flight path'''

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
        # apply each propagation path to input signal
        direct = self.direct_path.process(x)
        reflection = self.ground_reflection.process(x)

        # in samples
        offset = (self.ground_reflection._init_delay
                  - self.direct_path._init_delay)

        # calculate whole and fractional number of samples to delay reflection
        whole_offset, frac_offset = utils.nearest_whole_fraction(offset)
        # calculate fractional delay of reflection)
        reflection = np.array([
            i for i in interpolators.SincInterpolator(reflection, frac_offset)
        ])
        # add zeros to start of reflection to account for whole sample delay
        if whole_offset:
            reflection_zeros = np.zeros_like(reflection)
            reflection_zeros[:, whole_offset:] += reflection[:, :-whole_offset]
            reflection = reflection_zeros
        self._d = direct.T
        self._r = reflection.T
        output = direct.T + reflection.T
        return output

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
        self._flightpath = np.empty([3, 0])

        for p in params:
            self._flightpath = np.append(
                self._flightpath, utils.vector_t(*p), axis=1)

        self._setup_paths()
        self._flight_parameters = params

    def _setup_paths(self):
        # set up direct and reflected paths
        self.direct_path = PropagationPath(
            np.concatenate(
                (self._flightpath[:2],
                 self._flightpath[2:] - self.receiver_height)
            ),
            None,
            self.fs,
            loudspeaker_mapping=self.loudspeaker_mapping
        )

        self.ground_reflection = PropagationPath(
            np.concatenate(
                (self._flightpath[:2],
                 - self._flightpath[2:] - self.receiver_height)
            ),
            self.ground_material,
            self.fs,
            loudspeaker_mapping=self.loudspeaker_mapping
        )

        self._norm_scaling = np.max(abs(self.direct_path._inv_sqr_attn))
        self.direct_path._inv_sqr_attn /= self._norm_scaling
        self.ground_reflection._inv_sqr_attn /= self._norm_scaling


class PropagationPath():
    '''
    Defines a single propagation path from source to receiver.

    '''
    def __init__(
            self,
            flightpath,
            reflection_surface=None,
            fs=48_000,
            c=343.0,
            frame_len=512,
            loudspeaker_mapping='Octagon + Cube'
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

        # load loudspeaker layout
        self.loudspeaker_mapping = loudspeaker_mapping
        '''String to select layout of loudspeaker array for rendering
        (default `"Octagon + Cube"`).'''
        _, th, ph, r = utils.load_mapping(loudspeaker_mapping)
        self._ls_locs = utils.sph_to_cart(np.array([th, ph, r]).T)

        # calculate amplitude envelopes for each loudspeaker
        self.flightpath = flightpath
        '''Array describing position of source at every sample point.'''
        self._amp_envs = self._calculate_amp_envs(self._ls_locs)

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

    def _calculate_amp_envs(self, loudspeaker_locs):
        dbap = DBAP(loudspeaker_locs)
        fpath = np.copy(self.flightpath)
        # clip flightpath to (approximate) surface of array convex hull
        fpath = (fpath / np.linalg.norm(fpath, axis=0)) *\
            np.mean(np.linalg.norm(dbap._ls_pos, axis=0))
        return dbap.gains(fpath.T)

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

    def _apply_amp_envs(self, x):
        return x * self._inv_sqr_attn * self._amp_envs

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

        output = \
            self._apply_amp_envs(
                    self._filter(
                        self._apply_doppler(x)
                    )
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


class DBAP():
    '''
    Implements distance-based amplitude panning.
    Based on https://github.com/PasqualeMainolfi/Pannix/

    '''
    def __init__(self, loudspeaker_locs):
        '''
        Initialises DBAP.
        '''
        self.loudspeaker_locs = loudspeaker_locs
        '''Array defining cartesian locations of array loudspeakers.'''
        self._ls_pos = self.loudspeaker_locs.T
        self._spat_blur = np.mean(np.linalg.norm(self._ls_pos, axis=0)) + 0.2
        self._eta = self._spat_blur / len(self._ls_pos.T)

    def _loudspeaker_distance(self, pos_arr):
        return np.array(
            [
                np.sqrt(
                    np.sum(
                        (self._ls_pos.T - pos)**2, axis=1
                    ) + self._spat_blur**2
                )
                for pos in pos_arr
            ])

    def _b(self, d):
        u = d.T - d.max(axis=1)
        u_norm = np.linalg.norm(u, axis=0)
        u = u/u_norm
        u = u**2 + self._eta
        um = np.median(d, axis=1)
        return (2*u / um)**2 + 1

    def _k(self, b, d):
        k_den = np.sqrt(
            np.sum(
                (b**2).T /
                (d**2), axis=1
            )
        )
        return 1 / k_den

    def gains(self, pos_arr):
        '''
        Calculates gains applicable to each loudspeaker for input array of
        source positions.

        Parameters
        ----------
        `pos_arr`
            Array describing position of source at each sample time.

        Returns
        -------
        `gains`
            Array of gains for specified loudspeaker layout.
        '''
        d = self._loudspeaker_distance(pos_arr)
        b = self._b(d)
        k = self._k(b, d)
        gains = (k * b) / d.T
        return gains
