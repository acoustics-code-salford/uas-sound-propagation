import numpy as np
from scipy import signal
from . import interpolators, utils


class UASEventRenderer():
    def __init__(
            self,
            flight_parameters,
            ground_material='grass',
            fs=48000,
            receiver_height=1.5,
            ) -> None:

        self.fs = fs
        self.receiver_height = receiver_height
        self.ground_material = ground_material
        self.flight_parameters = flight_parameters

    def render(self, x):
        # apply each propagation path to input signal
        direct = self.direct_path.process(x)
        reflection = self.ground_reflection.process(x)

        # in samples
        offset = (self.ground_reflection.init_delay
                  - self.direct_path.init_delay)

        # calculate whole and fractional number of samples to delay reflection
        whole_offset, frac_offset = utils.nearest_whole_fraction(offset)
        # calculate fractional delay of reflection
        reflection = np.array([
            i for i in interpolators.SincInterpolator(reflection, frac_offset)
        ])
        # add zeros to start of reflection to account for whole sample delay
        reflection_zeros = np.zeros_like(reflection)
        reflection_zeros[whole_offset:] += reflection[:-whole_offset]
        reflection = reflection_zeros
        self.d = direct
        self.r = reflection
        return direct + reflection

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
            self.fs
        )

        self.ground_reflection = PropagationPath(
            np.concatenate(
                (self._flightpath[:2],
                 - self._flightpath[2:] - self.receiver_height)
            ),
            self.ground_material,
            self.fs
        )


class PropagationPath():
    def __init__(
            self,
            flightpath,
            reflection_surface=None,
            fs=48000,
            max_amp=1.0,
            c=330,
            frame_len=512,
            N=2
    ):

        self.max_amp = max_amp
        self.c = c
        self.fs = fs
        self.reflection_surface = reflection_surface
        self.theta, self.phi, self.r = utils.cart_to_sph(flightpath)

        # calculate delays and amplitude curve
        delays = (self.r / self.c) * self.fs
        self.init_delay = delays[0]
        self.delta_delays = np.diff(delays)
        self.amp_env = 1 / (self.r**2)

        self.frame_len = frame_len
        self.hop_len = frame_len // 2
        self.phi_per_frame = np.lib.stride_tricks.sliding_window_view(
                self.phi, self.frame_len)[::self.hop_len].mean(1)
        self.theta_per_frame = np.lib.stride_tricks.sliding_window_view(
                self.theta, self.frame_len)[::self.hop_len].mean(1)
        self.N = N

    def apply_doppler(self, x):
        # init output array and initial read position
        out = np.zeros(len(self.delta_delays) + 1)
        read_pointer = 0

        for i in range(len(out)):
            # find whole and fractional part
            n, s, = utils.nearest_whole_fraction(read_pointer)
            # negative s for correct indexing over n
            out[i] = interpolators.interpolate(x, n, -s)

            # update read pointer position
            read_pointer += (1 - self.delta_delays[i-1])

        return out

    def apply_amp_env(self, x):
        return x * self.amp_env

    def ground_effect(self, x):
        # do not apply filter if surface not set (direct path)
        if self.reflection_surface is None:
            return x

        # neat trick to get windowed frames
        x_windowed = np.lib.stride_tricks.sliding_window_view(
            x, self.frame_len)[::self.hop_len] * np.hanning(self.frame_len)

        refl_filter = GroundReflectionFilter(material=self.reflection_surface)

        x_out = np.zeros(len(x))
        for i, (angle, frame) \
                in enumerate(zip(self.phi_per_frame, x_windowed)):

            frame_index = i * self.hop_len
            angle = np.round(np.rad2deg(np.pi - angle))  # reflection angle

            x_out[frame_index:frame_index + self.frame_len] += \
                refl_filter.filter(frame, angle)

        return x_out

    def spatialise(self, x):
        # do not spatialise if order not set (mono output)
        if self.N is None:
            return x

        x_windowed = np.lib.stride_tricks.sliding_window_view(
            x, self.frame_len)[::self.hop_len] * np.hanning(self.frame_len)

        x_out = np.zeros((len(x), (self.N+1)**2))

        for i, (theta, phi, frame) \
                in enumerate(zip(self.theta_per_frame,
                                 self.phi_per_frame,
                                 x_windowed)):
            
            frame_index = i * self.hop_len
            Y_mn = utils.Y_array(self.N, np.array([theta]), np.array([phi]))
            x_out[frame_index:frame_index + self.frame_len] += (frame * Y_mn).T

        return x_out

    def process(self, x):
        if len(x) < len(self.r):
            raise ValueError('Input signal shorter than path to be rendered')

        return \
            self.spatialise(
                self.ground_effect(
                    self.apply_amp_env(
                        self.apply_doppler(x)
                    )
                )
            )


class GroundReflectionFilter():
    def __init__(
            self,
            freqs=np.geomspace(20, 24000),
            angles=np.arange(1, 91),
            material='asphalt',
            Z_0=413.26,
            fs=48000,
            n_taps=21
            ):

        self.freqs = freqs
        self.phi = np.deg2rad(angles)
        self.Z_0 = Z_0
        self.material = material
        self.fs = fs
        self.n_taps = n_taps
        self.filterbank = self._compute_filterbank()

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

    def _R(self):
        return np.real(
            np.array([
                (self._Z() * np.cos(p) - self.Z_0) /
                (self._Z() * np.cos(p) + self.Z_0)
                for p in self.phi
            ])
        ).squeeze()

    def _compute_filterbank(self):
        return np.array([
            signal.firls(self.n_taps, self.freqs, abs(r), fs=self.fs)
            for r in self._R()
        ])

    def filter(self, x, angle):
        angle = round(angle)
        h = self.filterbank[angle - 1]
        return signal.fftconvolve(x, h, 'same')
