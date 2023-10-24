import numpy as np
from scipy import signal
from . import interpolators
from .utils import nearest_whole_fraction


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
        whole_offset, frac_offset = nearest_whole_fraction(offset)
        # calculate fractional delay of reflection
        reflection = interpolators.SincInterpolator(reflection, frac_offset)
        # add zeros to start of reflection to account for whole sample delay
        reflection = np.concatenate(
            (
                np.zeros(whole_offset),
                np.array([i for i in reflection])[:-whole_offset]
            )
        )
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
        self._x_positions = np.empty(0)
        self._z_positions = np.empty(0)

        for p in params:
            x_next, z_next = self._xz_over_time(*p)
            self._x_positions = np.append(self._x_positions, x_next)
            self._z_positions = np.append(self._z_positions, z_next)

        self._setup_paths()
        self._flight_parameters = params

    def _setup_paths(self):
        # set up direct and reflected paths
        self.direct_path = PropagationPath(
            self._x_positions,
            self._z_positions,
            self.receiver_height,
            None,
            self.fs
        )

        self.ground_reflection = PropagationPath(
            self._x_positions,
            -self._z_positions,
            self.receiver_height,
            self.ground_material,
            self.fs
        )

    def _xz_over_time(self, start, end, speeds):
        v_0, v_T = speeds
        distance = np.linalg.norm(start - end)
        # heading of source
        vector = ((end - start) / distance)
        # acceleration
        a = ((v_T**2) - (v_0**2)) / (2 * distance)
        # number of time steps in samples for operation
        n_output_samples = self.fs * ((v_T-v_0) / a) if a != 0 else \
            (distance / v_0) * self.fs

        # array of positions at each time step
        x_t = np.array([
            [
                (v_0 * (t / self.fs)) 
                + ((a * (t / self.fs)**2) / 2) 
                for t in range(int(n_output_samples))
            ]
        ]).T

        # map to axes
        xyz = start + vector * x_t
        x, _, z = xyz.T
        return x, z


class PropagationPath():
    def __init__(
            self,
            x_distances,
            z_distances,
            receiver_height=1.5,
            reflection_surface=None,
            fs=48000,
            max_amp=1.0,
            c=330
    ):

        self.max_amp = max_amp
        self.c = c
        self.fs = fs
        self.reflection_surface = reflection_surface

        z_distances -= receiver_height

        self.hyp_distances = np.linalg.norm(
            np.array([x_distances, z_distances]).T, axis=1
        )
        self.incident_angles = abs(np.arctan(z_distances / x_distances))

        # calculate delays and amplitude curve
        delays = (self.hyp_distances / self.c) * self.fs
        self.init_delay = delays[0]
        self.delta_delays = np.diff(delays)
        self.amp_env = 1 / (self.hyp_distances**2)
        self.amp_env /= max(self.amp_env) / self.max_amp

    def apply_doppler(self, x):
        # init output array and initial read position
        out = np.zeros(len(self.delta_delays) + 1)
        read_pointer = 0

        for i in range(len(out)):
            # find whole and fractional part
            n, s, = nearest_whole_fraction(read_pointer)
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

        frame_len = 512
        hop_len = frame_len // 2
        window = np.hanning(frame_len)

        # neat trick to get windowed frames
        x_windowed = np.lib.stride_tricks.sliding_window_view(
            x, frame_len)[::hop_len] * window

        angle_per_frame = np.round(np.rad2deg(
            np.lib.stride_tricks.sliding_window_view(
                self.incident_angles, frame_len)[::hop_len].mean(1)))

        refl_filter = GroundReflectionFilter(material=self.reflection_surface)

        x_out = np.zeros(len(x))
        for i, (angle, frame) in enumerate(zip(angle_per_frame, x_windowed)):
            frame_index = i * hop_len
            x_out[frame_index:frame_index + frame_len] += \
                refl_filter.filter(frame, angle)
        return x_out

    def process(self, x):
        if len(x) < len(self.hyp_distances):
            raise ValueError('Input signal shorter than path to be rendered')

        return \
            self.ground_effect(
                self.apply_amp_env(
                    self.apply_doppler(x)
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
        # angle defined between ground plane and wave path
        # for definition between vertical, use cos
        return np.real(
            np.array([
                (self._Z() * np.sin(p) - self.Z_0) /
                (self._Z() * np.sin(p) + self.Z_0)
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
