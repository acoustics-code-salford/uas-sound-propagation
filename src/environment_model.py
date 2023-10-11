import numpy as np
from .interpolators import linear, lagrange


class UASEventRenderer():
    def __init__(
            self,
            flight_parameters,
            fs=192000,
            receiver_height=1.5,
            direct_amplitude=1.0,
            reflection_amplitude=0.2
            ) -> None:

        self.fs = fs
        self.receiver_height = receiver_height
        self.direct_amplitude = direct_amplitude
        self.reflection_amplitude = reflection_amplitude
        self.flight_parameters = flight_parameters

    def render(self, signal):
        # apply each propagation path to input signal
        direct = self.direct_path.process(signal)
        reflection = self.ground_reflection.process(signal)

        # in samples
        reflection_offset = (self.ground_reflection.init_delay
                              - self.direct_path.init_delay)

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
    def direct_amplitude(self):
        return self._direct_amplitude

    @direct_amplitude.setter
    def direct_amplitude(self, amp):
        self._direct_amplitude = amp
        if hasattr(self, 'flight_parameters'):
            self._setup_paths()

    @property
    def reflection_amplitude(self):
        return self._reflection_amplitude

    @reflection_amplitude.setter
    def reflection_amplitude(self, amp):
        self._reflection_amplitude = amp
        if hasattr(self, 'flight_parameters'):
            self._setup_paths()

    @property
    def flight_parameters(self):
        return self._flight_parameters

    @flight_parameters.setter
    def flight_parameters(self, params):
        self._x_positions = np.empty(0)
        self._y_positions = np.empty(0)

        for p in params:
            x_next, y_next = self._xy_over_time(*p)
            self._x_positions = np.append(self._x_positions, x_next)
            self._y_positions = np.append(self._y_positions, y_next)

        self._setup_paths()
        self._flight_parameters = params

    def _setup_paths(self):
        # set up direct and reflected paths
        self.direct_path = PropagationPath(
            self._x_positions,
            self._y_positions,
            self.direct_amplitude,
            self.fs,
            self.receiver_height
        )

        self.ground_reflection = PropagationPath(
            self._x_positions,
            -self._y_positions,
            self.reflection_amplitude,
            self.fs,
            self.receiver_height
        )

    def _xy_over_time(self, start, end, speed_ramp):
        t_interval = 1/self.fs

        # extract initial and final speeds
        s_start, s_end = speed_ramp
        x_start, y_start = start
        x_end, y_end = end

        # find out distance over which accel/deceleration takes place
        accel_distance = np.linalg.norm(np.array([start]) - np.array([end]))

        theta = np.arctan2((y_end - y_start), (x_end - x_start))

        # calculate acceleration
        acceleration = (
            lambda start, end, distance: ((end**2) - (start**2)) / (2*distance)
        )(s_start, s_end, accel_distance)

        if acceleration == 0:  # no accel / decel
            x_diff = s_start * np.cos(theta)
            y_diff = s_start * np.sin(theta)

            n_output_samples = (
                np.ceil((accel_distance / s_start) * self.fs).astype(int)
            )

            # construct x/y distance arrays
            if abs(x_diff) < 1e-10:
                x_distances = np.ones(n_output_samples) * x_start
            else:
                x_distances = np.arange(x_start, x_end, x_diff * t_interval)

            if abs(y_diff) < 1e-10:
                y_distances = np.ones(n_output_samples) * y_start
            else:
                y_distances = np.arange(y_start, y_end, y_diff * t_interval)

        else:
            # init position
            position = 0

            x_distances = np.empty(0)
            y_distances = np.empty(0)

            # this operates per-sample so can take a while with large fs
            while position < accel_distance:
                position += s_start * t_interval

                x_distances = np.append(x_distances, x_start)
                y_distances = np.append(y_distances, y_start)

                x_start += s_start * np.cos(theta) * t_interval
                y_start += s_start * np.sin(theta) * t_interval

                s_start += acceleration * t_interval

        return x_distances, y_distances


class PropagationPath():
    def __init__(
            self,
            x_distances,
            y_distances,
            max_amp=1.0,
            fs=44100,
            receiver_height=1.5,
            c=330
    ):

        self.max_amp = max_amp
        self.c = c
        self.fs = fs

        y_distances -= receiver_height

        self.hyp_distances = np.linalg.norm(
            np.array([x_distances, y_distances]).T, axis=1
        )
        
        # calculate delays and amplitude curve
        delays = (self.hyp_distances / self.c) * self.fs
        self.init_delay = delays[0]
        self.delta_delays = np.diff(delays)
        self.amp_env = 1 / (self.hyp_distances**2)
        self.amp_env /= max(self.amp_env) / self.max_amp

    def apply_doppler(self, x, interpolate=linear):
        # init output array and initial read position
        
        out = np.zeros(len(self.delta_delays) + 1)
        read_pointer = 0
        
        for i in range(len(out)):
            
            n = int(read_pointer)  # whole part
            s = read_pointer % 1  # fractional part

            # replace this with call to interpolator
            out[i] = interpolate(x, n, s)

            read_pointer += (1 - self.delta_delays[i-1])

        return out

    def apply_amp_env(self, signal):
        return signal * self.amp_env

    def process(self, signal):
        if len(signal) < len(self.hyp_distances):
            raise ValueError('Input signal shorter than path to be rendered')

        return self.apply_amp_env(
            self.apply_doppler(signal)
        )
