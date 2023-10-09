import numpy as np


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
        self.flight_parameters = flight_parameters
        self.receiver_height = receiver_height
        self.direct_amplitude = direct_amplitude
        self.reflection_amplitude = reflection_amplitude
        # TODO: make propagation parameters editable 
        # (presently path calculations will not change if flight parameters 
        # are altered after init)

        self.direct_path = PropagationPath(
            self.x_positions, 
            self.y_positions,
            self.direct_amplitude,
            self.fs,
            self.receiver_height
        )

        self.ground_reflection = PropagationPath(
            self.x_positions,
            -self.y_positions,
            self.reflection_amplitude,
            self.fs,
            self.receiver_height
        )

    def render(self, signal):
        # apply each propagation path to input signal
        direct = self.direct_path.process(signal)
        reflection = self.ground_reflection.process(signal)

        # calculate offset between paths
        delay_offset = (
            self.ground_reflection.tau_x_ax[0] - self.direct_path.tau_x_ax[0]
        )
        # combine direct and ground reflection path signals
        direct[delay_offset:] += reflection
        return direct

    @property
    def flight_parameters(self):
        return self._flight_parameters
    
    @flight_parameters.setter
    def flight_parameters(self, params):
        self.x_positions = np.empty(0)
        self.y_positions = np.empty(0)

        for p in params:
            x_next, y_next = self._xy_over_time(*p)
            self.x_positions = np.append(self.x_positions, x_next)
            self.y_positions = np.append(self.y_positions, y_next)
        
        self._flight_parameters = params

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
        self.t_interval = 1/fs

        y_distances -= receiver_height

        self.hyp_distances = np.linalg.norm(
            [[x_i, y_i] for x_i, y_i in zip(x_distances, y_distances)],
            axis=1
        )

        self.calc_delays()
        self.calc_amplitudes()

    def calc_delays(self):
        # delay times uncorrected for source movement
        self.straight_time_delays = self.hyp_distances/self.c

        # build array of time steps
        self.total_time_seconds = len(self.hyp_distances)/self.fs
        t_steps = np.arange(0, self.total_time_seconds, self.t_interval)
        t_steps = t_steps[:len(self.straight_time_delays)]

        # calculate tau (emission time)
        tau, self.tau_x_ax = self.calc_tau(t_steps)

        # correct delays for source movement
        self.delays = self.straight_time_delays[tau] * self.fs

    def calc_tau(self, t_steps):
        # tau calcaulated as per Rizzi/Sullivan
        tau = ((t_steps - self.straight_time_delays) * self.fs).astype(int)
        tau_x_ax = np.where(tau > 0)[0]
        tau = tau[tau_x_ax]

        return tau, tau_x_ax

    def calc_amplitudes(self):
        # calculate R_tau
        tau_shifted_distances = (self.delays / self.fs) * self.c

        # calculate amplitude curve
        inv_sqr_amplitudes = 1 / tau_shifted_distances**2

        # scale to specified maximum amplitude
        inv_sqr_amplitudes /= max(inv_sqr_amplitudes) / self.max_amp

        self.amp_env = inv_sqr_amplitudes

    def apply_doppler(self, signal):
        # establish output array - leave extra second at end for delay
        doppler_out = np.zeros(len(signal) + self.fs)

        # split delays into whole and fractional parts
        # linear interpolation requires oversampling to avoid 
        # excessive aliasing
        # TODO: implement more sophisticated interpolation
        whole_sample_delays = np.floor(self.delays).astype(int)
        fractional_delays = np.mod(self.delays, 1)

        for i, sample in enumerate(signal[:-1]):
            next_sample = signal[i + 1]
            doppler_out[i + whole_sample_delays[i]] = (
                sample * (fractional_delays[i])
                + next_sample * (1 - fractional_delays[i])
            )

        return doppler_out

    def apply_amp_env(self, signal):
        return signal[self.tau_x_ax] * self.amp_env

    def process(self, signal):
        if len(signal) < len(self.hyp_distances):
            raise ValueError('Input signal shorter than path to be rendered')

        return self.apply_amp_env(
                  self.apply_doppler(
                      signal[:self.delays.shape[0]]
                  )
        )
