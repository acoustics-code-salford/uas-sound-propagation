import numpy as np


class PropagationPath():
    def __init__(
            self,
            x_distances,
            y_distances,
            max_amp=1.0,
            refl_source=False,
            c=330,
            fs=44100,
            listener_height=1.5
    ):

        self.max_amp = max_amp
        self.c = c
        self.fs = fs
        self.t_interval = 1/fs

        if refl_source:
            y_distances = -y_distances

        y_distances -= listener_height

        self.hyp_distances = np.linalg.norm(
            [[x_i, y_i] for x_i, y_i in zip(
                x_distances, y_distances)], axis=1)

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

    def test_signal(self, f_0=440, A=0.75):
        # generates test sine wave

        t = int(self.total_time_seconds)
        t_samples = np.linspace(0, t, self.fs*t)
        sig = np.zeros_like(t_samples)

        for i, sample in enumerate(t_samples):
            sig[i] = np.sin(2*np.pi*f_0*sample) * A

        return sig

    def apply_doppler(self, signal):
        # establish output array - leave extra second at end for delay
        doppler_out = np.zeros(len(signal) + self.fs)

        # split delays into whole and fractional parts
        # linear interpolation requires oversampling to avoid excessive aiasing
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
                  self.apply_doppler(signal[:self.delays.shape[0]])
        )


def accel(s_end, s_start, d):
    return ((s_end**2) - (s_start**2)) / (2*d)


def xy_over_time(start, end, speed_ramp, fs=192000):

    t_interval = 1/fs

    # extract initial and final speeds
    s_start, s_end = speed_ramp
    x_start, y_start = start
    x_end, y_end = end

    # find out distance over which accel/deceleration takes place
    d = np.linalg.norm(np.array([start]) - np.array([end]))

    theta = np.arctan2((y_end - y_start), (x_end - x_start))

    # calculate acceleration
    acc = accel(s_end, s_start, d)

    if acc == 0:  # no accel / decel
        x_diff = s_start * np.cos(theta)
        y_diff = s_start * np.sin(theta)

        n_output_samples = (np.ceil((d / s_start) * fs).astype(int))

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
        pos = 0

        x_distances = np.empty(0)
        y_distances = np.empty(0)

        # this operates per-sample, so can take a long time with large fs
        while pos < d:
            pos += s_start * t_interval

            x_distances = np.append(x_distances, x_start)
            y_distances = np.append(y_distances, y_start)

            x_start += s_start * np.cos(theta) * t_interval
            y_start += s_start * np.sin(theta) * t_interval

            s_start += acc * t_interval

    return x_distances, y_distances


def render_flyby(
        signal,
        flight_params,
        fs=192000,
        listener_height=1.5,
        max_amp=1.0,
        relf_rel_amp=0.2
):

    # calculate x and y positions
    xp = np.empty(0)
    yp = np.empty(0)
    for p in flight_params:
        xstage, ystage = xy_over_time(*p)
        xp = np.append(xp, xstage)
        yp = np.append(yp, ystage)

    # objects to render direct path and ground reflection
    # very basic ground reflection implementation
    d = PropagationPath(xp, yp, fs=fs,
                        max_amp=max_amp,
                        listener_height=listener_height)
    r = PropagationPath(xp, yp, fs=fs,
                        max_amp=max_amp*relf_rel_amp,
                        refl_source=True,
                        listener_height=listener_height)

    # apply each propagation path to input signal
    direct = d.process(signal)
    reflection = r.process(signal)

    # calculate offset between paths
    delay_offset = r.tau_x_ax[0] - d.tau_x_ax[0]

    # combine direct and ground reflection path signals
    direct[delay_offset:] += reflection
    return direct
