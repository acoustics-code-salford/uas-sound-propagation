import random
import unittest
import numpy as np
import soundfile as sf
import uasevent.utils as utils
import uasevent.interpolators as interpolators
from uasevent.environment import UASEventRenderer, FlightPath
from numpy.testing import assert_array_almost_equal


class TestInterpolators(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        # one-second test sine signal
        self.x = utils.test_sine(1)

    def basic_linear(self, x, n, s):
        return x[n] + (s * (x[n - 1] - x[n]))

    def test_interpolators(self):
        frac_sample_pos = random.randint(0, len(self.x)) + random.random()
        n, s = utils.nearest_whole_fraction(frac_sample_pos)

        # basic linear interpolation for ballpark figure
        baseline = self.basic_linear(self.x, n, -s)

        lin_interp = interpolators.LinearInterpolator(self.x, -s)
        self.assertEqual(lin_interp[n], baseline)

        # assert various interpolators give close results
        sinc_interp = interpolators.SincInterpolator(self.x, -s)
        self.assertAlmostEqual(sinc_interp[n], baseline, 2)

        allpass_interp = interpolators.AllpassInterpolator(self.x, -s)
        self.assertAlmostEqual(allpass_interp[n], baseline, 2)

        windowed_interp = interpolators.interpolate(self.x, n, -s)
        self.assertAlmostEqual(windowed_interp, sinc_interp[n])

        # assert result is in correct range
        sample_a = self.x[n]
        sample_b = self.x[(n - 1) if s < 0 else (n + 1)]
        if sample_a > sample_b:
            self.assertGreater(sample_a, sinc_interp[n])
            self.assertGreater(sinc_interp[n], sample_b)
        else:
            self.assertLess(sample_a, sinc_interp[n])
            self.assertLess(sinc_interp[n], sample_b)


class TestTrajectoryCalc(unittest.TestCase):
    def test_trajectory(self):
        fast_accel = {
            'fast_accel': {
                'start': np.array([0, 0, 30]),
                'end': np.array([20, 0, 30]),
                'speeds': np.array([0, 10])
            }
        }

        slow_accel = {
            'slow_accel': {
                'start': np.array([0,  0, 30]),
                'end': np.array([20,  0, 30]),
                'speeds': np.array([0, 5])
            }
        }

        large_distance_const_speed = {
            'large_distance_const_speed': {
                'start': np.array([0, 0, 30]),
                'end': np.array([200, 200, 30]),
                'speeds': np.array([30, 30])
            }
        }

        large_distance_accel = {
            'large_distance_accel': {
                'start': np.array([-200, 0, 30]),
                'end': np.array([100, 50, 30]),
                'speeds': np.array([20, 30])
            }
        }

        large_distance_decel = {
            'large_distance_decel': {
                'start': np.array([-200, 0, 30]),
                'end': np.array([100, 50, 30]),
                'speeds': np.array([30, 20])
            }
        }

        fast_traj = FlightPath(fast_accel)(48000).T
        slow_traj = FlightPath(slow_accel)(48000).T
        long_dist_const_traj = FlightPath(large_distance_const_speed)(48000).T
        long_accel_traj = FlightPath(large_distance_accel)(48000).T
        long_decel_traj = FlightPath(large_distance_decel)(48000).T

        # lower acceleration should result in longer trajectories
        self.assertGreater(len(slow_traj), len(fast_traj))

        # accelerating and decelerating by the same amount over the
        # same distance should result in equal-length trajectories
        self.assertEqual(len(long_decel_traj), len(long_accel_traj))

        # constant speed trajectory should end very near where specified
        assert_array_almost_equal(
            long_dist_const_traj[-1],
            large_distance_const_speed['large_distance_const_speed']['end'],
            3)

        # accelerating trajectory should end very near where specified
        assert_array_almost_equal(
            long_accel_traj[-1],
            large_distance_accel['large_distance_accel']['end'],
            3)


class TestRender(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        self.x, fs = sf.read('tests/testsrc.wav')
        params = 'tests/test_flight.json'
        self.renderer = UASEventRenderer(params, fs, 'asphalt', 1.5)
        self.renderer.render(self.x)

        params_2 = 'tests/test_flight_2.json'
        self.renderer_2 = UASEventRenderer(params_2, fs, 'asphalt', 1.5)

    def test_output_sensible(self):
        # rendering should equal length of calculated trajectory
        self.assertEqual(
            len(self.renderer.direct_path.flightpath(48000).T),
            len(self.renderer._d + self.renderer._r))

        # direct path should have higher power than reflection
        self.assertGreater(np.sum(abs(self.renderer._d)),
                           np.sum(abs(self.renderer._r)))

        # first n samples of reflected signal should be zero
        n = int(np.floor(
            self.renderer.ground_reflection._init_delay
            - self.renderer.direct_path._init_delay
        ))
        self.assertTrue((self.renderer._r[:n] == 0).all())

        # rendering of greater distance should have larger scaling
        self.assertGreater(1/self.renderer_2._norm_scaling,
                           1/self.renderer._norm_scaling)
