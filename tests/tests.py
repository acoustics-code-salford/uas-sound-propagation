import random
import unittest
import numpy as np
import uasevent.utils as utils
import uasevent.interpolators as interpolators
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
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_trajectory(self):
        fast_accel = [np.array([0,    0,   30]),
                      np.array([20,   0,  30]),
                      np.array([0, 10])]

        slow_accel = [np.array([0,    0,   30]),
                      np.array([20,   0,  30]),
                      np.array([0, 5])]

        large_distance_const_speed = [np.array([0,    0,   30]),
                                      np.array([200,   200,  30]),
                                      np.array([30, 30])]

        large_distance_accel = [np.array([-200,    0,   30]),
                                np.array([100,   50,  30]),
                                np.array([20, 30])]

        large_distance_decel = [np.array([-200,    0,   30]),
                                np.array([100,   50,  30]),
                                np.array([30, 20])]

        fast_traj = utils.vector_t(*fast_accel).T
        slow_traj = utils.vector_t(*slow_accel).T
        long_dist_const_traj = utils.vector_t(*large_distance_const_speed).T
        long_accel_traj = utils.vector_t(*large_distance_accel).T
        long_decel_traj = utils.vector_t(*large_distance_decel).T

        self.assertGreater(len(slow_traj), len(fast_traj))
        assert_array_almost_equal(long_dist_const_traj[-1],
                                  large_distance_const_speed[1],
                                  3)
        self.assertEqual(len(long_decel_traj), len(long_accel_traj))
        assert_array_almost_equal(long_accel_traj[-1],
                                  large_distance_accel[1],
                                  3)
