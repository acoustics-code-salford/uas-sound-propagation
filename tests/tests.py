import random
import unittest
import uasevent.utils as utils
import uasevent.interpolators as interpolators


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
