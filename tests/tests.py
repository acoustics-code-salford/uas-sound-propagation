import uasevent.utils as utils
import random
import unittest
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
        baseline = self.basic_linear(self.x, n, s)

        lin_interp = interpolators.LinearInterpolator(self.x, s)
        self.assertEqual(lin_interp[n], baseline)

        # assert various interpolators give close results
        sinc_interp = interpolators.SincInterpolator(self.x, s)
        self.assertAlmostEqual(sinc_interp[n], baseline, 3)

        if s < 1:
            n += 1
            s += 1
        allpass_interp = interpolators.AllpassInterpolator(self.x, s)
        self.assertAlmostEqual(allpass_interp[n], baseline, 3)
