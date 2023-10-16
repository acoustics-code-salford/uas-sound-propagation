import uasevent.utils as utils
import random
import unittest
import uasevent.interpolators as interpolators
from uasevent.environment_model import nearest_whole_fraction


class TestInterpolators(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        # one-second test sine signal
        self.test_x = utils.test_sine(1)

    def basic_linear(self, x, n, s):
        return x[n] + (s * (x[n+1] - x[n]))

    def test_interpolators(self):
        frac_sample_pos = random.randint(0, len(self.test_x)) + random.random()
        n, s = nearest_whole_fraction(frac_sample_pos)
        
        # basic linear interpolation for ballpark figure
        baseline = self.basic_linear(self.x, n, s)

        # assert various interpolators give close results
        sinc_interp = interpolators.SincInterpolator(self.x, n)
        self.assertAlmostEqual(sinc_interp[n], baseline, 2)

        allpass_interp = interpolators.AllpassInterpolator(self.x, n)
        self.assertAlmostEqual(allpass_interp[n], baseline, 2)

        lin_interp = interpolators.LinearInterpolator(self.x, n)
        self.assertEqual(lin_interp[n], baseline)

## make sure function used in model is also working properly