import unittest

import numpy as np

from scripts import test_caesar_era5


class CaesarEra5OptionsTest(unittest.TestCase):
    def test_limit_channels_keeps_first_n_channels(self):
        data = np.zeros((268, 16, 4, 4), dtype=np.float32)

        limited = test_caesar_era5.limit_channels(data, 32)

        self.assertEqual(limited.shape, (32, 16, 4, 4))

    def test_limit_channels_rejects_empty_limit(self):
        data = np.zeros((268, 16, 4, 4), dtype=np.float32)

        with self.assertRaises(ValueError):
            test_caesar_era5.limit_channels(data, 0)


if __name__ == "__main__":
    unittest.main()
