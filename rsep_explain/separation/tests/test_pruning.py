import os
import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.datasets import load_iris

from ..pruning import adversarial_pruning


class TestPruning(unittest.TestCase):
    def setUp(self):
        X, y = load_iris(return_X_y=True)
        ind = np.logical_or(y==0, y==1)
        self.X, self.y = X[ind], y[ind]

    def test_adversarial_pruning(self):
        augX, augy = adversarial_pruning(self.X, self.y, 1.5, 2)
        self.assertEqual(len(augX), 73)
        assert_almost_equal(
            augX[:5],
            np.array([[5.1, 3.5, 1.4, 0.2],
                      [4.9, 3. , 1.4, 0.2],
                      [4.7, 3.2, 1.3, 0.2],
                      [4.6, 3.1, 1.5, 0.2],
                      [5. , 3.6, 1.4, 0.2]])
            )
        assert_almost_equal(
            augy,
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1, 1, 1])
        )


if __name__ == '__main__':
    unittest.main()