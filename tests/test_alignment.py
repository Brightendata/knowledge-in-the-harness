"""The forest tells two sets apart when each tag has the same share on both sides but pairs differ; the linear model
cannot. Run: python -m unittest tests.test_alignment"""
import unittest

import numpy as np

from rsi.data_preprocessing.alignment import fold_auc


class AlignmentTest(unittest.TestCase):
    def test_pairs(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 2, (400, 2))
        y = (a[:, 0] == a[:, 1]).astype(int)  # both tags at about half on each side, together only on one
        X = np.hstack([a, rng.integers(0, 2, (400, 3))]).astype(float)
        groups = np.arange(len(y))
        self.assertLess(fold_auc("linear", X, y, groups, 0), 0.65)
        self.assertGreater(fold_auc("forest", X, y, groups, 0), 0.9)


if __name__ == "__main__":
    unittest.main()
