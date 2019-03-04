import unittest

from anchor2.anchor2.multi_armed_bandit_solver import KullbackLeiblerLUCB, BernoulliArm
from functools import partial
import numpy as np


class TestKLLUCB(unittest.TestCase):

    def test_bernoulli(self):
        # TODO add probabilistic property to the test
        def bernoulli_draw(p):
            return 100, np.sum(np.random.binomial(n=100, p=p))

        ps = np.linspace(0, 100, num=100) / 100
        arms = [BernoulliArm(f"{p:.2f}", partial(bernoulli_draw, p=p)) for p in ps]
        solver = KullbackLeiblerLUCB(arms)
        best_arms = solver.get(10, 0.05, 0.1)
        best_probabilities = sorted([a.obj for a in best_arms])
        self.assertListEqual("0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.00".split(), best_probabilities)
