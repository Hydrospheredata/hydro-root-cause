import unittest
from functools import partial

import numpy as np
from anchor2.anchor2.multi_armed_bandit_solver import KullbackLeiblerLUCB, BernoulliArm
from loguru import logger

logger.disable("anchor2.anchor2")


class TestKLLUCB(unittest.TestCase):

    def test_bernoulli(self):
        def bernoulli_draw(p):
            return 100, np.sum(np.random.binomial(n=100, p=p))

        # This test is supposed to fail sometimes, so we count # of successes
        successes = []
        ps = np.linspace(0, 100, num=100) / 100

        for i in range(50):
            arms = [BernoulliArm(f"{p:.2f}", partial(bernoulli_draw, p=p)) for p in ps]
            solver = KullbackLeiblerLUCB(arms)
            best_arms = solver.get(10, 0.05, 0.1)
            best_probabilities = sorted([a.obj for a in best_arms])
            successes.append(set("0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.00".split()) == set(best_probabilities))
        self.assertGreater(np.mean(successes), 0.7)
