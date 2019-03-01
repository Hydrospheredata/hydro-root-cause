from anchor2.anchor2.multi_armed_bandit_solver import KullbackLeiblerLUCB, BernoulliArm
from functools import partial
import numpy as np


def bernoulli_draw(p):
    return 100, np.sum(np.random.binomial(n=100, p=p))


ps = np.linspace(0, 100, num=100) / 100
arms = [BernoulliArm(f"Probability: {p:.2f}", partial(bernoulli_draw, p=p)) for p in ps]
solver = KullbackLeiblerLUCB(arms)
best_arms = solver.get(10, 0.05, 0.1)
[print(str(a)) for a in best_arms]
