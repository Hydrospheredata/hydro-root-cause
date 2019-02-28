from typing import List
import numpy as np


class Arm:
    """
    Class represents the arm with bernoulli distributed reward
    in a multi-armed bandit scenario
    """

    def draw(self) -> float:
        pass


class MultiArmedBanditSolver:
    """
    Parent class for all algorithms subclasses, which goal is to
    select a single or subset of best arms from a multi-armed bandit
    """

    def __int__(self, arms: List[Arm]):
        self.arms = arms


class KullbackLeiblerLUCB(MultiArmedBanditSolver):
    """
    KL-LUCB method is introduced in http://proceedings.mlr.press/v30/Kaufmann13.pdf
    It is adaptive sampling pure exploration algorithm which returns a set of arms in
    multi-armed bandit problem.

    Algorithm samples best (by upper_bound) among not best arms and worst (by lower_bound)
    among best arms until the gap between estimated reward is tolerable. Tolerance for stopping
    and number of best arms returned as a hyperparameters.
    """

    @staticmethod
    def _kl_bernoulli(p: float, q: float):
        """
        Computes Kullback-Leibler divergence between two Bernoulli distributions.
        It is mentioned as 'd' in Kauffman&Kalyanakrishnan paper
        :param p:
        :param q:
        :return:
        """
        p = min(0.9999999999999999, max(0.0000001, p))  # Numerical stabilization
        q = min(0.9999999999999999, max(0.0000001, q))  # Numerical stabilization
        return p * np.log(float(p) / q) + (1 - p) * np.log(float(1 - p) / (1 - q))

    def _compute_beta_kl_lucb(self, ):
        """
        Beta is an exploration rate in KL-LUCB algorithm.
        Alpha and k is chosen as an in experiments of http://proceedings.mlr.press/v30/Kaufmann13.pdf
        Though, alpha can be any number > 1, and k can be any number if
        k > 2e + 1 + e/(a-1) + (e+1)/(a-1)^2. If alpha and k satisfy this inequalities,
        mistake probability will be at most delta.
        :return: Exploration rate
        """
        alpha = 1.1
        k = 405.5
        number_of_arms = len(self.arms)
        temperature = np.log(k * number_of_arms * (self.stage_number ** alpha) / self.delta)
        return temperature + np.log(temperature)

    def _update_bounds(self, ):
        pass

    def get(self, n: int, delta: float):
        """
        Computes best n arms in given set of arms
        :param n:
        :param delta: Mistake probability. It is the probability that chosen set of arms
        will not be a subset of optimal arms.
        """
        self.delta = delta
        self.n = n
