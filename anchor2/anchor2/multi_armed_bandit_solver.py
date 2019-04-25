from typing import List
import numpy as np
from loguru import logger


def _kl_bernoulli(p: float, q: float):
    """
    Computes Kullback-Leibler divergence between two Bernoulli distributions
    parametrized by p and q
    It is mentioned as 'd' in Kauffman&Kalyanakrishnan paper
    :param p: parameter of the first Bernoulli distribution
    :param q: parameter of the second Bernoulli distribution
    :return: Kullback-Leibler divergence between two Bernoulli distributions.
    """
    p = min(0.9999999999999999, max(0.0000001, p))  # Numerical stabilization
    q = min(0.9999999999999999, max(0.0000001, q))  # Numerical stabilization
    return p * np.log(float(p) / q) + (1 - p) * np.log(float(1 - p) / (1 - q))


class BernoulliArm:
    """
    Class represents the arm with bernoulli distributed reward
    in a multi-armed bandit scenario
    """

    def __init__(self, obj, draw_fn):
        self.draw_fn = draw_fn
        self.obj = obj
        self.n_samples = 0
        self.cumulative_reward = 0

    def draw(self) -> float:
        """
        Draws a sample from bernoulli distribution.
        :return:
        """
        n_samples, reward = self.draw_fn()
        self.n_samples += n_samples
        self.cumulative_reward += reward

    def mean(self) -> float:
        return self.cumulative_reward / self.n_samples

    def upper_kl_bound(self, beta):
        """
        Calculates eq(4) from K&K paper.
        It divides the range [theoretical_minimum, mean] into 17 consequent
        binary splits to find the minimum to eq(4)
        :param beta: exploration rate
        :return: upper bound for that arm's distribution with given beta
        """
        level = beta / self.n_samples
        # return qM>mean such that d(mean,qM)<=level
        lowest_edge = self.mean()
        upper_edge = min(min(1, self.mean() + np.sqrt(level / 2.)), 1)
        for j in range(1, 17):  # Why 17?
            next_candidate = (upper_edge + lowest_edge) / 2.
            if _kl_bernoulli(self.mean(), next_candidate) > level:
                upper_edge = next_candidate
            else:
                lowest_edge = next_candidate
        return upper_edge

    def lower_kl_bound(self, beta):
        # Level = beta\N_a. Why it is called level?
        # return lM<p such that d(p,lM)<=level
        """
        Calculates eq(5) from K&K paper.
        It divides the range [theoretical_minimum, mean] into 17 consequent
        binary splits to find the minimum to eq(5)
        :param beta: exploration rate, passed from MA Bandit Solver
        :return: lower bound for that arm's distribution with given beta
        """
        level = beta / self.n_samples
        theoretical_lowest_boundary = self.mean() - np.sqrt(level / 2.)
        upper_edge = self.mean()
        lowest_edge = max(min(1, theoretical_lowest_boundary), 0)
        for j in range(1, 17):  # Why 17?
            next_candidate = (upper_edge + lowest_edge) / 2.
            if _kl_bernoulli(self.mean(), next_candidate) > level:
                lowest_edge = next_candidate
            else:
                upper_edge = next_candidate
        return lowest_edge

    def __str__(self):
        return f"{str(self.obj)} with mean p = {self.mean():.2f}; sampled {self.n_samples}"


class MultiArmedBanditSolver:
    """
    Parent class for all algorithms subclasses, which goal is to
    select a single or subset of best arms from a multi-armed bandit
    """

    def __int__(self, arms: List[BernoulliArm]):
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

    def __init__(self, arms: List[BernoulliArm]):
        self.arms = np.array(arms)

    def _compute_exploration_rate(self, ):
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
        temperature = np.log(k * number_of_arms * (self.iteration ** alpha) / self.delta)
        return temperature + np.log(temperature)

    def get(self, n: int, delta: float, tolerance) -> List[BernoulliArm]:
        """
        Computes best n arms in a given set of arms.
        :param tolerance: minimum distance between lowest boundary of best-n arms
        and highest upper boundary of other arms to be achieved by an algorithm. Stopping
        criteria for an algorithm
        :param n: Number of arms to return
        :param delta: Mistake probability. It is the probability that chosen set of arms
        will not be a subset of optimal arms.
        """
        self.delta = delta
        self.n = n
        self.iteration = 1

        for arm in self.arms:
            arm.draw()  # Initialize each arms mean

        # Compute each arms mean +- bounds
        means = np.array([arm.mean() for arm in self.arms])
        beta = self._compute_exploration_rate()
        upper_bounds = np.array([arm.upper_kl_bound(beta) for arm in self.arms])
        lower_bounds = np.array([arm.lower_kl_bound(beta) for arm in self.arms])

        best_mean = np.argsort(means)[-n:]
        not_best_mean = np.argsort(means)[:-n]

        lower_bounds[not_best_mean] = 1.0  # make lower bounds of non-best arms equal one
        # to determine highest upper bound only for best arms

        upper_bounds[best_mean] = 0.0  # nullify upper bounds of best arms
        # to determine highest upper bound only for non-best arms

        l: int = np.argmax(upper_bounds)  # l from K&K paper. The worst among best arms
        u: int = np.argmin(lower_bounds)  # u from K&K paper. The best among not best arms

        lowest_lower_bound_among_best_arms = lower_bounds[u]
        highest_higher_bound_among_not_best_arms = upper_bounds[l]

        gap = highest_higher_bound_among_not_best_arms - lowest_lower_bound_among_best_arms
        logger.info(f"{self.iteration}-th iteration gap is {gap:.3f}")
        logger.info(f"L = {self.arms[u]}, U = {self.arms[l]}")
        logger.info(f"arms = {[str(a) for a in self.arms]}")

        logger.info(f"Means = {means}")
        # raise Exception("FIX THE GAP")

        while gap > tolerance:
            # Update iteration number, and draw samples from  l and u arms.
            self.iteration += 1
            self.arms[l].draw()
            self.arms[u].draw()

            # Compute each arms mean +- bounds
            means = np.array([arm.mean() for arm in self.arms])
            beta = self._compute_exploration_rate()
            upper_bounds = np.array([arm.upper_kl_bound(beta) for arm in self.arms])
            lower_bounds = np.array([arm.lower_kl_bound(beta) for arm in self.arms])

            best_mean = np.argsort(means)[-n:]
            not_best_mean = np.argsort(means)[:-n]

            lower_bounds[not_best_mean] = 1.0  # make lower bounds of non-best arms equal one
            # to determine highest upper bound only for best arms

            upper_bounds[best_mean] = 0.0  # nullify upper bounds of best arms
            # to determine highest upper bound only for non-best arms

            l: int = np.argmax(upper_bounds)  # l from K&K paper. The worst among best arms
            u: int = np.argmin(lower_bounds)  # u from K&K paper. The best among not best arms

            lowest_lower_bound_among_best_arms = lower_bounds[u]
            highest_higher_bound_among_not_best_arms = upper_bounds[l]

            gap = highest_higher_bound_among_not_best_arms - lowest_lower_bound_among_best_arms
            logger.info(f"{self.iteration}-th iteration gap is {gap:.3f}")

        return self.arms[best_mean]
