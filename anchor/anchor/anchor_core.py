"""Base anchor functions"""

import collections
import copy
from typing import Dict, List, Callable, Tuple

import numpy as np


def _matrix_subset(matrix, n_samples):
    if matrix.shape[0] == 0:
        return matrix
    n_samples = min(matrix.shape[0], n_samples)
    return matrix[np.random.choice(matrix.shape[0], n_samples, replace=False)]


def _normalize_tuple(x):
    """
    Return sorted tuple without duplicate values
    :param x: tuple to be sorted and filtered
    :return: normalized value
    """
    return tuple(sorted(set(x)))


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


def _dup_bernoulli(p, level):
    # KL upper confidence bound:
    # return qM>p such that d(p,qM)=level
    lm = p
    um = min(min(1, p + np.sqrt(level / 2.)), 1)
    for j in range(1, 17):
        qm = (um + lm) / 2.
        if _kl_bernoulli(p, qm) > level:
            um = qm
        else:
            lm = qm
    return um


def _dlow_bernoulli(p, level):
    # KL lower confidence bound
    # return lM<p such that d(p,lM)=level
    um = p
    lm = max(min(1, p - np.sqrt(level / 2.)), 0)
    for j in range(1, 17):
        qm = (um + lm) / 2.
        if _kl_bernoulli(p, qm) > level:
            lm = qm
        else:
            um = qm
    return lm


def _compute_beta_kl_lucb(number_of_arms: int, stage_number: int, delta: float):
    """
    Beta is an exploration rate in KL-LUCB algortihm.
    Alpha and k is chosen as an in experiments of http://proceedings.mlr.press/v30/Kaufmann13.pdf
    Though, alpha can be any number > 1, and k can be any number if
    k > 2e + 1 + e/(a-1) + (e+1)/(a-1)^2. If alpha and k satisfy this inequalities,
    mistake probability will be at most delta.
    :param number_of_arms: Number of anchors, or arms in multi-armed bandit setting
    :param stage_number: Stage number in LUCB algorithm
    :param delta: Mistake probability. It is the probability that chosen set of arms
    will not be a subset of optimal arms.
    :return: Exploration rate
    """
    alpha = 1.1
    k = 405.5
    temperature = np.log(k * number_of_arms * (stage_number ** alpha) / delta)
    return temperature + np.log(temperature)


class AnchorBaseBeam(object):
    def __init__(self):
        pass

    @staticmethod
    def kl_lucb(sampling_functions,
                num_samples_evaluated,
                anchor_precision_score,
                tolerance,
                delta,
                batch_size,
                number_of_anchors):
        """
        KL-LUCB method is introduced in http://proceedings.mlr.press/v30/Kaufmann13.pdf
        It is adaptive sampling pure exploration algorithm which returns a set of arms in
        multi-armed bandit problem.

        TODO: Complete documentation for this method!
        :param anchor_precision_score:
        :param num_samples_evaluated:
        :param sampling_functions:
        :param tolerance: Tolerance
        :param delta: Width
        :param batch_size:
        :param number_of_anchors:
        :return:
        """
        n_features = len(sampling_functions)

        anchor_precision_score = np.array(anchor_precision_score)
        num_samples_evaluated = np.array(num_samples_evaluated)

        upper_bounds = np.zeros_like(num_samples_evaluated)
        lower_bounds = np.zeros_like(num_samples_evaluated)

        for f in np.where(num_samples_evaluated == 0)[0]:
            num_samples_evaluated[f] += 1
            anchor_precision_score[f] += sampling_functions[f](1)
        if n_features == number_of_anchors:
            return range(n_features)
        anchor_means = anchor_precision_score / num_samples_evaluated
        stage_number = 1

        def update_bounds(i):
            beta = _compute_beta_kl_lucb(n_features, i, delta)

            _sorted_means = np.argsort(anchor_means)
            best_anchors_idxs = _sorted_means[-number_of_anchors:]
            not_best_anchors_idxs = _sorted_means[:-number_of_anchors]
            for anchor_idx in not_best_anchors_idxs:
                upper_bounds[anchor_idx] = _dup_bernoulli(anchor_means[anchor_idx], beta / num_samples_evaluated[anchor_idx])
            for anchor_idx in best_anchors_idxs:
                lower_bounds[anchor_idx] = _dlow_bernoulli(anchor_means[anchor_idx], beta / num_samples_evaluated[anchor_idx])

            new_upper_bound_idx = not_best_anchors_idxs[np.argmax(upper_bounds[not_best_anchors_idxs])]
            new_lower_bound_idx = best_anchors_idxs[np.argmin(lower_bounds[best_anchors_idxs])]

            return new_upper_bound_idx, new_lower_bound_idx

        upper_bound_idx, lower_bound_idx = update_bounds(stage_number)
        gap = upper_bounds[upper_bound_idx] - lower_bounds[lower_bound_idx]

        while gap > tolerance:
            stage_number += 1

            num_samples_evaluated[upper_bound_idx] += batch_size
            num_samples_evaluated[lower_bound_idx] += batch_size

            anchor_precision_score[upper_bound_idx] += sampling_functions[upper_bound_idx](batch_size)
            anchor_precision_score[lower_bound_idx] += sampling_functions[lower_bound_idx](batch_size)

            anchor_means[upper_bound_idx] = anchor_precision_score[upper_bound_idx] / num_samples_evaluated[upper_bound_idx]
            anchor_means[lower_bound_idx] = anchor_precision_score[lower_bound_idx] / num_samples_evaluated[lower_bound_idx]

            upper_bound_idx, lower_bound_idx = update_bounds(stage_number)

            gap = upper_bounds[upper_bound_idx] - lower_bounds[lower_bound_idx]

        sorted_means = np.argsort(anchor_means)
        return sorted_means[-number_of_anchors:]

    @staticmethod
    def make_tuples(previous_best, state):
        # alters state, computes support for new tuples
        all_features = range(state['n_features'])
        coverage_data = state['coverage_data']
        current_idx = state['current_idx']
        data = state['data'][:current_idx]
        labels = state['labels'][:current_idx]
        if len(previous_best) == 0:
            tuples = [(x,) for x in all_features]
            for x in tuples:
                pres = data[:, x[0]].nonzero()[0]
                # NEW
                state['t_idx'][x] = set(pres)
                state['t_nsamples'][x] = float(len(pres))
                state['t_positives'][x] = float(labels[pres].sum())
                state['t_order'][x].append(x[0])
                # NEW
                state['t_coverage_idx'][x] = set(
                    coverage_data[:, x[0]].nonzero()[0])
                state['t_coverage'][x] = (
                        float(len(state['t_coverage_idx'][x])) /
                        coverage_data.shape[0])
            return tuples
        new_tuples = set()
        for f in all_features:
            for t in previous_best:
                new_t = _normalize_tuple(t + (f,))
                if len(new_t) != len(t) + 1:
                    continue
                if new_t not in new_tuples:
                    new_tuples.add(new_t)
                    state['t_order'][new_t] = copy.deepcopy(state['t_order'][t])
                    state['t_order'][new_t].append(f)
                    state['t_coverage_idx'][new_t] = (
                        state['t_coverage_idx'][t].intersection(
                            state['t_coverage_idx'][(f,)]))
                    state['t_coverage'][new_t] = (
                            float(len(state['t_coverage_idx'][new_t])) /
                            coverage_data.shape[0])
                    t_idx = np.array(list(state['t_idx'][t]))
                    t_data = state['data'][t_idx]
                    present = np.where(t_data[:, f] == 1)[0]
                    state['t_idx'][new_t] = set(t_idx[present])
                    idx_list = list(state['t_idx'][new_t])
                    state['t_nsamples'][new_t] = float(len(idx_list))
                    state['t_positives'][new_t] = np.sum(
                        state['labels'][idx_list])
        return list(new_tuples)

    @staticmethod
    def get_sampling_fns(sample_fn,  # : Callable[[List[Tuple[int]], int], TODO: Add return value],
                         tuples: List[Tuple[int]],
                         state) -> List[Callable[[int, List[int]], float]]:
        # each sample fn returns number of positives
        sampling_fns = []

        def complete_sample_fn(t: Tuple[int], n: int) -> float:
            # TODO add pyDoc
            raw_data, data, labels = sample_fn(list(t), n)
            current_idx = state['current_idx']
            # idxs = range(state['data'].shape[0], state['data'].shape[0] + n)
            indexes = range(current_idx, current_idx + n)
            state['t_idx'][t].update(indexes)
            state['t_nsamples'][t] += n
            state['t_positives'][t] += labels.sum()
            state['data'][indexes] = data
            state['raw_data'][indexes] = raw_data
            state['labels'][indexes] = labels
            state['current_idx'] += n
            if state['current_idx'] >= state['data'].shape[0] - max(1000, n):
                prealloc_size = state['prealloc_size']
                state['data'] = np.vstack((state['data'], np.zeros((prealloc_size, data.shape[1]), data.dtype)))
                state['raw_data'] = np.vstack(
                    (state['raw_data'], np.zeros((prealloc_size, raw_data.shape[1]), raw_data.dtype)))
                state['labels'] = np.hstack((state['labels'], np.zeros(prealloc_size, labels.dtype)))

            return labels.sum()

        for t in tuples:
            sampling_fns.append(lambda n, t=t: complete_sample_fn(t, n))
        return sampling_fns

    @staticmethod
    def get_initial_statistics(tuples, state: Dict):
        """
        TODO: Fill pyDoc
        :param tuples:
        :param state:
        :return:
        """
        stats = {
            'n_samples': [],
            'positives': []
        }
        for t in tuples:
            stats['n_samples'].append(state['t_nsamples'][t])
            stats['positives'].append(state['t_positives'][t])
        return stats

    @staticmethod
    def get_anchor_from_tuple(t, state):
        # TODO: (Author) This is wrong, some of the intermediate anchors may not exist. (WTF, dude (?))

        anchor = dict(feature=[],
                      mean=[],
                      precision=[],
                      coverage=[],
                      examples=[],
                      all_precision=0,
                      num_preds=state['data'].shape[0])

        current_t = tuple()
        for f in state['t_order'][t]:
            current_t = _normalize_tuple(current_t + (f,))

            mean = (state['t_positives'][current_t] / state['t_nsamples'][current_t])
            anchor['feature'].append(f)
            anchor['mean'].append(mean)
            anchor['precision'].append(mean)
            anchor['coverage'].append(state['t_coverage'][current_t])
            raw_idx = list(state['t_idx'][current_t])
            raw_data = state['raw_data'][raw_idx]
            covered_true = (state['raw_data'][raw_idx][state['labels'][raw_idx] == 1])
            covered_false = (state['raw_data'][raw_idx][state['labels'][raw_idx] == 0])

            exs = {'covered': _matrix_subset(raw_data, 10),
                   'covered_true': _matrix_subset(covered_true, 10),
                   'covered_false': _matrix_subset(covered_false, 10),
                   'uncovered_true': np.array([]),
                   'uncovered_false': np.array([])}

            anchor['examples'].append(exs)
        return anchor

    @staticmethod
    def anchor_beam(sample_fn, delta=0.05,
                    epsilon=0.1, batch_size=10,
                    min_shared_samples=0, desired_confidence=1, beam_size=1,
                    tolerance=0.05, min_samples_start=0,
                    max_anchor_size=None,
                    stop_on_first=False, coverage_samples=10000):

        anchor = dict(feature=[], mean=[], precision=[], coverage=[], examples=[], all_precision=0)

        _, coverage_data, _ = sample_fn([], coverage_samples, compute_labels=False)
        raw_data, data, labels = sample_fn([], max(1, min_samples_start))
        mean = labels.mean()
        beta = np.log(1. / delta)
        lower_bound = _dlow_bernoulli(mean, beta / data.shape[0])
        while mean > desired_confidence and lower_bound < desired_confidence - epsilon:
            nraw_data, ndata, nlabels = sample_fn([], batch_size)
            data = np.vstack((data, ndata))
            raw_data = np.vstack((raw_data, nraw_data))
            labels = np.hstack((labels, nlabels))
            mean = labels.mean()
            lower_bound = _dlow_bernoulli(mean, beta / data.shape[0])
        if lower_bound > desired_confidence:
            anchor['num_preds'] = data.shape[0]
            anchor['all_precision'] = mean
            return anchor
        prealloc_size = batch_size * 10000
        current_idx = data.shape[0]
        data = np.vstack((data, np.zeros((prealloc_size, data.shape[1]),
                                         data.dtype)))
        raw_data = np.vstack(
            (raw_data, np.zeros((prealloc_size, raw_data.shape[1]),
                                raw_data.dtype)))
        labels = np.hstack((labels, np.zeros(prealloc_size, labels.dtype)))
        n_features = data.shape[1]
        state = {'t_idx': collections.defaultdict(lambda: set()),
                 't_nsamples': collections.defaultdict(lambda: 0.),
                 't_positives': collections.defaultdict(lambda: 0.),
                 'data': data,
                 'prealloc_size': prealloc_size,
                 'raw_data': raw_data,
                 'labels': labels,
                 'current_idx': current_idx,
                 'n_features': n_features,
                 't_coverage_idx': collections.defaultdict(lambda: set()),
                 't_coverage': collections.defaultdict(lambda: 0.),
                 'coverage_data': coverage_data,
                 't_order': collections.defaultdict(lambda: list())
                 }
        current_size = 1
        best_of_size = {0: []}
        best_coverage = -1
        best_tuple = ()
        if max_anchor_size is None:
            max_anchor_size = n_features
        while current_size <= max_anchor_size:
            tuples = AnchorBaseBeam.make_tuples(
                best_of_size[current_size - 1], state)
            tuples = [x for x in tuples
                      if state['t_coverage'][x] > best_coverage]
            if len(tuples) == 0:
                break
            sample_fns = AnchorBaseBeam.get_sampling_fns(sample_fn, tuples,
                                                         state)
            initial_stats = AnchorBaseBeam.get_initial_statistics(tuples,
                                                                  state)
            chosen_tuples = AnchorBaseBeam.kl_lucb(sample_fns, initial_stats['n_samples'], initial_stats['positives'],
                                                   epsilon, delta, batch_size, min(beam_size, len(tuples)))

            best_of_size[current_size] = [tuples[x] for x in chosen_tuples]
            stop_this = False
            for i, t in zip(chosen_tuples, best_of_size[current_size]):
                # I can choose at most (beam_size - 1) tuples at each step,
                # and there are at most n_feature steps
                beta = np.log(1. /
                              (delta / (1 + (beam_size - 1) * n_features)))
                mean = state['t_positives'][t] / state['t_nsamples'][t]
                lower_bound = _dlow_bernoulli(mean, beta / state['t_nsamples'][t])
                upper_bound = _dup_bernoulli(mean, beta / state['t_nsamples'][t])
                coverage = state['t_coverage'][t]
                while ((mean >= desired_confidence and lower_bound < desired_confidence - tolerance) or
                       (mean < desired_confidence and upper_bound >= desired_confidence + tolerance)):
                    sample_fns[i](batch_size)
                    mean = state['t_positives'][t] / state['t_nsamples'][t]
                    lower_bound = _dlow_bernoulli(mean, beta / state['t_nsamples'][t])
                    upper_bound = _dup_bernoulli(mean, beta / state['t_nsamples'][t])

                if mean >= desired_confidence and lower_bound > desired_confidence - tolerance:
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_tuple = t
                        if best_coverage == 1 or stop_on_first:
                            stop_this = True
            if stop_this:
                break
            current_size += 1
        if best_tuple == ():
            # Could not find an anchor, will now choose the highest precision
            # amongst the top K from every round
            tuples = []
            for i in range(0, current_size):
                tuples.extend(best_of_size[i])
            sample_fns = AnchorBaseBeam.get_sampling_fns(sample_fn, tuples, state)
            initial_stats = AnchorBaseBeam.get_initial_statistics(tuples, state)
            chosen_tuples = AnchorBaseBeam.kl_lucb(sample_fns, initial_stats['n_samples'], initial_stats['positives'],
                                                   epsilon, delta, batch_size, 1)
            best_tuple = tuples[chosen_tuples[0]]

        return AnchorBaseBeam.get_anchor_from_tuple(best_tuple, state)


class AnchorBaseGreed(object):
    def __init__(self, epsilon, sigma, precision_threshold):
        self.epsilon = epsilon
        self.sigma = sigma
        self.precision_threshold = precision_threshold

        self.current_anchor = []

        pass

    def generate_candidates(self, anchor, predicate_generator, coverage_threshold):
        # A_r = set()
        # append anchor with one new predicate which is not present inanchor
        pass

    def greedy_best_candidate(self, ):  # A, D, epsilon, δ):
        # init prec, prec_lb, prec_ub,
        # A <- arg max prec(A)
        # A' <- arg max (A != A') prec_ub(A', δ)
        # while prec_ub(A') - prec_lb(A) > epsilon:
        # sample z~ D(z|A), z' ~ D(z'|A')
        # update prec, prec_ub, prec_lb
        # A <- arg max prec(A)
        # A' <- arg max (A != A') prec_ub(A', δ)

        pass
