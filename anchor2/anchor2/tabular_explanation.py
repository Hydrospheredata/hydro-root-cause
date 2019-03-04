from typing import Generator, List

from anchor2.anchor2.explanation import Explanation, Predicate
import numpy as np


class TabularExplanation(Explanation):
    def __init__(self):
        self.predicates: List = []
        pass

    def __str__(self):
        return " AND ".join([str(p) for p in self.predicates])

    def increment(self, predicate_generator: Generator):
        new_predicate = next(predicate_generator)
        while new_predicate in self.predicates:
            new_predicate = next(predicate_generator)
        self.predicates.append(new_predicate)

    def __eq__(self, other):
        if isinstance(other, TabularExplanation):
            return len(set(self.predicates).difference(other.predicates)) == 0
        else:
            return False

    def __hash__(self):
        return hash(tuple([hash(x) for x in self.predicates]))

    def numpy_selector(self):
        return lambda x: np.logical_and(*[p.check_against_sample(x) for p in self.predicates])


class TabularPredicate(Predicate):
    def __init__(self, value: float, feature_idx, feature_name):
        self.value = value
        self.feature_id = feature_idx
        self.feature_name = feature_name

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            # Feature name is deliberately not considered in equation check
            return self.value == other.value and self.feature_id == other.feature_id
        else:
            return False

    def __hash__(self):
        return hash((self.value, self.feature_id))

    def check_against_sample(self, x: np.array):
        raise NotImplementedError()


class EqualityPredicate(TabularPredicate):
    def __str__(self):
        return f"{self.feature_name} == {self.value}"

    def check_against_sample(self, x: np.array):
        return x[self.feature_id] == self.value


class InequalityPredicate(TabularPredicate):
    def __str__(self):
        return f"{self.feature_name} != {self.value}"

    def check_against_sample(self, x: np.array):
        return x[self.feature_id] != self.value


class GreaterOrEqualPredicate(TabularPredicate):
    def __str__(self):
        return f"{self.feature_name} >= {self.value}"

    def check_against_sample(self, x: np.array):
        return x[self.feature_id] >= self.value


class LessPredicate(TabularPredicate):
    def __str__(self):
        return f"{self.feature_name} < {self.value}"

    def check_against_sample(self, x: np.array):
        return x[self.feature_id] < self.value
