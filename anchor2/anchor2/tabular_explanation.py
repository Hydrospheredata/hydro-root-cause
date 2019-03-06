from typing import Generator, List

from anchor2.anchor2.explanation import Explanation, Predicate
import numpy as np


class TabularExplanation(Explanation):
    def __init__(self, x, predicate_generator: Generator):
        """
        :param x: The sample we try to explain
        """
        self.predicates: List[TabularPredicate] = []
        self.x = x
        self.predicate_generator = predicate_generator

    def __str__(self):
        return " AND ".join([str(p) for p in self.predicates])

    def increment(self, ):
        new_predicate = next(self.predicate_generator)
        is_contradictory = any([predicate.is_contradictory_to(new_predicate) for predicate in self.predicates])
        present_in_anchor = new_predicate.check_against_sample(self.x)

        while new_predicate in self.predicates or is_contradictory or not present_in_anchor:
            new_predicate = next(self.predicate_generator)
            is_contradictory = any([predicate.is_contradictory_to(new_predicate) for predicate in self.predicates])
            present_in_anchor = new_predicate.check_against_sample(self.x)

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
        """
        Parent class for all predicates.
        :param value: Value used in the predicate construction
        :param feature_idx: Index of the feature
        :param feature_name: Name of the feature. Used in __str__ method only
        iow How many possible values can that feature take given this predicate
        """
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
        """
        Returns True, if predicate holds for sample x, False otherwise
        :param x:
        :return:
        """
        raise NotImplementedError()

    def check_against_column(self, x: np.array):
        """
        Returns True, if predicate holds for every element of x, False otherwise
        :param x:
        :return:
        """
        raise NotImplementedError()

    def is_contradictory_to(self, new_predicate) -> bool:
        """
        Returns True if two predicates contradict with each other, False otherwise
        :param new_predicate:
        :return:
        """
        raise NotImplementedError()


class EqualityPredicate(TabularPredicate):
    def __str__(self):
        return f"{self.feature_name} == {self.value}"

    def check_against_sample(self, x: np.array):
        return x[self.feature_id] == self.value

    def check_against_column(self, x: np.array):
        return x == self.value

    def is_contradictory_to(self, new_predicate: TabularPredicate) -> bool:
        if new_predicate.feature_id == self.feature_id:
            if type(new_predicate) is EqualityPredicate and new_predicate.value != self.value:
                return True
            if type(new_predicate) is InequalityPredicate and new_predicate.value == self.value:
                return True
        return False


class InequalityPredicate(TabularPredicate):
    def __str__(self):
        return f"{self.feature_name} != {self.value}"

    def check_against_column(self, x: np.array):
        return x != self.value

    def check_against_sample(self, x: np.array):
        return x[self.feature_id] != self.value

    def is_contradictory_to(self, new_predicate) -> bool:
        if new_predicate.feature_id == self.feature_id:
            if type(new_predicate) is EqualityPredicate and new_predicate.value == self.value:
                return True
        return False


class GreaterOrEqualPredicate(TabularPredicate):
    def __str__(self):
        return f"{self.feature_name} >= {self.value}"

    def check_against_column(self, x: np.array):
        return x >= self.value

    def check_against_sample(self, x: np.array):
        return x[self.feature_id] >= self.value

    def is_contradictory_to(self, new_predicate) -> bool:
        if new_predicate.feature_id == self.feature_id:
            if type(new_predicate) is GreaterOrEqualPredicate and new_predicate.value < self.value:
                return True
            if type(new_predicate) is LessPredicate and new_predicate.value <= self.value:
                return True
        return False


class LessPredicate(TabularPredicate):
    def __str__(self):
        return f"{self.feature_name} < {self.value}"

    def check_against_sample(self, x: np.array):
        return x[self.feature_id] < self.value

    def check_against_column(self, x: np.array):
        return x <= self.value

    def is_contradictory_to(self, new_predicate) -> bool:
        if new_predicate.feature_id == self.feature_id:
            if type(new_predicate) is LessPredicate and new_predicate.value > self.value:
                return True
            if type(new_predicate) is GreaterOrEqualPredicate and new_predicate.value >= self.value:
                return True
        return False
