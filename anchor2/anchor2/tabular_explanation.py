from typing import Generator, List, Dict, Any

from anchor2.anchor2.explanation import Explanation, Predicate
import numpy as np


class TabularExplanation(Explanation):
    def __init__(self, x, predicate_generator: Generator, feature_values: List[np.array]):
        """
        :type predicate_generator: infinite generator of random predicates
        :type feature_values: List of possible feature values
        :param x: The sample we try to explain
        """
        self.predicates: List[TabularPredicate] = []
        self.x = x
        self.predicate_generator = predicate_generator
        self._coverages = []
        self._precisions = []
        self._feature_values = feature_values
        self.str = None

    def __str__(self):
        if self.str is None:
            return " AND ".join([str(p) for p in self.predicates])
        else:
            return self.str

    def precision(self):
        if len(self._precisions) > 0:
            return np.round(self._precisions[-1], decimals=3)
        else:
            return None

    def coverage(self):
        if len(self._coverages) > 0:
            return np.round(self._coverages[-1], decimals=3)
        else:
            return None

    def increment(self, ):
        new_predicate = next(self.predicate_generator)
        is_contradictory = any([predicate.is_contradictory_to(new_predicate) for predicate in self.predicates])
        present_in_anchor = new_predicate.check_against_sample(self.x)

        current_predicates = self.predicates.copy()
        valid_predicate_found = False
        while not valid_predicate_found:
            while new_predicate in self.predicates or is_contradictory or not present_in_anchor:
                new_predicate = next(self.predicate_generator)
                is_contradictory = any([predicate.is_contradictory_to(new_predicate) for predicate in self.predicates])
                present_in_anchor = new_predicate.check_against_sample(self.x)

            filtered_feature_values = []
            for feature_id, feature_values in enumerate(self._feature_values):
                suitable_values_masks = [np.ones((len(feature_values), 1))]
                for predicate in filter(lambda p: p.feature_id == feature_id, current_predicates):
                    suitable_values_masks.append(predicate.check_against_column(feature_values)[:, np.newaxis])
                suitable_values_mask = np.all(np.concatenate(suitable_values_masks, axis=1), axis=1)
                # TODO if suitable_values_mask are identical among predicates => we can safely delete one of them, isn't it?
                filtered_feature_values.append(feature_values[suitable_values_mask])

            if any([len(x) == 0 for x in filtered_feature_values]):
                del current_predicates[-1]
                print("Some feature is destroyed")
            else:
                self.predicates.append(new_predicate)
                valid_predicate_found = True

        self.simplify_predicates()

        return self

    def __eq__(self, other):
        if isinstance(other, TabularExplanation):
            return len(set(self.predicates).difference(other.predicates)) == 0
        else:
            return False

    def __hash__(self):
        return hash(tuple([hash(x) for x in self.predicates]))

    def numpy_selector(self, x):
        return np.all([p.check_against_sample(x) for p in self.predicates])

    def copy(self):
        new_explanation = TabularExplanation(self.x, self.predicate_generator, self._feature_values)
        new_explanation.predicates = self.predicates.copy()
        return new_explanation

    def simplify_predicates(self):
        """
        Replace two or more overlapping predicates with the single strongest one.
        e.g. x > 3 and x > 5 => x > 5
        :return:
        """
        for feature_id, feature_values in enumerate(self._feature_values):

            geq_predicates = list(filter(lambda p: p.feature_id == feature_id and type(p) == GreaterOrEqualPredicate, self.predicates))
            if len(geq_predicates) > 1:
                strongest_predicate = max(geq_predicates, key=lambda p: p.value)
                compressed_predicates = list(filter(lambda p: p.feature_id != feature_id or
                                                              type(p) is LessPredicate or
                                                              p is strongest_predicate,
                                                    self.predicates))
                self.predicates = compressed_predicates

            less_predicates = list(filter(lambda p: p.feature_id == feature_id and type(p) == LessPredicate, self.predicates))
            if len(less_predicates) > 1:
                strongest_predicate = min(less_predicates, key=lambda p: p.value)
                compressed_predicates = list(filter(lambda p: p.feature_id != feature_id or
                                                              type(p) is GreaterOrEqualPredicate or
                                                              p is strongest_predicate,
                                                    self.predicates))
                self.predicates = compressed_predicates


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

    def is_contradictory_to(self, other: TabularPredicate) -> bool:
        if other.feature_id == self.feature_id:
            if type(other) is EqualityPredicate and other.value != self.value:
                return True
            elif type(other) is InequalityPredicate and other.value == self.value:
                return True
            elif type(other) is GreaterOrEqualPredicate and other.value > self.value:
                return True
            elif type(other) is LessPredicate and other.value < self.value:
                return True

        return False


class InequalityPredicate(TabularPredicate):
    def __str__(self):
        return f"{self.feature_name} != {self.value}"

    def check_against_column(self, x: np.array):
        return x != self.value

    def check_against_sample(self, x: np.array):
        return x[self.feature_id] != self.value

    def is_contradictory_to(self, other) -> bool:
        if other.feature_id == self.feature_id:
            if type(other) is EqualityPredicate and other.value == self.value:
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
            elif type(new_predicate) is LessPredicate and new_predicate.value <= self.value:
                return True
            elif type(new_predicate) is EqualityPredicate and new_predicate.value < self.value:
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
            elif type(new_predicate) is GreaterOrEqualPredicate and new_predicate.value >= self.value:
                return True
            elif type(new_predicate) is EqualityPredicate and new_predicate.value >= self.value:
                return True

        return False
