from typing import Generator, List, Dict, Any
from abc import ABC, abstractmethod
from .explanation import Explanation, Predicate
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
        # self.str is set in anchor_selector after anchor is selected as the best anchor
        if self.str is None:
            return " AND ".join([str(p) for p in self.predicates])
        else:
            return self.str

    def precision(self):
        """
        :return: precision of this anchor
        """
        if len(self._precisions) > 0:
            return np.round(self._precisions[-1], decimals=3)
        else:
            return None

    def coverage(self):
        """
        :return: coverage of this anchor
        """
        if len(self._coverages) > 0:
            return np.round(self._coverages[-1], decimals=3)
        else:
            return None

    def increment(self, ):
        """
        Increment this anchor by a single predicate. Incrementation is done iteratively.
        Random predicate is added and if it does not contradict with existing predicates and
        it exists in the explained sample it is added to the predicates list.
        Later, predicates list is simplified, and unnecessary predicates are removed.
        :return:
        """
        new_predicate = next(self.predicate_generator)
        is_contradictory = any([predicate.is_contradictory_to(new_predicate) for predicate in self.predicates])
        present_in_anchor = new_predicate.check_against_sample(self.x)

        while new_predicate in self.predicates or is_contradictory or not present_in_anchor:
            new_predicate = next(self.predicate_generator)
            is_contradictory = any([predicate.is_contradictory_to(new_predicate) for predicate in self.predicates])
            present_in_anchor = new_predicate.check_against_sample(self.x)

            ''' FIXME Dead code for checking whether predicate can reduce # of possible feature values to zero. But if predicate passes
             present_in_anchor test, it automatically has at least 1 possible feature value. 
             
             Even so, line 131 in anchor.selector.py do throw the error like some feature has no possible values.
        
             '''

            # filtered_feature_values = []
            # for feature_id, feature_values in enumerate(self._feature_values):
            #     suitable_values_masks = [np.ones((len(feature_values), 1))]
            #     for predicate in filter(lambda p: p.feature_id == feature_id, current_predicates):
            #         suitable_values_masks.append(predicate.check_against_column(feature_values)[:, np.newaxis])
            #     suitable_values_mask = np.all(np.concatenate(suitable_values_masks, axis=1), axis=1)
            #     #if suitable_values_mask are identical among predicates => we can safely delete one of them, isn't it?
            #     filtered_feature_values.append(feature_values[suitable_values_mask])
            #
            # if any([len(df) == 0 for df in filtered_feature_values]):
            #     del current_predicates[-1]
            #     raise Exception("Some feature is destroyed")
            # else:

        self.predicates.append(new_predicate)

        self.simplify_predicates()

        return self

    def __eq__(self, other):
        if isinstance(other, TabularExplanation):
            return len(set(self.predicates).difference(other.predicates)) == 0
        else:
            return False

    def __hash__(self):
        return hash(tuple([hash(x) for x in self.predicates]))

    def check_against_sample(self, x):
        return np.all([p.check_against_sample(x) for p in self.predicates])

    def copy(self):
        new_explanation = TabularExplanation(self.x, self.predicate_generator, self._feature_values)
        new_explanation.predicates = self.predicates.copy()
        return new_explanation

    def simplify_predicates(self):
        """
        Replace two or more overlapping predicates with the single strongest one.
        e.g. df > 3 and df > 5 => df > 5
        e.g. df = 3 and df < 10 => df = 3
        :return:
        """
        for feature_id, feature_values in enumerate(self._feature_values):

            # If Equality predicate is present for this feature, remove all other predicates for this feature
            eq_predicates = list(filter(lambda p: p.feature_id == feature_id and type(p) == EqualityPredicate, self.predicates))
            if len(eq_predicates) == 1:
                strongest_predicate = eq_predicates[0]
                compressed_predicates = list(filter(lambda p: p.feature_id != feature_id or
                                                              p is strongest_predicate,
                                                    self.predicates))
                self.predicates = compressed_predicates
            elif len(eq_predicates) > 1:
                raise Exception("Invalid predicates")

            # If multiple GEQ predicates are present for this feature,
            # remove all other predicates for this feature, but the strongest one
            geq_predicates = list(filter(lambda p: p.feature_id == feature_id and type(p) == GreaterOrEqualPredicate, self.predicates))
            if len(geq_predicates) > 1:
                strongest_predicate = max(geq_predicates, key=lambda p: p.value)
                compressed_predicates = list(filter(lambda p: p.feature_id != feature_id or
                                                              type(p) is LessPredicate or
                                                              p is strongest_predicate,
                                                    self.predicates))
                self.predicates = compressed_predicates

            # If multiple LE predicates are present for this feature,
            # remove all other predicates for this feature, but the strongest one
            less_predicates = list(filter(lambda p: p.feature_id == feature_id and type(p) == LessPredicate, self.predicates))
            if len(less_predicates) > 1:
                strongest_predicate = min(less_predicates, key=lambda p: p.value)
                compressed_predicates = list(filter(lambda p: p.feature_id != feature_id or
                                                              type(p) is GreaterOrEqualPredicate or
                                                              p is strongest_predicate,
                                                    self.predicates))
                self.predicates = compressed_predicates

            # Fetch all inequality predicates, next fetch all geq predicates. It's guaranteed by previous code that we will fetch
            # 0 or 1 geq predicates. If any of inequality predicates has the same value as geq, meaning geq is not geq but ge, remove
            # both inequality predicate and geq predicate and replace them with single new geq and leq.
            ineq_predicates = list(filter(lambda p: p.feature_id == feature_id and type(p) == InequalityPredicate, self.predicates))
            geq_predicates = list(filter(lambda p: p.feature_id == feature_id and type(p) == GreaterOrEqualPredicate, self.predicates))
            if len(geq_predicates) == 1:
                geq_value = geq_predicates[0].value
                ineq_geq_interception = list(filter(lambda p: p.value == geq_value, ineq_predicates))
                if len(ineq_geq_interception) > 0:
                    compressed_predicates = list(filter(lambda p: p is not ineq_geq_interception[0] and
                                                                  p is not geq_predicates[0],
                                                        self.predicates))
                    new_geq_predicate = geq_predicates[0].copy()
                    new_geq_predicate.value += 1

                    new_le_predicate = LessPredicate(value=ineq_geq_interception[0].value,
                                                     feature_idx=ineq_geq_interception[0].feature_id,
                                                     feature_name=ineq_geq_interception[0].feature_name)
                    new_geq_predicate.value += 1

                    compressed_predicates.append(new_geq_predicate)
                    compressed_predicates.append(new_le_predicate)
                    self.predicates = compressed_predicates

            # Symmetrical case as one before, but for le predicates.
            # Fetch all inequality predicates, next fetch all le predicates. It's guaranteed by previous code that we will fetch
            # 0 or 1 le predicates. If any of inequality predicates has the same value as le, meaning le is redundant, remove
            # le predicate
            ineq_predicates = list(filter(lambda p: p.feature_id == feature_id and type(p) == InequalityPredicate, self.predicates))
            le_predicates = list(filter(lambda p: p.feature_id == feature_id and type(p) == LessPredicate, self.predicates))
            if len(le_predicates) == 1:
                le_value = le_predicates[0].value
                ineq_le_interception = list(filter(lambda p: p.value == le_value, ineq_predicates))
                if len(ineq_le_interception) > 0:
                    compressed_predicates = list(filter(lambda p: p is not le_predicates[0], self.predicates))
                    self.predicates = compressed_predicates

    def get_possible_feature_values(self, feature_id):
        """
        Filter possible feature values according to the predicates list in this explanation
        :param feature_id:
        :return:
        """
        feature_values = self._feature_values[feature_id]
        suitable_values_masks = [np.ones((len(feature_values), 1))]
        for predicate in filter(lambda p: p.feature_id == feature_id, self.predicates):
            suitable_values_masks.append(predicate.check_against_column(feature_values)[:, np.newaxis])
        suitable_values_mask = np.all(np.concatenate(suitable_values_masks, axis=1), axis=1)
        return feature_values[suitable_values_mask]


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
            # Feature model_name is deliberately not considered in equation check
            return self.value == other.value and self.feature_id == other.feature_id
        else:
            return False

    def __hash__(self):
        return hash((self.value, self.feature_id))

    @abstractmethod
    def check_against_sample(self, x: np.array):
        """
        Returns True, if predicate holds for sample df, False otherwise
        :param x:
        :return:
        """
        pass

    @abstractmethod
    def check_against_column(self, x: np.array):
        """
        Returns True, if predicate holds for every element of df, False otherwise
        :param x:
        :return:
        """
        pass

    @abstractmethod
    def is_contradictory_to(self, other) -> bool:
        """
        Returns True if two predicates contradict with each other, False otherwise
        :param other:
        :return:
        """
        pass


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

    def copy(self):
        return EqualityPredicate(self.value, self.feature_id, self.feature_name)


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

    def copy(self):
        return InequalityPredicate(self.value, self.feature_id, self.feature_name)


class GreaterOrEqualPredicate(TabularPredicate):
    def __str__(self):
        return f"{self.feature_name} >= {self.value}"

    def check_against_column(self, x: np.array):
        return x >= self.value

    def check_against_sample(self, x: np.array):
        return x[self.feature_id] >= self.value

    def is_contradictory_to(self, other) -> bool:
        if other.feature_id == self.feature_id:
            if type(other) is GreaterOrEqualPredicate and other.value < self.value:
                return True
            elif type(other) is LessPredicate and other.value <= self.value:
                return True
            elif type(other) is EqualityPredicate and other.value < self.value:
                return True
        return False

    def copy(self):
        return GreaterOrEqualPredicate(self.value, self.feature_id, self.feature_name)


class LessPredicate(TabularPredicate):
    def __str__(self):
        return f"{self.feature_name} < {self.value}"

    def check_against_sample(self, x: np.array):
        return x[self.feature_id] < self.value

    def check_against_column(self, x: np.array):
        return x <= self.value

    def is_contradictory_to(self, other) -> bool:
        if other.feature_id == self.feature_id:
            if type(other) is LessPredicate and other.value > self.value:
                return True
            elif type(other) is GreaterOrEqualPredicate and other.value >= self.value:
                return True
            elif type(other) is EqualityPredicate and other.value >= self.value:
                return True
        return False

    def copy(self):
        return LessPredicate(self.value, self.feature_id, self.feature_name)
