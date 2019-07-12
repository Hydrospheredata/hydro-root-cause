import json
import os
import string
from io import open
from typing import List, Dict, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.utils.validation import check_is_fitted

from . import anchor_core
from . import anchor_explanation

Num = Union[int, float]


# TODO: Move to utils
class Discretizer(KBinsDiscretizer):

    def __int__(self, n_bins=5, strategy='quantile'):
        super().__init__(n_bins=n_bins, encode='ordinal', strategy=strategy)

    @staticmethod
    def __get_discretizer_names(b_edges: np.array, feature_name: str):
        bins_names = [f"{feature_name} <= {b_edges[1]}"]
        bins_names.extend(
            [f"{b1_edge} < {feature_name} <= {b2_edge}" for b1_edge, b2_edge in zip(b_edges[1:-2], b_edges[2:])])
        bins_names.append(f"{feature_name} > {b_edges[-2]}")
        return bins_names

    def get_feature_names(self, ):
        check_is_fitted(self, 'bin_edges_')
        feature_names = self.__get_discretizer_names(self.bin_edges_[0], "df")
        return np.array(feature_names, dtype=object)


# TODO: Move to utils
def id_generator(size=15):
    """
    Helper function to generate random div ids. This is useful for embedding
    HTML into ipython notebooks.
    """
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(np.random.choice(chars, size, replace=True))


class AnchorTabularExplainer(object):

    def __init__(self, ):
        """
        Initialize AnchorTabularExplainer object.
        """
        self.std: Dict[int, float] = {}  # Feature index to std
        self.max: Dict[int, float] = {}  # Feature index to max value
        self.min: Dict[int, float] = {}  # Feature index to min value

        self.feature_names = None
        self.categorical_names = None
        self.class_names = None
        self.categorical_features_idx = None
        self.ordinal_features_idx = None  # TODO: Check for appropirate usage.
        self.data = None
        self.labels = None  # TODO Check whether we can delete it? It seems other parts of code does not use it.
        self.encoder = None
        self.d_data = None

    def fit(self,
            data: pd.DataFrame,
            categorical_features_idx: List[int],
            categorical_names: Dict[int, List[str]],
            class_names: Dict[int, str],
            target: pd.Series = None,
            balance=False, ):
        """
        Fit AnchorTabularExplainer
        :param data: dataset which model have to explain
        :param target: predictions for this dataset. Only needed for balancing if balance == True. Ignored if balance == False
        :param categorical_features_idx: indices of categorical features in this data
        :param categorical_names: dictionary from feature_index to map[Int->String] from category __id into category model_name
        :param class_names: dictionary from label code into label model_name
        :param balance: whether to make stratified subsampled version of data
        """
        self.feature_names = list(data.columns)
        self.categorical_names = categorical_names
        self.class_names = class_names
        self.categorical_features_idx = categorical_features_idx
        self.ordinal_features_idx = list(set(range(data.shape[1])).difference(set(self.categorical_features_idx)))

        data = data.copy()
        data = np.array(data)

        if balance:  # Balance dataset according to target label
            if target is None:
                raise ValueError("Provide target labels for balancing the dataset. (balance=True)")
            labels = np.array(target)
            subsample_idx = np.array([], dtype='int')
            min_labels = np.min(np.bincount(labels))  # The number of labels in most under-represented class

            for label in np.unique(labels):
                idx = np.random.choice(np.where(labels == label)[0], min_labels)
                subsample_idx = np.hstack((subsample_idx, idx))

            # Subsample data, so it will be stratified according to target with min_labels in each strata
            data = data[subsample_idx]

        # One one-hot-encoder for all columns
        transformers = [("one-hot-encoding", OneHotEncoder(categories='auto', sparse=False), self.categorical_features_idx.copy())]

        # Each ordinal column has its own discretizer
        transformers.extend(
            [(f"{f_idx}_discretizer", Discretizer(n_bins=20, encode="ordinal"), [f_idx]) for f_idx in self.ordinal_features_idx])

        self.encoder = ColumnTransformer(transformers, n_jobs=-1, sparse_threshold=0.0)

        self.data = data
        self.d_data = self.encoder.fit_transform(data)

        discrete_category_names = dict()
        for transformer_name, transformer, f_idx in self.encoder.transformers_:
            if len(f_idx) > 1 or f_idx[0] not in self.ordinal_features_idx:
                continue
            f_idx = f_idx[0]
            f_name = self.feature_names[f_idx]
            discrete_category_names[f_idx] = [x.replace("df", f_name) for x in transformer.get_feature_names()]

        # self.d_data = self.encoder.transform(self.data, )  # Discretize data

        self.categorical_names.update(discrete_category_names)

        self.categorical_features_idx += self.ordinal_features_idx  # Since ordinal features now are discrete ranges, we
        # can treat each bin as a category.
        # No point in this variable, since its always will be range(data.shape[1]). Strange.... TODO: Check this!

        for feature_idx in range(self.data.shape[1]):
            if feature_idx in self.categorical_features_idx and feature_idx not in self.ordinal_features_idx:
                continue
            self.min[feature_idx] = np.min(data[:, feature_idx])
            self.max[feature_idx] = np.max(data[:, feature_idx])
            self.std[feature_idx] = np.std(data[:, feature_idx])

    def sample_from_data(self,
                         conditions_eq: Dict[int, Num],
                         conditions_neq: Dict[int, Num],
                         conditions_geq: Dict[int, Num],
                         conditions_leq: Dict[int, Num],
                         num_samples: int):
        """
        This function generates new samples which satisfy all conditions passed as arguments.
        There are 4 types of conditions, equality, inequality, greater and less. Inequality is not implemented right now.
        To generate new samples, discretized version of data passed during AnchorExplainer.fit is analyzed
        whether it meets conditions passed as arguments. If certain features does not meet these conditions, feature values
        change into new value sampled from distribution of  feature values which do satisfy these conditions. If such
        distribution cannot be found, new values are just selected from ~U(min, max) for corresponding feature.

        # TODO: either remove 'conditions_neq' argument, or implement it
        # TODO: the code in this function iterates over each feature and each condition. Candidate for vectorization.
        # TODO: Look at possible bug described in the next TODO

        :param conditions_eq:
        :param conditions_neq:
        :param conditions_geq:
        :param conditions_leq:
        :param num_samples:
        :return:
        """

        data = self.data
        digitized_data = self.d_data

        # Subsample data with replacement
        subsample_idx = np.random.choice(range(data.shape[0]), num_samples, replace=True)
        subsampled_data = data[subsample_idx]
        subsampled_digitized_data = digitized_data[subsample_idx]

        # For every equality condition, change the subsampled feature to the value specified in equality condition,
        # so that each sample in subsampled ata will satisfy conditions_eq predicates
        for feature_idx in conditions_eq:
            subsampled_data[:, feature_idx] = np.repeat(conditions_eq[feature_idx], num_samples)

        for feature_idx in conditions_geq:
            geq_condition_value = conditions_geq[feature_idx]

            # Boolean mask where condition for >= does not hold!
            unmet_geq_condition_mask = subsampled_digitized_data[:, feature_idx] <= geq_condition_value
            if feature_idx in conditions_leq:
                # If the same feature is also in <= conditions,
                leq_condition_value = conditions_leq[feature_idx]
                unmet_leq_condition_mask = subsampled_digitized_data[:, feature_idx] > leq_condition_value

                # Update boolean mask. Now it is mask for rows where any of <= and >= conditions does not hold
                unmet_geq_condition_mask = unmet_geq_condition_mask | unmet_leq_condition_mask

            if unmet_geq_condition_mask.sum() == 0:
                continue  # If all samples satisfy these conditions, no need change them.

            # possible_values is a boolean mask which marks correct values as 1 in a NOT subsampled version of dataset
            possible_values = digitized_data[:, feature_idx] > geq_condition_value
            if feature_idx in conditions_leq:
                leq_condition_value = conditions_leq[feature_idx]
                leq_possible_values = digitized_data[:, feature_idx] <= leq_condition_value
                possible_values = possible_values & leq_possible_values

            # If there are no samples which satisfy these conditions in the whole dataset,
            # sample from ~U(min, max)
            if possible_values.sum() == 0:
                min_ = conditions_geq.get(feature_idx, self.min[feature_idx])
                max_ = conditions_leq.get(feature_idx, self.max[feature_idx])
                replacement_values = np.random.uniform(min_, max_, unmet_geq_condition_mask.sum())
                # TODO: if possible values are categorical, (discretized values), what will happen?
                # TODO: Are there any more intelligent way to pick replacement value according to input distribution?
            else:
                # Or else, just select from values
                replacement_values = np.random.choice(data[possible_values, feature_idx],
                                                      unmet_geq_condition_mask.sum(), replace=True)
            subsampled_data[unmet_geq_condition_mask, feature_idx] = replacement_values

        for feature_idx in conditions_leq:
            if feature_idx in conditions_geq:
                # This feature was already processed, so no need to process it now
                continue
            unmet_leq_condition_mask = subsampled_digitized_data[:, feature_idx] > conditions_leq[feature_idx]
            if unmet_leq_condition_mask.sum() == 0:
                continue
            possible_values = digitized_data[:, feature_idx] <= conditions_leq[feature_idx]
            if possible_values.sum() == 0:
                min_ = conditions_geq.get(feature_idx, self.min[feature_idx])
                max_ = conditions_leq.get(feature_idx, self.max[feature_idx])
                replacement_values = np.random.uniform(min_, max_, unmet_leq_condition_mask.sum())
            else:
                replacement_values = np.random.choice(data[possible_values, feature_idx],
                                                      unmet_leq_condition_mask.sum(),
                                                      replace=True)
            subsampled_data[unmet_leq_condition_mask, feature_idx] = replacement_values
        return subsampled_data

    def transform_to_examples(self, examples, features_in_anchor=(),
                              predicted_label=None):
        # TODO: add pyDoc
        ret_obj = []
        if len(examples) == 0:
            return ret_obj
        weights = [int(predicted_label) if x in features_in_anchor else -1
                   for x in range(examples.shape[1])]
        examples = self.encoder.transform(examples)
        for example in examples:
            values = [self.categorical_names[i][int(example[i])] if i in self.categorical_features_idx else example[i] for i in
                      range(example.shape[0])]
            ret_obj.append(list(zip(self.feature_names, values, weights)))
        return ret_obj

    def to_explanation_map(self, exp):
        # TODO: add pyDoc

        instance = exp['instance']
        predicted_label = exp['prediction']
        predict_proba = np.zeros(len(self.class_names))
        predict_proba[predicted_label] = 1

        examples_obj = []
        for i, temp in enumerate(exp['examples'], start=1):
            features_in_anchor = set(exp['feature'][:i])
            ret = {
                'coveredFalse': self.transform_to_examples(temp['covered_false'], features_in_anchor, predicted_label),
                'coveredTrue': self.transform_to_examples(temp['covered_true'], features_in_anchor, predicted_label),
                'uncoveredTrue': self.transform_to_examples(temp['uncovered_true'], features_in_anchor, predicted_label),
                'uncoveredFalse': self.transform_to_examples(temp['uncovered_false'], features_in_anchor, predicted_label),
                'covered': self.transform_to_examples(temp['covered'], features_in_anchor, predicted_label)}

            examples_obj.append(ret)

        explanation = {'names': exp['names'],
                       'certainties': exp['precision'] if len(exp['precision']) else [exp['all_precision']],
                       'supports': exp['coverage'],
                       'allPrecision': exp['all_precision'],
                       'examples': examples_obj,
                       'onlyShowActive': False}

        weights = [-1 for x in range(instance.shape[0])]
        instance = self.encoder.transform(exp['instance'].reshape(1, -1))[0]
        values = [self.categorical_names[i][int(instance[i])] if i in self.categorical_features_idx else instance[i] for i in
                  range(instance.shape[0])]
        raw_data = list(zip(self.feature_names, values, weights))

        ret = {
            'explanation': explanation,
            'rawData': raw_data,
            'predictProba': list(predict_proba),
            'labelNames': list(map(str, self.class_names)),
            'rawDataType': 'tabular',
            'explanationType': 'anchor',
            'trueClass': False
        }

        return ret

    def as_html(self, exp, **kwargs):
        # TODO: add pyDoc

        exp_map = self.to_explanation_map(exp)

        def jsonize(x): return json.dumps(x)

        this_dir, _ = os.path.split(__file__)
        bundle = open(os.path.join(this_dir, 'bundle.js'), encoding='utf8').read()
        random_id = 'top_div' + id_generator()
        out = u'''<html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head><script>%s </script></head><body>''' % bundle
        out += u'''
        <div __id="{random_id}" />
        <script>
            div = d3.select("#{random_id}");
            lime.RenderExplanationFrame(div,{label_names}, {predict_proba},
            {true_class}, {explanation}, {raw_data}, "tabular", {explanation_type});
        </script>'''.format(random_id=random_id,
                            label_names=jsonize(exp_map['labelNames']),
                            predict_proba=jsonize(exp_map['predictProba']),
                            true_class=jsonize(exp_map['trueClass']),
                            explanation=jsonize(exp_map['explanation']),
                            raw_data=jsonize(exp_map['rawData']),
                            explanation_type=jsonize(exp_map['explanationType']))
        out += u'</body></html>'
        return out

    def get_sample_fn(self, explained_sample, classifier_fn, desired_label=None):
        # TODO: add pyDoc

        def predict_fn(x):
            return classifier_fn(x)

        true_label = desired_label
        if true_label is None:
            true_label = predict_fn(explained_sample.reshape(1, -1))[0]
        # Must map present here to include categorical features (for conditions_eq), and numerical features for geq
        # and leq
        mapping: Dict[int, Tuple[int, str, Num]] = {}
        explained_sample = self.encoder.transform(explained_sample.reshape(1, -1))[0]  # TODO: why so, it is already digitized (??
        for feature_idx in self.categorical_features_idx:  # Categorical features are all_features_idx now!
            if feature_idx in self.ordinal_features_idx:
                for v in range(len(self.categorical_names[feature_idx])):
                    idx = len(mapping)
                    if explained_sample[feature_idx] <= v != len(self.categorical_names[feature_idx]) - 1:
                        mapping[idx] = (feature_idx, 'leq', v)
                    elif explained_sample[feature_idx] > v:
                        mapping[idx] = (feature_idx, 'geq', v)
            else:
                idx = len(mapping)
                mapping[idx] = (feature_idx, 'eq', explained_sample[feature_idx])

        def sample_fn(present: List[int], num_samples: int, compute_labels: bool = True):
            # TODO: add pydoc, change this 'currying' to functools.partial
            conditions_eq: Dict[int, Num] = {}  # Conditions for equality FIXME: more meaningful description
            conditions_leq: Dict[int, Num] = {}  # Conditions for <
            conditions_geq: Dict[int, Num] = {}  # Conditions for >

            for x in present:
                f, op, value = mapping[x]
                if op == 'eq':
                    conditions_eq[f] = value
                if op == 'leq':
                    if f not in conditions_leq:
                        conditions_leq[f] = value
                    conditions_leq[f] = min(conditions_leq[f], value)
                if op == 'geq':
                    if f not in conditions_geq:
                        conditions_geq[f] = value
                    conditions_geq[f] = max(conditions_geq[f], value)

            raw_data = self.sample_from_data(
                conditions_eq=conditions_eq,
                conditions_neq={},
                conditions_geq=conditions_geq,
                conditions_leq=conditions_leq,
                num_samples=num_samples)

            digitized_raw_data = self.encoder.transform(raw_data)
            data = np.zeros((num_samples, len(mapping)), int)  # Binary Matrix, (n df m), where m is number of predicates
            # 1 if predicate is satisfied, 0 if predicate is not satisfied
            for i in mapping:
                f, op, value = mapping[i]
                if op == 'eq':
                    data[:, i] = (digitized_raw_data[:, f] == explained_sample[f]).astype(int)
                if op == 'leq':
                    data[:, i] = (digitized_raw_data[:, f] <= value).astype(int)
                if op == 'geq':
                    data[:, i] = (digitized_raw_data[:, f] > value).astype(int)
            labels = []
            if compute_labels:
                labels = (predict_fn(raw_data) == true_label).astype(int)
            return raw_data, data, labels

        return sample_fn, mapping

    # TODO: pass beam_size argument to inner function calls
    def explain_instance(self,
                         explained_sample,
                         classifier_fn,
                         threshold=0.95,
                         delta=0.1,
                         tau=0.15,
                         batch_size=100,
                         max_anchor_size=None,
                         desired_label=None,
                         beam_size=4,  # FIXME pass beam_size to the
                         **kwargs):
        # It's possible to pass in max_anchor_size
        sample_fn, mapping = self.get_sample_fn(explained_sample, classifier_fn, desired_label=desired_label)
        # return sample_fn, mapping
        exp = anchor_core.AnchorBaseBeam.anchor_beam(
            sample_fn,
            delta=delta,
            epsilon=tau,
            batch_size=batch_size,
            desired_confidence=threshold,
            max_anchor_size=max_anchor_size,
            **kwargs)

        self.add_names_to_exp(explained_sample, exp, mapping)
        exp['instance'] = explained_sample
        exp['prediction'] = classifier_fn(explained_sample.reshape(1, -1))[0]
        explanation = anchor_explanation.AnchorExplanation('tabular', exp, self.as_html)
        return explanation

    def add_names_to_exp(self, _, hoeffding_exp, mapping):
        # TODO: (Author todo) precision recall is all wrong, coverage functions wont work anymore due to ranges
        # TODO: (mine) what the fuck, dude?
        indexes = hoeffding_exp['feature']
        hoeffding_exp['names'] = []
        hoeffding_exp['feature'] = [mapping[idx][0] for idx in indexes]
        ordinal_ranges: Dict[int, List[float]] = {}  # Feature_idx -> range of values it should be
        for idx in indexes:
            f, op, v = mapping[idx]  # feature_id_old, operation, value
            if op == 'geq' or op == 'leq':
                if f not in ordinal_ranges:
                    ordinal_ranges[f] = [float('-inf'), float('inf')]  # initialize ordinal_ranges[f] with (-ing, +inf)
            if op == 'geq':
                ordinal_ranges[f][0] = max(ordinal_ranges[f][0], v)
            if op == 'leq':
                ordinal_ranges[f][1] = min(ordinal_ranges[f][1], v)
        handled: Set = set()  # TODO: What is handled?
        # print(list(ordinal_ranges.items()))  # Working fine here
        for idx in indexes:
            f, op, v = mapping[idx]
            if op == 'eq':
                feature_name = '%s = ' % self.feature_names[f]
                if f in self.categorical_names:
                    v = int(v)
                    if ('<' in self.categorical_names[f][v]
                            or '>' in self.categorical_names[f][v]):
                        feature_name = ''
                    feature_name = '%s%s' % (feature_name, self.categorical_names[f][v])
                else:
                    feature_name = '%s%.2f' % (feature_name, v)
            else:
                if f in handled:
                    continue
                geq, leq = ordinal_ranges[f]
                feature_name = ''
                geq_val = ''
                leq_val = ''
                if geq > float('-inf'):
                    if geq == len(self.categorical_names[f]) - 1:
                        geq = geq - 1
                    name = self.categorical_names[f][geq + 1]
                    if '<' in name:
                        geq_val = name.split()[0]
                    elif '>' in name:
                        geq_val = name.split()[-1]
                if leq < float('inf'):
                    name = self.categorical_names[f][leq]
                    if leq == 0:
                        leq_val = name.split()[-1]
                    elif '<' in name:
                        leq_val = name.split()[-1]
                if leq_val and geq_val:
                    feature_name = '%s < %s <= %s' % (geq_val, self.feature_names[f],
                                                      leq_val)
                elif leq_val:
                    feature_name = '%s <= %s' % (self.feature_names[f], leq_val)
                elif geq_val:
                    feature_name = '%s > %s' % (self.feature_names[f], geq_val)
                handled.add(f)
            hoeffding_exp['names'].append(feature_name)
