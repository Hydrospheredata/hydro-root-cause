from unittest import TestCase
import numpy as np
import pandas as pd
import anchor
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import unittest
import os

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class TestAnchorOnAdultDataset(TestCase):

    def setUp(self):

        def map_array_values(series, value_map):
            if series.dtype == 'object':
                ret = series.str.strip().copy()
            else:
                ret = series.copy()
            for src, target in value_map.items():
                ret[ret == src] = target
            return ret

        feature_names = ["Age", "Workclass", "fnlwgt", "Education",
                         "Education-Num", "Marital Status", "Occupation",
                         "Relationship", "Race", "Sex", "Capital Gain",
                         "Capital Loss", "Hours per week", "Country", 'Income']
        features_to_use = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        education_map = {
            '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
                'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
                'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
            'Some-college': 'High School grad', 'Masters': 'Masters',
            'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
            'Assoc-voc': 'Associates',
        }
        occupation_map = {
            "Adm-clerical": "Admin", "Armed-Forces": "Military",
            "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
            "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
                "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
                "Service", "Priv-house-serv": "Service", "Prof-specialty":
                "Professional", "Protective-serv": "Other", "Sales":
                "Sales", "Tech-support": "Other", "Transport-moving":
                "Blue-Collar",
        }
        country_map = {
            'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China':
                'China', 'Columbia': 'South-America', 'Cuba': 'Other',
            'Dominican-Republic': 'Latin-America', 'Ecuador': 'South-America',
            'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
            'France': 'Euro_1', 'Germany': 'Euro_1', 'Greece': 'Euro_2',
            'Guatemala': 'Latin-America', 'Haiti': 'Latin-America',
            'Holand-Netherlands': 'Euro_1', 'Honduras': 'Latin-America',
            'Hong': 'China', 'Hungary': 'Euro_2', 'India':
                'British-Commonwealth', 'Iran': 'Other', 'Ireland':
                'British-Commonwealth', 'Italy': 'Euro_1', 'Jamaica':
                'Latin-America', 'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico':
                'Latin-America', 'Nicaragua': 'Latin-America',
            'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru':
                'South-America', 'Philippines': 'SE-Asia', 'Poland': 'Euro_2',
            'Portugal': 'Euro_2', 'Puerto-Rico': 'Latin-America', 'Scotland':
                'British-Commonwealth', 'South': 'Euro_2', 'Taiwan': 'China',
            'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
            'United-States': 'United-States', 'Vietnam': 'SE-Asia'
        }
        married_map = {
            'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married',
            'Married-civ-spouse': 'Married', 'Married-spouse-absent':
                'Separated', 'Separated': 'Separated', 'Divorced':
                'Separated', 'Widowed': 'Widowed'
        }

        def cap_gains_fn(x):
            x = x.astype(float)
            d = np.digitize(x, [0, np.median(x[x > 0]), float('inf')], right=True)
            new_series = pd.Series(["None"] * len(d))
            new_series[d == 0] = 'None'
            new_series[d == 1] = 'Low'
            new_series[d == 2] = 'High'
            return new_series

        transformations = {
            'Education': lambda x: map_array_values(x, education_map),
            'Marital Status': lambda x: map_array_values(x, married_map),
            'Occupation': lambda x: map_array_values(x, occupation_map),
            'Capital Gain': cap_gains_fn,
            'Capital Loss': cap_gains_fn,
            'Country': lambda x: map_array_values(x, country_map),
        }

        df = pd.read_csv(os.path.join(FILE_PATH, "../../data/adult/adult.data"), header=None)
        df.columns = feature_names
        target_labels = pd.Series(df.iloc[:, -1], index=df.index)
        df = df.iloc[:, features_to_use]
        df.dropna(inplace=True)

        for feature, fun in transformations.items():
            df[feature] = fun(df[feature])

        categorical_features_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
        categorical_names = {}  # Dictionary with (Category id -> category classes)
        for f_idx in categorical_features_idx:
            le = sklearn.preprocessing.LabelEncoder()
            df.iloc[:, f_idx] = le.fit_transform(df.iloc[:, f_idx])
            categorical_names[f_idx] = le.classes_

        le = LabelEncoder()
        target_labels = le.fit_transform(target_labels)
        class_names = list(le.classes_)

        train_X, rest_X, train_y, rest_y = train_test_split(df, target_labels, stratify=target_labels,
                                                            test_size=0.5)
        val_X, test_X, val_y, test_y = train_test_split(rest_X, rest_y, stratify=rest_y, test_size=0.5)

        c = RandomForestClassifier(n_estimators=50, n_jobs=5)
        c.fit(train_X, train_y)

        explainer = anchor.AnchorTabularExplainer()
        explainer.fit(data=val_X,
                      target=val_y,
                      categorical_features_idx=categorical_features_idx,
                      categorical_names=categorical_names,
                      class_names=class_names,
                      balance=True)

        val_X = np.array(val_X)
        val_y = np.array(val_y)

        self.explainer = explainer
        self.predict_fn = c.predict
        self.val_X = val_X
        self.val_Y = val_y

    def test_prediction_names(self):
        self.assertTrue(all([type(class_name) is str for class_name in self.explainer.class_names]), "All class names must be strings")
        self.assertTrue(all([len(class_name) > 0 for class_name in self.explainer.class_names]), "Class names should not be empty strings")

    def test_explanation_name(self):
        x_idxs = np.random.choice(list(range(self.val_X.shape[0])), 5, replace=False)
        for i, x_idx in enumerate(x_idxs):
            x = self.val_X[x_idx]
            exp = self.explainer.explain_instance(x, self.predict_fn, threshold=0.95)
            with self.subTest(i=i, msg="Non-empty feature name in explanation"):
                self.assertTrue(all([len(name) > 0 for name in exp.names()]), "Each feature name should not be empty string")

    @unittest.skip("https://github.com/provectus/hydro-root-cause/issues/1")
    def test_explanation_metrics(self):

        x_idxs = np.random.choice(list(range(self.val_X.shape[0])), 5, replace=False)
        for i, x_idx in enumerate(x_idxs):
            x = self.val_X[x_idx]
            predicted_class = self.predict_fn(x.reshape(1, -1))
            exp = self.explainer.explain_instance(x, self.predict_fn, threshold=0.95)

            explanation_anchor = x[exp.features()]
            val_x_idx_where_anchor_suffice = np.where(np.all(self.val_X[:, exp.features()] == explanation_anchor, axis=1))[0]

            with self.subTest(i=i, msg="Precision coverage on val_X is true"):
                true_precision = np.mean(self.predict_fn(self.val_X[val_x_idx_where_anchor_suffice]) == predicted_class)
                self.assertAlmostEqual(exp.precision(), true_precision, places=3)

            with self.subTest(i=i, msg="Explanation coverage on val_X is true"):
                true_coverage = len(val_x_idx_where_anchor_suffice) / float(self.val_X.shape[0])
                self.assertAlmostEqual(exp.coverage(), true_coverage, places=3)

    # TODO write test for partial anchors
    # TODO write test  self.AssertTrue(threshold <= exp.precision())
