import os
import time

import numpy as np
import pandas as pd
import sklearn.ensemble
from anchor2 import TabularExplainer

FILE_PATH = os.path.dirname(os.path.realpath(__file__))
PRECISION_THRESHOLD = 0.95


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
categorical_features = [1, 3, 5, 6, 7, 8, 9, 10, 11, 13]
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

df = pd.read_csv(os.path.join(FILE_PATH, "data/adult/adult.data"), header=None)
df.columns = feature_names
target_labels = pd.Series(df.iloc[:, -1], index=df.index)
df = df.iloc[:, features_to_use]
df.dropna(inplace=True)

for feature, fun in transformations.items():
    df[feature] = fun(df[feature])

categorical_features_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
categorical_names = {}  # Dictionary with (Category __id -> category classes)
for f_idx in categorical_features_idx:
    le = sklearn.preprocessing.LabelEncoder()
    df.iloc[:, f_idx] = le.fit_transform(df.iloc[:, f_idx])
    categorical_names[f_idx] = le.classes_

le = sklearn.preprocessing.LabelEncoder()
target_labels = le.fit_transform(target_labels)
class_names = list(le.classes_)

train_X, rest_X, train_y, rest_y = sklearn.model_selection.train_test_split(df, target_labels, stratify=target_labels,
                                                                            test_size=0.5, random_state=42)
val_X, test_X, val_y, test_y = sklearn.model_selection.train_test_split(rest_X, rest_y, stratify=rest_y,
                                                                        test_size=0.5, random_state=42)

c = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5, random_state=42)
c.fit(train_X, train_y)

print('Train', sklearn.metrics.accuracy_score(train_y, c.predict(train_X)))
print('Validation', sklearn.metrics.accuracy_score(val_y, c.predict(val_X)))

explainer = TabularExplainer()
explainer.fit(val_X,
              label_decoders=categorical_names,
              ordinal_features_idx=[0, 10],
              oh_encoded_categories=dict(), )

print("Explanation 1: ")
explained_x = np.array(val_X.iloc[0])
print(f"Prediction: {class_names[c.predict(explained_x.reshape(1, -1))[0]]}")
time_start = time.time()
explanation = explainer.explain(explained_x, c.predict, threshold=PRECISION_THRESHOLD, verbose=False)
print(explanation)
print("Precision", explanation.precision())
print("Coverage", explanation.coverage())
print(f"Elapsed time for first explanation is {time.time() - time_start:.2f}s")

print("\n\n\nExplanation 2: ")
explained_x = np.array(val_X.iloc[1])
print(f"Prediction: {class_names[c.predict(explained_x.reshape(1, -1))[0]]}")
time_start = time.time()
explanation = explainer.explain(explained_x, c.predict, threshold=PRECISION_THRESHOLD, verbose=False)
print(explanation)
print("Precision", explanation.precision())
print("Coverage", explanation.coverage())
print(f"Elapsed time for second explanation is {time.time() - time_start:.2f}s")

print("\n\n\nExplanation 3: ")
explained_x = np.array([37, 4, 1, 0, 4, 0, 4, 1, 2, 2, 50, 9])
print(f"Prediction: {class_names[c.predict(explained_x.reshape(1, -1))[0]]}")
time_start = time.time()
explanation = explainer.explain(explained_x, c.predict, threshold=PRECISION_THRESHOLD, verbose=False)
print(explanation)
print("Precision", explanation.precision())
print("Coverage", explanation.coverage())
print(f"Elapsed time for third explanation is {time.time() - time_start:.2f}s")

# Example of output
#
# Explanation 1:
# Prediction:  <=50K
# Marital Status == Widowed
# Precision 0.974
# Coverage 0.029
# Elapsed time for first explanation is 2.35s
#
# Explanation 2:
# Prediction:  >50K
# Capital Gain == None AND Hours per week < 76.78787878787898 AND Capital Loss == None AND Occupation == Sales AND Race ==  White AND Workclass ==  Federal-gov
# Precision 1.0
# Coverage 0.0
# Elapsed time for second explanation is 40.95s
#
# Explanation 3:
# Prediction:  >50K
# Age > 33.0 AND Sex ==  Male AND Relationship ==  Husband AND Hours per week < 42.0 AND Capital Gain == None AND Capital Loss == None AND Marital Status == Married AND (Age < 28.0 OR 29.0 < Age) AND Race ==  White AND Education == Bachelors AND Hours per week > 37.66666666666674 AND Occupation == Other
# Precision 1.0
# Coverage 0.001
# Elapsed time for third explanation is 132.92s
