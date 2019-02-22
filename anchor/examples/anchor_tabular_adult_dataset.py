import os
import anchor
import numpy as np
import pandas as pd
import sklearn.ensemble

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

df = pd.read_csv(os.path.join(FILE_PATH, "../data/adult/adult.data"), header=None)
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

le = sklearn.preprocessing.LabelEncoder()
target_labels = le.fit_transform(target_labels)
class_names = list(le.classes_)

train_X, rest_X, train_y, rest_y = sklearn.model_selection.train_test_split(df, target_labels, stratify=target_labels, test_size=0.5)
val_X, test_X, val_y, test_y = sklearn.model_selection.train_test_split(rest_X, rest_y, stratify=rest_y, test_size=0.5)

c = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
c.fit(train_X, train_y)
predict_fn = lambda x: c.predict(x)
print('Train', sklearn.metrics.accuracy_score(train_y, predict_fn(train_X)))
print('Validation', sklearn.metrics.accuracy_score(val_y, predict_fn(val_X)))

explainer = anchor.AnchorTabularExplainer()
explainer.fit(data=val_X,
              target=val_y,
              categorical_features_idx=categorical_features_idx,
              categorical_names=categorical_names,
              class_names=class_names,
              balance=True)

val_X = np.array(val_X)
val_y = np.array(val_y)

idx = 1

print('Prediction: ', explainer.class_names[predict_fn(val_X[idx].reshape(1, -1))[0]])
exp = explainer.explain_instance(val_X[idx], c.predict, threshold=PRECISION_THRESHOLD)

# The precision and coverage are invalid!
print('Anchor: %s' % (' AND '.join(exp.names())))
print('Precision: %.2f' % exp.precision())  # Invalid!
print('Coverage: %.2f' % exp.coverage())  # Invalid!

test_X = np.array(test_X)
test_y = np.array(test_y)

predicted_class = predict_fn(test_X[idx].reshape(1, -1))
a = test_X[idx][exp.features()]

# fit_anchor is indices of samples which have satisfied the anchor
fit_anchor = np.where(np.all(test_X[:, exp.features()] == a, axis=1))[0]

print('Anchor true test coverage: %.4f' % (len(fit_anchor) / float(test_X.shape[0])))
print('Anchor true test precision: %.4f' % (np.mean(predict_fn(test_X[fit_anchor]) == predicted_class)))

print("\n\nPartial anchors: \n")
explanations = []  # (precision, coverage, explanation)
for i in range(len(exp.names())):
    print("-----" * 10)
    print(f"Partial anchor with length == {i}")

    a = test_X[idx][exp.features(i)]
    explanation = ' AND '.join(exp.names(i))
    print(f"Partial anchor({i}) = " + explanation)

    fit_anchor = np.where(np.all(test_X[:, exp.features(i)] == a, axis=1))[0]
    coverage = (len(fit_anchor) / float(test_X.shape[0]))
    precision = (np.mean(predict_fn(test_X[fit_anchor]) == predicted_class))
    print('Coverage: %.4f' % coverage)
    print('Precision: %.4f' % precision)
    explanations.append((coverage, precision, explanation))
explanations = pd.DataFrame(explanations, columns='coverage precision explanation'.split())

if explanations.precision.max() < PRECISION_THRESHOLD:
    print("Method has failed!")
else:
    explanations = explanations[explanations.precision >= PRECISION_THRESHOLD].sort_values(by="coverage", ascending=False)
    print("-----" * 10)
    print("True best explanation:")
    print(f"Explanation: {explanations.iloc[0].explanation}")
    print(f"Precision: {explanations.iloc[0].precision}")
    print(f"Coverage: {explanations.iloc[0].coverage}")