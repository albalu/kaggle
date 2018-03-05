import logging
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

logging.basicConfig(level=logging.INFO, format='%(message)s')

train = pd.read_csv('./data/train.csv', mangle_dupe_cols=True, na_values=None)
test = pd.read_csv('./data/test.csv', mangle_dupe_cols=True, na_values=None)

for df in [train, test]:
    df["Cabin"] =  df.Cabin.str.extract('([A-Za-z])', expand=False)
    df["Cabin"] = df["Cabin"].fillna("unknown")
    df["Cabin"] = df["Cabin"].map({ch:0 if ch=="unknown" else 1 for ch in list(set(df["Cabin"].values))})

train["Embarked"] = train["Embarked"].fillna(train.Embarked.dropna().mode()[0]) # only train has missing
for df in [train, test]:
    df["Embarked"] = df["Embarked"].map({"Q": 0, "S": 1, "C": 2})

missing_fare_replacement = test[["Pclass", "Fare"]].groupby(["Pclass"], as_index=False).mean().values[2][1]
test.loc[test["Fare"].isnull(), "Fare"] = missing_fare_replacement # same as below
# test["Fare"].transform(lambda x: x.fillna(missing_fare_replacement)) # same as above

# there are also some people for whom Fare price was (assuming accidentally) 0.0! we set those to mean of each Pclass
for df in [train, test]:
    df.loc[df["Fare"] == 0, "Fare"] = None
    df["Fare"] = df[["Pclass", "Fare"]].groupby(["Pclass"]).transform(lambda x: x.fillna(x.mean())).astype(float)

# we then get it from title after observing that Miss. and Master. used for minors/youngster as opposed to Mr. and Mrs.
for df in [train, test]:
    df["Title"] = df.Name.str.extract(" ([A-Za-z]+)\.", expand=False)
    # df["Title"][df["Title"]=="Ms"] = "Miss" # don't use it this way, in addition to getting a warning, it is confusion
    df.loc[df["Title"] == "Ms", "Title"] = "Miss"
    df["Age"] = df[["Title", "Age"]].groupby(["Title"]).transform(lambda x: x.fillna(x.mean())).astype(float)

for df in [train, test]:
    df["Sex"] = df["Sex"].map({s:0 if s=="male" else 1 for s in df["Sex"].values})
    df["Family"] = df["SibSp"] + df["Parch"]

qcuts = {"Fare": [7.925, 14.5, 31.275],
         "Age": [21.816, 30.5, 35.898]}

# splitting Fare and Age to 4 groups to normalize features assuming qcuts>=2:
for df in [train, test]:
    for feature in qcuts.keys():
        df.loc[df[feature] <= qcuts[feature][0], feature] = 0
        df.loc[df[feature] > qcuts[feature][-1], feature] = len(qcuts[feature])-1
        for i in range(len(qcuts[feature])-1):
            df.loc[(df[feature] > qcuts[feature][i]) & (df[feature] <= qcuts[feature][i+1]), feature] = i+1
        df[feature] = df[feature].astype(int)

submission_x = df["PassengerId"]
for drop_name in ["Title", "Name", "Ticket", "PassengerId", "SibSp", "Parch"]:
    train = train.drop([drop_name], axis=1)
    test = test.drop([drop_name], axis=1)

logging.info(train.head())
logging.info(test.head())

# learning
y = train["Survived"]
X = train.drop(["Survived"], axis=1)
X_test = test
# cv = KFold(n_splits=10, shuffle=True, random_state=1)
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
rf = RandomForestClassifier(n_estimators=10)
param_grid = {'min_samples_leaf': range(1, 10), 'max_depth': range(1,10), "min_samples_split": range(5,15)}
gcv = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, n_jobs=-1)
rfg = gcv.fit(X, y)
y_pred = rfg.predict(X_test)
logging.info("best score for grid search: {}".format(rfg.best_score_))
logging.info("best parameters for grid search: {}".format(rfg.best_params_))

# submission:
submission = pd.DataFrame({"PassengerId": submission_x, "Survived": y_pred})
submission.to_csv("albalu_submission.csv", index=False)

