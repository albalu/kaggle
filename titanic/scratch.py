import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pylab as plt

def write_nulls(df):
    """df: a pandas DataFrame"""
    logging.info(df.isnull().sum())


logging.basicConfig(level=logging.INFO, format='%(message)s')

train = pd.read_csv('train.csv', mangle_dupe_cols=True, na_values=None)
test = pd.read_csv('test.csv', mangle_dupe_cols=True, na_values=None)
feature_names = train.columns.tolist()
print "feature names: \n {}".format(feature_names)
logging.info(train.shape)
logging.info(test.shape)

for df in [train, test]:
    df["Cabin"] =  df.Cabin.str.extract('([A-Za-z])', expand=False)
    df["Cabin"] = df["Cabin"].fillna("unknown")
    df["Cabin"] = df["Cabin"].map({ch:0 if ch=="unknown" else 1 for ch in list(set(df["Cabin"].values))})

train["Embarked"] = train["Embarked"].fillna(train.Embarked.dropna().mode()[0]) # only train has missing
# plt.scatter(test["Pclass"].values, test["Fare"].values)
# plt.show()
for df in [train, test]:
    # print list(set(df["Embarked"].values))
    df["Embarked"] = df["Embarked"].map({"Q": 0, "S": 1, "C": 2})

na_Fare_idx = test["Fare"][test["Fare"].isnull()].index
# print na_Fare_idx[0]
# print test[["Pclass", "Fare"]].groupby(["Pclass"], as_index=False).mean().values[2][1]
test["Fare"][na_Fare_idx[0]] = test[["Pclass", "Fare"]].groupby(["Pclass"], as_index=False).mean().values[2][1]

# there are also some people for whom Fare price was (assuming accidentally) 0.0! we set those to mean of each Pclass
for df in [train, test]:
    nul_idx = df["Fare"] == 0
    df["Fare"][nul_idx] = None
    # print df[["Fare", "Pclass"]][nul_idx]
    df["Fare"] = df[["Pclass", "Fare"]].groupby(["Pclass"]).transform(lambda x: x.fillna(x.mean())).astype(float)
    # print df[["Fare", "Pclass"]][nul_idx]
    # write_nulls(df)


# trying to find a way to fill out the missing Age information which is not rare! (a lot of them are missing)
# hypothesis is that Age and Pclass are related, but the following shows that it is WRONG!!
# print train[["Age", "Pclass"]].groupby(["Pclass"]).mean()
# plt.scatter(test["Pclass"].values, test["Age"].values)
# plt.show()

# we then get it from title after observing that Miss. and Master. used for minors/youngster as opposed to Mr. and Mrs.
for df in [train, test]:
    df["Title"] = df.Name.str.extract(" ([A-Za-z]+)\.", expand=False)
    df["Title"][df["Title"]=="Ms"] = "Miss"
    # print df[["Title", "Age"]].groupby(["Title"]).mean()
    # print df[["Title", "Age"]].groupby(["Title"]).count()
    df["Age"] = df[["Title", "Age"]].groupby(["Title"]).transform(lambda x: x.fillna(x.mean())).astype(float)

for df in [train, test]:
    df["Sex"] = df["Sex"].map({s:0 if s=="male" else 1 for s in df["Sex"].values})
    df["Family"] = df["SibSp"] + df["Parch"]

print list(set(pd.qcut(train['Fare'], 4).values)) # qcuts = [7.925, 14.5, 31.275, >31.275]
print list(set(pd.qcut(train['Age'], 4).values)) # qcuts = [0.419, 21.816, 30.5, >35.898]

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


# for drop_name in ["Title", "Name", "Ticket", "PassengerId"]:
submission_x = df["PassengerId"]
for drop_name in ["Title", "Name", "Ticket", "PassengerId", "SibSp", "Parch"]:
    train = train.drop([drop_name], axis=1)
    test = test.drop([drop_name], axis=1)


#
# train["Cabin"] = train["Cabin"].fillna("unknown")
# train["Cabin"][train["Cabin"] == "unknown"] = 0
# train["Cabin"][train["Cabin"] != "unknown"] = 1


for df in [train, test]:
    write_nulls(df)
    print df.head()
    print

y = train["Survived"]
X = train.drop(["Survived"], axis=1)
X_test = test
# cv = KFold(n_splits=10, shuffle=True, random_state=1)
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

svc = SVC()
svc.fit(X, y)
score = np.mean(cross_val_score(svc, X, y, scoring='accuracy', cv=cv, n_jobs=1))
# y_pred = svc.predict(X_test)
print "SVC score= {}".format(score)

rf = RandomForestClassifier(n_estimators=10)
score = np.mean(cross_val_score(rf, X, y, scoring='accuracy', cv=cv, n_jobs=1))
print "RandomForest score= {}".format(score)

param_grid = {'min_samples_leaf': range(1, 10), 'max_depth': range(1,10), "min_samples_split": range(5,15)}
gcv = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, n_jobs=-1)
rfg = gcv.fit(X, y)
y_pred = rfg.predict(X_test)
print "best score for grid search: {}".format(rfg.best_score_)
print "best parameters for grid search: {}".format(rfg.best_params_)

# when including SibSp and Parch separately
# best score for grid search: 0.83687150838
# best parameters for grid search: {'min_samples_split': 9, 'max_depth': 6, 'min_samples_leaf': 3}

# when excluding SibSp and Parch but combining them (adding) to "Family" instead
# best score for grid search: 0.842458100559
# best parameters for grid search: {'min_samples_split': 5, 'max_depth': 9, 'min_samples_leaf': 3}


# best so far in terms of kaggle score:
# param_grid = {'min_samples_leaf': range(1, 10), 'max_depth': range(1,10), "min_samples_split": range(5,15)}
# best parameters for grid search: {'min_samples_split': 6, 'max_depth': 8, 'min_samples_leaf': 4}

# submission:
submission = pd.DataFrame({"PassengerId": submission_x, "Survived": y_pred})
submission.to_csv("albalu_submission_scratch.csv", index=False)

