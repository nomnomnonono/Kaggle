import pandas as pd
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# print(train.shape, test.shape)
# print(train.head())

train["is_train"] = 1
test["is_train"] = 0
data = pd.concat([train.drop(columns=["Survived"]), test])  # connect train and test
# print(data.isnull().sum())  # sum NaN by each raw

# filling in missing values
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Fare"] = data["Fare"].fillna(data["Fare"].median())
data["Embarked"] = data["Embarked"].fillna("S")

# categorical -> int
data = pd.concat([data, pd.get_dummies(data["Embarked"], prefix="Embarked")], axis=1).drop(columns=["Embarked"])  # axis=1 == connect column
data["Sex"] = pd.get_dummies(data["Sex"], drop_first=True)  # drop_first: remove 1st column

# split the data into training and validation
feature_column = ["Pclass", "Sex", "Age", "Embarked_C", "Embarked_Q", "Embarked_S"]
feature_train = data[data["is_train"] == 1].drop(columns=["is_train"])[feature_column]
feature_test = data[data["is_train"] == 0].drop(columns=["is_train"])[feature_column]

# target of training data
target_train = train["Survived"]

# learning
model = DecisionTreeClassifier()
model.fit(feature_train, target_train)
pred_train = model.predict(feature_train)
print(model.score(feature_train, target_train))

# evaluation
pred_test = model.predict(feature_test)
my_pred = pd.DataFrame(pred_test, index=test["PassengerId"], columns=["Survived"])
my_pred.to_csv("my_pred.csv", index_label=["PassengerId"])
