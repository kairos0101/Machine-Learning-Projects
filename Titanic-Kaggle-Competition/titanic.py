import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_ids = test["PassengerId"]

def clean(data):
    data = data.drop(["PassengerId", "Ticket", "Name", "Cabin"], axis=1)
    cols = ["SibSp", "Parch", "Fare", "Age"]
    for col in cols: data[col].fillna(float(data[col].mean()), inplace=True)
    data.Embarked.fillna("U", inplace=True)
    return data

data = clean(data)
test = clean(test)
le = preprocessing.LabelEncoder()
cols = ["Sex", "Embarked"]

for col in cols:
    data[col] = le.fit_transform(data[col])
    test[col] = le.transform(test[col])
    #print(le.classes_)

X = data.drop(["Survived"], axis=1)
y = data["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
predictions = clf.predict(X_val)
score = accuracy_score(y_val, predictions)
print(f"The Accuracy of Logistic Regression in the dataset is {score*100:.2f}%")

submission_preds = clf.predict(test)
df = pd.DataFrame({"PassengerId": test_ids.values, "Survived": submission_preds})
df.to_csv("submission.csv", index=False)
