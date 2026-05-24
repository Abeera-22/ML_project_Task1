# ==============================
# CREDIT SCORING MODEL
# ==============================

# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# LOAD DATASET

data = pd.read_csv("dataset.csv")

print("Dataset Preview:")
print(data.head())


# ENCODE TARGET LABEL

label_encoder = LabelEncoder()
data['Status'] = label_encoder.fit_transform(data['Status'])

# SPLIT FEATURES AND TARGET

X = data.drop('Status', axis=1)
y = data['Status']

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# CREATE MODEL

model = RandomForestClassifier(n_estimators=100)


# TRAIN MODEL

model.fit(X_train, y_train)


# PREDICTION

y_pred = model.predict(X_test)


# EVALUATION

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# CUSTOM PREDICTION

print("\nCUSTOM PREDICTION")

age = int(input("Enter Age: "))
income = int(input("Enter Income: "))
loan = int(input("Enter Loan Amount: "))
history = int(input("Credit History (1=Good, 0=Bad): "))
debt = int(input("Enter Debt: "))

new_data = [[age, income, loan, history, debt]]

prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Person is CREDITWORTHY")
else:
    print("Person is NOT CREDITWORTHY")