import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('../Dataset/all_website_streams.csv')

##########################   TIME TO RUN THE MODEL   ##########################

# each label corresponds to a website
# 1 - Glassdoor
# 2 - X
# 3 - Stack Overflow
y = data["Label"]

X = data.copy()
X.drop("Label", axis=1, inplace=True)

# splitting data into training and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# scaling datasets for better model fit
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# using logistic regression for classification
model = LogisticRegression()

# fit data to training data
model.fit(X_train_scaled, y_train)

# make predictions for testing data and check accuracy
y_test_predictions = model.predict(X_test_scaled)

test_accuracy = accuracy_score(y_test, y_test_predictions)
test_f1s = f1_score(y_test, y_test_predictions, average="macro")
print("Test accuracy:", test_accuracy)
print("Test F1 scores:", test_f1s)

##########################   VISUALIZING ACCURACY   ##########################

# compute the confusion matrix for test values
cm = confusion_matrix(y_test, y_test_predictions)

# initialize heatmap
sns.heatmap(cm,
            annot=True,
            fmt='g',
            xticklabels=["Glassdoor", "X", "Stack Overflow"],
            yticklabels=["Glassdoor", "X", "Stack Overflow"])

# add labels and show confusion matrix
plt.xlabel("Actual Website", fontsize=15)
plt.ylabel("Prediction", fontsize=15)
plt.title("Logistic Regression", fontsize=18)
plt.show()