import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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

# using decision tree for classification
model = DecisionTreeClassifier(random_state=0)

# fit data to training data
model.fit(X_train, y_train)

# check accuracy of training data predictions
y_train_predictions = model.predict(X_train)

train_accuracy = accuracy_score(y_train, y_train_predictions)
train_f1s = f1_score(y_train, y_train_predictions, average="macro")
print("Training accuracy:", train_accuracy)
print("Training F1 scores:", train_f1s)

# make predictions for validation data and check accuracy
y_val_predictions = model.predict(X_val)

validation_accuracy = accuracy_score(y_val, y_val_predictions)
validation_f1s = f1_score(y_val, y_val_predictions, average="macro")
print("Validation accuracy:", validation_accuracy)
print("Validation F1 scores:", validation_f1s)

# make predictions on testing data
y_test_predictions = model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_predictions)
test_f1s = f1_score(y_test, y_test_predictions, average="macro")
print("Test accuracy:", test_accuracy)
print("Test F1 scores:", test_f1s)

##########################   VISUALIZING ACCURACY   ##########################

# show tree
plot_tree(model)
plt.show()

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
plt.title("Decision Tree", fontsize=18)
plt.show()