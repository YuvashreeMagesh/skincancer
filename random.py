import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the image folder
image_folder =  r'C:\softcom_project\skin'

# Define the target labels
target_labels =  ["Test", "Train"]

# Define the image size and color space
image_size = (64, 64)
color_space = cv2.COLOR_BGR2GRAY

# Load the images and their labels into arrays
X = []
y = []
for label in target_labels:
    label_folder = os.path.join(image_folder, label)
    for filename in os.listdir(label_folder):
        image_path = os.path.join(label_folder, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, color_space)
        image = cv2.resize(image, image_size)
        X.append(image)
        y.append(target_labels.index(label))

# Convert the arrays to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

# Print the classification report and confusion matrix
print(classification_report(y_test, y_pred, target_names=target_labels))
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='g', xticklabels=target_labels, yticklabels=target_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
