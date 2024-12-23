import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import  confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics import  confusion_matrix, classification_report


# Load the training dataset
train_data = pd.read_csv('sign_mnist_13bal_train.csv')

# Separate the data (features) and the  classes
X_train = train_data.drop('class', axis=1)  # Features (all columns except the first one)
X_train = X_train / 255.0
y_train = train_data['class']   # Target (first column)

#Create the validation dataset
validate_data = pd.read_csv('sign_mnist_13bal_train.csv')

# Separate the data (features) and the  classes
X_validate = validate_data.drop('class', axis=1)  # Features (all columns except the first one)
X_validate = X_validate / 255.0
y_validate = validate_data['class']   # Target (first column)

X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 10, random_state = 4, stratify = y_train)


# Load the testing dataset
test_data = pd.read_csv('sign_mnist_13bal_test.csv')

# Separate the data (features) and the  classes
X_test = test_data.drop('class', axis=1)  # Features (all columns except the first one)
X_test = X_test / 255.0
y_test = test_data['class']   # Target (first column)

#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=40, random_state=0)


#Build the neural network
neural_net_model = MLPClassifier( hidden_layer_sizes=(22),random_state=42,tol=0.005)

neural_net_model.fit(X_train, y_train)
# Determine model architecture 
layer_sizes = [neural_net_model.coefs_[0].shape[0]]  # Start with the input layer size
layer_sizes += [coef.shape[1] for coef in neural_net_model.coefs_]  # Add sizes of subsequent layers
layer_size_str = " x ".join(map(str, layer_sizes))
print(f"Training set size: {len(y_train)}")
print(f"Layer sizes: {layer_size_str}")


# predict the classes from the training and test sets
y_pred_train = neural_net_model.predict(X_train)
y_pred = neural_net_model.predict(X_test)

# Create dictionaries to hold total and correct counts for each class
correct_counts = defaultdict(int)
total_counts = defaultdict(int)
overall_correct = 0

# Count correct test predictions for each class
for true, pred in zip(y_test, y_pred):
    total_counts[true] += 1
    if true == pred:
        correct_counts[true] += 1
        overall_correct += 1

# For comparison, count correct _training_ set predictions
total_counts_training = 0
correct_counts_training = 0
for true, pred in zip(y_train, y_pred_train):
    total_counts_training += 1
    if true == pred:
        correct_counts_training += 1


# Calculate and print accuracy for each class and overall test accuracy
for class_id in sorted(total_counts.keys()):
    accuracy = correct_counts[class_id] / total_counts[class_id] *100
    print(f"Accuracy for class {class_id}: {accuracy:3.0f}%")
print(f"----------")
overall_accuracy = overall_correct / len(y_test)*100
print(f"Overall Test Accuracy: {overall_accuracy:3.1f}%")
overall_training_accuracy = correct_counts_training / total_counts_training*100
print(f"Overall Training Accuracy: {overall_training_accuracy:3.1f}%")


#Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
class_ids = sorted(total_counts.keys())

# For better formatting
print("Confusion Matrix:")
print(f"{'':9s}", end='')
for label in class_ids:
    print(f"Class {label:2d} ", end='')
print()  # Newline for next row

for i, row in enumerate(conf_matrix):
    print(f"Class {class_ids[i]}:", " ".join(f"{num:8d}" for num in row))


print()  # Newline for next row
#Print the most misidentified letters
print("In this neural network model, the letters 'A' and 'E' are the most misidentified. While 'A' gets the most false positives, 'E' gets the most false negatives. In sign language, these two letters look very similar. That is why the majority of the false positives for 'A' were actually supposed to be classifed as 'E'.")