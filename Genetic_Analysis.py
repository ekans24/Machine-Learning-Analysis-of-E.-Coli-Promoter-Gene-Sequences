import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import motifs
from Bio.Seq import Seq
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

# Replace this with the path to your "promoters.data" file
data_file_path = "C:\\Users\\sekra\\Desktop\\Bioinformatics_project\\data\\promoters.data"

# Initialize an empty list to store the data
data = []

# Read and process the file with a more flexible approach
with open(data_file_path, 'r') as file:
    for line in file:
        parts = line.strip().split(',')  # Splitting by comma to handle format variations
        if len(parts) >= 2:
            class_label = parts[0].strip()  # Class label is the first part
            sequence = parts[-1].strip()  # Sequence is the last part
            data.append({'class': class_label, 'sequence': sequence})

# Convert the list to a DataFrame
df = pd.DataFrame(data)

# Initialize the OneHotEncoder with the updated parameter
encoder = OneHotEncoder(sparse_output=False, categories=[['A', 'C', 'G', 'T']])

# Function to encode a single sequence
def encode_sequence(seq):
    seq_upper = seq.upper()  # Convert the sequence to uppercase
    encoded = encoder.fit_transform([[nuc] for nuc in seq_upper])
    return encoded.flatten()

# Apply the encoding to each sequence
encoded_sequences = [encode_sequence(seq) for seq in df['sequence']]

# Convert the list of arrays into a DataFrame
encoded_df = pd.DataFrame(encoded_sequences)

print(encoded_df.head())

################
#### EDA #####

# Assuming df is your DataFrame with sequences

# Basic statistics
print(df.describe())

# Nucleotide frequency
nucleotide_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
for seq in df['sequence']:
    for nucleotide in seq:
        nucleotide_counts[nucleotide.upper()] += 1

print(nucleotide_counts)

# Visualizing nucleotide frequency
plt.bar(nucleotide_counts.keys(), nucleotide_counts.values())
plt.xlabel('Nucleotide')
plt.ylabel('Frequency')
plt.title('Nucleotide Frequency in Sequences')
plt.show()


# Ensure all sequences are in uppercase
df['sequence'] = df['sequence'].str.upper()

# Motif Analysis
instances = [Seq(seq) for seq in df['sequence']]
m = motifs.create(instances)
print(m.counts)

# Calculate entropy for each position
def calculate_entropy(motif_counts):
    total = sum(motif_counts)
    probabilities = [freq / total for freq in motif_counts]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

# Assume that m.counts is a dictionary with keys 'A', 'C', 'G', 'T'
entropy_by_position = [calculate_entropy([m.counts[nuc][i] for nuc in "ACGT"])
                       for i in range(len(df['sequence'][0]))]

# Visualize entropy by position
sns.lineplot(data=entropy_by_position)
plt.xlabel('Position')
plt.ylabel('Entropy')
plt.title('Entropy by Position in Sequences')
plt.show()

# One-Hot Encoding of the sequences
encoder = OneHotEncoder(sparse=False)
encoded_sequences = encoder.fit_transform(df['sequence'].apply(lambda x: list(x)).tolist())

# PCA Visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(encoded_sequences)
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Encoded Sequences')
plt.show()

####################################################################################################
##########################
### FEATURE SELECTION ####
# Assuming 'entropy_by_position' is a list of entropy values for each position
# and 'encoded_df' is your DataFrame after one-hot encoding

# Define your entropy thresholds
high_entropy_threshold = np.percentile(entropy_by_position, 75)  # top 25% high entropy
low_entropy_threshold = np.percentile(entropy_by_position, 25)   # bottom 25% low entropy

# Identify positions with low entropy (conserved)
conserved_positions = [i for i, e in enumerate(entropy_by_position) if e <= low_entropy_threshold]

# Assuming each position in the sequence corresponds to 4 columns in the encoded_df (one for each nucleotide)
# We need to calculate the actual column indices in the encoded DataFrame
conserved_column_indices = []
for pos in conserved_positions:
    conserved_column_indices.extend(range(pos*4, (pos+1)*4))

# Select only the columns corresponding to the conserved positions
selected_features = encoded_df.iloc[:, conserved_column_indices]

# Now 'selected_features' DataFrame contains only the features you've chosen based on entropy

#############
### MODEL ###


### LOGISTIC REGRESSION ###

from sklearn.model_selection import train_test_split

# Extracting features and target variable
X = selected_features
y = df['class'].map({'+': 1, '-': 0})  # Convert class labels to binary (assuming '+' is positive)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Confusion Matrix:\n{conf_matrix}")


### RANDOM FOREST ###

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Define a grid of hyperparameters to tune
param_grid_rf = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Setup GridSearchCV
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy')

# Fit the model
grid_search_rf.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters:", grid_search_rf.best_params_)
print("Best score:", grid_search_rf.best_score_)

###########
### SVM ###

from sklearn.svm import SVC

# Initialize the SVM model
svm_model = SVC(random_state=42)

# Define a grid of hyperparameters to tune
param_grid_svm = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}

# Setup GridSearchCV
grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=5, scoring='accuracy')

# Fit the model
grid_search_svm.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters:", grid_search_svm.best_params_)
print("Best score:", grid_search_svm.best_score_)

#######################
### NEURAL NETWORK ###

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Create the model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=10, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy}")