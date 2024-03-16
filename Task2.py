import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras as k

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

test_pd = pd.read_csv("Task_2_data/test.csv")
train_pd = pd.read_csv("Task_2_data/train.csv")
gender_submission = pd.read_csv("Task_2_data/gender_submission.csv")

input_data_names = ["Pclass", "Sex", "SibSp", "Fare", "Parch", "Embarked"]

train_input_pd = train_pd[input_data_names]
train_output_pd = train_pd["Survived"]
test_input_pd = test_pd[input_data_names]
test_output_pd = gender_submission["Survived"]


normalisation = False
def normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())

def preprocess_data(df):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # One-hot encode 'Embarked' and 'Pclass'
    df = pd.get_dummies(df, columns=['Embarked', 'Pclass'], prefix=['Embarked', 'Pclass'])

    if normalisation:
        df['Fare'] = df['Fare'].apply(lambda x: (x - df['Fare'].min()) / (df['Fare'].max() - df['Fare'].min()))
        df['SibSp'] = df['SibSp'].apply(lambda x: (x - df['SibSp'].min()) / (df['SibSp'].max() - df['SibSp'].min()))
        df['Parch'] = df['Parch'].apply(lambda x: (x - df['Parch'].min()) / (df['Parch'].max() - df['Parch'].min()))

    return df.values.astype(np.float32)


# Preprocess train and test data
train_input_processed = preprocess_data(train_input_pd.copy())
test_input_processed = preprocess_data(test_input_pd.copy())

# Define the model
model = k.Sequential([
    k.layers.Dense(units=200, activation="relu"),
    k.layers.Dense(units=50, activation="relu"),
    k.layers.Dense(units=1, activation="sigmoid")  # Sigmoid for binary classification
])

model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])

# Train the model
fit_results = model.fit(train_input_processed, train_output_pd, epochs=80, batch_size=32)

# Evaluate the model on test data
print("\nModel evaluation on test data:")
test_loss, test_accuracy = model.evaluate(test_input_processed, test_output_pd)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

# Plot training & validation accuracy
plt.plot(fit_results.history['accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train'], loc='upper left')
plt.show()
