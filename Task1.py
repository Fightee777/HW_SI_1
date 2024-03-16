import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0: all messages, 1: info (default), 2: warnings, 3: errors
import tensorflow as tf


gpus = tf.config.list_physical_devices('GPU')
print(tf.config.list_physical_devices('GPU'))
if gpus:
    for gpu in gpus:
        print("Device Name:", gpu.name)
        print("Device Type:", gpu.device_type)
else:
    print("No GPU devices found.")


file_path = 'Task_1_data/SI1.csv'

# an empty list to store rows
rows = []

with open(file_path, 'r') as file_x:
    csv_reader = csv.reader(file_x, delimiter=',')

    for row in csv_reader:
        rows.append(row)

df = pd.DataFrame(rows[1:], columns=rows[0])

X = df.iloc[:, :-4]  # input data
Y = df.iloc[:, -4:]  # output data
print("X shape = ", X.shape)
print("Y shape = ", Y.shape)

training_data_input, test_data_input, training_data_output,  test_data_output = (
    train_test_split(X, Y, test_size=0.3334, random_state=42))

training_data_input_np = np.array(training_data_input.values).astype(float)
test_data_input_np = np.array(test_data_input.values).astype(float)
training_data_output_np = np.array(training_data_output.values).astype(float)
test_data_output_np = np.array(test_data_output.values).astype(float)

print("training_data_input_np.shape", training_data_input_np.shape)
print("test_data_input_np.shape", test_data_input_np.shape)
print("training_data_output_np.shape", training_data_output_np.shape)
print("test_data_output_np.shape", test_data_output_np.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax'),
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

history = model.fit(training_data_input_np, training_data_output_np, epochs=20, batch_size=1)

test_loss, test_accuracy = model.evaluate(test_data_input_np, test_data_output_np)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train'], loc='upper left')

plt.show()
