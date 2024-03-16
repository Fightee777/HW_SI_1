import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0: all messages, 1: info (default), 2: warnings, 3: errors

data = pd.read_csv("Task_2_data_invalid/csgo_round_snapshots.csv")

input_filtred_data = data[["time_left", "ct_score", "t_score", "map", "bomb_planted", "ct_health", "t_health",
                           "ct_armor", "t_armor", "ct_money", "t_money", "ct_helmets", "t_helmets", "ct_defuse_kits",
                           "ct_players_alive", "t_players_alive"]]


input_filtred_data_copy = input_filtred_data.copy()  # do copy
input_filtred_data_copy['bomb_planted'] = input_filtred_data_copy['bomb_planted'].astype(int)  # convert copy
input_filtred_data = input_filtred_data_copy*1  # return all values

output_filtred_data = data["round_winner"]
output_filtred_data = output_filtred_data.to_frame()
word_mapping = {'CT': 0, 'T': 1}  # czy wygrają T?
output_filtred_data['round_winner'] = output_filtred_data['round_winner'].map(word_mapping)
print(output_filtred_data.columns.tolist())
print(output_filtred_data['round_winner'])


# Create a sample DataFrame with a categorical column
required_columns = ['de_dust2', 'de_mirage', 'de_nuke', 'de_inferno',
                    'de_overpass', 'de_vertigo', 'de_train', 'de_cache']

OHE_map = pd.get_dummies(input_filtred_data['map'])
OHE_map = OHE_map.astype(float)
print(type(OHE_map))
print("OHE_map", OHE_map.shape)


input_filtred_data = input_filtred_data.drop('map', axis=1)
print("OHE_map = ", OHE_map)

print("\n\n\ninput_filtred_data", input_filtred_data.shape)
print("OHE_map\n\n\n", OHE_map.shape)


print("columns = ", input_filtred_data.columns.tolist())

input_filtered_data_pd = pd.concat([input_filtred_data.astype(float), OHE_map.astype(float)],
                                   ignore_index=False, axis=1)
print("columns = ", input_filtered_data_pd.columns.tolist())

print("input_filtered_data_pd", input_filtered_data_pd)
print("input_filtered_data_pd", input_filtered_data_pd.shape)
print("OHE_map", OHE_map.shape)

print(output_filtred_data['round_winner'].describe())
print(input_filtered_data_pd['bomb_planted'].describe())

print(input_filtered_data_pd.iloc[34, :].head(23))

head_data_out = output_filtred_data.head(50)
head_data = input_filtered_data_pd.head(50)

X_train_raw = input_filtered_data_pd.to_numpy()
y_train_raw = output_filtred_data.to_numpy()
print("X_train = ", X_train_raw.shape)
print("Y_train = ", y_train_raw.shape)


head_data.to_csv('head_data.csv', index=False)
head_data_out.to_csv('head_data.output.csv', index=False)

X_train, X_test, y_train, y_test = train_test_split(X_train_raw, y_train_raw, test_size=0.3334, random_state=143)

model = tf.keras.models.Sequential([
    Dense(2048, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')

])

custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=custom_optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2)


test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
