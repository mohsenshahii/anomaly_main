!wget http://www.timeseriesclassification.com/Downloads/ECG5000.zip
!unzip ECG5000.zip

!cat ECG5000_TRAIN.txt ECG5000_TEST.txt > ecg_final.txt

!head ecg_final.txt



import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
mpl.rcParams['figure.figsize'] = (10,8)
mpl.rcParams['axes.grid'] = False


df = pd.read_csv('ecg_final.txt', sep='  ', header=None)

df

df.columns

df = df.add_prefix('c')

df['c0'].value_counts()

df.describe()

train_data, test_data, train_labels, test_labels = train_test_split(df.values, df.values[:,0:1], test_size=0.2, random_state=111)

scaler = MinMaxScaler()
data_scaled = scaler.fit(train_data)

train_data_scaled = data_scaled.transform(train_data)
test_data_scaled = data_scaled.transform(test_data)

train_data_scaled

normal_train_data = pd.DataFrame(train_data_scaled).add_prefix('c').query('c0 == 0').values[:,1:]
anomaly_train_data = pd.DataFrame(train_data_scaled).add_prefix('c').query('c0 > 0').values[:,1:]

normal_test_data = pd.DataFrame(test_data_scaled).add_prefix('c').query('c0 == 0').values[:,1:]
anomaly_test_data = pd.DataFrame(test_data_scaled).add_prefix('c').query('c0 > 0').values[:,1:]

print(normal_train_data.shape)
print(normal_test_data.shape)

print(anomaly_train_data.shape)
print(anomaly_test_data.shape)

plt.plot(normal_train_data[0])
plt.plot(normal_train_data[1])
plt.plot(normal_train_data[2])

plt.plot(anomaly_train_data[0])
plt.plot(anomaly_train_data[1])
plt.plot(anomaly_train_data[2])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(140, activation='sigmoid'))

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=2,
                                                  mode='min')
model.compile(optimizer='adam', loss='mae')

history = model.fit(normal_train_data, normal_train_data,
                    epochs=50,
                    batch_size=128,
                    validation_data=(train_data_scaled[:,1:], train_data_scaled[:,1:]),
                    shuffle=True,
                    callbacks=[early_stopping])


reconstructions = model.predict(normal_test_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_test_data)

plt.hist(train_loss, bins=50)

np.mean(train_loss)

np.std(train_loss)

threshold = np.mean(train_loss) + 2*np.std(train_loss)

threshold

reconstructions_a = model.predict(anomaly_test_data)
train_loss_a = tf.keras.losses.mae(reconstructions_a, anomaly_test_data)

plt.hist(train_loss_a, bins = 50)

plt.hist(train_loss, bins=50, label='normal')
plt.hist(train_loss_a, bins=50, label='anomaly')
plt.axvline(threshold, color='r', linewidth=3, linestyle='dashed', label= '{:0.3f}'.format(threshold))
plt.legend(loc='upper right')
plt.show()

np.mean(train_loss_a)

np.std(train_loss_a)

tf.math.less(train_loss, threshold)
preds = tf.math.less(train_loss, threshold)
tf.math.count_nonzero(preds)

preds.shape

preds_a = tf.math.greater(train_loss_a, threshold)
tf.math.count_nonzero(preds_a)

preds_a.shape




