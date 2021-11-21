import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import RobustScaler


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

# sns.set(style='whitegrid', palette='muted', font_scale=1.5)

# rcParams['figure.figsize'] = 16, 10

# RANDOM_SEED = 42

# np.random.seed(RANDOM_SEED)
# tf.random.set_seed(RANDOM_SEED)

# time = np.arange(0, 100, 0.1)
# sin = np.sin(time) + np.random.normal(scale=0.5, size=len(time))

# plt.plot(time, sin, label='sine (with noise)');
# plt.legend();
# plt.show();


df = pd.read_csv(
  "dailyInPatientsData.csv",
  parse_dates=['Date'],
  index_col='Date'
)
print(df.head())

train_size = int(len(df) * 0.9)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(len(train), len(test))

# f_columns = ['Date']
# f_transformer = RobustScaler()
# f_transformer = f_transformer.fit(train[f_columns].to_numpy())
# train.loc[:, f_columns] = f_transformer.transform(
#   train[f_columns].to_numpy()
# )
# test.loc[:, f_columns] = f_transformer.transform(
#   test[f_columns].to_numpy()
# )

cnt_transformer = RobustScaler()
cnt_transformer = cnt_transformer.fit(train[['Number']])
train['Number'] = cnt_transformer.transform(train[['Number']])
test['Number'] = cnt_transformer.transform(test[['Number']])

time_steps = 10
# reshape to [samples, time_steps, n_features]
X_train, y_train = create_dataset(train[['Number']], train.Number, time_steps)
X_test, y_test = create_dataset(test[['Number']], test.Number, time_steps)
print(X_train.shape, y_train.shape)

print(X_train.shape[1])
print(X_train.shape[2])

# print(test[['Number']])


model = keras.Sequential()
model.add(keras.layers.LSTM(
  units=128,
  input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(keras.layers.Dense(units=1))
model.compile(
  loss='mean_squared_error',
  optimizer=keras.optimizers.Adam(0.001)
)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=16,
    validation_split=0.1,
    verbose=1,
    shuffle=False
)

y_pred = model.predict(X_test)

# y_train_inv = cnt_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = cnt_transformer.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = cnt_transformer.inverse_transform(y_pred)

print(y_test_inv.flatten())
print('')
print('')
print('GAP')
print('')
print('')
print('')
print('')
print(y_pred_inv.flatten());

plt.plot(y_test_inv.flatten(), marker='.', label="true")
plt.plot(y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Number of Patients')
plt.xlabel('Date')
plt.legend()
plt.show();