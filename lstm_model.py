import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

def main():
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)
    rcParams['figure.figsize'] = 16, 10

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    df = merge_data('interpolated_data')
    df = df.rename(columns={'Heart rate': 'Heart_rate'})
    df.set_index('Timestamp',inplace=True)

    train_size = int(len(df) * 0.8)
    test_size = len(df) - train_size
    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

    # History of 3 hours
    time_steps = 36

    X_train, y_train = create_dataset(train, train.Heart_rate, time_steps)
    X_test, y_test = create_dataset(test, test.Heart_rate, time_steps)

    print(X_train.shape, y_train.shape)

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
        epochs=100,
        batch_size=16,
        validation_split=0.1,
        verbose=1,
        shuffle=False
    )

    y_pred = model.predict(X_test)
    ys = np.squeeze(np.asarray(y_pred))
    xs = np.linspace(0, len(ys), num=len(ys))

    sns.lineplot(x=xs, y=y_test, markers=True, label='true')
    sns.lineplot(x=xs, y=ys, label='prediction')
    plt.show()

def merge_data(directory):
    df = pd.DataFrame(columns=('Timestamp','Heart rate'))
    direct = os.listdir(directory)
    direct.sort()
    for filename in direct:
        if filename.endswith(".csv"):
            temp_df = pd.read_csv('interpolated_data/{}'.format(filename))
            df = df.append(temp_df)
    return df

def create_dataset(X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)


if __name__ == '__main__':
    main()

