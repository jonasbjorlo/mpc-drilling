import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tens
# The time series prediction utilities have been based on TensorFlow time-series forecasting utilities.  
# Source: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/structured_data/time_series.ipynb
# Date Accessed: 10.02-23.

def load_heave():
    colnames = ['TIME', 'MRU_VALUE']
    dcols = ['TIME']
    df = pd.read_csv('mod_mru.csv',
                    names=colnames, header=0, parse_dates=True)
    df["TIME"] = pd.to_datetime(df["TIME"])
    df.insert(1,"TIMESEC", pd.to_timedelta(df["TIME"].dt.time.astype(str)).dt.total_seconds())

    mask = (df['TIMESEC'] >7000) & (df['TIMESEC'] < 70000)
 
    s = df[mask]
    s["TIMESEC"]-=s["TIMESEC"].iloc[0]


    f = interp1d(s["TIMESEC"],s["MRU_VALUE"])
    t_new = np.arange(s["TIMESEC"].iloc[0],s["TIMESEC"].iloc[-1],0.5)
    heave_new = f(t_new)
    plt.plot(t_new,heave_new,'r.')
    plt.plot(s["TIMESEC"],s["MRU_VALUE"])
    new_df = pd.DataFrame({'MRU_VALUE': heave_new})
    s = new_df
    print(s,t_new)

def split_dataset(s, testsize, trainsize):
    column_indices = {name: i for i, name in enumerate(s.columns)}

    n = len(s)
    train_s = s[int(n*testsize):int(n*trainsize)]
    val_s = s[int(n*trainsize):int(n*1)]
    test_s = s[:int(n*testsize)]

    return train_s, val_s, test_s

class Window():
    def __init__(self, input_width, label_width, shift,
               train_df=train_s, val_df=val_s, test_df=test_s,
               label_columns=None):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

def plot(self, model=None, plot_col='MRU_VALUE', max_subplots=1):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index], edgecolors='k', label='Labels', c='#2ca02c', s=64)

        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index], marker='X', edgecolors='k', 
                        label='Predictions',c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

    plt.xlabel('Time [s]')

def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=10,)

    ds = ds.map(self.split_window)

    return ds

@property
def train(self):
    return self.make_dataset(self.train_df)

@property
def val(self):
    return self.make_dataset(self.val_df)

@property
def test(self):
    return self.make_dataset(self.test_df)

@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result

def train_model(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
    return history

def construct_model():
    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(OUT_STEPS*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])

    #lstm_model.summary()

    history = train_model(lstm_model, wide_window)

    #IPython.display.clear_output()
    #val_performance['LSTM'] = lstm_model.evaluate( wide_window.val)
    #performance['LSTM'] = lstm_model.evaluate( wide_window.test, verbose=0)