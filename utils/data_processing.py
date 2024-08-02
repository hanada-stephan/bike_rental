import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def lagged_data(vector, lag):
    """Generate lagged data 

    Args:
        vector (arr): array with numerical data
        lag (int): number of lag

    Returns:
        array: arrays with lagged data
    """
    X_lag, y_lag = [], []
    for i in range(lag,vector.shape[0]):
      X_lag.append(list(vector.loc[i-lag:i-1]))
      y_lag.append(vector.loc[i])
    X_lag, y_lag = np.array(X_lag), np.array(y_lag) 
    return X_lag, y_lag


def plot_timeseries(x, y1, y2, data1, data2, label1, label2, xlabel,
                    ylabel, title, marker=None):
    sns.lineplot(
        x=x,
        y=y1,
        data=data1,
        label=label1
    )
    sns.lineplot(
        x=x,
        y=y2,
        data=data2,
        label=label2,
        marker=marker
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=70)
    plt.show()
