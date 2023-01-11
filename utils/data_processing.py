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