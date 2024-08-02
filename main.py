import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, LSTM

from utils.data_processing import lagged_data, plot_timeseries

def main():
    # Setting plot parameters
    mpl.rcParams["figure.figsize"] = (10,6)
    mpl.rcParams["font.size"] = 15


    # Importing the data set
    bike = pd.read_csv("./bike_rental.csv")
    # Converting the date column to datetime
    bike["date"] = pd.to_datetime(bike["date"])

    # Instantiating the StandardScaler
    sc = StandardScaler()
    # Fitting and scaling the data
    sc.fit(bike["count"].values.reshape(-1,1))
    X = sc.transform(bike["count"].values.reshape(-1,1))
    # Splitting the data into training and test sets
    training_size = int(len(bike)*0.9)
    X_train = X[0:training_size]
    X_test = X[training_size:len(bike)]   

    plot_timeseries(
        x="date",
        y1 = X_train[:,0],
        y2 = X_test[:,0],
        data1 = bike[0:training_size],
        data2 = bike[training_size:len(bike)],
        label1 = "Train",
        label2 = "Test",
        xlabel = "Date",
        ylabel = "Scaled number of rented bikes",
        title = "Number of rented bikes over time"
    )

    # Lagging the train and test sets
    X_train_df = pd.DataFrame(X_train)[0]
    X_test_df = pd.DataFrame(X_test)[0]
    # Splitting the data into X and y
    X_train_lag, y_train_lag = lagged_data(X_train_df, 10)
    X_test_lag, y_test_lag = lagged_data(X_test_df , 10)
    # Reshaping the input data to include a third dimension
    X_train_lag = X_train_lag.reshape(
        (X_train_lag.shape[0], X_train_lag.shape[1], 1)
    )
    X_test_lag = X_test_lag.reshape((
        X_test_lag.shape[0], X_test_lag.shape[1], 1)
    )

    # Building the LSTM model
    lstm_model = Sequential()
    lstm_model.add(
        LSTM(
            128, 
            input_shape=(
                X_train_lag.shape[1], 
                X_train_lag.shape[2]
            )
        )
    )
    lstm_model.add(Dense(units=1))
    lstm_model.compile(loss="mean_squared_error", optimizer="RMSProp")
    # Fitting the model
    result_lstm = lstm_model.fit(
        X_train_lag,
        y_train_lag,
        validation_data=(X_test_lag, y_test_lag),
        epochs=100
    )
    # Prediction
    y_pred_train = lstm_model.predict(X_train_lag)
    y_pred_test = lstm_model.predict(X_test_lag)

    # Plotting the the training set's prediction
    plot_timeseries(
        x = "date",
        y1 = X_train[:,0],
        y2 = y_pred_train[:,0],
        data1 = bike[0:training_size],
        data2 = bike[0:15662],
        label1 = "Training set",
        label2 = "Prediction",
        xlabel = "Date",
        ylabel = "Scaled number of rented bikes",
        title = "Lagged LSTM's number of rented bikes over time"
    )

    # Scalling back
    y_pred_test_inv = sc.inverse_transform(y_pred_test)
    X_test_inv = sc.inverse_transform(X_test)

    plot_timeseries(
        x = "date",
        y1 = X_test_inv[:,0],
        y2 = y_pred_test_inv[:,0],
        data1 = bike[training_size:len(bike)],
        data2 = bike[training_size+10:len(bike)],
        label1 = "Test set",
        label2 = "Prediction",
        xlabel = "Date",
        ylabel = "Number of rented bikes",
        title = "Scale back's number of rented bikes over time",
        marker = "."
    )

    # Building the GRU model
    gru_model = Sequential()
    gru_model.add(
        GRU(
            128,
            input_shape=(
                X_train_lag.shape[1],
                X_train_lag.shape[2]
            )
        )
    )
    gru_model.add(Dense(units=1))
    gru_model.compile(loss='mean_squared_error',optimizer='RMSProp')
    # Fitting and predicting the GRU model 
    result_gru = gru_model.fit(X_train_lag,
                y_train_lag,
                validation_data=(X_test_lag,y_test_lag),
                epochs=100
    )
    y_pred_gru = gru_model.predict(X_test_lag)
    # Invesing the standard scaling
    y_pred_gru_inv = sc.inverse_transform(y_pred_gru)

    # Plotting the test set's predictions for the GRU model
    plot_timeseries(
        x = "date",
        y1 = X_test_inv[:,0],
        y2 = y_pred_gru_inv,
        data1 = bike[training_size:len(bike)],
        data2 = bike[training_size+10:len(bike)],
        label1 = "Test set",
        label2 = "Prediction",
        xlabel = "Date",
        ylabel = "Number of rental bikes",
        title = "GRU model's number of rented bikes over time",
        marker = "."      
    )

    # Plotting the loss and validation loss over epochs of LSTM and GRU models
    plt.plot(result_lstm.history['loss'], label = "LSTM Training set")
    plt.plot(result_lstm.history['val_loss'], label = "LSTM Test set")
    plt.plot(result_gru.history['loss'], label = "GRU Test set")
    plt.plot(result_gru.history['val_loss'], label = "GRU Test set")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    