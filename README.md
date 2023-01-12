# Bike rental Analysis: Project Overview

**Tags: neural network, GRU, LSTM, deep learning, glorot uniform, RMSProp, EDA, Keras, TensorFlow, deep learning, time series**

This notebook is part of Alura's courses [Deep Learning: previsão com Keras) (Deep Learning: predictions with Keras)](https://cursos.alura.com.br/course/deep-learning-previsao-keras) by Allan Segovia Spadini.


## Code and resources

Platform: Google Colab

Python version: 3.7.6

Packages: matplotlib, numpy, pandas, seaborn, sklearn, TensorFlow


## Data set

The data set contains information about the number of rented bikes per hour from January 4th, 2015 to January 3rd, 2017. Also, there is information about thermal sensation, humidity, wind speed, climate, season, and if it is a holiday and weekend. Thus, there are 17414 records and 10 columns. All of them are numerical besides the column "date". 


## Data cleaning

The data cleaning steps were:
- convert the date column from object to datetime;
- scale the number of rented bikes before using them as the model's input. 
- lag the data in 10 rows, i.e., the model uses the last 10 data points the predict the next. 

## Model building

The first neural network (NN) model has 2 layers:
- The LSTM with 128 units;
- Dense layer with 1 unit.
The loss function and the optimizer are, respectively, mean squared error and RMSProp.

The second NN model also has 2 layers:
- The GRU with 128 units;
- Dense layer with 1 unit.
This model has the same loss function and optimizer as the LSTM model. 

## Model performance

Both models have similar loss function values for both the training and test set (around 0.01). Since the GRU model has fewer parameters to train, it spends less time to run, and it is advisable to use this model rather than the LSTM. 
