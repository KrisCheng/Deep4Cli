## ENSO Forecasting Using ~~Deep~~ Convolutional LSTM Network with Different Optimization Algorithms

***Manuscript Version***

1. different optimizers

2. data fusion methods

3. XXX preprocessing
### Abstract

TBD
### Keywords

Spatiotemporal Sequence Forecasting, Convolutional LSTM, Optimization Algorithm, ENSO.
### 1. Introduction

Approximately every 4 years, the sea surface temperature (SST) is higher than average in the eastern equatorial Pacific. This phenomenon is called El Niño-Southern Oscillation (ENSO) and is considered as the dominant mode of interannual climate variability observed globally (Wunsch 1990). ENSO is associated with many climate changes (Fraedrich 1994; Wilkinson et al.1999), affecting climate of much of the tropics and subtropics, then cause enormous damage worldwide, so a skillful forecasting of ENSO is strongly needed.

---
1. What is ENSO and its extremely impacts (why a skillful forecasting is very important);


2. Methods to study ENSO currently (Climate method and it should be improved with longer prediction ahead);

	predict result not well enough + computationally expensive --> remain room for further study of this problem.


3. ENSO is a spatiotemporal sequence forecasting problem and DL (machine learning) methods have great potential to handle this problem. exist work have been attempet to this problem. However, little work have been done with this problem for exploring the spatial and temporal information of SST pattern.

2 complex factors:

* Spatial dependencies

* Temporal dependencies


we formulize the grid SST pattern problem and use the Convolutional LSTM model to capture the spatial and temporal information of ENSO development simultaneously,

In summary, The contributions of our work are 3-fold: 

* We formulize the ENSO SST pattern forecasting problem, a I * J grid map based on longitude and latitude where a grid donates a region of NINO3.4, which can be converted as a multi-channels physical parameters setting spatiotemporal sequence forecasting problem;

* We apply Convolutional LSTM network, which can capture the spatial and temporal information of SST data effectively, to predict ENSO with -6, -9, -12 monthly ahead respectively, the result show that our model outperform other neural network models and conventional statistical model.

* We apply different gradient-based optimizer algorithms for training process

---

### 2. Related Work

#### 2.1 Machine Learning for ENSO Forecasting

TBD

#### 2.2 STSF with Deep Learning

TBD
### 3. Methodology

#### 3.1 Formulation of ENSO Forecasting ProblemA multi-channel image like grid pattern, and the specified value stand for concrete physical information.
#### 3.2 Convolutional LSTM Network

TBD### 4. Experiment

#### 4.1 Experiment Setting

We conduct experiments on the real-world monthly (1850.01~2015.12) SST grid dataset, which covers the Niño 3.4 (5N-5S, 170W-120W) region with 1° latitude multiplied by 1° (50*10)[1]. Considering the limited size of dataset, we apply different sliding windows with 6, 9 and 12 months ahead to construct the input sequence respectively, then use rolling-forecasting method to generate new forecasting sequence. 80% of data is used for training, 10% are used for testing while the remaining 10% for validation. To evaluate the performance of our model, we adopt the root mean square error (RMSE), mean absolute error (MAE) and mean absolute percentage error (MAPE) as metrics (formulas as follow), all are the lower the better.

	Formulas of RMSE/MAE/MAPE 

We use 0-1 normalization to scale the input SST data before training. In the following experiments, we first compare different forecasting methods to validate the effect of our model, then compare the different hyperparameters setting in our model and discuss the influence of different optimization algorithms to the final result. Finally, we explore the model interpretation with the generated SST patterns. All neural network based approaches are implemented using Keras, and all the codes are available on GitHub[2].

[1] https://www.esrl.noaa.gov/psd/data/gridded/data.cobe2.html

[2] https://github.com/KrisCheng/Deep4Cli

#### 4.2 Effect of Spatiotempoal Modeling

We compare ConvLSTM based network with other widely used time series regression models, including (1) HA: Historical Average, which models the development of SST as a seasonal process, and uses the passed sequence as the prediction; (2) ARIMA: Auto-Regression Integrated Moving Average, which is a well-known model for understanding and forecasting future; (3) SVR: Support Vector Regression, which applies linear support vector machine for regression task. The following deep neural network based models are also included: (4) Feed forward Neural Network (FNN): Feed forward neural network with equal magnitude of parameters for comparision; (5) Convolutional Neural Network (CNN) with 3 hidden layers; (6) Fully connected LSTM with 3 hidden layers and similar parameter magnitude. 

Table 1 shows the result of different approaches for -6 months, -9 months and -12 months ahead forecasting.


- Table1: Performance comparision of different approaches, -6, -9, -12 months ahead

|  *T* | Metric | HA | ARIMA | SVR | FNN | CNN | FC-LSTM | ConvLSTM |
| ---- | ---- | ---- | ---- |---- | ---- | ---- | ---- | ---- |
|             | RMSE |  |  |  |  |  |  |  |
| *-6 Month*  | MAE  |  |  |  |  |  |  |  |
|             | MAPE |  |  |  |  |  |  |  |
|             | RMSE |  |  |  |  |  |  |  |
| *-9 Month*  | MAE  |  |  |  |  |  |  |  |
|             | MAPE |  |  |  |  |  |  |  |
|             | RMSE |  |  |  |  |  |  |  |
| *-12 Month* | MAE  |  |  |  |  |  |  |  |
|             | MAPE |  |  |  |  |  |  |  |


#### 4.3  Different ConvLSTM Structures Comparison

TBD

#### 4.4 Different Optimization Algorithms Comparision

TBD
#### 4.5 Model Interpretation with Generated SST Patterns

TBD
### 5. Conclusion and Future WorkTBD### Reference[1] TBD
