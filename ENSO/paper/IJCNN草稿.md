## ENSO Forecasting Using ~~Deep~~ Convolutional LSTM Network with Different Optimization Algorithms

***Manuscript Version***

TBD:

1. different optimizers

2. Innet network structue

3. different preprocessing strategy
### Abstract

TBD
### Keywords

Spatiotemporal Sequence Forecasting, Convolutional LSTM, Optimization Algorithm, ENSO.
### 1. Introduction

Approximately every 4 years, the sea surface temperature (SST) is higher than average in the eastern equatorial Pacific. This phenomenon is called El Niño-Southern Oscillation (ENSO) and is considered as the dominant mode of interannual climate variability observed globally (Wunsch 1990). ENSO is associated with many climate changes (Fraedrich 1994; Wilkinson et al.1999), affecting climate of much of the tropics and subtropics, then cause enormous damage worldwide, so a skillful forecasting of ENSO is strongly needed.

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

### 2. Related Work

#### 2.1 Machine Learning for ENSO Forecasting

TBD

#### 2.2 STSF with Deep Learning

TBD
### 3. Methodology

#### 3.1 Formulation of ENSO Forecasting ProblemA multi-channel image like grid pattern, and the specified value stand for concrete physical information.
#### 3.2 Convolutional LSTM Network

TBD### 4. Experiments

#### 4.1 Experiment Settings

We conduct experiments on the real-world monthly (1850.01~2015.12) SST grid dataset, which covers the Niño 3.4 (5N-5S, 170W-120W) region with 1° latitude multiplied by 1° (50*10)[1]. Considering the limited size of dataset, we apply different sliding windows with 6, 9 and 12 month ahead to construct the input sequence respectively, then use rolling-forecasting method to generate new forecasting sequence. 80% of data is used for training, 10% are used for testing while the remaining 10% for validation. We use 0-1 normalization to scale the input SST data before training. To evaluate the performance of our model, we adopt 3 commonly used metrics: Root Mean Square Error (RMSE), Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE), all are the lower the better (formulations as follow).

	Formulas of RMSE/MAE/MAPE 

In the following experiments, we first compare different forecasting methods to validate the effect of our model, then compare the hyperparameters settings and discuss the influence of different optimization algorithms to the final result. Finally, we explore the model interpretation with the generated SST patterns. All neural network based approaches are implemented using Keras, and all the codes are available on GitHub[2].

[1] https://www.esrl.noaa.gov/psd/data/gridded/data.cobe2.html

[2] https://github.com/KrisCheng/Deep4Cli

#### 4.2 Effect of Spatiotempoal Modeling

We compare ConvLSTM based network with other widely used time series regression models, including (1) HA: Historical Average, which models the development of SST as a periodic variation, and uses the passed observation as the prediction; (2) ARIMA: Auto-Regression Integrated Moving Average, which is a well-known model for understanding and forecasting future; (3) SVR: Support Vector Regression, which applies linear support vector machine for regression task. The following deep neural network based models are also included: (4) Feed forward Neural Network (FNN): Feed forward neural network with equal magnitude of parameters; (5) Convolutional Neural Network (CNN); (6) Fully connected LSTM (FC-LSTM). All deep neural networks with 3 hidden layers and similar magnitude of parameters for comparision.

Table 1 shows the result of different approaches for 6-, 9- and 12-month lead time forecasting. we observe the following phenomenon on this process: (1) ConvLSTM outperforms all other baselines regarding all the metrics for all forecasting horizon, which suggests the effectiveness of handling spatiotemporal dependencies. (2) Deep neural network based methods, especially CNN and ConvLSTM, tend to have a better performance than other baselines. One intuitive reason is that the development of SST is irregular and highly spatial- correlated, so it is hard for a model to give accurate predictions on test set without learning the inner dynamics development of the climate system. (3) The performance of different models did not show consistently tendency with the increasing of forecasting horizon, and the performance of CNN is better than FC-LSTM. The intuition is that the temporal dependencies of ENSO is hard to capture than spatial dependencies on this experiment.

- Table 1: Performance comparision of different approaches, 6-, 9- and 12-month lead.

|  *T* | Metric | HA | ARIMA | SVR | FNN | CNN | FC-LSTM | ConvLSTM |
| ---- | ---- | ---- | ---- |---- | ---- | ---- | ---- | ---- |
|         | RMSE | 1.555 | 1.300  | 2.056 | 1.261 | 0.896 | 1.341 |  |
| 6-Month | MAE  | 1.271 | 1.053  | 1.767 | 0.860 | 0.688 | 1.004 |  |
|         | MAPE | 4.78% | 3.95%  | 6.44% | 3.15% | 2.59% | 3.85% |  |
|         | RMSE | 1.506 | 1.314  | 2.056 | 1.248 | 1.147 | 1.313 |  |
| 9-Month | MAE  | 1.224 | 1.051  | 1.791 | 0.997 | 0.920 | 0.981 |  |
|         | MAPE | 4.59% | 3.95%  | 6.54% | 3.73% | 3.38% | 3.75% |  |
|         | RMSE | 1.251 | 1.158  | 2.119 | 1.295 | 1.039 | 1.079 |  |
| 12-Month| MAE  | 0.969 | 0.905  | 1.882 | 1.034 | 0.801 | 0.814 |  |
|         | MAPE | 3.64% | 3.39%  | 6.86% | 3.86% | 3.00% | 3.06% |  |


#### 4.3  Different ConvLSTM Structures Comparison

To investigate the different structure of 

1. Deeper models can produce better results with fewer parameters.
2. More layer and more filters does not always reach better performance.

- Table 2: Performance comparision of different network structure,xL-yN-zS refes the network with x layer, y filters on each layer and the size of filter is z*z, 6-, 9- and 12-month lead.

|  *T* | Metric | xL-yN-zS | xL-yN-zS | xL-yN-zS | xL-yN-zS | xL-yN-zS | xL-yN-zS |
| ---- | ---- | ---- | ---- |---- | ---- | ---- | ---- | ---- |
|      | Parammeters |   |   |   |  |  |  |  |
|              | RMSE |   |   |   |  |  |  |  |
| 6-Month      | MAE  |   |   |   |  |  |  |  |
|              | MAPE |   |   |   |  |  |  |  |
|              | RMSE |   |   |   |  |  |  |  |
| 9-Month      | MAE  |   |   |   |  |  |  |  |
|              | MAPE |   |   |   |  |  |  |  |
|              | RMSE |   |   |   |  |  |  |  |
| 12-Month     | MAE  |   |   |   |  |  |  |  |
|              | MAPE |   |   |   |  |  |  |  |


#### 4.4 Different Optimization Algorithms Comparision

SGD/RMSprop/Adagrad/Adadelta/Adam/Nadam

Learning Curve and result

- Fig1: Different learning curve of different optimization algorithms
#### 4.5 Model Interpretation with Generated SST Patterns

TBD (这个不知道咋写~~~)
### 5. Conclusion and Future WorkTBD### Reference[1] TBD
