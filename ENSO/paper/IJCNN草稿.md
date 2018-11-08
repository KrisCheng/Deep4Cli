## Deep Convolutional LSTM Network with Different Optimization Algorithms for ENSO Forecasting


1. different optimizers

2. data fusion methods

3. XXX preprocessing

***Manuscript Version***### Abstract

TBD
### Keywords

Spatiotemporal Sequence Forecasting, Convolutional LSTM, Optimization Algorithms, ENSO.
### 1. Introduction

Approximately every 4 years, the sea surface temperature (SST) is higher than average in the eastern equatorial Pacific. This phenomenon is called El Niño-Southern Oscillation (ENSO) and is considered as the dominant mode of interannual climate variability observed globally (Wunsch 1990). ENSO is associated with many climate changes (Fraedrich 1994; Wilkinson et al.1999), affecting climate of much of the tropics and subtropics, then cause enormous damage worldwide, so a skillful forecasting of ENSO is strongly needed.

---
1. What is ENSO and its extremely impacts (why a skillful forecasting is very important);


2. Methods to study ENSO currently (Climate method and it should be improved with longer prediction ahead);

	predict result not well enough + computationally expensive --> remain room for further study of this problem.


3. ENSO is a spatiotemporal sequence forecasting problem and DL (machine learning) methods have great potential to handle this problem. exist work have been attempet to this problem. However, little work have been done with this problem for exploring the spatial and temporal information of SST pattern.

2 (or 3) complex factors:

* Spatial dependencies

* Temporal dependencies

* Exernal influence (TBD)

we formulize the grid SST pattern problem and use the Convolutional LSTM model to capture the spatial and temporal information of ENSO development simultaneously,

In summary, The contributions of our work are 2 (or 3)-fold: 

* We formulize the ENSO SST pattern forecasting problem, a I * J grid map based on longitude and latitude where a grid donates a region of NINO3.4, which can be converted as a multi-channels physical parameters setting spatiotemporal sequence forecasting problem;

* We apply Convolutional LSTM network, which can capture the spatial and temporal information of SST data effectively, to predict ENSO with -6, -9, -12 monthly ahead respectively, the result show that our model outperform other neural network models and conventional statistical model.

* TBD

---

### 2. Related Work

#### 2.1 Machine Learning for ENSO Forecasting

TBD


#### 2.2 STSF with Deep Learning

TBD---



---
### 3. Methodology

#### 3.1 Formulation of ENSO Forecasting ProblemTBD#### 3.2 Convolutional LSTM Network

TBD---



---
### 4. Experiment

brief introduction of the whole experiment setting.

#### 4.1 Experiment Setting

3 Metrics

RMSE / MAE / MAPE

1. Comparision between different methods

	Historical Average

	ARIMA (todo) (grid-point)

	FNN

	CNN 

	FC-LSTM

	ConvLSTM

	ConvLSTM + XXX (TBD)


2. Comparison between ConvLSTM with inner structures ( with(out) BN ) or  different optimizers



#### 4.2 Effect of Spatial and Tempoal Dependency Modeling


Number of units and width of kernel (effects of different parameters)


Learning Curve

#### 4.3 Model Interpretation with Generated SST Pattern (Case: ENSO 15~16)

TBD


---



---
### 5.Conclusion and Future WorkTBD---



---

### Reference[1] TBD
