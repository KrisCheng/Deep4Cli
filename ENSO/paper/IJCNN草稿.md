## ENSO Forecasting Over Multiple Time Horizons Using Deep Convolutional LSTM Network and Rolling Mechanism

***Manuscript Version***

TBD:

1. different optimizers

2. inner network structure

3. different preprocessing strategy
### Abstract

TBD
### Keywords

Spatiotemporal Sequence Forecasting, Convolutional LSTM, Rolling Mechanism, ENSO.
### 1. Introduction

"fully coupled models are computationally expensive and only just starting to become publicly available. From a practical point of view, it remains challenging for the typhoon community to replace existing operational forecast systems with coupled models at this stage."

Approximately every 4 years, the sea surface temperature (SST) is higher than average in the eastern equatorial Pacific. This phenomenon is called El Niño-Southern Oscillation (ENSO) and is considered as the dominant mode of interannual climate variability observed globally (Wunsch 1990). ENSO is associated with many climate changes (Fraedrich 1994; Wilkinson et al.1999), affecting climate of much of the tropics and subtropics, then cause enormous damage worldwide, so a skillful forecasting of ENSO is strongly needed.

1. What is ENSO and its extremely impacts (why a skillful forecasting is very important);

2. Methods to study ENSO currently (Climate method and it should be improved with longer prediction ahead);

	predict result not well enough + computationally expensive --> remain room for further study of this problem.

3. ENSO is a spatiotemporal sequence forecasting problem and DL (machine learning) methods have great potential to handle this problem. exist work have been attempet to this problem. However, little work have been done with this problem for exploring the spatial and temporal information of SST pattern.

2 complex factors:

* Spatial dependencies

* Temporal dependencies


Why Rolling? Direct is not good and rolling is a better choice for multi-step forecasting.

we formulize the grid SST pattern problem and use the Convolutional LSTM model to capture the spatial and temporal information of ENSO development simultaneously,

In summary, The contributions of our work are 3-fold: 

* We formulize the ENSO SST pattern forecasting problem, a I * J grid map based on longitude and latitude where a grid donates a region of NINO3.4, which can be converted as a multi-channels physical parameters setting spatiotemporal sequence forecasting problem;

* We apply Convolutional LSTM network, which can capture the spatial and temporal information of SST data effectively, to predict ENSO with -6, -9, -12 monthly ahead respectively, the result show that our model outperform other neural network models and conventional statistical model.


* Visualization of ENSO Pattern

The remainder of the paper is structured as follow:  

### 2. Related Work

#### 2.1 Machine Learning for ENSO Forecasting

TBD

#### 2.2 STSF with Deep Learning

TBD
### 3. Methodology

#### 3.1 Formulation of ENSO Forecasting ProblemA multi-channel image like grid pattern, and the specified value stand for concrete physical information.
#### 3.2 Convolutional LSTM and Rolling Mechanism (ConvLSTM-RM)

Each ConvLSTM block is composed of a convolutional LSTM layer and a batch normalization layer.

We use 0-1 normalization to scale the input SST data before training. 
then use rolling mechanism to generate new forecasting sequence.

"Theoretically, the proposed RMGM-BP hybrid model in this study consists of four essential components: data input, data pre-processing, neural network algorithm and the data output space."
"Next, the RM process is used to group the normalized data as the set of new inputs via:

Following this, the GM method is applied to transform the original chaotic streamflow patterns"

### 4. Experiments

#### 4.1 Experiment Settings

We conduct experiments on the real-world monthly (1850.01~2015.12) SST grid dataset, which covers the Niño 3.4 (5N-5S, 170W-120W) region with 1° latitude multiplied by 1° (50*10)[1]. Considering the limited size of dataset, we apply different sliding windows with 6, 9 and 12 month ahead to construct the input sequence respectively. 80% of data is used for training, 10% are used for testing while the remaining 10% for validation. To evaluate the performance of our model, we adopt 3 commonly used metrics: Root Mean Square Error (RMSE), Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE), all are the lower the better (formulations as follow).

	Formulas of RMSE/MAE/MAPE 

In the following experiments, we first compare different forecasting methods to validate the effect of our model, then discuss the influence of different hyperparameters settings to the final result. Finally, we take the ENSO during 2015/16 as the case, explore the model interpretation with the generated SST patterns. All neural network based approaches are implemented using Keras, and all the codes are available on GitHub[2]. We run all the experiments on a computer with two NVIDIA 1080Ti GPUs.

[1] https://www.esrl.noaa.gov/psd/data/gridded/data.cobe2.html

[2] https://github.com/KrisCheng/Deep4Cli

#### 4.2 Effect of Spatiotemporal Modeling

We compare ConvLSTM-RM network with other widely used time series regression models, including (1) HA: Historical Average, which models the development of SST as a periodic variation, and uses the passed observation as the prediction; (2) ARIMA: Auto-Regression Integrated Moving Average, which is a well-known model for understanding and forecasting future; (3) SVR: Support Vector Regression, which applies linear support vector machine for regression task. The following deep neural network based models are also included: (4) Feed forward Neural Network (FNN): Feed forward neural network with equal magnitude of parameters; (5) Convolutional Neural Network (CNN); (6) Fully connected LSTM (FC-LSTM). Those deep neural networks with 3 hidden layers and roughly same amount of parameters, and they are fully trained with a fixed number of epochs (e.g., 10000 epochs) for fair comparision.

Table 1 shows the result of different approaches for 6-, 9- and 12-month lead time forecasting. we observe the following phenomenon on this process: (1) ConvLSTM outperforms all other baselines regarding all the metrics for all forecasting horizon, which suggests the effectiveness of handling spatiotemporal dependencies. (2) Deep neural network based methods, especially CNN and ConvLSTM, tend to have a better performance than other baselines. One intuitive reason is that the development of SST is irregular and highly spatial- correlated, so it is hard for a model to give accurate predictions on test set without learning the inner dynamics development of the climate system. (3) The performance of different models did not show consistently tendency with the growth of forecasting horizon, and the performance of CNN is better than FC-LSTM. The intuition is that the temporal dependencies of ENSO is hard to capture than spatial dependencies in this experiment.

- Table 1: Performance comparision of different approaches, 6-, 9- and 12-month lead, and the boldface items in the table represent the best performance.

|  *T* | Metric | HA | ARIMA | SVR | FNN | CNN | FC-LSTM | ConvLSTM | ConvLSTM-RM |
| ---- | ---- | ---- | ---- |---- | ---- | ---- | ---- | ---- | ---- |
|         | RMSE | 1.555 | 1.300  | 2.056 | 1.261 | 0.896 | 1.341 |0.947|0.729|
| 6-month | MAE  | 1.271 | 1.053  | 1.767 | 0.860 | 0.688 | 1.004 |0.749|0.555|
|         | MAPE | 4.78% | 3.95%  | 6.44% | 3.15% | 2.59% | 3.85% |2.72%|1.45%|
|         | RMSE | 1.506 | 1.314  | 2.056 | 1.248 | 1.147 | 1.313 |0.976|0.807|
| 9-month | MAE  | 1.224 | 1.051  | 1.791 | 0.997 | 0.920 | 0.981 |0.769|0.605|
|         | MAPE | 4.59% | 3.95%  | 6.54% | 3.73% | 3.38% | 3.75% |2.86%|2.27%|
|         | RMSE | 1.251 | 1.158  | 2.119 | 1.295 | 1.039 |1.079|1.033| 0.789 |
| 12-month| MAE  | 0.969 | 0.905  | 1.882 | 1.034 | 0.801 |0.814|0.805| 0.607 |
|         | MAPE | 3.64% | 3.39%  | 6.86% | 3.86% | 3.00% |3.06%|2.98%| 2.25% |


#### 4.3 Different ConvLSTM-RM Structures Comparison

To further investigate the influence of different network structures to the final result and figure out the best network structure, we compare ConvLSTM with the following two aspects: (1) Num of Layer, which is a fundamental setting for deep neural network structure. (2) Num of kernel and kernel size, which can stand for the collectors of spatial correlation between SST grid data. We apply grid search strategy in this experiment, all with Adam optimizer during training process to accelerate learning process.

Fig 1 shows the comparison between different number of ConvLSTM block. We found that (1) deeper models can produce better results with fewer parameters; (2) More layer and more filters does not always reach better performance.


- Fig 1: Performance comparision of different network layers, 6-, 9- and 12-month lead.

Layers

|  *T* | Metric | 1 | 2 | 3 | 4 | 5 |
| ---- | ---- | ---- | ---- |---- | ---- | ---- |
|         | RMSE | 0.878  | 0.823  | 0.737  | 0.729  | 0.812  | 
| 6-Month | MAE  | 0.681  | 0.627  | 0.562  | 0.555  | 0.615  | 
|         | MAPE | 2.54%  | 2.36%  | 2.11%  | 1.45%  | 1.93%  | 
|         | RMSE | 0.931 | 0.892 | 0.879  | 0.807  | 0.883  | 
| 9-Month | MAE  | 0.729 | 0.682 | 0.691  | 0.605  |  0.673 |
|         | MAPE | 2.71% | 2.55% | 2.56%  | 2.27%  | 2.49%  | 
|         | RMSE | 0.888  |  0.844 | 0.828  | 0.789  | 1.035  | 
| 12-Month| MAE  | 0.681  |  0.649 | 0.633  | 0.607  | 0.780  | 
|         | MAPE | 2.53%  |  2.42% | 2.34%  | 2.25%  | 2.86%  | 

Next, we explore the chose of different number of kernel and kernel size. Fig 1 shows final result with layer X. K roughtly corresponds to the size of filters' reception field while the number of units corresponds to the number of filters. Larger K enables the model to capture broader spatial dependency at the cost of increasing learning complexity. We observe that with the increase of K, the error on the testing dataset first quickly decrease, and then slightly increase. Similar behavior is observed for varying the number of units.

- Fig 2: Performance comparision of different kernel number .

Histogram (kernel num and kernel size) or Box-Whisker Plot is better (many models).

Kernel Num with 4 Layer

|  *T* | Metric | 4 | 8 | 16 | 32 |
| ---- | ---- | ---- | ---- |---- | ---- |
| 12-Month | RMSE  | 1.070  |  0.789 | 0.861  |  1.007 |



Kernel Num with 4 Layer and 8 Kernels

- Fig 3: Performance comparision of different kernel size.

|  *T* | Metric | 1 | 2 | 3 | 4 | 5 |
| ---- | ---- | ---- | ---- |---- | ---- | ---- |
| 12-Month | RMSE  | 1.094  | 0.882 | 0.789 | 0.868  | 0.906  | 

#### 4.4 Model Interpretation with Generated SST Patterns

Why ENSO 15/16?

To better understand the behavior of our model, In this part, we take the ENSO occured during 2015/2016 as the case, which is considered as the most extreme ENSO since records began, and the fluctuation of SST is a good candidate for investigate. We visualize the generated SST patterns from 2015.01 to 2016.01. Fig 3 shows the visualization of 12-month lead forecasting. We have the following observations: 

(1) ...

(2) we calculate the NINO3.4 index during this period, and compared it with Climate Model Prediction From IRI CPC.

(3) .... which can be considered as a successful prediction of ENSO.


Outline:

pattern of ENSO during 2015/2016 --> our model capture the development of SST pattern --> regression result, predict the peak value and can be considered as a successful prediction of ENSO 15/16.	

(https://iri.columbia.edu/our-expertise/climate/forecasts/enso/2015-January-quick-look/) 

(https://iri.columbia.edu/our-expertise/climate/forecasts/enso/2015-January-quick-look/?enso_tab=enso-sst_table)



- Fig 4: The generated SST pattern from ..., which indicate a strong ENSO and predictly the time accuracy.


- Fig 5: Nino index with Model Comparision.


~~~

Experiment Can Be Done (TBD)

1. Different Optimization Algorithms Comparison

The optimizer of SGD/RMSprop/Adagrad/Adadelta/Adam/Nadam during training process.

Adam is the most stable one.

- Fig 2: Learning curve of different optimization algorithms in validation set.


2. Robustness of the Network Structure

different data split, a lot of experiment to figure out the RMSE dictribution with different time horizons.~~~
### 5. Conclusion and Future WorkTBD### Reference[1] TBD
