# Using Deep Neural Networks to Predict ENSO

***Manuscript Version***Bin Mu, Cheng Peng and Shijin Yuan(✉) 
School of Software Engineering, Tongji University, Shanghai 201804, China
binmu@tongji.edu.cn,  tjupengcheng@163.com, yuanshijin2003@163.com### Abstract.

Todo

The abstract should summarize the contents of the paper in short terms, i.e. 150-250 words.
### Keywords: 

time series prediction, deep neural networks, ENSO.
### 1	Introduction***Outline:***1. the reason (significance) of study ENSO, traditional methods (statistical methods && regular machine learning methods) and limitations.

	* *"The dynamical models use physical equations of the ocean and atmosphere to forecast ENSO event, which are computationally very expensive and not available outside the atmospheric scientific community."*

2. deep learning method’s success on almost all the data-driven problems[4]

	* "capturing long-term temporal dependencies" -- LSTM
	* CNN+LSTM has been applied on climate problem, like Sea Level Anomalies (SLA), which show great pontential for such kind of problem.[7]

3. ENSO SST prediction is a suitable (reasonable) candidate for deep learning method, some work have been done on this problem[1][2], which show great potential for this problem, but limitation as follow:

	* only SST prediction, no specified climate problem analysis.[1]
	* only single nino index (NINO3.4) prediction, no spatio information (pattern) considered, and both network model and data input size can be improved.[2]
	* Todo***Ref:***
1. 《Prediction of Sea Surface Temperature using Long Short-Term Memory》

	* Single SST prediction (Bohai SST)

	* two aspects -- the structure of the model (different layers, memory cells in each layer) && machine learning algorithms (SVR, MLPR)

	* need more training data

2. 《El niño-southern oscillation forecasting using complex networks analysis of LSTM neural networks》 ***（highly related）***

	* Use climate network to construct input and LSTM network as model to predict ENSO, but just priedict single NINO3.4 index, no spatial information discussed, for the experiment, forecast the  NINO3.4 index from 6-month lead, 9-month lead and 12-month lead, use RMSE and MAE, no contrast experiment. the conclusion said that more data and more complex LSTM neural network is needed. 

	* Nino3.4 + Climate Network (Preprocessing)  --> predict Nino34 with different time leads, and no contrast experiment done.

	* *"We believe this approach has great potential performance skills to augment the ENSO forecasting activity of climate scientists and meteorologists. And our vision is to improve the model by increasing the data sample size and the complexity of the LSTM architecture."*

	* *"To the best of our knowledge, this is the first time this approach has been applied to forecast ENSO phenomenon."*

	* **pyunicorn** --> which is used for construct Climate Network in this paper.

3.  NOAA related blogs
	
	* [How Good Have ENSO Forecasts Been Lately?](https://www.climate.gov/news-features/blogs/enso/how-good-have-enso-forecasts-been-lately) (a good reference when inroduce the prediction situation of ENSO )

		* *"how well the forecasts have matched reality"*
		* lead time --> *" The month from which forecasts are made is often called the start month, and the season that the forecast is for is often called the target season."*
		* measurements (MAE && RMSE && Correlation Coefficient)
		* *"we have a long way to go in improving their performance and utility beyond that.  It is especially hard to predict the timing of ENSO transitions and the correct strength. "*4. 《Deep learning》

5. 《Prediction of Sea Surface Temperature by Combining Numerical and Neural Techniques》

6. 《Using Network Theory and Machine Learning to predict El Niño》

7. 《SEASONAL PREDICTION USING UNSUPERVISED FEATURE LEARNING AND REGRESSION》

	* *"CNNs are particularly suitable for gridded data exploiting the spatial correlations"*
	* LSTM / CNN+ConvLSTM / Sequence LSTM
	* *"reoccuring spatial patterns" (ENSO)*
	* Both in single grid cell (get one grid on the whole region) and grid spatial pattern capture (predict a single img-like grid pattern)
	* *"All the used network designs outperform the regression."*
8. 《Statistical Prediction of ENSO from Subsurface Sea Temperature Using a Nonlinear Dimensionality Reduction》

9. 《Deep learning for predicting the monsoon over the homogeneous regions of India》

10. 《ClimateLearn - A machine-learning approach for climate prediction using network measures》

11. 《Bayesian Recurrent Neural Network Models for Forecasting and Quantifying Uncertainty in Spatial-Temporal Data》

12. 《2018自然基金申请书》

13. 《Unsupervised Discovery of El Nino˜Using Causal Feature Learning on Microlevel Climate Data》


14. 《Sequence to Sequence Weather Forecasting with Long Short-Term Memory Recurrent Neural Networks》

### 2	Preliminaries 
#### 2.1	Formulation of ENSO Prediction Problem
***Outline:***1.	Goal of ENSO prediction (NINO index and simulation pattern)2.	From the perspective of machine learning, this is a spatiotemporal regression prediction problem***Ref:***
1. 《Convolutional LSTM Network - A Machine Learning Approach for Precipitation Nowcasting》#### 2.2	Special Neural Networks for Sequence Modeling

***Outline:***
1. LSTM and CNN-LSTM network for spatiotemporal prediction, which have show great effectiveness on spatiotemporal problems  2. Structure of CNN and RNN, and combine them together, view as a single variable prediction and a sequence generation problem***Ref:***
1. 《Deep Learning for Precipitation Nowcasting - A Benchmark and A New Model》### 3	Model
#### 3.1	Multi-layer LSTM Blocks

***Outline:***the structure of multi LSTM for single variable prediction***Ref:***#### 3.2	CNN–LSTM Spatiotemporal Model

***Outline:***
The structure and detail description of the network (how to capture spatio and temporal structure independently)***Ref:***### 4 Experiment and Results
#### 4.1	Single Niño Index Prediction
***Outline:***1.	Connect those single nino index experiment together

	* the single nino index predition with different LSTM networks
	* different LSTM cell && different LSTM layers
	* different training methods 
2.	Compare the result with different statistical approaches. (todo)

	* Baseline prediction ( naive method )
	* ARIMA model
	* Auto regression model***Ref:***
1. Final Report for “Machine learning applications in oceanography” ( on [GitHub](https://github.com/Yongyao/enso-forcasting) )

	*  Four other index ( SOI, PNA, Nino3, precipitation ) --> ONI
	*  LSTM model, random forest, linear regression

2. [Nino3 / 4 / 3.4 index Data (1870~2018)](https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Nino34/) （ Raw Data ）

3. [Time Series Forecasting with the Long Short-Term Memory Network in Python](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/)
	* Raw data --> Difference (stationary) --> Transfer Time series to Supervised learning problem (pairs) --> Scale [-1~1] --> Model --> Output

4. [Multi-step Time Series Forecasting with Long Short-Term Memory Networks in Python](https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/) ( how to make multistep forecasting )
	* *"The LSTM is stateful; this means that we have to manually reset the state of the network at the end of each training epoch. The network will be fit for 1500 epochs."*

5. [Multivariate Time Series Forecasting with LSTMs in Keras](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/) ( how to make multivariate forecasting )
	* figure out related ( seem related ) raw data
	* *"multvariate time series forecasting with multiple lag inputs"*


6. [NINO SST INDICES (NINO 1+2, 3, 3.4, 4; ONI AND TNI)](https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni) ( different SST anomalies for different regions)
	* *"The Niño 3.4 index and the Oceanic Niño Index (ONI) are the most commonly used indices to define El Niño and La Niña events."*

7. [enso-forecast](https://github.com/lohancock/enso-forecast) ( GitHub, based on R) (todo)



#### 4.2	Grid Spatiotemporal SST Region Prediction

***Outline:***1.	Different models for grid experiment (todo)***Ref:***
### 5 Conclusion and Future WorkTodo### Writing Skill:
1. 《芝加哥大学论文写作指南》

2. 《The-Science-of-Scientific-Writing》