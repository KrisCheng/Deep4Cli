# Using Deep Neural Networks to Study ENSO Case

***Manuscript Version***Bin Mu, Cheng Peng and Shijin Yuan(✉) 
School of Software Engineering, Tongji University, Shanghai 201804, China
binmu@tongji.edu.cn,  tjupengcheng@163.com, yuanshijin2003@163.com### Abstract.

The abstract should summarize the contents of the paper in short terms, i.e. 150-250 words.
### Keywords: 

time series prediction, deep neural networks, ENSO.
### Writing Skill:
* 《芝加哥大学论文写作指南》

	* 	 
* 《The-Science-of-Scientific-Writing》

### 1	IntroductionTodo1.1 the reason (significance) of study ENSO, traditional methods (statistical method && regular machine learning methods) and limitations.

*"The dynamical models use physical equations of the ocean and atmosphere to forecast ENSO event, which are computationally very expensive and not available outside the at- mospheric scientific community."*

1.2	 deep learning method’s success on data-driven problem

"capturing long-term temporal dependencies" -- LSTM


1.3	 ENSO SST prediction is a suitable (reasonable) candidate for deep learning（machine learning） method*Ref:*
* 《Prediction of Sea Surface Temperature using Long Short-Term Memory》



* 《El niño-southern oscillation forecasting using complex networks analysis of LSTM neural networks》 ***（highly related）***

	* Use climate network to construct input and LSTM network as model to predict ENSO, but just priedict single NINO3.4 index, no spatial information discussed, for the experiment, forecast the  NINO3.4 index from 6-month lead, 9-month lead and 12-month lead, use RMSE and MAE, no contrast experiment. the conclusion said that more data and more complex LSTM neural network is needed. 

	* **pyunicorn** --> which is used for construct Climate Network in this paper.

	* Nino34 + Climate Network (Preprocessing)  --> predict Nino34 with different time leads
*  NOAA related blogs
	
	* [How Good Have ENSO Forecasts Been Lately?](https://www.climate.gov/news-features/blogs/enso/how-good-have-enso-forecasts-been-lately) ( a good reference when inroduce the prediction situation of ENSO )

		* *"how well the forecasts have matched reality"*
		* lead time --> *" The month from which forecasts are made is often called the start month, and the season that the forecast is for is often called the target season."*
		* measurements (MAE && Correlation Coefficient)
		* *"we have a long way to go in improving their performance and utility beyond that.  It is especially hard to predict the timing of ENSO transitions and the correct strength. "** Deep learning

* Prediction of Sea Surface Temperature by Combining Numerical and Neural Techniques

* Using Network Theory and Machine Learning to predict El Niño

* SEASONAL PREDICTION USING UNSUPERVISED FEATURE LEARNING AND REGRESSION

* Statistical Prediction of ENSO from Subsurface Sea Temperature Using a Nonlinear Dimensionality Reduction

* Deep learning for predicting the monsoon over the homogeneous regions of India

* ClimateLearn- A machine-learning approach for climate prediction using network measures

* Bayesian Recurrent Neural Network Models for Forecasting and Quantifying Uncertainty in Spatial-Temporal Data

* 《2018自然基金申请书》

* Unsupervised Discovery of El Nino˜Using Causal Feature Learning on Microlevel Climate Data


* Sequence to Sequence Weather Forecasting with Long Short-Term Memory Recurrent Neural Networks


### 2	Preliminaries 
#### 2.1	Formulation of ENSO Prediction Problem
Todo1.	Goal of ENSO prediction (NINO index and simulation pattern)2.	From the perspective of machine learning， a spatiotemporal regression prediction problem*Ref:*
* Convolutional LSTM Network- A Machine Learning Approach for Precipitation Nowcasting

* #### 2.2	Special Neural Networks for Sequence Modeling
LSTM and CNN-LSTM network for spatiotemporal prediction, which have show great effectiveness on spatiotemporal problems  1. Structure of CNN and RNN, and combine them together, view as a single variable prediction and a sequence generation problem*Ref:*
* Deep Learning for Precipitation Nowcasting - A Benchmark and A New Model### 3	Model
#### 3.1	Multi-layer LSTM Blocks
Todothe structure of multi LSTM for single variable prediction*Ref:*
* #### 3.2	CNN –LSTM Spatiotemporal Model
The structure and detail description of the network (how to capture spatio and temporal structure independently)*Ref:*

* ### 4 Experiment and Results
#### 4.1	Single Niño Index Prediction
Todo1.	Connect those single nino index experiment together
2.	Compare with different statistical approaches (todo)*Ref:*
* Final Report for “Machine learning applications in oceanography” (GitHub)

#### 4.2	Grid Spatiotemporal SST Region Prediction
Todo1.	Different models for grid experiment (todo)*Ref:*### 5 Conclusion and Future WorkTodo### References
1.	Author, F.: Article title. Journal 2(5), 99–110 (2016).2.	Author, F., Author, S.: Title of a proceedings paper. In: Editor, F., Editor, S. (eds.) CONFERENCE 2016, LNCS, vol. 9999, pp. 1–13. Springer, Heidelberg (2016). 3.	Author, F., Author, S., Author, T.: Book title. 2nd edn. Publisher, Location (1999).4.	Author, F.: Contribution title. In: 9th International Proceedings on Proceedings, pp. 1–2. Publisher, Location (2010).5.	LNCS Homepage, http://www.springer.com/lncs, last accessed 2016/11/21.