import pandas as pd
import numpy as np
np.random.seed(42)

 
# Matplotlib and seaborn for plotting
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['figure.figsize'] = (9, 9)

import seaborn as sns

from IPython.core.pylabtools import figsize

# Scipy helper functions
from scipy.stats import percentileofscore
from scipy import stats
# Standard ML Models for comparison
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

# Splitting data into training/testing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

# Distributions
import scipy

# PyMC3 for Bayesian Inference
import pymc3 as pm

df = pd.read_csv('../../data/oni/csv/all.csv')

# print(df.corr()['Nino3_4'].sort_values())
# print(df.describe())


# # Bar plot of grades
# plt.bar(df['Nino3_4'].value_counts().index, 
#         df['Nino3_4'].value_counts().values,
#          fill = 'navy', edgecolor = 'k', width = 1)
# plt.xlabel('Nino3_4'); plt.ylabel('Count'); plt.title('Distribution of Final Grades');
# plt.xticks(list(range(-3, 3)));

def format_data(df):
    # Targets are final grade of student
    labels = df['Nino3_4']
    
    # Find correlations with the Grade
    most_correlated = df.corr().abs()['Nino3_4'].sort_values(ascending=False)
    
    # Maintain the top 6 most correlation features with Grade
    most_correlated = most_correlated[:8]
    
    df = df.ix[:, most_correlated.index]
    
    # Split into training/testing sets with 25% split
    X_train, X_test, y_train, y_test = train_test_split(df, labels, 
                                                        test_size = 0.2,
                                                        random_state=42)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = format_data(df)
print(X_train.shape)
print(X_test.shape)

# Calculate correlation coefficient
def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .6), xycoords=ax.transAxes,
               size = 12)
cmap = sns.cubehelix_palette(light=1, dark = 0.1,
                             hue = 0.5, as_cmap=True)

sns.set_context(font_scale=2)
# Pair grid set up
g = sns.PairGrid(X_train)
# Scatter plot on the upper triangle
g.map_upper(plt.scatter, s=10, color = 'red')
# Distribution on the diagonal
g.map_diag(sns.distplot, kde=False, color = 'red')
# Density Plot and Correlation coefficients on the lower triangle
g.map_lower(sns.kdeplot, cmap = cmap)
g.map_lower(corrfunc);
# plt.show()

# Create relation to the median grade column
X_plot = X_train.copy()
X_plot['relation_median'] = (X_plot['Nino3_4'] >= 0)
X_plot['relation_median'] = X_plot['relation_median'].replace({True: 'above', False: 'below'})
X_plot = X_plot.drop(columns='Nino3_4')

plt.figure(figsize=(12, 12))
# Plot the distribution of each variable colored
# by the relation to the median grade
for i, col in enumerate(X_plot.columns[:-1]):
    plt.subplot(4, 2, i + 1)
    subset_above = X_plot[X_plot['relation_median'] == 'above']
    subset_below = X_plot[X_plot['relation_median'] == 'below']
    sns.kdeplot(subset_above[col], label = 'Above Median', color = 'green')
    sns.kdeplot(subset_below[col], label = 'Below Median', color = 'red')
    plt.legend(); plt.title('Distribution of %s' % col)
    
plt.tight_layout()

# # Calculate mae and rmse
# def evaluate_predictions(predictions, true):
#     mae = np.mean(abs(predictions - true))
#     rmse = np.sqrt(np.mean((predictions - true) ** 2))
#     return mae, rmse

# # Naive baseline is the median
# median_pred = X_train['Nino3_4'].median()
# median_preds = [median_pred for _ in range(len(X_test))]
# true = X_test['Nino3_4']

# def evaluate(X_train, X_test, y_train, y_test):
#     # Names of models
#     model_name_list = ['LR', 'ER',
#                       'RF', 'ET', 'SVM',
#                        'GB', 'Baseline']
#     X_train = X_train.drop(columns='Nino3_4')
#     X_test = X_test.drop(columns='Nino3_4')
    
#     # Instantiate the models
#     model1 = LinearRegression()
#     model2 = ElasticNet(alpha=1.0, l1_ratio=0.5)
#     model3 = RandomForestRegressor(n_estimators=50)
#     model4 = ExtraTreesRegressor(n_estimators=50)
#     model5 = SVR(kernel='rbf', degree=3, C=1.0, gamma='auto')
#     model6 = GradientBoostingRegressor(n_estimators=20)
    
#     # Dataframe for results
#     results = pd.DataFrame(columns=['mae', 'rmse'], index = model_name_list)
    
#     # Train and predict with each model
#     for i, model in enumerate([model1, model2, model3, model4, model5, model6]):
#         model.fit(X_train, y_train)
#         predictions = model.predict(X_test)
        
#         # Metrics
#         mae = np.mean(abs(predictions - y_test))
#         rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        
#         # Insert results into the dataframe
#         model_name = model_name_list[i]
#         results.ix[model_name, :] = [mae, rmse]
    
#     # Median Value Baseline Metrics
#     baseline = np.median(y_train)
#     baseline_mae = np.mean(abs(baseline - y_test))
#     baseline_rmse = np.sqrt(np.mean((baseline - y_test) ** 2))
    
#     results.ix['Baseline', :] = [baseline_mae, baseline_rmse]
    
#     return results


# # Display the naive baseline metrics
# mb_mae, mb_rmse = evaluate_predictions(median_preds, true)
# print('Median Baseline  MAE: {:.4f}'.format(mb_mae))
# print('Median Baseline RMSE: {:.4f}'.format(mb_rmse))

# results = evaluate(X_train, X_test, y_train, y_test)
# figsize(12, 8)
# matplotlib.rcParams['font.size'] = 16
# # Root mean squared error
# ax =  plt.subplot(1, 2, 1)
# results.sort_values('mae', ascending = True).plot.bar(y = 'mae', color = 'b', ax = ax)
# plt.title('Model MAE'); plt.ylabel('MAE');

# # Median absolute percentage error
# ax = plt.subplot(1, 2, 2)
# results.sort_values('rmse', ascending = True).plot.bar(y = 'rmse', color = 'r', ax = ax)
# plt.title('Model RMSE'); plt.ylabel('RMSE');

# # plt.tight_layout()
# # plt.show()

# lr = LinearRegression()
# lr.fit(X_train.drop(columns='Nino3_4'), y_train)

# ols_formula = 'Nino3_4 = %0.2f +' % lr.intercept_
# for i, col in enumerate(X_train.columns[1:]):
#     ols_formula += ' %0.2f * %s +' % (lr.coef_[i], col)
    
# print(' '.join(ols_formula.split(' ')[:-1]))

# formula = 'Nino3_4 ~ ' + ' + '.join(['%s' % variable for variable in X_train.columns[1:]])
# print(formula)

# # Context for the model
# with pm.Model() as normal_model:    
#     # The prior for the model parameters will be a normal distribution
#     family = pm.glm.families.Normal()
#     # Creating the model requires a formula and data (and optionally a family)
#     pm.GLM.from_formula(formula, data = X_train, family = family)
#     # Perform Markov Chain Monte Carlo sampling
#     normal_trace = pm.sample(draws=2000, chains = 10, tune = 500, njobs=-1)

# # Shows the trace with a vertical line at the mean of the trace
# def plot_trace(trace):
#     # Traceplot with vertical lines at the mean value
#     ax = pm.traceplot(trace, figsize=(14, len(trace.varnames)*1.8),
#                       lines={k: v['mean'] for k, v in pm.summary(trace).iterrows()})
#     matplotlib.rcParams['font.size'] = 16
#     # Labels with the median value
#     for i, mn in enumerate(pm.summary(trace)['mean']):
#         ax[i, 0].annotate('{:0.2f}'.format(mn), xy = (mn, 0), xycoords = 'data', size = 10,
#                           xytext = (-18, 18), textcoords = 'offset points', rotation = 90,
#                           va = 'bottom', fontsize = 'large', color = 'red')

# plot_trace(normal_trace)
# pm.forestplot(normal_trace)
# pm.plot_posterior(normal_trace, figsize = (14, 14), text_size=20);
# plt.show()
# print("==.==")
# print(pm.summary(normal_trace))
# # def evaluate_trace(trace, X_train, X_test, y_train, y_test, model_results):
    
# #     # Dictionary of all sampled values for each parameter
# #     var_dict = {}
# #     for variable in trace.varnames:
# #         var_dict[variable] = trace[variable]
        
# #     # Results into a dataframe
# #     var_weights = pd.DataFrame(var_dict)
    
# #     # Means for all the weights
# #     var_means = var_weights.mean(axis=0)
    
# #     # Create an intercept column
# #     X_test['Intercept'] = 1
    
# #     # Align names of the test observations and means
# #     names = X_test.columns[1:]
# #     X_test = X_test.ix[:, names]
# #     var_means = var_means[names]
    
# #     # Calculate estimate for each test observation using the average weights
# #     results = pd.DataFrame(index = X_test.index, columns = ['estimate'])

# #     for row in X_test.iterrows():
# #         results.ix[row[0], 'estimate'] = np.dot(np.array(var_means), np.array(row[1]))
        
# #     # Metrics 
# #     actual = np.array(y_test)
# #     errors = results['estimate'] - actual
# #     mae = np.mean(abs(errors))
# #     rmse = np.sqrt(np.mean(errors ** 2))
    
# #     print('Model  MAE: {:.4f}\nModel RMSE: {:.4f}'.format(mae, rmse))
    
# #     # Add the results to the comparison dataframe
# #     model_results.ix['BLR', :] = [mae, rmse]
    
# #     plt.figure(figsize=(12, 8))
    
# #     # Plot median absolute percentage error of all models
# #     ax = plt.subplot(1, 2, 1)
# #     model_results.sort_values('mae', ascending = True).plot.bar(y = 'mae', color = 'r', ax = ax)
# #     plt.title('Model MAE Comparison'); plt.ylabel('MAE'); 
# #     plt.tight_layout()
    
# #     # Plot root mean squared error of all models
# #     ax = plt.subplot(1, 2, 2)
# #     model_results.sort_values('rmse', ascending = True).plot.bar(y = 'rmse', color = 'b', ax = ax)
# #     plt.title('Model RMSE Comparison'); plt.ylabel('RMSE')
    
# #     return model_results

# # all_model_results = evaluate_trace(normal_trace, X_train, X_test, y_train, y_test, results)
# # print(results)
# # plt.show()

# # Examines the effect of changing a single variable
# # Takes in the name of the variable, the trace, and the data
# def model_effect(query_var, trace, X):
    
#     # Variables that do not change
#     steady_vars = list(X.columns)
#     steady_vars.remove(query_var)
    
#     # Linear Model that estimates a grade based on the value of the query variable 
#     # and one sample from the trace
#     def lm(value, sample):
        
#         # Prediction is the estimate given a value of the query variable
#         prediction = sample['Intercept'] + sample[query_var] * value
        
#         # Each non-query variable is assumed to be at the median value
#         for var in steady_vars:
            
#             # Multiply the weight by the median value of the variable
#             prediction += sample[var] * X[var].median()
        
#         return prediction
    
#     figsize(6, 6)
    
#     # Find the minimum and maximum values for the range of the query var
#     var_min = X[query_var].min()
#     var_max = X[query_var].max()
    
#     # Plot the estimated grade versus the range of query variable
#     pm.plot_posterior_predictive_glm(trace, eval=np.linspace(var_min, var_max, 100), 
#                                      lm=lm, samples=100, color='blue', 
#                                      alpha = 0.4, lw = 2)
    
#     # Plot formatting
#     plt.xlabel('%s' % query_var, size = 16)
#     plt.ylabel('Nino3_4', size = 16)
#     plt.title("Posterior of Nino3_4 vs %s" % query_var, size = 18)
#     plt.show()

# model_effect('Nino3', normal_trace, X_train.drop(columns='Nino3_4'))
# model_effect('Nino4', normal_trace, X_train.drop(columns='Nino3_4'))
# model_effect('TPI', normal_trace, X_train.drop(columns='Nino3_4'))
# model_effect('AMO', normal_trace, X_train.drop(columns='Nino3_4'))
# model_effect('DMI', normal_trace, X_train.drop(columns='Nino3_4'))
# model_effect('Nino1_2', normal_trace, X_train.drop(columns='Nino3_4'))
# model_effect('SOI', normal_trace, X_train.drop(columns='Nino3_4'))
plt.show()