import numpy as np

# from sklearn.utils import check_array

def mean_absolute_percentage_error(y_true, y_pred): 
    # y_true, y_pred = check_array(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

y_true = np.array([ [1,2,3,4] ])
y_pred = np.array([ [4,3,2,1] ])
print(mean_absolute_percentage_error(y_true, y_pred))

