### Functions that automate the creation of linear regression models and extracting their associated values

import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from scipy.stats import stats
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy import stats

import warnings
warnings.filterwarnings("ignore")



def onetime_ols(X_train, y_train):
    # Create x constants
    Xconst = sm.add_constant(X_train)

    # Create OLS model and summary
    ols_model = sm.OLS(y_train, Xconst, hasconst= True)
    model = ols_model.fit()
    results = model.summary()
    
    return model, results



def ols_loop(X_train, y_train, run):

    # Create x constants
    Xconst = sm.add_constant(X_train)

    # Create OLS model and summary
    ols_model = sm.OLS(y_train, Xconst, hasconst= True)
    est = ols_model.fit()
    results = est.summary()

    # Capture r2_adj 
    r2_adj = est.rsquared_adj

    # Load summary info into dataframe for processing
    results_as_html = results.tables[1].as_html()
    results_df = pd.read_html(results_as_html, header=0, index_col=0)[0]
    results_df = results_df.reset_index()
    results_df = results_df.rename(columns={'index':'feature'})
    results_df = results_df[1:]

    # Identify highest p-value in data set
    kill_cols = results_df[results_df['P>|t|'] == results_df['P>|t|'].max()]
    highest_p_col = kill_cols['feature'].values
    highest_p = kill_cols['P>|t|'].values

    # Remove feature with highest p-value
    X_chop = X_train.drop(columns=highest_p_col)

    # Capture values from test in dictionary
    run_dict = {'test_run': run, \
            'r2_adj': r2_adj, \
            'highest_feature': highest_p_col,\
            'highest_pval': highest_p[0]}
    
    return X_chop, run_dict   



    def get_pvals(X_train, y_train):
	    cols = np.array(X_train.columns.values)
	    cols = np.insert(cols, 0, 'coef', axis=0)

	    lm = LinearRegression()
	    lm.fit(X_train, y_train)
	    params = np.append(lm.intercept_,lm.coef_)
	    predictions = lm.predict(X_train)

	    newX = np.append(np.ones((len(X_train),1)), X_train, axis=1)
	    MSE = (sum((y_train - predictions)**2)) / (len(newX) - len(newX[0]))

	    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
	    sd_b = np.sqrt(var_b)
	    ts_b = params/ sd_b

	    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

	    sd_b = np.round(sd_b,3)
	    ts_b = np.round(ts_b,3)
	    p_values = np.round(p_values,3)
	    params = np.round(params,4)

	    pvals = pd.DataFrame()
	    pvals['feature'], pvals['coef'], pvals['standard_error'], pvals['t_vals'], pvals['pval'] = \
	        [cols, params,sd_b,ts_b,p_values]
	    
	    return pvals