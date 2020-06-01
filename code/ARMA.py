import pandas as pd
import datetime
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
import seaborn as sns
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats.diagnostic import unitroot_adf
import itertools
import warnings
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf
from read_table import *
from statsmodels.tsa.arima_model import ARMA

def run_arma(time_series, start, n_steps):
	(p, q) = select_order(time_series)
	return build_model(time_series, n_steps, p, q, start)

def select_order(time_series):
	(p, q) =(sm.tsa.arma_order_select_ic(time_series,max_ar=3,max_ma=3,ic='aic')['aic_min_order'])
	if p <= 0:
		p = 1
	if q <= 0:
		q = 1
	return (p, q)

def show_prediction(arma, time_series):
	predict_data = arma.predict(start=time_series.index[0], end=time_series.index[-1], dynamic=False)
	fig = plt.figure(figsize=(60, 18), dpi=100, facecolor="white")
	plt.plot(predict_data)
	plt.plot(time_series)
	plt.xticks(rotation=-90)
	plt.show()

def show_forecast(fc_all):
	fig = plt.figure(figsize=(60, 18), dpi=100, facecolor="white")
	plt.plot(fc_all.forecast)
	plt.xticks(rotation=-90)
	plt.show()

def build_model(time_series, n_steps, p, q, start):
	# arma = sm.tsa.ARMA(time_series,(p,q)).fit()
	# show_prediction(arma, time_series)
	p,d,q = (1,3,1)
	arma = ARMA(time_series,(0,1)).fit(disp=-1,maxiter=100)

	f, err95, ci95 = arma.forecast(steps=n_steps) # 95% CI
	_, err99, ci99 = arma.forecast(steps=n_steps, alpha=0.01) # 99% CI

	idx = pd.date_range(start, periods=n_steps, freq='D')
	fc_95 = pd.DataFrame(np.column_stack([f, ci95]), 
	                     index=idx, columns=['forecast', 'lower_ci_95', 'upper_ci_95'])
	fc_99 = pd.DataFrame(np.column_stack([ci99]), 
	                     index=idx, columns=['lower_ci_99', 'upper_ci_99'])
	fc_all = fc_95.combine_first(fc_99)
	# show_forecast(fc_all)
	return fc_all 

# table = read_table()
# data = table
# data.index = pd.to_datetime(data['date'])
# time_series = pd.Series(data.volume)
# time_series.index = pd.Index(table.date)
# print(run_arma(time_series, "2020-04-01", 21))
