from scipy.optimize import minimize
import numpy as np
from ARMA import run_arma
from read_table import *

class stocks():
	def __init__(self, M, B, S, table, start, n):
		"""
		Params:
		B: 佣金，list
		M: 总资金（持有金额）
		T: 印花说，float
		O: 可负担的总费用
		S：每支股票买入时的股价，list
		P: 每支股票预测收益率
		table: 10支股票的数据表，dist
		X: 每个股票的投入资金(包括印花税和佣金额)
		Xi/Si: 每个股票的持股数
		m: 用于购买股票的总金额
		"""
		self.n = n
		self.start = start
		self.M = M
		self.T = 0.0001
		self.B = np.array(B)+self.T
		self.S = np.array(S)
		self.table = table 

	def find_P(self):
		# Here we got a possible dividend ratio
		table = read_table()
		price = pd.DataFrame(columns=list(range(1, 11)))
		for i in range(1, 11):
			print(i)
			data = table[str(i)]
			data.index = pd.to_datetime(data.date)
			time_series = pd.Series(data.close)
			time_series.index = pd.Index(data.index)
			fc = run_arma(time_series, self.start, self.n).forecast
			fc = pd.DataFrame(fc)
			fc.index.name = 'date'
			price.iloc[:, i-1] = -(pd.Series(np.log(fc.forecast.shift(-1))-np.log(fc.forecast)*100).dropna(axis=0, how=any))
		self.P = price

	def find_w(self):
		# generate a table that contains mean and standard deviation of : 
		# high-low, possible dividend ratio, (open-close)/open
		graph = []
		for i in range(1, 11):
		    high_low_diff = self.table[str(i)].high-self.table[str(i)].low
		    oc = (self.table[str(i)].open-self.table[str(i)].close)/self.table[str(i)].open
		    graph.append([high_low_diff.std(), high_low_diff.mean(), oc.mean(), oc.std(), self.P[i].std(), self.P[i].mean()])
		graph = pd.DataFrame(graph, columns=["diff_mean", "diff_std", "oc_ratio_mean", "oc_ratio_std", "exp_std", "exp_mean"])

		# Because the columns are different in scale
		# So we use the standard scaler from sklearn to scale the dataset
		from sklearn.preprocessing import StandardScaler
		scaler = StandardScaler()
		graph = scaler.fit_transform(graph).round(4)
		graph = pd.DataFrame(graph, columns=["diff_mean", "diff_std", "oc_ratio_mean", "oc_ratio_std", "exp_std", "exp_mean"])

		# We cut off the open_close_ratio with comparative large changes and negative expectation
		# then we got the shoud-purchase stocks table
		g1 = graph[graph.exp_mean>=0]
		self.purchased = g1
		if g1.shape[0] == 0:
			print("You don't need to purchase now...")
		# We calculate the weights of different should-purchase stock here
		# Sum the features in proportion diff_std:or_ratio_mean:exp_std = 1:1:4 to calculate the weights 
		# that allocate the real purchase price
		w = []
		for i in g1.index:
			# w1 = g1.diff_std[i]
			w2 = g1.oc_ratio_mean[i]
			w3 = g1.exp_mean[i]*4
			print(w2,w3)
			w.append(w2+w3)
		w = np.array(w)
		w = w/w.sum().round(4)

		W = np.zeros(10)
		for i,j in zip(g1.index, range(g1.index.shape[0])):
			W[i] = w[j]
		self.W = W

	def find_params(self):
		# Here we generate a table params, which can use these data convenientlt
		dd  = []
		for w, s, p, b in zip(self.W, self.S, self.P, self.B):
			dd.append([w,s,p,b])
		dd = pd.DataFrame(data=dd, index=list(range(1, 11)), columns=["W", "S", "P", "B"])
		self.params = dd

	def find_m(self):
		# Here we calculate the real amount for purchasing the stocks
		self.m = self.M/(self.params["W"]*(1+self.params["B"])).sum()
		self.params["X"] = self.m*self.params["W"] 

	def get_dividend(self):
		# Here we calculate the possible dividend that the client can get
		dividend = (self.params["X"]*self.params["P"]).sum()-(self.params['X']*(self.params['P']+1)*self.params["B"]).sum()
		return dividend

	def summary(self):
		# We got a report for purchasing stocks
		dividend = (self.params["X"]*self.params["P"]).sum()-(self.params['X']*(self.params['P']+1)*self.params["B"]).sum()
		print("您购买的股票为：{}".format(list(self.purchased.index)))
		print("________________________________________________________")
		print("您所持有的股票份额:")
		for i in range(1, 11):
			if self.params.X[i]!=0:
				print("股票: {0}, 份额: {1}, 金额: {2}".format(i, self.params.X[i]/self.params.S[i], self.params.X[i]))
		print("总投入: {0}".format((self.params.X/self.params.S).sum()))
		print("________________________________________________________")
		print("预计每支收入:")
		for i in range(1, 11):
			if self.params.X[i]!=0:
				print("股票: {0}, 收入: {1}".format(i, self.params.X[i]*self.params.P[i]))
		print("预计总收入: {}".format((self.params.X*self.params.P).sum()))
		print("预计净收入: {}".format(dividend))
		print("________________________________________________________")

		
