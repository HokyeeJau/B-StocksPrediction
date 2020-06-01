import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
cols = ['date', 'open', 'high', 'low', 'close', 'volume']
def read_table():
	table = {}
	for i in range(1, 11):
		num = lambda x: "0"+str(x) if x < 10 else str(x)
		df = pd.read_csv("full"+num(i)+".csv")
		df.columns = cols
		table[str(i)] = df
	# table = pd.read_csv("abc01.csv", header=None)
	# table.columns = cols
	return table

def fresh_table():
	table = read_table()
	for i in range(1, 11):
		# table[str(i)]['date'] = pd.to_datetime(table[str(i)]['time'], format = "%Y/%m/%d", errors = 'coerce')
		temp = table[str(i)]["time"].apply(lambda x: x.split("/"))
		temp = np.array(temp)
		demp = []
		for j in range(0, temp.shape[0]):
			temp[j][0] = str(int(temp[j][0]))
			temp[j][1] = str(int(temp[j][1]))
			temp[j][2] = str(int(temp[j][2]))
			demp.append("/".join(temp[j]))
		table[str(i)]["time"] = demp
		table[str(i)].to_csv(str(i)+".csv", index=0)
	return table

def get_table():
	table = {}
	for i in range(1, 11):
		df = pd.read_csv(str(i)+".csv")
		df.columns = cols
		table[str(i)] = df
	return table

# print(read_table().columns)