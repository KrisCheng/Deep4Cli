import numpy as np
import pandas as pd
from matplotlib import pyplot
result=[]
with open('200_epochs.txt','r') as f:
	for line in f:
		result.append(list(line.split(',')))
result = result[0]
while ' ' in result:
    result.remove(' ')
for i in range(len(result)):
    result[i] = float(result[i])
pyplot.plot(result)
pyplot.show()
# print(result)
