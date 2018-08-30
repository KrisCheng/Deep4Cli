# txt --> csv
import numpy
import pandas as pd

raw = '../data/oni/txt/nino1+2.txt'  
raw_data = numpy.loadtxt(raw)
raw_data = numpy.delete(raw_data, 0, 1)
raw_values = raw_data.reshape(12 * 148)
raw_values = raw_values.tolist()
timeList = []
beginYear = 1870
beginMonth = 1
for i in raw_values:
    if beginMonth < 10:
        time = str(str(beginYear)+'-'+str('0'+str(beginMonth)))
    else: 
        time = str(str(beginYear)+'-'+str(beginMonth))
    timeList.append(time)
    beginMonth = beginMonth + 1
    if(beginMonth is 13):
        beginMonth = 1
        beginYear = beginYear + 1
    
dataframe = pd.DataFrame({'Date':timeList,'Temp':raw_values})

dataframe.to_csv("nino1+2.csv",index=False,sep=',')