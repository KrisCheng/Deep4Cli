
# get NWP and NCT index
import numpy, pandas
from pandas import Series
from pandas import DataFrame

# get nwt and nct
series3 = Series.from_csv('../data/oni/csv/nino3_anomaly.csv')
series4 = Series.from_csv('../data/oni/csv/nino4_anomaly.csv')
nwpList = []
nctList = []
timeList = []
beginYear = 1870
beginMonth = 1
for (nino3, nino4) in zip(series3, series4):
    if(nino3 * nino4 > 0):
        nwp = nino4 - 0.4 * nino3
        nct = nino3 - 0.4 * nino4
    else:
        nwp = nino4
        nct = nino3
    nctList.append(nct)
    nwpList.append(nwp)
    if beginMonth < 10:
        time = str(str(beginYear)+'-'+str('0'+str(beginMonth)))
    else: 
        time = str(str(beginYear)+'-'+str(beginMonth))
    timeList.append(time)
    beginMonth = beginMonth + 1
    if(beginMonth is 13):
        beginMonth = 1
        beginYear = beginYear + 1
# print(nctList)
dataframe = pandas.DataFrame({'TIME':timeList,'NWP':nwpList})
dataframe.to_csv("nwp.csv",index=False,sep=',')

