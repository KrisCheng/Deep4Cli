from pandas import Series
from matplotlib import pyplot
from pandas import read_csv
from pywt import wavedec
import matplotlib.pyplot as plt
import pywt
import numpy

series = read_csv('../data/oni/csv/nino3_4_anomaly.csv', header=0, parse_dates=[0], index_col=0,
    squeeze=True)

X = series.values

(cA, cD) = pywt.dwt(X, 'db1')
print(len(X))
print(len(cA))
print(len(cD))

fig = plt.figure(figsize=(16, 8))
fig.suptitle("Monthly Niño3.4 Discrete Wavelet Transform", fontsize=16)
ax = plt.subplot(3,1,1)
plt.plot(series)
ax.set_title("Monthly Niño3.4 Data")
plt.ylabel('°C')

ax = plt.subplot(3,1,2)
ax.set_title("Approximations Reconstructed Sequence")
plt.plot(cA)

ax = plt.subplot(3,1,3)
ax.set_title("Detail Reconstructed Sequence")
plt.plot(cD)

plt.show()