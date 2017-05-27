from matplotlib import pyplot
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA

# Path to csv file
path = '/home/sethu/CS-535- Data/Proj/product_distribution_training_set.txt'
data = pd.read_csv(path,sep='\t',header=None)

# Transpose the dataframe
data = data.T

# Grab the first row for the header
new_header = data.iloc[0]
# Make the new_header as the column header
data.columns = new_header
data = data.drop(data.index[[0]])

# Sum of the products sold at each day
sum = np.sum(data,axis=1).astype(float)
sumarray = np.array(sum)
# Create a time series object with the sumarray(total products sold for each day)
ts = pd.Series(sumarray)
# ts.plot(color = 'yellow')
# pyplot.show()

# Pass the time series object to ARIMA model to train the test data
ts.index = pd.to_datetime(ts.index,unit = 'D')
model = ARIMA(ts, order=(3,0,0))
results_ARIMA = model.fit(disp=0)

# Open a file called output.txt
file = open('output.txt','w+')

 # Predict using forecast meethod of ARIMA
foreCast = results_ARIMA.forecast(steps = 29)[0]
# Write to file first line
w = '0  '
for i in foreCast:
    i = int(round(i))
    w += str(i) + '  '
file.write(w)
file.write('\n\n')

yarr = np.concatenate((sumarray,foreCast))
yarr = yarr.round()

# Plot the graph for sum of products sold over next 29 days

# x = pd.Series(yarr)
# x.plot(color='red')
# pyplot.plot(x="time",y=[ts,x])
# pyplot.show()

head_count = 0
# print new_header[head_count]

for col in data:
    rarray = np.array(data[col]).astype(float)
    # print rarray
    nts = pd.Series(rarray)
    nts.index = pd.to_datetime(nts.index,unit = 'D')
    model = ARIMA(nts, order=(3,0,0))
    results_ARIMA = model.fit(disp=0)
    foreCast = results_ARIMA.forecast(steps = 29)[0]
    s = str(new_header[head_count]) + ' '
    head_count += 1
    for i in foreCast:
        if i < 0:
            i = 0
        i = int(round(i))
        s += str(i) + '  '
    file.write(s)
    file.write('\n\n')
    # Plot the graph for every product sold over next 29 days
    # yarr = np.concatenate((rarray,foreCast))
    # yarr = yarr.round()
    # x = pd.Series(yarr)
    # x.plot(color='red')
    # pyplot.plot(x="time",y=[nts,x])
    # pyplot.show()


