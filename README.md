# Factor-model
## replicate the results from anomalies paper
### back test input:

Please input the serial number of signal:
210 

Please enter the start time YYYYMMDD:
19200101

Please enter the end time YYYYMMDD
20200101

Please input signal lag (1 month is zero):
1

Please input quantile to divide stocks (For example, decile is 10):
10

Please input the exchanges: [all/ NYSE/ NYSE MKT/ NASDAQ/ NYSE Arca]
all

Please input the weight scheme: [value/equal]
equal

do you want to drop penny stocks?[y/n]
n

Please input holding period .You can input many number (For example: 1 2 means 1 month and 2 month):
1

Please input the formation period:
1


### back test output:

3214351 observations in original dataset

0.0000% has been drop when quantilize stocks

0.0% has been drop when calculate weight

Long top and short botton portfolio analysis:
return analysis:
holding period                     1
observations             1101.000000
mean                       -0.004727
std                         0.057136
min                        -0.571195
skewness                   -0.883687
kurtosis                   11.816876
t value                    -2.744976
p value                     0.006150
annualized sharpe ratio    -0.286573
max drawdown               -0.999257


Net value of Long top short bottom portfolio:

