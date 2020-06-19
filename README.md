# Factor-model
## replicate the results from anomalies paper

This program need dataset from CRSP. 

After you have factors, naming these factors file as __Formation period-Lag-Holding Period.csv__(such as _1-0-1.csv_). Then run this file.  

You can specify:
* 1. Back test time period  
* 2. Lag between formation period and holding period
* 3. Exchange name
* 4. Equal weighted or Market value weighted
* 5. Whether need to drop panny stocks
* 6. Holding Period


### back test input:
<pre>
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
</pre>

### back test output:
<pre>
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

</pre>
![ls port](https://user-images.githubusercontent.com/61814814/84616812-9b159d80-aeff-11ea-83b3-8aaa6bfaa79e.png)
<pre>
Quantile portfolio analysis:
return analysis:
holding period                      1                              \
factor_quantile                  1.0           2.0           3.0    
observations             1.101000e+03  1.101000e+03  1.101000e+03   
mean                     1.263773e-02  1.390053e-02  1.323081e-02   
std                      5.991054e-02  6.881101e-02  6.774948e-02   
min                     -2.104137e-01 -2.420875e-01 -2.890893e-01   
skewness                 5.387399e+00  4.566029e+00  2.393709e+00   
kurtosis                 7.093862e+01  5.528368e+01  2.438665e+01   
t value                  6.999377e+00  6.702969e+00  6.479988e+00   
p value                  4.462506e-12  3.256188e-11  1.381302e-10   
annualized sharpe ratio  7.307290e-01  6.997842e-01  6.765052e-01   
max drawdown            -7.079925e-01 -8.035419e-01 -8.437447e-01   

holding period                                                     \
factor_quantile                  4.0           5.0           6.0    
observations             1.101000e+03  1.101000e+03  1.101000e+03   
mean                     1.362519e-02  1.333806e-02  1.339635e-02   
std                      7.068294e-02  7.502677e-02  7.772728e-02   
min                     -3.034585e-01 -3.139332e-01 -3.305338e-01   
skewness                 1.900267e+00  1.849884e+00  1.643324e+00   
kurtosis                 1.773545e+01  1.801111e+01  1.593894e+01   
t value                  6.396195e+00  5.898886e+00  5.718826e+00   
p value                  2.351108e-10  4.865995e-09  1.381210e-08   
annualized sharpe ratio  6.677573e-01  6.158387e-01  5.970405e-01   
max drawdown            -8.225115e-01 -8.610854e-01 -8.641918e-01   

holding period                                                                
factor_quantile                  7.0          8.0          9.0          10.0  
observations             1.101000e+03  1101.000000  1101.000000  1101.000000  
mean                     1.271751e-02     0.011989     0.011050     0.007911  
std                      7.928806e-02     0.081680     0.083797     0.086003  
min                     -3.257314e-01    -0.332185    -0.348755    -0.365572  
skewness                 1.155028e+00     0.691420     0.331907     0.007153  
kurtosis                 1.072725e+01     7.552704     5.078673     3.068001  
t value                  5.322161e+00     4.870359     4.375474     3.052231  
p value                  1.242236e-07     0.000001     0.000013     0.002326  
annualized sharpe ratio  5.556291e-01     0.508461     0.456796     0.318650  
max drawdown            -8.335607e-01    -0.909737    -0.900045    -0.943055  

Net value of different quantile:
</pre>
![quan](https://user-images.githubusercontent.com/61814814/84619425-9523ba80-af07-11ea-9bc5-6cccfaadde1e.png)
