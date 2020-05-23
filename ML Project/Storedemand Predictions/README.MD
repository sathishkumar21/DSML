Predict Store demand forecast
1.0 Objective
The objective of this project is to predict 3 months of item-level sales data at different store locations.
Data fields description
date - Date of the sale data. There are no holiday effects or store closures.
store - Store ID
item - Item ID
sales - Number of items sold at a particular store on a particular date.
2.0 Data Loading and Visualization
Import data set
3.0 Decomposition
4.0 Stationary Check
4.1 Rolling Statistics Methodology
4.2 ADF(Augmented Dickey-Fuller) Test
For a Time series to be stationary, its ADF test should have:
low p-value (according to the null hypothesis)
Critical values at 1%, 5%, 10% confidence intervals should be as close as possible to the Test Statistic
5. Data Transformation to achieve Stationarity¶
We can try to achive stationary using different methods like log scale, moving average,Exponential decay transformation,Time shift transformation.
5.1 Log Scale Transformation
5.2 Log Scale - Moving Average Transformation
5.3 Exponential Decay Transformation
5.4 Time Shift Transformation
6. Plotting ACF & PACF
7. Building Models
7.1 AR MODEL
7.2 ARMA Model
7.3 ARIMA Model
8. Prediction & Reverse transformations of fittedvalues
Forecasting the Store Sales for next 3 Months
We have 3(existing data of 3 months) data points.
To forecast additional 3 month data points or 6 months.
Using plot_predict method
9. Conclusion
Using ARIMA model our forcast and actual almost in line with the predictions
