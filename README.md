# project-2-ML_bike_sharing_demand_prediction

In this project, we train a model to predict the number rental bikes at any hour of the given year in different weather condition like Temprature, Humidity, Windspeed, Dew Point Temprature, Radiation, Snowfall, and Rainfall.
First we did EDA (Exploratory Data Analysis) on given data set to get insight of data and after knowing all variable factors and target variable and relation between them we will apply different kind of Regression models. 
We evaluated different models on the basis of evaluation metrics and compared different models are, Linear Regression, Lasso Regression, Ridge REgression, Elastic Net REgression, Desicion Tree Regressor, Random Forest Regressor. 
We also did Hyperparameter Tunning to get the optimal values for to get best accuracy of the model.

## Objective 
The main Objective of our project is to predict future bike sharing demand with best regression model to get the best accuracy. 
## Data Description 
The dataset contains weather information (Temperature, Humidity, Windspeed, Visibility, Dewpoint, Solar radiation, Snowfall, Rainfall), the number of bikes rented per hour and date information.
Attribute Information:

•	Date: year-month-day
•	Rented Bike count - Count of bikes rented at each hour
•	Hour - Hour of the day
•	Temperature-Temperature in Celsius
•	Humidity - %
•	Windspeed - m/s
•	Visibility - 10m
•	Dew point temperature - Celsius
•	Solar radiation - MJ/m2
•	Rainfall - mm
•	Snowfall - cm
•	Seasons - Winter, Spring, Summer, Autumn
•	Holiday - Holiday/No holiday
•	Functional Day - NoFunc(Non Functional Hours), Fun(Functional hours)


## Libraries and Tools 
1. Numpy 
2. Pandas 
3. Scipy 
4. sklearn 
5. Seaborn 
5. Matplotlib 
# Steps Involved
●	Exploratory Data Analysis 
After loading the dataset, we performed this method by comparing our target variable that is Rented Bike Count with other independent variables. This process helped us figuring out various aspects and relationships among the target and the independent variables. It gave us a better idea of which feature behaves in which manner compared to the target variable.

## Null values Treatment
Our dataset contains a no null values so we do not have to concerned about missing values. 

## Outlier Treatment 
Our dataset does not contain any outliers so we do not have to worry about outliers. 

## Explore our Numerical columns

1. Skewness - We have seen some of the features are skewed positively or negatively. we should treat skewness as it will mislead the results while applying algorithms.

2. Correlation - we are able to see this temperature(°C) and dev point temperature(°C) column are highly correlated i.e. 0.91.
We need to drop this column then it will not affect the outcome of our analysis and also having same variation.

3. Multicollinearity – we have seen there is high multicollinearity between columns and also having high VIF and affects the results. 

## Encoding of categorical columns 
We used One Hot Encoding to produce dummy variable which uses binary integers of 0 and 1 to encode our categorical features because categorical features that are in string format cannot be understood by the machine and needs to be converted to numerical format.
Our categorical values are Hour, Seasons, Holidays, Functional days etc.

## Normalization of features
Our dependent variable ‘Rented bike count” is right skewed so it needs to be normalized and the methods used for normalization is log10, square, square root.


# Model Training
## Test Train Split
We need to split the data into train and test data for estimating the performance of machine learning algorithms. We split in 75-25 % of the dataset.

## Fitting different models
For modelling we tried various regression algorithms like:
1.	Logistic Regression
2.	Lasso Regressor
3.	Ridge Regressor
4.	Decision Tree Regressor
5.	Random Forest Classifier

## Tuning the hyperparameters for better accuracy
Tuning the hyperparameters of respective algorithms is necessary for getting better accuracy and to avoid overfitting in case of tree-based models. 
1.	Gradient Boosting
2.	Ada Boosting
3. XG Boosting

# Evaluation metrices 
1.	Mean Absolute error
2.	Mean Squared error
3.	Root Mean Squared error
4.	R Squared 
5.	Adjusted R Squared

    
## Conclusion 
1. In holidays or non-working days there is demand in rented bikes.
2. There is a surge of high demand in the morning 8AM and in evening 6PM as the people might be going to their office at morning 8AM and returning from their office at the 6PM.
3. People preferred more rented bikes in the morning compared with evening.
4. When the rainfall was less, people have booked more bikes except some few cases.
5. The temperature, Hour are the most important features that positively drive the total rented bikes count.
6. It is observed that highest number of rental bikes counts in Autumn and summer seasons and the lowest in winter season.
7. We observed that the highest number of rental bikes counts on a clear day or little windy day and the lowest on a snowy and rainy day.
8. In the given dataset there was no strong relationship present between dependent variable "Rented bike count" and independent variables.
9. Out of all models we apply Decision tree and Random Forest model are most accurate and the reason is there are no specific relation between features.
10.	Random Forest worked best in predicting the count of rented bikes as its R2 score is maximum from the tried model.
11. We are getting best results using Gradient Boost Regressor.


