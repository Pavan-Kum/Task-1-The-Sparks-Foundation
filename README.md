# Task-1-The-Sparks-Foundation
Task 1 - Prediction using Supervised Machine Learning
Our goal here is to predict the percentage a student gets based on the number of study hours
Bolisetty Pavan Kumar
#importing all the libraries required
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline
the 
# Reading the data from the link 
url = "http://bit.ly/w-data"
student_data = pd.read_csv(url)
print("Data has been imported successfully")
Data has been imported successfully
5
#checking if data has been imported
student_data.head(5)
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
tudent
# Plotting the distribution of scores
student_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

We can see that it is a positive linear regression between the number of hours studied and percentage scored
X = student_data.iloc[:, :-1].values  
y = student_data.iloc[:, 1].values  
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
Now we have split our data into training and testing sets, we shall now train the algorithm
d
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
​
print("Training has been completed.")
Training has been completed.
# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_
​
# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()

Since our data has been trained, let us make some predictions
print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores
[[1.5]
 [3.2]
 [7.4]
 [2.5]
 [5.9]]
# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 
Actual	Predicted
0	20	16.884145
1	27	33.732261
2	69	75.357018
3	30	26.794801
4	62	60.491033
]]
# Predicting according to our requirement 
hours = [[9.25]]
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))
No of Hours = [[9.25]]
Predicted Score = 93.69173248737538
Evaluating our model
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 
Mean Absolute Error: 4.183859899002975
