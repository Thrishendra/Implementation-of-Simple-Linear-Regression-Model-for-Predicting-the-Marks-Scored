# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: T.Thrishendra
RegisterNumber: 212223230227
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

print("NAME: TELLA THRISHENDRA")
print("REG NO: 212223230227")
```
## Output:
## Displaying the content in datafield
## head:
![image](https://github.com/user-attachments/assets/ce7a689d-e0ff-4c90-8e92-045c2ea9b597)
## tail:
![image](https://github.com/user-attachments/assets/5fe75bb7-f41f-48ad-8784-46f0ed678240)
## Segregating data to variables
![image](https://github.com/user-attachments/assets/1925267a-4d10-4572-af8f-1a01d5f87a79)
## Displaying predicted values
![image](https://github.com/user-attachments/assets/63a91a54-7c60-4bb9-bf53-bbd55138b9be)
## Displaying actual values
![image](https://github.com/user-attachments/assets/b5f84e76-33c4-4fbb-8627-ab7828cf3737)
## Graph plot for training data
![image](https://github.com/user-attachments/assets/a6febe0d-9b26-4b03-bda3-bae4a177994e)
## Graph plot for test data
![image](https://github.com/user-attachments/assets/9ab8375c-ac3b-47d2-b782-ab3d1f820db2)
## MSE MAE RMSE:
![Screenshot 2025-04-28 154034](https://github.com/user-attachments/assets/60a646fd-16d0-4d2f-a010-d81be1e5c5d0)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
