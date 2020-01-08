import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Salary_Data.csv")

# print(dataset)

# Divide the Dataset into Two Variable (Dependent and Independent variable)

# X is a Independent variable (Year of experience)
x = dataset.iloc[:, :-1].values

# Y is a Dependent variable [Salary (Based on a year of experience)]
y = dataset.iloc[:, 1:2].values

# Spliting the data based on a training and test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)



# Implement or classifier based on simple linear Regression
from sklearn.linear_model import LinearRegression

simpleRegression = LinearRegression()
simpleRegression.fit(x_train, y_train)

y_predict = simpleRegression.predict(x_test)

# If you Predict the variable year of experince people salary then pass their value
# like we want to find 12 year of experience people salary then

y_predict_val = simpleRegression.predict([[11]])

# Implement the Graph
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, simpleRegression.predict(x_train))
plt.show()