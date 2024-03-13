# %% [markdown]
# Importing the required libraries for the project

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# %% [markdown]
# Reading the dataset from the .csv file

# %%
sales_data = pd.read_csv("Advertising.csv")

# %% [markdown]
# Getting the insights from the dataset

# %%
sales_data.head()

# %%
sales_data.shape

# %% [markdown]
# Removing the useless columns from the dataset i.e. which is not required for the project

# %%
sales_data = sales_data.drop("Unnamed: 0",axis=1)

# %%
sales_data.head()

# %% [markdown]
# Getting the information about the dataset

# %%
sales_data.info()

# %% [markdown]
# Data preprossing

# %%
sales_data.isna().sum()

# %%
sales_data.duplicated().sum()

# %%
sales_data.tail()

# %% [markdown]
# Getting the satistical analysis about the data

# %%
sales_data.describe()

# %% [markdown]
# Getting the insights about the data using the pairplot graph

# %%
sns.pairplot(sales_data)
plt.show()

# %% [markdown]
# Plotting the Heatmap and identifying the relations about various columns of the dataset.

# %%
correlation_matrix = sales_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# %% [markdown]
# Seperating out the features and target from the dataset

# %%
X = sales_data.drop("Sales",axis=1)
Y = sales_data["Sales"]

# %%
print(X.shape)
print(Y.shape)

# %% [markdown]
# Splitting the dataset for the trainning and testing data

# %%
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=7,test_size=0.2)

# %%
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# %% [markdown]
# Using the Random Forest Regressor algorithm for the predicton of the Sales.

# %%
model = RandomForestRegressor(n_estimators=99)

# %%
model.fit(X_train,Y_train)

# %% [markdown]
# Predicting on the trainning data

# %%
train_prediction = model.predict(X_train)

# %%
mae = mean_absolute_error(Y_train, train_prediction)
mse = mean_squared_error(Y_train, train_prediction)
r2 = r2_score(Y_train, train_prediction)

# %%
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared (R2) Score:", r2)

# %%
plt.scatter(Y_train, Y_train, color='red', label='Actual')
plt.scatter(train_prediction, train_prediction, color='green', label='Predicted')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual Values vs Predicted Values (Training data)")
plt.legend
plt.show()

# %% [markdown]
# Predicting on the Testing Data

# %%
test_prediction = model.predict(X_test)

# %%
mae_test = mean_absolute_error(Y_test, test_prediction)
mse_test = mean_squared_error(Y_test, test_prediction)
r2_test = r2_score(Y_test, test_prediction)

# %%
print("Mean Absolute Error:", mae_test)
print("Mean Squared Error:", mse_test)
print("R-squared (R2) Score:", r2_test)

# %%
plt.scatter(Y_test, Y_test, color='red', label='Actual')
plt.scatter(test_prediction, test_prediction, color='green', label='Predicted')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual Values vs Predicted Values (Testing data)")
plt.legend
plt.show()


