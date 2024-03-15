# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score


# %%
data_in_India = pd.read_csv("Unemployment in India.csv")

# %%
data_in_India.head()

# %%
data_in_India[" Date"].value_counts()

# %%
data_upto_nov_2020 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

# %%
data_upto_nov_2020.head()

# %%
data_upto_nov_2020.shape

# %%
data_upto_nov_2020.isna().sum()

# %%
data_upto_nov_2020.duplicated().sum()

# %%
data_upto_nov_2020.info()

# %%
data_upto_nov_2020.describe()

# %%
sns.pairplot(data_upto_nov_2020)
plt.show()

# %%
data_upto_nov_2020.columns

# %%
data_upto_nov_2020[" Frequency"].value_counts()

# %%
data_upto_nov_2020["Region.1"].value_counts()

# %%
data_upto_nov_2020["Region"].value_counts()

# %%
columns_remove = ['Region',' Frequency','Region.1']

# %%
data_upto_nov_2020 = data_upto_nov_2020.drop(columns=columns_remove,axis=1)

# %%
data_upto_nov_2020.head()

# %%
data_upto_nov_2020[" Date"].value_counts()

# %%
correlation_matrix = data_upto_nov_2020.drop(" Date",axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# %%
data_upto_nov_2020[" Date"] = pd.to_datetime(data_upto_nov_2020[" Date"])

# %%
data_upto_nov_2020.head()

# %%
data_upto_nov_2020.sort_values(by=" Date",inplace=True)

# %%
data_upto_nov_2020.head()

# %%
plt.figure(figsize=(10, 6))
plt.plot(data_upto_nov_2020[" Date"], data_upto_nov_2020[' Estimated Unemployment Rate (%)'])
plt.xlabel('Date')
plt.ylabel('Unemployment Rate')
plt.title('Unemployment Rate Time Series')
plt.grid(True)
plt.show()

# %%
today = pd.to_datetime('today').normalize()
data_upto_nov_2020['Days Since Date'] = (today -data_upto_nov_2020[" Date"]).dt.days
data_upto_nov_2020.drop(" Date", axis=1, inplace=True)

# %%
data_upto_nov_2020.head()

# %%
sns.pairplot(data_upto_nov_2020)
plt.show()

# %%
correlation_matrix = data_upto_nov_2020.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# %%
scalar = MinMaxScaler()

# %%
normalized_data = pd.DataFrame(scalar.fit_transform(data_upto_nov_2020),columns=data_upto_nov_2020.columns)

# %%
normalized_data.head()

# %%
normalized_data.tail()

# %%
data_upto_nov_2020 = normalized_data

# %%
X = data_upto_nov_2020.drop(' Estimated Unemployment Rate (%)',axis=1)
Y = data_upto_nov_2020[" Estimated Labour Participation Rate (%)"]

# %%
X.shape

# %%
Y.shape

# %%
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=7,test_size=0.2)

# %%
print(X.shape)
print(X_train.shape)
print(X_test.shape)

# %%
print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)

# %%
model = SVR()

# %%
model.fit(X_train,Y_train)

# %%
train_prediction = model.predict(X_train)

# %%
mse_train = mean_squared_error(Y_train,train_prediction)
r2_train = r2_score(Y_train,train_prediction)

# %%
print(f"The mean squared error on training data is {mse_train}")
print(f"The R squared score on training data is {r2_train}")

# %%
plt.scatter(Y_train, Y_train, color='red', label='Actual')
plt.scatter(train_prediction, train_prediction, color='green', label='Predicted')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual Values vs Predicted Values (Training data)")
plt.legend()
plt.show()

# %%
test_prediction = model.predict(X_test)

# %%
mse_test = mean_squared_error(Y_test,test_prediction)
r2_test = r2_score(Y_test,test_prediction)

# %%
print(f"The mean squared error on test data is {mse_test}")
print(f"The R squared score on test data is {r2_test}")

# %%
plt.scatter(Y_test, Y_test, color='red', label='Actual')
plt.scatter(test_prediction, test_prediction, color='green', label='Predicted')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual Values vs Predicted Values (Training data)")
plt.legend()
plt.show()


