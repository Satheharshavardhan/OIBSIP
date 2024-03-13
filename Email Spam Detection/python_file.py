# %% [markdown]
# Importing the Required Libraries for the Project

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# Reading the dataset from the .csv file

# %%
spam_data = pd.read_csv("spam.csv", encoding='latin1')

# %% [markdown]
# Analyzing the dataset

# %%
spam_data.head()

# %%
spam_data.shape

# %%
spam_data.describe()

# %%
spam_data.info()

# %% [markdown]
# Deleting the useless columns for the project

# %%
columns_drop = ["Unnamed: 2","Unnamed: 3","Unnamed: 4"]
spam_data = spam_data.drop(columns_drop,axis=1)

# %%
spam_data.head()

# %%
spam_data.shape

# %% [markdown]
# Reformatting the dataset

# %%
spam_data.columns = ["Target","Data"]

# %%
spam_data.head()

# %%
X = spam_data["Data"]
Y = spam_data["Target"]

# %%
X.shape

# %%
Y.shape

# %% [markdown]
# Encodding the data (Data Preprocessing)

# %% [markdown]
# Here 0 means it is Ham and 1 means it is a Spam

# %%
le = LabelEncoder()
Y = le.fit_transform(Y)

# %%
Y

# %% [markdown]
# Converting text data into numerical form (feature extraction)

# %%
X.head()

# %%
feature_extraction = TfidfVectorizer(min_df=1,stop_words="english",lowercase=True)

# %%
X = feature_extraction.fit_transform(X)

# %%
print(X)

# %% [markdown]
# Splitting the data into training and testing

# %%
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=7,test_size=0.2)

# %%
print(X_train.shape)
print(X_test.shape)
print(X.shape)

# %%
print(Y_train.shape)
print(Y_test.shape)
print(Y.shape)

# %% [markdown]
# Creating the logistic Regression model as this is best for the classification of 2 objects

# %%
Logistic_Model = LogisticRegression()

# %%
Logistic_Model.fit(X=X_train,y=Y_train)

# %% [markdown]
# Prediction on the tranning data

# %%
train_data_prediction = Logistic_Model.predict(X_train)

# %%
training_accuracy = accuracy_score(Y_train,train_data_prediction)

# %%
print(f"The accuracy on training data is {training_accuracy}")

# %% [markdown]
# Predicting on the testing data

# %%
conf_matrix = confusion_matrix(Y_train, train_data_prediction)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Classification: Confusion Matrix")
plt.show()

# %%
test_data_prediction = Logistic_Model.predict(X_test)

# %%
test_accuracy = accuracy_score(Y_test,test_data_prediction)

# %%
print(f"The accuracy on test data is {test_accuracy}")

# %%
conf_matrix = confusion_matrix(Y_test, test_data_prediction)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Classification: Confusion Matrix")
plt.show()


