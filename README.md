# EXNO:2
# DATA SCIENCE
# AIM:
To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
# Developed By: MOHAMED ABRAR M
# REGISTER  NO: 212223040111
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

drive.mount('/content/drive')
```
![image](https://github.com/user-attachments/assets/d452575d-c289-449e-a322-6396274a9c49)
```py
ls 'drive/MyDrive/Colab Notebooks'
```
![image](https://github.com/user-attachments/assets/eec2ef52-543c-4320-bf13-436dad2d3a28)
```py
dt=pd.read_csv("drive/MyDrive/Colab Notebooks/titanic_dataset.csv")
dt
```
![image](https://github.com/user-attachments/assets/a0602d70-aa10-4c1a-a9a4-d237c80a5b39)
```py
dt.info()
```
![image](https://github.com/user-attachments/assets/34567522-0bf6-411f-bd2e-166c5305f116)
## DISPLAY NO OF ROWS AND COLUMNS
```py
print(f"Number of rows = {dt.shape[0]}")
print(f"Number of columns = {dt.shape[1]}")
```
![image](https://github.com/user-attachments/assets/d99c1a9b-4140-43db-a7d9-24b9dcfda152)
## SET PASSENGER ID AS INDEX COLUMN
```py
dt.set_index('PassengerId', inplace=True)
dt
```
![image](https://github.com/user-attachments/assets/badaeab5-4b83-4281-96e6-62ff89a04f88)
```py
dt.describe()
```
![image](https://github.com/user-attachments/assets/922af2a3-f17d-482f-b9ea-c44cbb3f37bb)
# CATEGORICAL DATA ANALYSIS
## USE VALUE COUNT FUNCTION AND PERFROM CATEGORICAL ANALYSIS
```py
dt["Survived"].value_counts()
```
![image](https://github.com/user-attachments/assets/abe96b7c-1978-4426-8a11-4d1ce1b8041e)
```py
per = ((dt["Survived"].value_counts())/dt.shape[0]*100).round(2)
per
```
![image](https://github.com/user-attachments/assets/ef42df9f-d34d-4de2-a0a6-2d14cb9d6d92)
# UNIVARIATE ANALYSIS
## USE COUNTPLOT AND PERFORM UNIVARIATE ANALYSIS FOR THE "SURVIVED" COLUMN IN TITANIC DATASET
```py
sns.countplot(data=dt,x="Survived")
```
![image](https://github.com/user-attachments/assets/bd5b0f36-7128-4e12-88f3-c78b6f7f46ec)
## IDENTIFY UNIQUE VALUES IN "PASSENGER CLASS" COLUMN
```py
dt.Pclass.unique()
```
![image](https://github.com/user-attachments/assets/6b8ba472-af0f-408f-91bc-5faad65baf06)

## RENAMING COLUMN
```py
dt.rename(columns = {'Sex':'Gender'}, inplace = True)
dt
```
![image](https://github.com/user-attachments/assets/7b528339-afa6-41ae-b53f-39f0cc4845c5)

# BIVARIATE ANALYSIS
## USE CATPLOT METHOD FOR BIVARIATE ANALYSIS
```py
sns.catplot(x="Gender", col="Survived", kind="count", data=dt, height=5, aspect=0.7, palette="coolwarm")
```
![image](https://github.com/user-attachments/assets/fc61f999-599d-45f0-8780-b5b4187c764f)
```py
 sns.catplot(x="Survived",hue="Gender",data=dt,kind="count")
```
![image](https://github.com/user-attachments/assets/803bd17c-2572-42d5-a53c-e3465e523952)
```py
fig, ax1 = plt.subplots(figsize=(8,5))
graph = sns.countplot(data=dt, x="Survived", palette='coolwarm')
graph.set_xticklabels(graph.get_xticklabels())
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2, height + 20.8,height ,ha="left")
```
![image](https://github.com/user-attachments/assets/444dc3a9-32c1-4f24-b49d-f26d6bfe7e46)
## USE BOXPLOT METHOD TO ANALYZE AGE AND SURVIVED COLUMN
```py
dt.boxplot(column="Age",by="Survived")
```
![image](https://github.com/user-attachments/assets/ea69de29-24b6-4833-8530-d6ab3fb5a20d)
# MULTIVARIATE ANALYSIS
## USE BOXPLOT METHOD AND ANALYZE THREE COLUMNS(PCLASS,AGE,GENDER)
```py
plt.figure(figsize=(8,6))
sns.boxplot(x="Pclass", y="Age", hue="Gender", data=dt, palette="coolwarm")
```
![image](https://github.com/user-attachments/assets/044a5714-852a-410e-b78f-f9d1938df7d6)
## USE CATPLOT METHOD AND ANALYZE THREE COLUMNS(PCLASS,SURVIVED,GENDER)
```py
sns.catplot(x="Pclass", hue="Survived", col="Gender", data=dt, kind="count", palette="coolwarm", height=5, aspect=0.8)
```
![image](https://github.com/user-attachments/assets/066fe2c5-ba99-45c0-bac5-f324b90455d5)
## IMPLEMENT HEATMAP AND PAIRPLOT FOR THE DATASET
```py
numeric_df = dt.select_dtypes(include=np.number)
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True)
```
![image](https://github.com/user-attachments/assets/d216d1f7-c894-4b1d-bb2b-827e5f5daac0)
# RESULT
We have performed Exploratory Data Analysis on the given data set successfully.
