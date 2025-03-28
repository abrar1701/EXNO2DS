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
# Developed By: YUVAN SUNDAR S
# REGISTER  NO: 212223040250
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("/content/titanic_dataset.csv")
df
```
![image](https://github.com/user-attachments/assets/9df3b6ee-cde2-471b-853f-757c6084c1f4)
```py
df.info()
```
![image](https://github.com/user-attachments/assets/79123962-b784-42d5-b436-754268388e9c)
```py
df.shape
```
![image](https://github.com/user-attachments/assets/40d7e572-65b0-4fb6-9e45-1240f88bafc9)
```py
df.describe()
```
![image](https://github.com/user-attachments/assets/efdd96e6-928c-435f-9045-3bc1c1837fcc)
```py
df.shape
```
![image](https://github.com/user-attachments/assets/f06ec038-64c5-49bf-b018-1e358629a2ac)
```py
df.nunique()
```
![image](https://github.com/user-attachments/assets/1e51651f-6336-4f9b-a298-ee6d8334f038)
```py
df["Survived"].value_counts()
```
![image](https://github.com/user-attachments/assets/2c991851-4d1a-4bf9-8a97-fd1ed33d6c4a)
```py
per=(df["Survived"].value_counts()/df.shape[0]*100).round(2)
per
```
![image](https://github.com/user-attachments/assets/bebac8b7-01e9-464d-bbae-a983d8c3c469)
```py
sns.countplot(x="Survived",data=df)
```
![image](https://github.com/user-attachments/assets/b7b83a3d-3b64-4939-ad4f-99def1ce476e)
```py
df
```
![image](https://github.com/user-attachments/assets/779eb9b5-2a7a-420a-a2dd-dfc55c59a05a)
```py
df.Pclass.unique()
```
![image](https://github.com/user-attachments/assets/282eee5e-dce4-4d26-843c-5964a2ecb85c)
```py
df.rename(columns={'Sex':'Gender'},inplace=True)
df
```
![image](https://github.com/user-attachments/assets/f02bda5c-c96d-41e7-82fa-878fc083cf07)
# Bivariate Analysis
```py
 sns.catplot(x="Gender",col="Survived",kind="count",data=df,height=5,aspect=.7)
```
![image](https://github.com/user-attachments/assets/eb125a7d-0442-4a2c-9c60-6bfedd5d5326)
```py
 sns.catplot(x="Survived",hue="Gender",data=df,kind="count")
```
![image](https://github.com/user-attachments/assets/b093977b-4709-4f98-b540-8e130f8eb11a)
```py
df.boxplot(column="Age",by="Survived")
```
![image](https://github.com/user-attachments/assets/11f1657f-a10d-4865-b0cf-cc12b60ba0d4)
```py
sns.scatterplot(x=df["Age"],y=df["Fare"])
```
![image](https://github.com/user-attachments/assets/8d695780-ed5e-4497-aaba-113618c41086)
```py
sns.jointplot(x="Age",y="Fare",data=df)
```
![image](https://github.com/user-attachments/assets/f529f34c-d577-4a9a-b9cd-675b52d5deaf)
#  Multivariate Analysis
```py
fig, ax1 = plt.subplots(figsize=(8,5))
sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=df)
```
![image](https://github.com/user-attachments/assets/f965001f-7193-4399-823c-69751bde4027)
```py
 sns.catplot(data=df,col="Survived",x="Gender",hue="Pclass",kind="count")
```
![image](https://github.com/user-attachments/assets/27b36b0e-b86f-4d7e-bc82-104c8af5ab19)
# Co-Relation
```py
numeric_df = df.select_dtypes(include=np.number)
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True)
```
![image](https://github.com/user-attachments/assets/12b79584-a05c-4774-be87-4840505024ff)
```py
sns.pairplot(df)
```
![image](https://github.com/user-attachments/assets/2ea67bc7-8ca1-4beb-b539-16fd628101d1)

# RESULT
We have performed Exploratory Data Analysis on the given data set successfully.
