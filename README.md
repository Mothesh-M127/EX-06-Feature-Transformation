# EX-06-Feature-Transformation
## AIM
To Perform the various feature transformation techniques on a dataset and save the data to a file.

## Explanation
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## Method For Data Tranformation
1. FUNCTION TRANSFORMATION
2. POWER TRANSFORMATION
3. POWER TRANSFORMATION
## ALGORITHM
STEP 1
Read the given Data

STEP 2
Clean the Data Set using Data Cleaning Process

STEP 3
Apply Feature Transformation techniques to all the feature of the data set

STEP 4
Save the data to the file

## CODE
Developed by: MOTHESH.M
Register Number: 212221230066
## Data_To_Transform.csv:
```
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats 
df=pd.read_csv("Data_To_Transform.csv")  
df 
df.skew() 

#Log Transformation  
np.log(df["Highly Positive Skew"])  

#Reciprocal Transformation  
np.reciprocal(df["Moderate Positive Skew"])

#Square Root Transformation  
np.sqrt(df["Highly Positive Skew"])

#Square Transformation  
np.square(df["Highly Negative Skew"])

# POWER TRANSFORMATION
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])  
df 
df["Moderate Positive Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Positive Skew"])  
df
df["Moderate Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Negative Skew"])  
df
df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"])  
df

#QUANTILE TRANSFORMATION:  
from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])  
sm.qqplot(df['Moderate Negative Skew'],line='45')  
plt.show()
sm.qqplot(df['Moderate Negative Skew_1'],line='45')  
plt.show()
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])  
sm.qqplot(df['Highly Negative Skew'],line='45')  
plt.show()  
sm.qqplot(df['Highly Negative Skew_1'],line='45')  
plt.show()
df["Moderate Positive Skew_1"]=qt.fit_transform(df[["Moderate Positive Skew"]])  
sm.qqplot(df['Moderate Positive Skew'],line='45')  
plt.show()  
sm.qqplot(df['Moderate Positive Skew_1'],line='45')  
plt.show() 
df["Highly Positive Skew_1"]=qt.fit_transform(df[["Highly Positive Skew"]])  
sm.qqplot(df['Highly Positive Skew'],line='45')  
plt.show()  
sm.qqplot(df['Highly Positive Skew_1'],line='45')  
plt.show() 
df.skew()  
df 
```
## titanic_dataset.csv:
```
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats  

df=pd.read_csv("titanic_dataset.csv")  
df  

df.drop("Name",axis=1,inplace=True)  
df.drop("Cabin",axis=1,inplace=True)  
df.drop("Ticket",axis=1,inplace=True)  
df.isnull().sum()  

df["Age"]=df["Age"].fillna(df["Age"].median())  
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])  
df.info()  

from sklearn.preprocessing import OrdinalEncoder  
 
embark=["C","S","Q"]  
emb=OrdinalEncoder(categories=[embark])  
df["Embarked"]=emb.fit_transform(df[["Embarked"]])  

df  

#FUNCTION TRANSFORMATION:  
#Log Transformation  
np.log(df["Fare"])  

#ReciprocalTransformation  
np.reciprocal(df["Age"])  

#Squareroot Transformation:  
np.sqrt(df["Embarked"])  

#POWER TRANSFORMATION:  
df["Age _boxcox"], parameters=stats.boxcox(df["Age"])  
df  

df["Pclass _boxcox"], parameters=stats.boxcox(df["Pclass"])    
df    

df["Fare _yeojohnson"], parameters=stats.yeojohnson(df["Fare"])  
df  

df["SibSp _yeojohnson"], parameters=stats.yeojohnson(df["SibSp"])  
df  

df["Parch _yeojohnson"], parameters=stats.yeojohnson(df["Parch"])  
df  


#QUANTILE TRANSFORMATION  

from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)  


df["Age_1"]=qt.fit_transform(df[["Age"]])  
sm.qqplot(df['Age'],line='45')  
plt.show()  

sm.qqplot(df['Age_1'],line='45')  
plt.show()  

df["Fare_1"]=qt.fit_transform(df[["Fare"]])  
sm.qqplot(df["Fare"],line='45')  
plt.show()  

sm.qqplot(df['Fare_1'],line='45')  
plt.show()  

df.skew()  
df  
```

# OUPUT:
## Data_To_Transform.csv:
![q1](https://user-images.githubusercontent.com/94170892/169091056-13f864c2-ad8e-4600-b498-98b614564cb8.png)
![q2](https://user-images.githubusercontent.com/94170892/169091079-ccdba71e-aa2e-4d25-ad5c-a3cb37ec5021.png)
![q3](https://user-images.githubusercontent.com/94170892/169091105-d70d101a-8608-4bb5-ab76-4911203d957f.png)
![q4](https://user-images.githubusercontent.com/94170892/169091124-44f376b7-3d5f-49c6-ae7b-7b4206e1b1c2.png)
![q5](https://user-images.githubusercontent.com/94170892/169091149-c57f7359-5055-441f-8bfe-4c09416bcc4e.png)
![q6](https://user-images.githubusercontent.com/94170892/169091175-5c1800c3-f19a-44bf-b780-c2d6f31222fa.png)
![q7](https://user-images.githubusercontent.com/94170892/169091238-cbf5f907-b633-45c5-aedb-bb7562fda155.png)
![q8](https://user-images.githubusercontent.com/94170892/169091263-c5654a40-b762-4216-a608-7289510cbae7.png)
![q9](https://user-images.githubusercontent.com/94170892/169091296-bd8463bd-c26c-42eb-a6af-467485de2a9f.png)
![q10](https://user-images.githubusercontent.com/94170892/169091330-711b0400-d533-46a0-994d-f863e162ca49.png)
![q11](https://user-images.githubusercontent.com/94170892/169091355-9d56412a-cd34-4e08-a6d7-b13552008f03.png)
![q12](https://user-images.githubusercontent.com/94170892/169091392-1ff9029c-71c4-4ccc-875e-4db8b035703b.png)
![q13](https://user-images.githubusercontent.com/94170892/169091420-9e84289a-9011-4a91-92c6-3aa5d15293f0.png)
![q14](https://user-images.githubusercontent.com/94170892/169091457-c388fabb-0601-4bac-806e-3e716c52b11a.png)
![q15](https://user-images.githubusercontent.com/94170892/169091480-15ac30ed-1bc7-4c4b-a202-702cdb887026.png)
![q16](https://user-images.githubusercontent.com/94170892/169091519-6c11a509-b03f-4874-acef-c4ac10844403.png)
![q17](https://user-images.githubusercontent.com/94170892/169091545-390f61a8-bb95-49c8-83e7-0810e4ca98ea.png)
![q18](https://user-images.githubusercontent.com/94170892/169091575-459e9b44-4d2c-45c4-a96d-1dc003003ad3.png)

## titanic_dataset.csv:
![a1](https://user-images.githubusercontent.com/94170892/169091714-39e385cd-0317-4124-a191-3ede2a57b6d2.png)
![a2](https://user-images.githubusercontent.com/94170892/169091731-2b229145-5dda-410d-9943-59496cd5afc7.png)
![a3](https://user-images.githubusercontent.com/94170892/169091754-5d51b2ae-9cf8-42e1-a7ff-b8aabf7bc80b.png)
![a4](https://user-images.githubusercontent.com/94170892/169091778-23180e1a-6133-4de6-b994-5f02dd3b0bea.png)
![a5](https://user-images.githubusercontent.com/94170892/169091835-80b58ff2-67f7-4ee2-8127-007b7f8e8f6d.png)
![a6](https://user-images.githubusercontent.com/94170892/169091853-65ef9ff8-82dd-4597-9eb3-c6774c56e784.png)
![a7](https://user-images.githubusercontent.com/94170892/169091876-409d8dfd-90e9-4cfb-9943-5cacadaed73d.png)
![a8](https://user-images.githubusercontent.com/94170892/169091905-05e5093f-ebbf-4c7c-90ff-f070ae6668d6.png)
![a9](https://user-images.githubusercontent.com/94170892/169091920-99ef3348-fe94-44cc-85b4-580669f932ca.png)
![a10](https://user-images.githubusercontent.com/94170892/169091935-906ea2c4-9009-4af7-935d-726f5a8005de.png)
![a11](https://user-images.githubusercontent.com/94170892/169091969-0c559e77-4567-4862-8ee4-62175c931ab4.png)
![a12](https://user-images.githubusercontent.com/94170892/169092003-7428d7b3-e66c-42f7-ae32-e5edddc251cc.png)
![a13](https://user-images.githubusercontent.com/94170892/169092029-42968eba-bb98-42f7-9795-517f6d217dee.png)
![a14](https://user-images.githubusercontent.com/94170892/169092068-9e97a276-6fbe-47e2-8500-37af1baf4311.png)
![a15](https://user-images.githubusercontent.com/94170892/169092095-e0bad243-c293-4ad0-9110-9ff92978c829.png)
![a16](https://user-images.githubusercontent.com/94170892/169092133-e3c5bf0d-ccd5-467e-9651-a4be7ec3b0ca.png)
![a17](https://user-images.githubusercontent.com/94170892/169092154-4584b62d-ab6c-47d7-807c-f0d7b9eaef63.png)

## RESULT:
Hence, Feature transformation techniques is been performed on given dataset and saved into a file successfully.
