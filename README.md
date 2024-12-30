# Analysis-of-the-Titanic-ship-disaster

## Table of Content

- [Project Overview](#project-overview)

- [Data Source](#data-source)

- [Tools](#tools)

- [Data Cleaning/Preparation](#data-cleaningpreparation)

- [Exploratory Data Analysis](#exploratory-data-analysis)

- [Results/Findings](#resultsfindings)

- [Key Challenges](#Key-Challenges)
  
- [Conclusion](#Conclusion)

- [References](#references)

### Project Overview

I analyzed the Titanic ship disaster using Python and primarily focused on understanding the factors that contributed to passenger survival and identifying key insights from the available dataset to first, understand the distribution of passengers based on demographic and socio-economic factors. Secondly, identify the key factors that influenced survival rates. And lastly, draw actionable insights for safety improvements in maritime travel.

### Data Source
Titanic data: The primary dataset used for this analysis is the titanic.csv file, which contains detailed information about the Titanic ship's survival rate and the contributing factors.

### Tools
Kaggle is the source of the dataset in the CSV file. Python programming language on Kaggle is used to clean the dataset, format, transform, group, and analyze. Libraries used include Pandas, NumPy, Matplotlib, and Scikit-learn.

### Data Cleaning/Preparation

It is important to ensure the dataset is clean enough to gain insights from a dataset. I ensured the dataset was thoroughly cleaned to ensure the accuracy and integrity of my findings. The data cleaning and preparation processes I carried out are shown in the codes below:

 ```python
#load in some packages
%matplotlib inline

import numpy as np
import pandas as pd
import os

# Read the data
titanic_train = pd.read_csv("/kaggle/input/titanic/train.csv")

# Sorting out the name column
sorted(titanic_train["Name"])[0:15]

#check the dimensions
titanic_train.shape
```
![check the dimesion](https://github.com/user-attachments/assets/509330dc-4d33-423f-ad2b-618d234a91f6)
       
```python
# Check the summary of the dataset
titanic_train.info()
```
![check summary of data](https://github.com/user-attachments/assets/59078906-173b-4bc2-b3ac-a573bacd162a)

```python
 # Give summary statistics
titanic_train.describe()

# Check categorical columns for unique items
categorical = titanic_train.select_dtypes(include=['object']).columns
print(categorical)
titanic_train[categorical].describe()

#convert Cabin column to string
char_cabin = titanic_train["Cabin"].astype(str)             

#take first letter
new_Cabin = np.array([cabin[0] for cabin in char_cabin])    
new_Cabin = pd.Categorical(new_Cabin)
new_Cabin.describe()

# Rename new cabin column
titanic_train["Cabin"] = new_Cabin

#detect missing values
dummy_vector = pd.Series([1,None,3,None,7,8])
dummy_vector.isnull()

# Check the age column
titanic_train["Age"].describe()

# Check for missing values in the age column
missing = np.where(titanic_train["Age"].isnull() == True)
missing

# Plot a histogram on the age column to check for a pattern
titanic_train.hist(column='Age',   #column to plot
                  figsize=(5,5),   #plot size
                  bins=20)         #number of histogram bins
```
![histogram for age](https://github.com/user-attachments/assets/92d0c929-25e7-418e-ac4e-9ad1b4b35677)


```python
# Replace the missing values in the age column with the mean age(28)
new_age_var = np.where(titanic_train["Age"].isnull(),   
                      28,                              
                      titanic_train["Age"])             
titanic_train["Age"] = new_age_var
titanic_train["Age"].describe()

# Create a new variable 'Family' from the SibSp and Parch
#creating new variables is known as feature engineering
titanic_train["Family"] = titanic_train["SibSp"] + titanic_train["Parch"]

# Convert cabin to str and take the first letter, then save the new cabin
titanic_train=pd.read_csv("/kaggle/input/titanic/train.csv")  
char_cabin=titanic_train["Cabin"].astype(str)      
new_cabin=np.array([cabin[0] for cabin in char_cabin])  
titanic_train["Cabin"]=pd.Categorical(new_cabin)

# Changing 1 and 0 in the Survived column into 'Survived' and 'Died'
new_survived = pd.Categorical(titanic_train["Survived"])
new_survived = new_survived.rename_categories(["Died","Survived"])
new_survived.describe()
```

## Exploratory Data Analysis

```python
# Count of number of dead and survived passengers with their percentages
new_survived = pd.Categorical(titanic_train["Survived"])
new_survived = new_survived.rename_categories(["Died","Survived"])
new_survived.describe()
```
![count of dead and survived passengers](https://github.com/user-attachments/assets/0f74083e-b051-4239-b198-c6a71f02e0e1)

```python
# Count of passengers by sex
pd.crosstab(index=titanic_train["Sex"],
columns="count")
```
![count of sex](https://github.com/user-attachments/assets/ebb8c10b-1e55-4e42-9545-3471933d179d)

```python
#table of survival vs sex
survived_sex=pd.crosstab(index=titanic_train["Survived"],
                        columns=titanic_train["Sex"])
survived_sex.index=["Died","Survived"]
survived_sex
```
![table of survival vs sex](https://github.com/user-attachments/assets/8a7c4008-8f1b-4f5b-ae5c-113938a3c091)

```python
#table of survival vs passenger class
survived_class=pd.crosstab(index=titanic_train["Survived"],
                       columns=titanic_train["Pclass"])
survived_class.column = ["Class1","Class2","Class3"]
survived_class.index = ["Died","Survived"]
survived_class
```
![survival vs passenger class](https://github.com/user-attachments/assets/c4b006de-3cdf-4a14-9fc1-8a4005792f87)


```python
#divide by column totals to get the percentage of survival vs sex passenger class
surv_sex_class/surv_sex_class.loc["All"]
```
![percentage of survival vs sex and passenger class](https://github.com/user-attachments/assets/e5e8e810-4dd1-4741-97a2-61c1da17b5dd)

```python
# Table of Survival vs sex and passenger class
surv_sex_class=pd.crosstab(index=titanic_train["Survived"],
                           columns=[titanic_train["Pclass"],
                                   titanic_train["Sex"]],
                           margins=True)   #include rows and column total
surv_sex_class
```
![table of survival vs sex and passenger class](https://github.com/user-attachments/assets/78bd8863-0759-47a5-8cec-e64a88be255c)

```python
#table of survival vs cabin
survived_cabin=pd.crosstab(index=titanic_train["Survived"],
                        columns=titanic_train["Cabin"])
survived_cabin.index=["Died","Survived"]
survived_cabin
```
![table of survival vs cabin](https://github.com/user-attachments/assets/8c7d2a0a-ff5c-4540-8d5e-ac62351267e9)
```

```python
#table of survival vs family
survived_family=pd.crosstab(index=titanic_train["Survived"],
                        columns=titanic_train["Family"])
survived_family.index=["Died","Survived"]
survived_family
```
![table of survival vs family](https://github.com/user-attachments/assets/997179c2-ec14-4714-971a-ad916d101b96)

```python
#table of survival vs age
survived_age=pd.crosstab(index=titanic_train["Survived"],
                        columns=titanic_train["AgeGroup"])
survived_age.index=["Died","Survived"]
survived_age
```
![table of survival vs age](https://github.com/user-attachments/assets/79d3a3be-3a60-4af2-93b1-84e217135300)

### Results/Findings
1. The number of females and males on board was 314 and 577, totaling 891 passengers.
2. Total number of passengers that died is 549 which is 62% and for passengers that survived is 342 which is 38%.
3. The class of passengers largely influenced survival as 76% of the third class passengers died followed by second class with 53% death and the least being first class with just 37% death. For their survival records, the first class has the highest with 63% survival, followed by the second class with 47% and the least being the third class with 24%.
4. Further analysis of passenger class with sex shows that the sex of the passengers in each passenger class influenced survival. In the first class, out of 94 females, only 3 died while 91 survived, and out of 122 males, 77 died while only 45 survived. This trend follows for both second class and third class too.
5. The survival rate for the cabins was not of much help as a very high number of passenger cabins was not recorded. However, from the data provided for the cabin, we could see that cabins B and C recorded the highest number of deaths, and the same B and C had the highest survival.
6. Passengers with the smaller family experienced more survival. Passengers with no family had the highest survival, followed by passengers with 1 and 2 families. Passengers with over 5 Familes experienced little or no survival.
7. Passengers who are adults between the ages 18 to 34 recorded more deaths and survival, while passengers who are 60 years and above recorded the least deaths and survival. This is perhaps because there were more passengers between the ages 18 to 34 on board and fewer passengers who were over 60.

### Key Challenges
1. Missing data in the age and cabin columns affecting accurate analysis.
2. Bias in survival rate caused by favoring women and children skewed survival pattern.

### Conclusion
Analyzing the Titanic disaster provides valuable insights into the factors influencing survival during emergencies. These findings are capable of informing future safety protocols and disaster response strategies in maritime and other transportation industries.

### References
1. [ChatGPT](https://chat.openai.com)
2. [Kaggle Notebook](https://www.kaggle.com)   


