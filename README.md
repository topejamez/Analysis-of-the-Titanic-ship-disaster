# Analysis-of-the-Titanic-ship-disaster

### Project Overview

I analyzed the Titanic ship disaster using Python and primarily focused on understanding the factors that contributed to passenger survival and identifying key insights from the available dataset to first, understand the distribution of passengers based on demographic and socio-economic factors. Secondly, identify the key factors that influenced survival rates. And lastly, draw actionable insights for safety improvements in maritime travel.

### Data Source

Kaggle is the source of the dataset in CSV file. Python programming language on Kaggle is used to clean the dataset, format, transform, group, and carry out the analysis. Libraries used include Pandas, NumPy, Matplotlib, and Scikit-learn.

### Data Cleaning/Preparation

To gain insights from a dataset, it is important to ensure the dataset is clean enough for doing that. I ensured the dataset was thoroughly cleaned to ensure the accuracy and integrity of my findings. The data cleaning and preparation processes I carried out include are shown in the codes below:

 ```
#load in some packages
%matplotlib inline

import numpy as np
import pandas as pd
import os

``
# Read the data
titanic_train = pd.read_csv("/kaggle/input/titanic/train.csv")
```

```
# Sorting out the name column
sorted(titanic_train["Name"])[0:15]
``

``
#check the dimensions
titanic_train.shape           

# Check the summary of the dataset
titanic_train.info()

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





