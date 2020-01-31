# ML Project - Supervised Learning 
### Insurance Data Set
<a id= section1> </a>
### 1. Problem Statement

The goal is to __Predict Whether__ we can provide __Insurance to the applicant__ using different __Supervised Algorithm.__

### 2. Data Loading and Description
- The Dataset consists of the information about the applicatant like mentioned below table
- The Dataset comprises of __46k observations with 128 columns__


| Column Name| Description                                                                                          |
| ---------- | -----------------------------------------------------------------------------------------------------|
| Id			| A unique identifier associated with an application.                                                    |
| Product_Info_1-7| A set of normalized variables relating to the product applied for                                 |
| Ins_Age		 | Normalized age of applicant                                                                      |
| Ht			     | Normalized height of applicant                                                                   |
| Wt			     | Normalized weight of applicant                                                                   |
| BMI			 | Normalized BMI of applicant                                                                      |
| Employment_Info_1-6 |	A set of normalized variables relating to the employment history of the applicant.          |
| InsuredInfo_1-6     |	A set of normalized variables providing information about the applicant.                    |
| Insurance_History_1-9|	A set of normalized variables relating to the insurance history of the applicant.           |
| Family_Hist_1-5      |	A set of normalized variables relating to the family history of the applicant.              |
| Medical_History_1-41 |	A set of normalized variables relating to the medical history of the applicant.             |
| Medical_Keyword_1-48 |	A set of dummy variables relating to the presence of/absence of a medical keyword being associated with the application.|
| Response		      |This is the target variable, an ordinal variable relating to the final decision associated with an application |


### The following variables are all categorical (nominal):

Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41


### The following variables are continuous:

Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5


### The following variables are discrete:

Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32

Medical_Keyword_1-48 are dummy variables.


## 3. Preprocessing the data
- Dealing with missing values<br/>
    - Dropping/Replacing missing entries 
    - Replacing missing NaN values of  with mean values.
    - Dropped column  __ "Employement_Info_6"__ This has lot of null values 
    - Dropped column  __ "Insurance_History_5" __ This has 50% null values
    - Dropped column  __ "Family_Hist_2"__ This has 50% null values
    - Dropped column  __ "Family_Hist_3"__ This has 60% null values
    - Dropped column  __ "Family_Hist_5"__ This has 70% null values<br>
	
- Grouped Insurance entire data set into 2 Categories
    - Category 1. __Insurance_cont__ of all __continous__ data
    - Category 2. __Insurance_cat__ of all __Nominal__ data
	
	
### 4.0 EDA Observation
- We can observe that we have most of the values as __Positively__ correlation
- We can see a strong correlation between __BMI & Weight__ with __0.85__
- We can see a Good correlation between __BMI & height __ with __0.61__
- Others features are not having that much strong correlation
	
### 5.0 Linear Regression Algorithm applied & Observations
- Using __MSE__ & __RMSE__ method we observe the accurancy difference between train and Test data is very less 
- Both the model predictions are similar in nature 

### 6.0 Logistic Regression Algorithm Applied & Observation
- By using model to predict it has an accuracy score of __0.36__
- By using Proba method to predict with a value of  __0.75__ it gives an accuracy of __0.0__
- By using Proba method with a value of __.10__ it gives an accuracy of __0.058__

### 7.0 Decision Tree Algorithm Applied & Observation
- We got an accuracy of __0.31__ when we use Basemodel prediction 
- we see an improvement when we use GridsearchCV the accuracy has improved from __0.31__ To __0.35__

### 8.0 RandomForest Algorithm Applied & Observation
- When we use basemodel Ramdomforest we got an accuracy score of __0.33__
- When we use RamdomsearchCV we got an accuracy score of __0.32__

### 9.0 Conclusion & Future Action Summary
- After Analysing the data and applying different algorithms like __Linear Regression__,__Logistic Regression__,__DecisionTree__,and __RandomForest__
- We found when we apply __Logistic Regression Algorithm __ we got a high accuracy score of __0.36__
- We found when we apply __"DeicsionTree" Alogrithm__ we got a high accuracy score of __0.35__
- We have to fine tune further on the parameters to check whether we can improve the accuracy score
- We need to discuss with business and understand the importance of all the fields and if there is a possiblity, we can drop few more fields and then try to apply algorithm to improve the accuracy score. 


