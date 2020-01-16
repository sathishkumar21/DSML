#!/usr/bin/env python
# coding: utf-8

# <img src="https://github.com/insaid2018/Term-1/blob/master/Images/INSAID_Full%20Logo.png?raw=true" width="240" height="360" />
# 
# # LINEAR REGRESSION

# ## Table of Content
# 
# 1. [Problem Statement](#section1)<br>
# 2. [Data Loading and Description](#section2)<br>
# 3. [Exploratory Data Analysis](#section3)<br>
# 4. [Introduction to Linear Regression](#section4)<br>
#     - 4.1 [Linear Regression Equation with Errors in consideration](#section401)<br>
#         - 4.1.1 [Assumptions of Linear Regression](#sectionassumptions)<br>
#     - 4.2 [Preparing X and y using pandas](#section402)<br>
#     - 4.3 [Splitting X and y into training and test datasets](#section403)<br>
#     - 4.4 [Linear regression in scikit-learn](#section404)<br>
#     - 4.5 [Interpreting Model Coefficients](#section405)<br>
#     - 4.3 [Using the Model for Prediction](#section406)<br>
# 5. [Model evaluation](#section5)<br>
#     - 5.1 [Model evaluation using metrics](#section501)<br>
#     - 5.2 [Model Evaluation using Rsquared value.](#section502)<br>
# 6. [Feature Selection](#section6)<br>
# 7. [Handling Categorical Features](#section7)<br>

# <a id=section1></a>

# ## 1. Problem Statement
# 
# __Sales__ (in thousands of units) for a particular product as a __function__ of __advertising budgets__ (in thousands of dollars) for _TV, radio, and newspaper media_. Suppose that in our role as __Data Scientist__ we are asked to suggest.
# 
# - We want to find a function that given input budgets for TV, radio and newspaper __predicts the output sales__.
# 
# - Which media __contribute__ to sales?
# 
# - Visualize the __relationship__ between the _features_ and the _response_ using scatter plots.

# <a id=section2></a>

# ## 2. Data Loading and Description
# 
# The adverstising dataset captures sales revenue generated with respect to advertisement spends across multiple channles like radio, tv and newspaper.
# - TV        - Spend on TV Advertisements
# - Radio     - Spend on radio Advertisements
# - Newspaper - Spend on newspaper Advertisements
# - Sales     - Sales revenue generated

# __Importing Packages__

# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn import metrics

import numpy as np

# allow plots to appear directly in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Importing the Dataset

# In[4]:


data = pd.read_csv('https://raw.githubusercontent.com/insaid2018/Term-2/master/CaseStudy/Advertising.csv', index_col=0)
data.head()


# What are the **features**?
# - TV: advertising dollars spent on TV for a single product in a given market (in thousands of dollars)
# - Radio: advertising dollars spent on Radio
# - Newspaper: advertising dollars spent on Newspaper
# 
# What is the **response**?
# - Sales: sales of a single product in a given market (in thousands of widgets)

# <a id=section3></a>

# ## 3. Exploratory Data Analysis

# In[44]:


data.shape


# In[45]:


data.info()


# In[46]:


data.describe()


# There are 200 **observations**, and thus 200 markets in the dataset.

# __Distribution of Features__

# In[7]:


f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)                                      # Set up the matplotlib figure
sns.despine(left=True)

sns.distplot(data.sales, color="b", ax=axes[0, 0])

sns.distplot(data.TV, color="r", ax=axes[0, 1])

sns.distplot(data.radio, color="g", ax=axes[1, 0])

sns.distplot(data.newspaper, color="m", ax=axes[1, 1])


# __Observations__<br/>
# _Sales_ seems to be __normal distribution__. Spending on _newspaper advertisement_ seems to be __right skewed__. Most of the spends on _newspaper_ is __fairly low__ where are spend on _radio and tv_ seems be __uniform distribution__. Spends on _tv_ are __comparatively higher__ then spend on _radio and newspaper_.

# ### Is there a relationship between sales and spend various advertising channels?

# In[10]:


JG1 = sns.jointplot("newspaper", "sales", data=data, kind='reg')
JG2 = sns.jointplot("radio", "sales", data=data, kind='reg')
JG3 = sns.jointplot("TV", "sales", data=data, kind='reg')
#subplots migration
f = plt.figure()
for J in [JG1, JG2,JG3]:
    for A in J.fig.axes:
        f._axstack.add(f._make_key(A), A)
        


# __Observation__<br/>
# _Sales and spend on newpaper_ is __not__ highly correlaed where are _sales and spend on tv_ is __highly correlated__.

# ### Visualising Pairwise correlation

# In[49]:


sns.pairplot(data, size = 2, aspect = 1.5)


# In[50]:


sns.pairplot(data, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', size=5, aspect=1, kind='reg')


# __Observation__
# 
# - Strong relationship between TV ads and sales
# - Weak relationship between Radio ads and sales
# - Very weak to no relationship between Newspaper ads and sales
# 
# 

# ### Calculating and plotting heatmap correlation

# In[51]:


data.corr()


# In[52]:


sns.heatmap( data.corr(), annot=True );


# __Observation__
# 
# - The diagonal of the above matirx shows the auto-correlation of the variables. It is always 1. You can observe that the correlation between __TV and Sales is highest i.e. 0.78__ and then between __sales and radio i.e. 0.576__.
# 
# - correlations can vary from -1 to +1. Closer to +1 means strong positive correlation and close -1 means strong negative correlation. Closer to 0 means not very strongly correlated. variables with __strong correlations__ are mostly probably candidates for __model builing__.
# 

# <a id=section4></a>

# ## 4. Introduction to Linear Regression
# 
# __Linear regression__ is a _basic_ and _commonly_ used type of __predictive analysis__.  The overall idea of regression is to examine two things: 
# - Does a set of __predictor variables__ do a good job in predicting an __outcome__ (dependent) variable?  
# - Which variables in particular are __significant predictors__ of the outcome variable, and in what way they do __impact__ the outcome variable?  
# 
# These regression estimates are used to explain the __relationship between one dependent variable and one or more independent variables__.  The simplest form of the regression equation with one dependent and one independent variable is defined by the formula :<br/>
# $y = \beta_0 + \beta_1x$
# 
# ![image.png](attachment:image.png)
# 
# What does each term represent?
# - $y$ is the response
# - $x$ is the feature
# - $\beta_0$ is the intercept
# - $\beta_1$ is the coefficient for x
# 
# 
# Three major uses for __regression analysis__ are: 
# - determining the __strength__ of predictors,
#     - Typical questions are what is the strength of __relationship__ between _dose and effect_, _sales and marketing spending_, or _age and income_.
# - __forecasting__ an effect, and
#     - how much __additional sales income__ do I get for each additional $1000 spent on marketing?
# - __trend__ forecasting.
#     - what will the __price of house__ be in _6 months_?

# <a id=section401></a>

# ### 4.1 Linear Regression Equation with Errors in consideration
# 
# While taking errors into consideration the equation of linear regression is: 
# ![image.png](attachment:image.png)
# Generally speaking, coefficients are estimated using the **least squares criterion**, which means we are find the line (mathematically) which minimizes the **sum of squared residuals** (or "sum of squared errors"):
# 

# What elements are present in the diagram?
# - The black dots are the **observed values** of x and y.
# - The blue line is our **least squares line**.
# - The red lines are the **residuals**, which are the distances between the observed values and the least squares line.
# ![image.png](attachment:image.png)
# 

# How do the model coefficients relate to the least squares line?
# - $\beta_0$ is the **intercept** (the value of $y$ when $x$ = 0)
# - $\beta_1$ is the **slope** (the change in $y$ divided by change in $x$)
# 
# Here is a graphical depiction of those calculations:
# ![image.png](attachment:image.png)

# <a id = sectionassumptions></a>

# #### 4.1.1 Assumptions of Linear Regression

# 1. There should be a linear and additive relationship between dependent (response) variable and independent (predictor) variable(s). A linear relationship suggests that a change in response Y due to one unit change in X¹ is constant, regardless of the value of X¹. An additive relationship suggests that the effect of X¹ on Y is independent of other variables.
# 2. There should be no correlation between the residual (error) terms. Absence of this phenomenon is known as Autocorrelation.
# 3. The independent variables should not be correlated. Absence of this phenomenon is known as multicollinearity.
# 4. The error terms must have constant variance. This phenomenon is known as homoskedasticity. The presence of non-constant variance is referred to heteroskedasticity.
# 5. The error terms must be normally distributed.

# <a id=section402></a>

# ### 4.2 Preparing X and y using pandas

# - __Standardization__. <br/>
# Standardize features by removing the _mean_ and scaling to _unit standard deviation_.

# In[18]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(data)
data1 = scaler.transform(data)


# In[6]:


data = pd.DataFrame(data1)
data.head()


# In[7]:


data.columns = ['TV','radio','newspaper','sales']
data.head()


# In[8]:


feature_cols = ['TV', 'radio', 'newspaper']                # create a Python list of feature names
X = data[feature_cols]                                     # use the list to select a subset of the original DataFrame-+


# - Checking the type and shape of X.

# In[9]:


print(type(X))
print(X.shape)


# In[10]:


y = data.sales
y.head()


# - Check the type and shape of y

# In[11]:


print(type(y))
print(y.shape)


# <a id=section403></a>

# ### 4.3 Splitting X and y into training and test datasets.

# In[20]:


from sklearn.model_selection import train_test_split

def split(X,y):
    return train_test_split(X, y, test_size=0.20, random_state=1) # Random state = first time random value after that it takes the same value


# In[21]:


X_train, X_test, y_train, y_test=split(X,y)
print('Train cases as below')
print('X_train shape: ',X_train.shape)
print('y_train shape: ',y_train.shape)
print('\nTest cases as below')
print('X_test shape: ',X_test.shape)
print('y_test shape: ',y_test.shape)


# - Import the model class
#  - Call the model and create object
# - Do the .fit by passing (x_train,Y-Train) and get model trained
# - to model predict by passing x_test and get y_predict
# - Calculate Accuracy by comparing (Y_predict,Y_Test)
# 
# 

# <a id=section404></a>

# ### 4.4 Linear regression in scikit-learn

# To apply any machine learning algorithm on your dataset, basically there are 4 steps:
# 1. Load the algorithm
# 2. Instantiate and Fit the model to the training dataset
# 3. Prediction on the test set
# 4. Calculating Root mean square error 
# The code block given below shows how these steps are carried out:<br/>
# 
# ``` from sklearn.linear_model import LinearRegression
#     linreg = LinearRegression()
#     linreg.fit(X_train, y_train) 
#     RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))```

# In[22]:


def linear_reg( X, y, gridsearch = False):
    
    X_train, X_test, y_train, y_test = split(X,y)
    
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()
    
    if not(gridsearch):
        linreg.fit(X_train, y_train) 

    else:
        from sklearn.model_selection import GridSearchCV
        parameters = {'normalize':[True,False], 'copy_X':[True, False]}
        linreg = GridSearchCV(linreg,parameters, cv = 10,refit = True)
        linreg.fit(X_train, y_train)                                                           # fit the model to the training data (learn the coefficients)
        print("Mean cross-validated score of the best_estimator : ", linreg.best_score_)  
        
        y_pred_test = linreg.predict(X_test)                                                   # make predictions on the testing set

        RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))                          # compute the RMSE of our predictions
        print('RMSE for the test set is {}'.format(RMSE_test))

    return linreg


# In[ ]:





# ### Linear Regression Model without GridSearcCV
# Note:  Linear Regression Model with GridSearcCV is implemented at Table of Contents: 8

# In[15]:


X = data[feature_cols]  
y = data.sales
linreg = linear_reg(X,y)


# <a id=section405></a>

# ### 4.5 Interpreting Model Coefficients

# In[16]:


print('Intercept:',linreg.intercept_)          # print the intercept 
print('Coefficients:',linreg.coef_)  


# Its hard to remember the order of the feature names, we so we are __zipping__ the features to pair the feature names with the coefficients

# In[65]:


feature_cols.insert(0,'Intercept')
coef = linreg.coef_.tolist()            
coef.insert(0, linreg.intercept_)       


# In[66]:


eq1 = zip(feature_cols, coef)

for c1,c2 in eq1:
    print(c1,c2)


# __y = 0.00116 + 0.7708 `*` TV + 0.508 `*` radio + 0.010 `*` newspaper__

# How do we interpret the TV coefficient (_0.77081_)
# - A "unit" increase in TV ad spending is **associated with** a _"0.7708_ unit" increase in Sales.
# - Or more clearly: An additional $1,000 spent on TV ads is **associated with** an increase in sales of 770.8 widgets.
# 
# Important Notes:
# - This is a statement of __association__, not __causation__.
# - If an increase in TV ad spending was associated with a __decrease__ in sales,  β1  would be __negative.__

# <a id=section406></a>

# ### 4.6 Using the Model for Prediction

# In[67]:


y_pred_train = linreg.predict(X_train)  


# In[68]:


y_pred_test = linreg.predict(X_test)                                                           # make predictions on the testing set


# - We need an evaluation metric in order to compare our predictions with the actual values.

# <a id=section5></a>

# ## 5. Model evaluation 

# __Error__ is the _deviation_ of the values _predicted_ by the model with the _true_ values.<br/>
# For example, if a model predicts that the price of apple is Rs75/kg, but the actual price of apple is Rs100/kg, then the error in prediction will be Rs25/kg.<br/>
# Below are the types of error we will be calculating for our _linear regression model_:
# - Mean Absolute Error
# - Mean Squared Error
# - Root Mean Squared Error

# <a id=section501></a>

# ### 5.1 Model Evaluation using __metrics.__

# __Mean Absolute Error__ (MAE) is the mean of the absolute value of the errors:
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
# Computing the MAE for our Sales predictions

# In[69]:


MAE_train = metrics.mean_absolute_error(y_train, y_pred_train)
MAE_test = metrics.mean_absolute_error(y_test, y_pred_test)


# In[70]:


print('MAE for training set is {}'.format(MAE_train))
print('MAE for test set is {}'.format(MAE_test))


# __Mean Squared Error__ (MSE) is the mean of the squared errors:
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
# 
# Computing the MSE for our Sales predictions

# In[71]:


MSE_train = metrics.mean_squared_error(y_train, y_pred_train)
MSE_test = metrics.mean_squared_error(y_test, y_pred_test)


# In[72]:


print('MSE for training set is {}'.format(MSE_train))
print('MSE for test set is {}'.format(MSE_test))


# __Root Mean Squared Error__ (RMSE) is the square root of the mean of the squared errors:
# 
# $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
# 
# Computing the RMSE for our Sales predictions

# In[73]:


RMSE_train = np.sqrt( metrics.mean_squared_error(y_train, y_pred_train))
RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))


# In[74]:


print('RMSE for training set is {}'.format(RMSE_train))
print('RMSE for test set is {}'.format(RMSE_test))


# Comparing these metrics:
# 
# - __MAE__ is the easiest to understand, because it's the __average error.__ 
# - __MSE__ is more popular than MAE, because MSE "punishes" larger errors.
# - __RMSE__ is even more popular than MSE, because RMSE is _interpretable_ in the "y" units.
#     - Easier to put in context as it's the same units as our response variable.

# <a id=section502></a>

# ### 5.2 Model Evaluation using Rsquared value.

# - There is one more method to evaluate linear regression model and that is by using the __Rsquared__ value.<br/>
# - R-squared is the **proportion of variance explained**, meaning the proportion of variance in the observed data that is explained by the model, or the reduction in error over the **null model**. (The null model just predicts the mean of the observed response, and thus it has an intercept and no slope.)
# 
# - R-squared is between 0 and 1, and higher is better because it means that more variance is explained by the model. But there is one shortcoming of Rsquare method and that is **R-squared will always increase as you add more features to the model**, even if they are unrelated to the response. Thus, selecting the model with the highest R-squared is not a reliable approach for choosing the best linear model.
# 
# There is alternative to R-squared called **adjusted R-squared** that penalizes model complexity (to control for overfitting).

# In[75]:


yhat = linreg.predict(X_train)
SS_Residual = sum((y_train-yhat)**2)
SS_Total = sum((y_train-np.mean(y_train))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1)
print(r_squared, adjusted_r_squared)


# In[76]:


yhat = linreg.predict(X_test)
SS_Residual = sum((y_test-yhat)**2)
SS_Total = sum((y_test-np.mean(y_test))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print(r_squared, adjusted_r_squared)


# <a id=section6></a>

# ## 6. Feature Selection
# 
# At times some features do not contribute much to the accuracy of the model, in that case its better to discard those features.<br/> 
# - Let's check whether __"newspaper"__ improve the quality of our predictions or not.<br/> 
# To check this we are going to take all the features other than "newspaper" and see if the error (RMSE) is reducing or not.
# - Also Applying __gridsearch__ method for exhaustive search over specified parameter values of  estimator.

# In[77]:


feature_cols = ['TV','radio']                                                          # create a Python list of feature names
X = data[feature_cols]  
y = data.sales
linreg=linear_reg(X,y,gridsearch=True)


# - _Before_ doing feature selection _RMSE_ for the test dataset was __0.271182__.<br/>
# - _After_ discarding 'newspaper' column, RMSE comes to be __0.268675__.<br/>
#     - As you can see there is __no significant improvement__ in the quality, therefore, the 'newspaper' column shouldn't be discarded. But if in some other case if there is significant decrease in the RMSE, then you must discard that feature.
# - Give a try to other __features__ and check the RMSE score for each one.

# <a id=section7></a>

# ## 7.  Handling Categorical Features
# 
# Let's create a new feature called **Area**, and randomly assign observations to be **rural, suburban, or urban** :

# In[78]:


np.random.seed(123456)                                                # set a seed for reproducibility
nums = np.random.rand(len(data))
mask_suburban = (nums > 0.33) & (nums < 0.66)                         # assign roughly one third of observations to each group
mask_urban = nums > 0.66
data['Area'] = 'rural'
data.loc[mask_suburban, 'Area'] = 'suburban'
data.loc[mask_urban, 'Area'] = 'urban'
data.head()


# We want to represent Area numerically, but we can't simply code it as:<br/>
# - 0 = rural,<br/>
# - 1 = suburban,<br/>
# - 2 = urban<br/>
# Because that would imply an **ordered relationship** between suburban and urban, and thus urban is somehow "twice" the suburban category.<br/> Note that if you do have ordered categories (i.e., strongly disagree, disagree, neutral, agree, strongly agree), you can use a single dummy variable to represent the categories numerically (such as 1, 2, 3, 4, 5).<br/>
# 
# Anyway, our Area feature is unordered, so we have to create **additional dummy variables**. Let's explore how to do this using pandas:

# In[79]:


area_dummies = pd.get_dummies(data.Area, prefix='Area')                           # create three dummy variables using get_dummies
area_dummies.head()


# However, we actually only need **two dummy variables, not three**. 
# __Why???__
# Because two dummies captures all the "information" about the Area feature, and implicitly defines rural as the "baseline level".
# 
# Let's see what that looks like:

# In[80]:


area_dummies = pd.get_dummies(data.Area, prefix='Area').iloc[:, 1:]
area_dummies.head()


# Here is how we interpret the coding:
# - **rural** is coded as  Area_suburban = 0  and  Area_urban = 0
# - **suburban** is coded as  Area_suburban = 1  and  Area_urban = 0
# - **urban** is coded as  Area_suburban = 0  and  Area_urban = 1
# 
# If this sounds confusing, think in general terms that why we need only __k-1 dummy variables__ if we have a categorical feature with __k "levels"__.
# 
# Anyway, let's add these two new dummy variables onto the original DataFrame, and then include them in the linear regression model.

# In[81]:


# concatenate the dummy variable columns onto the DataFrame (axis=0 means rows, axis=1 means columns)
data = pd.concat([data, area_dummies], axis=1)
data.head()


# In[82]:


feature_cols = ['TV', 'radio', 'newspaper', 'Area_suburban', 'Area_urban']             # create a Python list of feature names
X = data[feature_cols]  
y = data.sales
linreg = linear_reg(X,y)


# In[83]:


feature_cols.insert(0,'Intercept')
coef = linreg.coef_.tolist()
coef.insert(0, linreg.intercept_)

eq1 = zip(feature_cols, coef)

for c1,c2 in eq1:
    print(c1,c2)


# __y = - 0.00218 + 0.7691 `*` TV + 0.505 `*` radio + 0.011 `*` newspaper - 0.0311 `*` Area_suburban + 0.0418 `*` Area_urban__<br/>
# How do we interpret the coefficients?<br/>
# - Holding all other variables fixed, being a **suburban** area is associated with an average **decrease** in Sales of 0.0311 widgets (as compared to the baseline level, which is rural).
# - Being an **urban** area is associated with an average **increase** in Sales of 0.0418 widgets (as compared to rural).

# <a id=section8></a>
