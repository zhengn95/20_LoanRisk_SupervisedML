# Evaluating Loan Risk using Supervised Machine Learning  
In this repository, various techniques of Supervised Machine Learning were used to evaluate a model based on loan risk.  

### Introduction: Supervised Machine Learning  
### Overview of Analysis 
**Purpose:** A dataset of historical lending activity was used from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers. See the Jupyter notebook `credit_risk_classification.ipynb` for the code.  
  
**Targets & Features:** The dataset contains several columns of borrower financial and loan information including loan size, interest rate,	income, debt-to-income ratio, number of accounts,	derogatory marks,	total debt, and	loan status. We are predicting the creditworthiness of borrowers by analyzing the historical if past borrowers' loan status (target) was approved or denied based on their other loan and financial information (features).

**Preprocessing:**
* Step 1: Clean the data (in this case, the dataset was pretty clean to start)
* Step 2: Assign `loan_status` to y, our target variable. Assigned all other columns as features to X.
* Step 3: Split the Data into Training and Testing Sets using `train_test_split` and assign a random state of 1.
   
**Method Used:**
* Logistic Regression Model with the Original Data

**Training & Testing:**
* With our dataset split into training and testing, we can now make predictions using our training dataset and then make predictions using our testing dataset.
  * By holding out a subset of data (testing dataset), we can measure the performance of our model and parameter selections with a subset of data not used to train the model.
* Generated a confusion matrix for the testing & training datasets using the `confusion_matrix()` function from the `scikit-learn` library.
* Created and saved the testing classification report to see how well the logistic regression model predicts both the 0 (healthy loan) and 1 (high-risk loan). See the results section for the answer based on the Machine Learning Model used.

### Results  
  * **Accuracy:** is how often the model is correct —-the ratio of correctly predicted observations to the total number of observations. The logistic regression model found a 0.99 for the test and training dataset. 
  * **Precision:** is the ratio of correctly predicted positive observations to the total predicted positive observations. (i.e., of all the samples classified as having diabetes, how many actually have diabetes?). High precision relates to low false-positives. The logistic regression model found that healthy loans 
  * **Recall:** is the ratio of correctly predicted positive observations to all predicted observations for that class. (i.e., of all of the actual diabetes samples, how many were correctly classified as having diabetes?)
High recall correlates to a more comprehensive output and a low false negative rate.


### Summary  
Question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?
Answer: The model is very good at predicting healthy loans and decently good at predicting high risk loans.
Looking at the two classification reports for the training and test data, it looks as if model performance declined--albeit very slightly--on the test data. This is to be expected: this is how well the model is performing on data that the model hasn't seen before. If we're still getting strong precision and recall on the test dataset, this is a good indication of how well the model is likely to perform in real life.


Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )  
