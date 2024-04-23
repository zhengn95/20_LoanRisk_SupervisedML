# Evaluating Loan Risk using Supervised Machine Learning  
In this repository, various techniques of Supervised Machine Learning were used to evaluate a model based on loan risk.  

### Introduction: Supervised Machine Learning  
### Overview of Analysis 
**Purpose** A dataset of historical lending activity was used from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers. See the Jupyter notebook `credit_risk_classification.ipynb` for the code.  
  
**Targets & Features** The dataset contains several columns of borrower financial and loan information including loan size, interest rate,	income, debt-to-income ratio, 	number of accounts,	derogatory marks,	total debt, and	loan status. We are predicting the creditworthiness of borrowers by analyzing historical data to see if a borrower's loan status (target) was approved or denied based on the borrower's other loan and financial information (features).

**Preprocessing:**
* Step 1: Clean the data (in this case, the dataset was pretty clean to start)
* Step 2: Assign `loan_status` to y, our target variable.
* Step 3: Split the Data into Training and Testing Sets using `train_test_split` and assign a random state of 1.
   
**Methods Used:**
* Logistic Regression Model with the Original Data
* Decision Tree Model

**Training & Testing**
* With our dataset split into training and testing, we will now make predictions using our training dataset and then make predictions using our testing dataset.
  * By holding out a subset of data (testing dataset), we can measure the performance of our model and parameter selections with a subset of data not used to train the model.
* Generate a confusion matrix for the testing & training datasets. 
* Create and save the testing classification report to see how well the logistic regression model predicts both the 0 (healthy loan) and 1 (high-risk loan). See the results section for the answer based on the Machine Learning Model used.

### Results  
Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.
* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.

### Summary  
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )  
