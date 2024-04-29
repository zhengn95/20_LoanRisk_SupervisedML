# Evaluating Loan Risk using Supervised Machine Learning  
In this repository, various techniques of Supervised Machine Learning were used to evaluate a model based on loan risk.  In this Loan Risk repo, Python libraries sci-kit learn for ML, pandas, and numpy were used to implement these algorithms, and build and deploying machine learning models.  
  
Please follow the links for documentation of the libraries:
Sci-kit learn
Pandas
Numpy

### Introduction: Supervised Machine Learning  
### Overview of Analysis 
**Purpose:** A dataset of historical lending activity was used from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers. See the Jupyter notebook `credit_risk_classification.ipynb` for the code.  
  
**Targets & Features:** The dataset contains several columns of borrower financial and loan information including loan size, interest rate, income, debt-to-income ratio, number of accounts, derogatory marks, total debt, and loan status. We are predicting the creditworthiness of borrowers by analyzing the loan status (target) of past borrowers and seeing of the loan was healthy or at high risk for default based on their other loan and financial information (features).

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
  * **Accuracy:** is how often the model is correct —-the ratio of correctly predicted observations to the total number of observations. The logistic regression model found that the overall accuracy of the testing and training model is 0.99, indicating that it correctly predicts loan approval or denial about 99% of the time.  
  * **Precision:** is the ratio of correctly predicted positive observations to the total predicted positive observations. (i.e., of all of the actual healthy loans, how many were actually healthy loans). High precision relates to low false-positives.  
  
The logistic regression model found that healthy loans had a precision of 1.00 for the testing and training model. This means that when the model predicts a loan will be healthy, it is correct 100% of the time.  
  
The logistic regression model found that high-risk loans had a precision of 0.86 for healthy loans and 0.85 for high-risk loans. This indicates that when the model predicts a loan will be high-risk for default, it is correct 86%% of the time in our training dataset and 85% of the time in our testing dataset.  

  * **Recall:** is the ratio of correctly predicted positive observations to all predicted observations for that class. (i.e., of all of the actual healthy loans, how many were correctly classified as healthy loans). High recall correlates to a more comprehensive output and a low false negative rate.  
In the training, we found a recall of 1.00 for class 0, which means that the model correctly identifies all instances of healthy loans. For class 1, the recall is 0.90, indicating that the model correctly identifies about 90% of the loans at high-risk for default. In the testing model, we found that the recall for class 0 is 0.99, indicating that the model correctly identifies about 99% of the healthy loans. For class 1, the recall is 0.91, meaning that the model correctly identifies about 91% of the loans that are high-risk for default.

* **The F1-score:** is the harmonic mean of precision and recall. It gives you the balance between precision and recall. It's useful when you have imbalanced classes (which seems to be the case here). This will not be discussed in our results section, please see the code for more details.
  
### Summary  
The model is very good at predicting healthy loans and decently good at predicting high-risk loans. Looking at the two classification reports for the training and test data, it looks as if model performance declined--albeit very slightly--on the test data. This is to be expected: this is how well the model is performing on data that the model hasn't seen before. If we're still getting strong precision and recall on the test dataset, this is a good indication of how well the model is likely to perform in real life.  
  
Based on the results from both the testing and training datasets, the logistic regression model performs quite well in predicting both the "healthy loan" (label 0) and "high-risk loan" (label 1) categories, but with some variation.  
  
**For the "healthy loan" category (label 0):**
* The precision for this category is consistently high, indicating that when the model predicts a loan as "healthy," it is correct almost all of the time.
* The recall is also high, indicating that the model effectively captures most of the actual "healthy loans" in the dataset.
* Both precision and recall are consistently above 0.99, suggesting that the model performs very well in identifying "healthy loans."
  
**For the "high-risk loan" category (label 1):**
* The precision is slightly lower compared to the "healthy loan" category but still relatively high. This means that when the model predicts a loan as "high-risk," it is correct most of the time.  
* The recall is also slightly lower compared to the "healthy loan" category but still reasonably high. This indicates that the model captures a significant portion of the actual "high-risk loans" in the dataset.
* Although precision and recall are slightly lower for the "high-risk loan" category compared to the "healthy loan" category, they are still at acceptable levels, with precision around 0.85 and recall around 0.91 in our testing dataset.
  
**In summary** the logistic regression model predicts both "healthy loans" and "high-risk loans" quite well, with high precision and recall scores for both categories. However, there is a slight drop in performance for the "high-risk loan" category compared to the "healthy loan" category, as reflected in slightly lower precision and recall scores. Nonetheless, the model demonstrates effectiveness in identifying both types of loans, making it suitable for predicting loan creditworthiness.

**Recommendation:**
Based on the results, the logistic regression model seems to be a suitable choice for predicting loan creditworthiness. Its high accuracy, precision, and recall make it a reliable model for identifying both "healthy loans" and "high-risk loans." However, if there are other models available for comparison, it's worth evaluating their performance as well to ensure that the logistic regression model is indeed the best choice for the specific problem and dataset.  
  
The choice of model may depend on the specific requirements and priorities of the problem. For instance, if it's more critical to accurately predict high-risk loans (label 1), models with higher recall for this class might be preferred, even if it comes at the cost of slightly lower precision or accuracy.  
  
In conclusion, the logistic regression model performs well in predicting borrower creditworthiness based on the provided metrics. However, further evaluation and comparison with alternative models could provide additional insights into model selection, particularly considering the specific goals and priorities of the problem at hand.

