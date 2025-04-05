#!/usr/bin/env python
# coding: utf-8

# Importing libraries

# In[ ]:


# importing required libraries 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc


# Loading & reviewing the dataset

# In[ ]:


# loading the dataset 
diabetes_df = pd.read_csv('Dataset_of_Diabetes.csv')


# In[ ]:


# checking dat structure
print(diabetes_df.head())


# In[ ]:


#rechecking summary stats
diabetes_df.describe()


# In[ ]:


# a summary of dataset, including data types and non-null counts
print(diabetes_df.info())


# In[ ]:


# checks for missing values
print(diabetes_df.isnull().sum())


# In[ ]:


# checks for any duplicate rows
print(diabetes_df.duplicated().sum())


# In[ ]:


# checking class balance in the dataset
print(diabetes_df['CLASS'].value_counts())


# In[ ]:


# cleaning class labels to remove spaces and make all uppercase
diabetes_df['CLASS'] = diabetes_df['CLASS'].str.strip().str.upper()


# In[ ]:


# visualising the cleaned class distribution
sns.countplot(x=diabetes_df['CLASS'])
plt.title('Class Distribution')
plt.xlabel('N = Non-Diabetic - Y = Diabetic')
plt.ylabel('Count')
plt.show()


# In[ ]:


# dropping irrelevant features that do not contribute to prediction
diabetes_df = diabetes_df.drop(columns = ['Gender','ID', 'No_Pation'])


# In[ ]:


# checking remaining features after dropping irrelevant columns
print(diabetes_df.columns)


# Preprocessing 1 : outlier handling, filtering, feature capping, encoding and correlation analysis 

# IQR Method for outliers

# In[ ]:


# handling outliers using the IQR method
for col in ['Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']:
    q1 = diabetes_df[col].quantile(0.25)
    q3 = diabetes_df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    #identifying outliers
    outliers = diabetes_df[(diabetes_df[col] < lower) | (diabetes_df[col] > upper)] 

    # printing number of outliers detected for each column
    if not outliers.empty:
       print(f'{col}: {len(outliers)} outliers')
       print('Head:\n', outliers[[col]].sort_values(by=col).head(10))  
       print('Tail:\n', outliers[[col]].sort_values(by=col).tail(10))


# Filtering rows to remove extreme values

# In[ ]:


# filtering/removing medically unrealistic values based on researched clinical thresholds
diabetes_df = diabetes_df[
    (diabetes_df['Chol'] > 0) &  
    (diabetes_df['HbA1c'] >= 1) &  
    (diabetes_df['HDL'] >= 0.5)] 


# In[ ]:


# rechecking summary stats after removing extreme values
print(diabetes_df.describe())


# Capping the extreme values 

# In[ ]:


# capping extreme values using medically informed thresholds to reduce outlier impact
col_caps = {
    'Urea' : (5,30),
    'Cr' : (45,200), 
    'HbA1c' : (None, 16),
    'Chol' : (None, 9),
    'TG' : (None, 10), 
    'HDL' : (None, 4), 
    'LDL' : (None, 7), 
    'VLDL' : (None, 10), 
    'BMI' : (None, 45)
}

#  # apply clipping to capped values
for col, (lower, upper) in col_caps.items():
    diabetes_df[col] = diabetes_df[col].clip(lower=lower, upper=upper)


# In[ ]:


# converting class labels from strings to binary integers 0 and 1
diabetes_df['CLASS'] = diabetes_df['CLASS'].replace({'N': 0 , 'Y': 1}).astype(int)

# confirming successful encoding of class
print(diabetes_df['CLASS'].unique()) 


# Plotting the correlations as a Heatmap 

# In[ ]:


# calculating feature correlations to detect multicollinearity
diabetes_corr = diabetes_df.corr()

# visualising correlation matrix as a heatmap
sns.heatmap(diabetes_corr, annot=True, fmt='.2f')
plt.title('Feature Correlations')
plt.show()


# In[ ]:


# dropping BMI for multicollinearity
diabetes_df = diabetes_df.drop(columns=['BMI'])


# Preprocessing 2: encoding, stratified train-test split, smote and feature scaling

# In[ ]:


# defining feature variables X and target variable y
X = diabetes_df.drop(columns=['CLASS'])
y = diabetes_df['CLASS']


# In[ ]:


# splitting data using stratified shuffle split to maintain class balance in train and test sets
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)


# In[ ]:


# creating train and test sets using the split indices
for train_index, test_index in strat_split.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

print('Train class distribution:\n', y_train.value_counts())
print('Test class distribution:\n', y_test.value_counts())


# Applying SMOTE to balance class

# In[ ]:


# applying smote to balance the class distribution in the training set
smote_sample = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote_sample.fit_resample(X_train, y_train)

# confirming class balance after SMOTE and verifying test set remains unchanged
print('After SMOTE (y_train_bal distribution):\n', y_train_bal.value_counts())
print('Test class distribution:\n', y_test.value_counts())


# Scaling the data

# In[ ]:


# scaling feature values to standardise the data for model training
std_scaler = StandardScaler()
X_train_scaled = std_scaler.fit_transform(X_train_bal)
X_test_scaled = std_scaler.transform(X_test)


# Training models - Logistic Regression, KNN, SVC, Naive Bayes & XGBoost

# --- Logistic Regression ---

# In[ ]:


# initialising the logistic regression model and setting regularisation parameter C to 0.01
log_reg = LogisticRegression(class_weight='balanced', C=0.01, random_state=42)


# In[ ]:


# fitting the logistic reg model to the scaled training data
log_reg.fit(X_train_scaled, y_train_bal)


# In[ ]:


# predicting using the trained logistic regression model with scaled test data
y_predict = log_reg.predict(X_test_scaled)


# In[ ]:


# evaluating the logistic regression model performance on the test set
train_acc_log = log_reg.score(X_train_scaled, y_train_bal)
test_acc_log = log_reg.score(X_test_scaled, y_test)

# print accuracies
print(f'Train Accuracy: {train_acc_log:.2f}')
print(f'Test Accuracy: {test_acc_log:.2f}')

# print the classification report for precision, recall, and F1 score
print('Logistic Regression Classification Report\n',classification_report(y_test, y_predict))


# Logistic Regression Confusion Matrix Visualised

# In[ ]:


# creating the confusion matrix for Logistic Regression
log_reg_cm = confusion_matrix(y_test, y_predict) 

# plotting confusion matrix for visual interpretation of misclassifications
plt.figure()
sns.heatmap(log_reg_cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix')
plt.show()


# --- KNN- --

# In[ ]:


# initialising the KNN model with 5 neighbors default value for simplicity
k_nearest = KNeighborsClassifier(n_neighbors=5)


# In[ ]:


# fitting the KNN model to the scaled training data
k_nearest.fit(X_train_scaled, y_train_bal)


# In[ ]:


# predicting using the trained KNN model with scaled test data
y_predict_knn = k_nearest.predict(X_test_scaled)


# In[ ]:


# evaluating KNN model performance on the test set
train_acc_knn = k_nearest.score(X_train_scaled, y_train_bal)
test_acc_knn = k_nearest.score(X_test_scaled, y_test)

# print accuracies
print(f'Train Accuracy: {train_acc_knn:.2f}')
print(f'Test Accuracy: {test_acc_knn:.2f}')

# print the classification report for precision, recall, and F1 score
print('KNN Classification Report\n', classification_report(y_test, y_predict_knn))


# KNN Confusion Matrix Visualised

# In[ ]:


# creating the confusion matrix for KNN
knn_cm = confusion_matrix(y_test, y_predict_knn)

# print the confusion matrix to see true/false positives and negatives
plt.figure()
sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN Confusion Matrix')
plt.show()


# Visualising best Ks

# In[ ]:


# creating empty lists to store accuracy and corresponding k values
acc = []
md = []

# iterating over k values from 1 to 20
for i in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=i) # initialise KNN model with current k
    knn.fit(X_train_scaled, y_train_bal)  # Train on balanced & scaled data
    acc.append(knn.score(X_test_scaled, y_test))  # test accuracy
    md.append(i)

# plotting KNN accuracy for different k values
plt.figure()
plt.plot(md, acc, label='KNN', marker='o', linestyle='-')
plt.xlabel('n_neighbors (k)')
plt.ylabel('Test Accuracy (%)')
plt.title('KNN Accuracy vs. K-Value on Diabetes Data')
plt.grid(True)
plt.show()


# --- SVC ---

# In[ ]:


# initialising the SVC model with default parameters
svc_rbf = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)


# In[ ]:


# training the SVC model on scaled training data
svc_rbf.fit(X_train_scaled, y_train_bal)


# In[ ]:


# predicting with the SVC model using the scaled test data
svc_y_predict = svc_rbf.predict(X_test_scaled)


# In[ ]:


# calculating train and test accuracy for SVC model
svc_trACC = svc_rbf.score(X_train_scaled, y_train_bal)
svc_tesACC = svc_rbf.score(X_test_scaled, y_test)

# print accuracies
print(f'Train Accuracy: {svc_trACC:.2f}')
print(f'Test Accuracy: {svc_tesACC:.2f}')


# print the classification report
print('SVC Classification Report\n', classification_report(y_test, svc_y_predict))


# In[ ]:


# creating the confusion matrix for SVC
svc_cm = confusion_matrix(y_test, svc_y_predict)

# plotting confusion matrix for visual interpretation of misclassifications
plt.figure()
sns.heatmap(svc_cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVC Confusion Matrix')
plt.show()


# --- Naïve Bayes ---

# In[ ]:


# initialising the Naive Bayes model
nb_gaussian = GaussianNB()


# In[ ]:


# training the Naive Bayes model on scaled training data
nb_gaussian.fit(X_train_scaled, y_train_bal)


# In[ ]:


# predicting with the Naive Bayes model using the scaled test data
nb_y_predict = nb_gaussian.predict(X_test_scaled)


# In[ ]:


# calculating train and test accuracy for Naive Bayes model
nb_trACC = nb_gaussian.score(X_train_scaled, y_train_bal)
nb_tesACC = nb_gaussian.score(X_test_scaled, y_test)

# print accuracies
print(f'Train Accuracy: {nb_trACC:.2f}')
print(f'Test Accuracy: {nb_tesACC:.2f}')

# print the classification report
print('Naive Bayes Classification Report\n',classification_report(y_test, nb_y_predict))


# In[ ]:


# creating the confusion matrix for Naive Bayes
nb_cm = confusion_matrix(y_test, nb_y_predict)

# plotting confusion matrix for visual interpretation of misclassifications
plt.figure()
sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Naive Bayes Confusion Matrix')
plt.show()


# --- XGBoost ---

# In[ ]:


# initialising the XGBoost model with default parameters
xgbst = XGBClassifier(objective = 'binary:logistic', random_state=42)


# In[ ]:


# training the XGBoost model on scaled training data
xgbst.fit(X_train_scaled, y_train_bal)


# In[ ]:


# predicting with the XGBoost model using the scaled test data
xgbst_y_predict = xgbst.predict(X_test_scaled)


# In[ ]:


# calculating train and test accuracy for XGBoost model
xgb_trACC = xgbst.score(X_train_scaled, y_train_bal)
xgb_tesACC = xgbst.score(X_test_scaled, y_test)

# print accuracies
print(f'Train Accuracy: {xgb_trACC:.2f}')
print(f'Test Accuracy: {xgb_tesACC:.2f}')

# print the classification report
print('XGBoost Classification Report:\n',classification_report(y_test, xgbst_y_predict))


# In[ ]:


# creating the confusion matrix for XGBoost
xgbst_cm = confusion_matrix(y_test, xgbst_y_predict)

# plotting confusion matrix for visual interpretation of misclassifications
plt.figure()
sns.heatmap(xgbst_cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('XGBoost Confusion Matrix')
plt.show()


# Hyperperameter Tuning

# --- KNN Tuning---

# In[ ]:


# defining parameter grid for KNN - range of k values
param_knn = {'n_neighbors' : list(range(2,21))}



# In[ ]:


# performing grid search with 5-fold cross-validation for KNN
k_nearest_gs = GridSearchCV(knn, param_knn, cv=5, scoring='f1', n_jobs=-1, return_train_score=True)


# In[ ]:


# fitting the grid search model on scaled training data
k_nearest_gs.fit(X_train_scaled, y_train_bal)


# In[ ]:


# retrieving the best k value from grid search
best_k_tuned = k_nearest_gs.best_params_['n_neighbors']
print(f'Best K Value: {best_k_tuned}')


# In[ ]:


# extracting the best KNN model from grid search
best_knn = k_nearest_gs.best_estimator_
print(f'Best Tuned KNN F1 score on Train Data: {k_nearest_gs.best_score_}')


# In[ ]:


# predicting with the best KNN model using the scaled test data
knn_tuned_y_predict = best_knn.predict(X_test_scaled)


# In[ ]:


# calculating train and test accuracy for tuned KNN model
knn_gs_trACC = best_knn.score(X_train_scaled, y_train_bal)
knn_gs_tesACC = best_knn.score(X_test_scaled, y_test)

# print accuracies
print(f'Train Accuracy: {knn_gs_trACC:.2f}')
print(f'Test Accuracy: {knn_gs_tesACC:.2f}')

# Tuned Classification Report
print('Tuned KNN Classification Report\n', classification_report(y_test, knn_tuned_y_predict))


# KNN Validation Curve

# In[ ]:


# extracting cross-validation results into a dataframe
results = pd.DataFrame(k_nearest_gs.cv_results_)

# gettting k values and corresponding mean train and test scores
k_values = results['param_n_neighbors']
train_scores = results['mean_train_score']
test_scores = results['mean_test_score']

# plot the validation curve for knn
plt.plot(k_values, train_scores, label='train', marker='o')
plt.plot(k_values, test_scores, label='test', marker='o')
plt.xlabel('number of neighbors (k)')
plt.ylabel('f1 score')
plt.title('validation curve for knn')
plt.legend()
plt.show()


# Hypermater Tuning SVC

# In[ ]:


# defining parameter grid for SVC - testing different C, gamma and kernel values
param_svc = {'C': [0.1, 1, 10],
                  'gamma': [0.001, 0.01, 0.1, 1, 'scale'],
                  'kernel': ['rbf', 'poly', 'sigmoid']}


# In[ ]:


# performing grid search with 5-fold cross-validation for SVC
svc_gs = GridSearchCV(SVC(class_weight='balanced'), param_svc, cv=5, scoring='f1', n_jobs=-1, return_train_score=True)


# In[ ]:


# fitting the grid search model on scaled training data
svc_gs.fit(X_train_scaled, y_train_bal)


# In[ ]:


# retrieving the best parameters from grid search
svc_best_param = svc_gs.best_params_
print(f'Best SVC Parameters: {svc_best_param}')


# In[ ]:


# extracting the best SVC model from grid search
svc_best_model = svc_gs.best_estimator_
print(f'Best SVC F1 score: {svc_best_model}')


# In[ ]:


# predicting with the best SVC model using the scaled test data
svc_tuned_y_predict = svc_best_model.predict(X_test_scaled)


# In[ ]:


# calculating train and test accuracy for tuned SVC model
svc_gs_trACC = svc_best_model.score(X_train_scaled, y_train_bal)
svc_gs_tesACC = svc_best_model.score(X_test_scaled, y_test)

# print accuracies
print(f'Train Accuracy: {svc_gs_trACC:.2f}')
print(f'Test Accuracy: {svc_gs_tesACC:.2f}')

print('Tuned SVC Classification Report\n', classification_report(y_test, svc_tuned_y_predict))


# SVC Validation Curve

# In[ ]:


# extract grid search results into a dataframe
results_svc = pd.DataFrame(svc_gs.cv_results_)

# get C values and mean test scores
c_values = results_svc['param_C']
test_scores = results_svc['mean_test_score']

# plot validation curve for svc
plt.figure()
plt.plot(c_values, test_scores, marker='o')
plt.xlabel('C value')
plt.ylabel('f1 score')
plt.title('validation curve for svc')
plt.grid(True)
plt.show()


# Hyperameter Tuning XGBoost

# In[ ]:


# defining parameter grid for XGBoost to tune tree depth, learning rate and subsampling
param_xgboost = {'n_estimators': [50, 100],
                  'max_depth': [3, 5, 7],
                  'learning_rate': [0.01, 0.1],
                  'subsample': [0.8, 1.0]}


# In[ ]:


# performing grid search with 5-fold cross-validation for XGBoost
xgboost_gs = GridSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'),
                               param_xgboost, cv=5, scoring='f1', n_jobs=-1)


# In[ ]:


# fitting the grid search model on scaled training data
xgboost_gs.fit(X_train_scaled, y_train_bal)


# In[ ]:


# retrieving the best parameters from grid search
xgboost_best_param = xgboost_gs.best_params_
print(f'Best XGBoost Parameters:\n {xgboost_best_param}')


# In[ ]:


# extracting the best XGBoost model from grid search
xgboost_best_model = xgboost_gs.best_estimator_


# In[ ]:


# predicting with the best XGBoost model using the scaled test data
xgboost_tuned_y_predict = xgboost_best_model.predict(X_test_scaled)


# In[ ]:


# calculating train and test accuracy for tuned XGBoost model
xgb_gs_trACC = xgboost_best_model.score(X_train_scaled, y_train_bal)
xgb_gs_tesACC = xgboost_best_model.score(X_test_scaled, y_test)

# print accuracies
print(f'Train Accuracy: {xgb_gs_trACC:.2f}')
print(f'Test Accuracy: {xgb_gs_tesACC:.2f}')

# print the classification report
print('Tuned XGBoost Classification Report:\n', classification_report(y_test, xgboost_tuned_y_predict))


# In[ ]:


# creating the confusion matrix for tuned XGBoost
xgboost_gs_cm = confusion_matrix(y_test, xgboost_tuned_y_predict)

# plotting confusion matrix for visual interpretation of misclassifications
plt.figure(figsize=(6, 5))
sns.heatmap(xgboost_gs_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Tuned XGBoost Confusion Matrix')
plt.show()


# F1 Score Barplot for all Models

# In[ ]:


# f1 scores for each tuned model based on final outputs
f1_scores = [0.86, 0.88, 0.86, 0.84, 0.90]
model_names = ['Logistic Regression', 'KNN', 'SVC', 'Naive Bayes', 'XGBoost']

# create bar plot for f1 scores
plt.figure()
plt.bar(model_names, f1_scores)
plt.xlabel('model')
plt.ylabel('f1 score')
plt.title('f1 scores of tuned models')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


# Comparing Models ROC and AUC 

# In[ ]:


# getting predicted probabilities or decision scores for ROC curve
log_reg_probs = log_reg.predict_proba(X_test_scaled)[:, 1]
knn_probs = best_knn.predict_proba(X_test_scaled)[:, 1] 
svc_probs = svc_best_model.decision_function(X_test_scaled)
nb_probs = nb_gaussian.predict_proba(X_test_scaled)[:, 1]
xgboost_probs = xgboost_best_model.predict_proba(X_test_scaled)[:, 1]

# storing model names and their probabilities/scores
models = {
    'Logistic Regression': log_reg_probs,
    'KNN': knn_probs,
    'SVM': svc_probs,
    'Naive Bayes': nb_probs,
    'XGBoost': xgboost_probs}

# plotting ROC curves for each model
plt.figure()
for name, probs in models.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')

# plotting baseline random classifier
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')

# formatting plot
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()

