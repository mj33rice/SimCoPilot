import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm as svm_model
from sklearn import linear_model
import warnings
from pandas.core.common import SettingWithCopyWarning
# Disable the specific warning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Get the directory containing this script
current_dir = os.path.dirname(os.path.realpath(__file__)) 
# Construct the full path to the dataset
BANK_PATH = os.path.join(current_dir, 'bankfullclean.csv')

# Load the dataset
bank = pd.read_csv(BANK_PATH)
df = pd.DataFrame(bank)
df =  df.sample(n=500, random_state=1234).reset_index(drop=True)
dummy = df[['catAge', 'job', 'marital','education','balance','day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'Fmonth', 'biDefault', 'biHousing', 'biLoan']]
dummy = pd.get_dummies(data=dummy)


# Machine Learning Classification Model
# Split the dataset into training and testing sets
def get_naive_dataset(dataset):
    dataset = dataset.sample(frac=1, random_state=1234).reset_index(drop=True)
    X = dummy
    y = dataset['biY']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = get_naive_dataset(df)
# Print the shape of the training and testing sets
print('x_train.shape: ', x_train.shape)
print('x_test.shape: ', x_test.shape)
# Print the columns of the training and testing sets
print('x_train.columns.values: ', x_train.columns.values)
print('y_train.values: ', y_train.values)
print('x_test.columns.values: ', x_test.columns.values)
print('y_test.values: ', y_test.values)

# Naive Bayes Model
gnb = GaussianNB()
gnb_pred = gnb.fit(x_train, y_train).predict(x_test)

# Result
test_df = x_test.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(gnb_pred, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])


# Confusion Matrix
confusion_mat = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix: ', confusion_mat)
# Calculate Accuracy
accuracy_naive_bayes = (confusion_mat[0][0] + confusion_mat[1][1]) / (confusion_mat[0][0] + confusion_mat[0][1] + confusion_mat[1][0] + confusion_mat[1][1])
print('accuracy_naive_bayes: ', accuracy_naive_bayes)

# Split dataset based on marital status

test_married = x_test[x_test['marital_married'] == 1]
test_single = x_test[x_test['marital_single'] == 1]
test_divorced = x_test[x_test['marital_divorced'] == 1]

# Test on single subset with Naive Bayes Model
gnb_pred_single = gnb.fit(x_train, y_train).predict(test_single)
test_df = test_single.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(gnb_pred_single, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
confusion_matrix_single = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix_single: ', confusion_matrix_single)
accuracy_naive_bayes_single = (confusion_matrix_single[0][0] + confusion_matrix_single[1][1]) / (confusion_matrix_single[0][0] + confusion_matrix_single[0][1] + confusion_matrix_single[1][0] + confusion_matrix_single[1][1])
print('accuracy_naive_bayes_single: ', accuracy_naive_bayes_single)

# Test on married subset with Naive Bayes Model
gnb_pred_married = gnb.fit(x_train, y_train).predict(test_married)
test_df = test_married.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(gnb_pred_married, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
confusion_matrix_married = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix_married: ', confusion_matrix_married)
accuracy_naive_bayes_married = (confusion_matrix_married[0][0] + confusion_matrix_married[1][1]) / (confusion_matrix_married[0][0] + confusion_matrix_married[0][1] + confusion_matrix_married[1][0] + confusion_matrix_married[1][1])
print('accuracy_naive_bayes_married: ', accuracy_naive_bayes_married)

# Test on divorced subset with Naive Bayes Model
gnb_pred_divorced = gnb.fit(x_train, y_train).predict(test_divorced)
test_df = test_divorced.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(gnb_pred_divorced, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
confusion_matrix_divorced = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix_divorced: ', confusion_matrix_divorced)
accuracy_naive_bayes_divorced = (confusion_matrix_divorced[0][0] + confusion_matrix_divorced[1][1]) / (confusion_matrix_divorced[0][0] + confusion_matrix_divorced[0][1] + confusion_matrix_divorced[1][0] + confusion_matrix_divorced[1][1])
print('accuracy_naive_bayes_divorced: ', accuracy_naive_bayes_divorced)

# Demographic Parity
# positive rate = TP + TN
PR_Naive_Bayes_married = confusion_matrix_married[0,0] +confusion_matrix_married[1,1]
print('PR_Naive_Bayes_married: ', PR_Naive_Bayes_married)
PR_Naive_Bayes_single = confusion_matrix_single[0,0] +confusion_matrix_single[1,1]
print('PR_Naive_Bayes_single: ',PR_Naive_Bayes_single)
PR_Naive_Bayes_divorced = confusion_matrix_divorced[0,0] +confusion_matrix_divorced[1,1]
print('PR_Naive_Bayes_divorced: ', PR_Naive_Bayes_divorced)

# Equalized Opportunity
# TPR = TP/TP+FN
TPR_Naive_Bayes_married = confusion_matrix_married[0,0] / (confusion_matrix_married[0,0] + confusion_matrix_married[1,0])
print('TPR_Naive_Bayes_married: ', TPR_Naive_Bayes_married)
TPR_Naive_Bayes_single = confusion_matrix_single[0,0] / (confusion_matrix_single[0,0] + confusion_matrix_single[1,0])
print('TPR_Naive_Bayes_single: ', TPR_Naive_Bayes_single)
TPR_Naive_Bayes_divorced = confusion_matrix_divorced[0,0] / (confusion_matrix_divorced[0,0] + confusion_matrix_divorced[1,0])
print('TPR_Naive_Bayes_divorced: ', TPR_Naive_Bayes_divorced)

# Equalized Odds
# TPR = TP/TP+FN
# FNR = FN/FN+TP
FNR_Naive_Bayes_married = confusion_matrix_married[1,0] / (confusion_matrix_married[1,0] + confusion_matrix_married[0,0])
print('FNR_Naive_Bayes_married: ', FNR_Naive_Bayes_married)
FNR_Naive_Bayes_single = confusion_matrix_single[1,0] / (confusion_matrix_single[1,0] + confusion_matrix_single[0,0])
print('FNR_Naive_Bayes_single: ', FNR_Naive_Bayes_single)
FNR_Naive_Bayes_divorced = confusion_matrix_divorced[1,0] / (confusion_matrix_divorced[1,0] + confusion_matrix_divorced[0,0])
print('FNR_Naive_Bayes_divorced: ', FNR_Naive_Bayes_divorced)

# Fairness Through Unwareness
x_train_unawareness = x_train.copy()
x_train_unawareness.drop('marital_married', inplace=True, axis=1)
x_train_unawareness.drop('marital_single', inplace=True, axis=1)
x_train_unawareness.drop('marital_divorced', inplace=True, axis=1)

# Fairness through unawareness
x_test_unawareness = x_test.copy()
x_test_married = x_test_unawareness[x_test_unawareness['marital_married'] == 1]
x_test_single = x_test_unawareness[x_test_unawareness['marital_single'] == 1]
x_test_divorced = x_test_unawareness[x_test_unawareness['marital_divorced'] == 1]
# Drop the marital status columns for x_test_married,
x_test_married.drop('marital_married', inplace=True, axis=1)
x_test_married.drop('marital_single', inplace=True, axis=1)
x_test_married.drop('marital_divorced', inplace=True, axis=1)
# Drop the marital status columns for x_test_single
x_test_single.drop('marital_married', inplace=True, axis=1)
x_test_single.drop('marital_single', inplace=True, axis=1)
x_test_single.drop('marital_divorced', inplace=True, axis=1)
# Drop the marital status columns for x_test_divorced
x_test_divorced.drop('marital_married', inplace=True, axis=1)
x_test_divorced.drop('marital_single', inplace=True, axis=1)
x_test_divorced.drop('marital_divorced', inplace=True, axis=1)

# Test on single set with Naive Bayes Model
gnb_pred_single = gnb.fit(x_train_unawareness, y_train).predict(x_test_single)
test_df = x_test_single.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(gnb_pred_single, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])

confusion_matrix_single = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix_single: ', confusion_matrix_single)
accuracy_naive_bayes_single = (confusion_matrix_single[0][0] + confusion_matrix_single[1][1]) / (confusion_matrix_single[0][0] + confusion_matrix_single[0][1] + confusion_matrix_single[1][0] + confusion_matrix_single[1][1])
print('accuracy_naive_bayes_single: ', accuracy_naive_bayes_single)

# Test on married set with Naive Bayes Model
gnb_pred_married = gnb.fit(x_train_unawareness, y_train).predict(x_test_married)
test_df = x_test_married.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(gnb_pred_married, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
confusion_matrix_married = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix_married: ', confusion_matrix_married)
accuracy_naive_bayes_married = (confusion_matrix_married[0][0] + confusion_matrix_married[1][1]) / (confusion_matrix_married[0][0] + confusion_matrix_married[0][1] + confusion_matrix_married[1][0] + confusion_matrix_married[1][1])
print('accuracy_naive_bayes_married: ', accuracy_naive_bayes_married)

# Test on divorced set with Naive Bayes Model
gnb_pred_divorced = gnb.fit(x_train_unawareness, y_train).predict(x_test_divorced)
test_df = x_test_divorced.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(gnb_pred_divorced, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
confusion_matrix_divorced = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix_divorced: ', confusion_matrix_divorced)
accuracy_naive_bayes_divorced = (confusion_matrix_divorced[0][0] + confusion_matrix_divorced[1][1]) / (confusion_matrix_divorced[0][0] + confusion_matrix_divorced[0][1] + confusion_matrix_divorced[1][0] + confusion_matrix_divorced[1][1])
print('accuracy_naive_bayes_divorced: ', accuracy_naive_bayes_divorced)

# Random Forest Model 
rf = RandomForestClassifier(max_depth=2, random_state=1234, class_weight='balanced')
rf_pred = rf.fit(x_train, y_train).predict(x_test)
test_df = x_test.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(rf_pred, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
confusion_mat = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix: ', confusion_mat)
# Accuracy
accuracy_rf = (confusion_mat[0][0] + confusion_mat[1][1]) / (confusion_mat[0][0] + confusion_mat[0][1] + confusion_mat[1][0] + confusion_mat[1][1])
print('accuracy_rf: ', accuracy_rf)

# Split dataset based on marital status
test_married = x_test[x_test['marital_married'] == 1]
test_single = x_test[x_test['marital_single'] == 1]
test_divorced = x_test[x_test['marital_divorced'] == 1]

# Test on single subset with Random Forest Model
rf_pred_single = rf.fit(x_train, y_train).predict(test_single)
test_df = test_single.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(rf_pred_single, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
confusion_matrix_single = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix_single: ', confusion_matrix_single)
accuracy_rf_single = (confusion_matrix_single[0][0] + confusion_matrix_single[1][1]) / (confusion_matrix_single[0][0] + confusion_matrix_single[0][1] + confusion_matrix_single[1][0] + confusion_matrix_single[1][1])
print('accuracy_rf_single: ', accuracy_rf_single)

# Test on married subset with Random Forest Model
rf_pred_married = rf.fit(x_train, y_train).predict(test_married)
test_df = test_married.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(rf_pred_married, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
confusion_matrix_married = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix_married: ', confusion_matrix_married)
accuracy_rf_married = (confusion_matrix_married[0][0] + confusion_matrix_married[1][1]) / (confusion_matrix_married[0][0] + confusion_matrix_married[0][1] + confusion_matrix_married[1][0] + confusion_matrix_married[1][1])
print('accuracy_rf_married: ', accuracy_rf_married)

# Test on divorced subset with Random Forest Model
rf_pred_divorced = rf.fit(x_train, y_train).predict(test_divorced)
test_df = test_divorced.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(rf_pred_divorced, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
confusion_matrix_divorced = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix_divorced: ', confusion_matrix_divorced)
accuracy_naive_bayes_divorced = (confusion_matrix_divorced[0][0] + confusion_matrix_divorced[1][1]) / (confusion_matrix_divorced[0][0] + confusion_matrix_divorced[0][1] + confusion_matrix_divorced[1][0] + confusion_matrix_divorced[1][1])
print('accuracy_naive_bayes_divorced: ', accuracy_naive_bayes_divorced)

# Demographic Parity
# positive rate = TP + TN
PR_rf_married = confusion_matrix_married[0,0] +confusion_matrix_married[1,1]
print('PR_rf_married: ', PR_rf_married)
PR_rf_single = confusion_matrix_single[0,0] +confusion_matrix_single[1,1]
print('PR_rf_single: ', PR_rf_single)
PR_rf_divorced = confusion_matrix_divorced[0,0] +confusion_matrix_divorced[1,1]
print('PR_rf_divorced: ', PR_rf_divorced)

# Equalized Opportunity
# TPR = TP/TP+FN
TPR_rf_married = confusion_matrix_married[0,0] / (confusion_matrix_married[0,0] + confusion_matrix_married[1,0])
print('TPR_rf_married: ', TPR_rf_married)
TPR_rf_single = confusion_matrix_single[0,0] / (confusion_matrix_single[0,0] + confusion_matrix_single[1,0])
print('TPR_rf_single: ', TPR_rf_single)
TPR_rf_divorced = confusion_matrix_divorced[0,0] / (confusion_matrix_divorced[0,0] + confusion_matrix_divorced[1,0])
print('TPR_rf_divorced: ', TPR_rf_divorced)

# Equalized Odds
# TPR = TP/TP+FN
# FNR = FN/FN+TP
FNR_rf_married = confusion_matrix_married[1,0] / (confusion_matrix_married[1,0] + confusion_matrix_married[0,0])
print('FNR_rf_married: ', FNR_rf_married)
FNR_rf_single = confusion_matrix_single[1,0] / (confusion_matrix_single[1,0] + confusion_matrix_single[0,0])
print('FNR_rf_single: ', FNR_rf_single)
FNR_rf_divorced = confusion_matrix_divorced[1,0] / (confusion_matrix_divorced[1,0] + confusion_matrix_divorced[0,0])
print('FNR_rf_divorced: ', FNR_rf_divorced)

# Fairness Through Unwareness
x_train_unawareness = x_train.copy()
x_train_unawareness.drop('marital_married', inplace=True, axis=1)
x_train_unawareness.drop('marital_single', inplace=True, axis=1)
x_train_unawareness.drop('marital_divorced', inplace=True, axis=1)

# Fairness through unawareness
x_test_unawareness = x_test.copy()
x_test_married = x_test_unawareness[x_test_unawareness['marital_married'] == 1]
x_test_single = x_test_unawareness[x_test_unawareness['marital_single'] == 1]
x_test_divorced = x_test_unawareness[x_test_unawareness['marital_divorced'] == 1]
# Drop the marital status columns for x_test_married
x_test_married.drop('marital_married', inplace=True, axis=1)
x_test_married.drop('marital_single', inplace=True, axis=1)
x_test_married.drop('marital_divorced', inplace=True, axis=1)
# Drop the marital status columns for x_test_single
x_test_single.drop('marital_married', inplace=True, axis=1)
x_test_single.drop('marital_single', inplace=True, axis=1)
x_test_single.drop('marital_divorced', inplace=True, axis=1)
# Drop the marital status columns for x_test_divorced
x_test_divorced.drop('marital_married', inplace=True, axis=1)
x_test_divorced.drop('marital_single', inplace=True, axis=1)
x_test_divorced.drop('marital_divorced', inplace=True, axis=1)

# Test on single set with Random Forest Model
rf_pred_single = rf.fit(x_train_unawareness, y_train).predict(x_test_single)
test_df = x_test_single.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(rf_pred_single, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
confusion_matrix_single = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix_single: ', confusion_matrix_single)
accuracy_rf_single = (confusion_matrix_single[0][0] + confusion_matrix_single[1][1]) / (confusion_matrix_single[0][0] + confusion_matrix_single[0][1] + confusion_matrix_single[1][0] + confusion_matrix_single[1][1])
print('accuracy_rf_single: ', accuracy_rf_single)

# Test on married set with Random Forest Model
rf_pred_married = rf.fit(x_train_unawareness, y_train).predict(x_test_married)
test_df = x_test_married.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(rf_pred_married, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
confusion_matrix_married = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix_married: ', confusion_matrix_married)
accuracy_rf_married = (confusion_matrix_married[0][0] + confusion_matrix_married[1][1]) / (confusion_matrix_married[0][0] + confusion_matrix_married[0][1] + confusion_matrix_married[1][0] + confusion_matrix_married[1][1])
print('accuracy_rf_married: ', accuracy_rf_married)

# Test on divorced set with Random Forest Model
rf_pred_divorced = rf.fit(x_train_unawareness, y_train).predict(x_test_divorced)
test_df = x_test_divorced.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(rf_pred_divorced, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
confusion_matrix_divorced = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix_divorced: ', confusion_matrix_divorced)
accuracy_rf_divorced = (confusion_matrix_divorced[0][0] + confusion_matrix_divorced[1][1]) / (confusion_matrix_divorced[0][0] + confusion_matrix_divorced[0][1] + confusion_matrix_divorced[1][0] + confusion_matrix_divorced[1][1])
print('accuracy_rf_divorced: ', accuracy_rf_divorced)

# Support Vector Machine add class_weight='balanced' is to avoid classifying all data as 0
svm = svm_model.SVC(class_weight='balanced') 
svm_pred = svm.fit(x_train, y_train).predict(x_test)

test_df = x_test.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(svm_pred, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
confusion_mat = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix: ', confusion_mat)

# Accuracy
accuracy_svm = (confusion_mat[0][0] + confusion_mat[1][1]) / (confusion_mat[0][0] + confusion_mat[0][1] + confusion_mat[1][0] + confusion_mat[1][1])
print('accuracy_svm: ', accuracy_svm)

# Split dataset based on marital status

test_married = x_test[x_test['marital_married'] == 1]
test_single = x_test[x_test['marital_single'] == 1]
test_divorced = x_test[x_test['marital_divorced'] == 1]

# Test on single subset with SVM
svm_pred_single = svm.fit(x_train, y_train).predict(test_single)
test_df = test_single.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(svm_pred_single, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
confusion_matrix_single = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix_single: ', confusion_matrix_single)
accuracy_svm_single = (confusion_matrix_single[0][0] + confusion_matrix_single[1][1]) / (confusion_matrix_single[0][0] + confusion_matrix_single[0][1] + confusion_matrix_single[1][0] + confusion_matrix_single[1][1])
print('accuracy_svm_single: ', accuracy_svm_single)

# Test on married subset with SVM
svm_pred_married = svm.fit(x_train, y_train).predict(test_married)
test_df = test_married.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(svm_pred_married, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
confusion_matrix_married = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix_married: ', confusion_matrix_married)
accuracy_svm_married = (confusion_matrix_married[0][0] + confusion_matrix_married[1][1]) / (confusion_matrix_married[0][0] + confusion_matrix_married[0][1] + confusion_matrix_married[1][0] + confusion_matrix_married[1][1])
print('accuracy_svm_married: ', accuracy_svm_married)

# Test on divorced subset with SVM
svm_pred_divorced = svm.fit(x_train, y_train).predict(test_divorced)
test_df = test_divorced.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(svm_pred_divorced, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
confusion_matrix_divorced = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix_divorced: ', confusion_matrix_divorced)
accuracy_svm_divorced = (confusion_matrix_divorced[0][0] + confusion_matrix_divorced[1][1]) / (confusion_matrix_divorced[0][0] + confusion_matrix_divorced[0][1] + confusion_matrix_divorced[1][0] + confusion_matrix_divorced[1][1])
print('accuracy_svm_divorced: ', accuracy_svm_divorced)

# Demographic Parity
# positive rate = TP + TN
PR_svm_married = confusion_matrix_married[0,0] +confusion_matrix_married[1,1]
print('PR_svm_married: ', PR_svm_married)
PR_svm_single = confusion_matrix_single[0,0] +confusion_matrix_single[1,1]
print('PR_svm_single: ', PR_svm_single)
PR_svm_divorced = confusion_matrix_divorced[0,0] +confusion_matrix_divorced[1,1]
print('PR_svm_divorced:', PR_svm_divorced)

# Equalized Opportunity
# TPR = TP/TP+FN
TPR_svm_married = confusion_matrix_married[0,0] / (confusion_matrix_married[0,0] + confusion_matrix_married[1,0])
print('TPR_svm_married: ', TPR_svm_married)
TPR_svm_single = confusion_matrix_single[0,0] / (confusion_matrix_single[0,0] + confusion_matrix_single[1,0])
print('TPR_svm_single: ', TPR_svm_single)
TPR_svm_divorced = confusion_matrix_divorced[0,0] / (confusion_matrix_divorced[0,0] + confusion_matrix_divorced[1,0])
print('TPR_svm_divorced: ', TPR_svm_divorced)

# Equalized Odds
# TPR = TP/TP+FN
# FNR = FN/FN+TP
FNR_svm_married = confusion_matrix_married[1,0] / (confusion_matrix_married[1,0] + confusion_matrix_married[0,0])
print('FNR_svm_married: ', FNR_svm_married)
FNR_svm_single = confusion_matrix_single[1,0] / (confusion_matrix_single[1,0] + confusion_matrix_single[0,0])
print('FNR_svm_single: ', FNR_svm_single)
FNR_svm_divorced = confusion_matrix_divorced[1,0] / (confusion_matrix_divorced[1,0] + confusion_matrix_divorced[0,0])
print('FNR_svm_divorced: ', FNR_svm_divorced)

# Fairness Through Unwareness
x_train_unawareness = x_train.copy()
# x_train_unawareness
x_train_unawareness.drop('marital_married', inplace=True, axis=1)
x_train_unawareness.drop('marital_single', inplace=True, axis=1)
x_train_unawareness.drop('marital_divorced', inplace=True, axis=1)

#Fairness through unawareness
x_test_unawareness = x_test.copy()
# x_test_unawareness
x_test_married = x_test_unawareness[x_test_unawareness['marital_married'] == 1]
x_test_single = x_test_unawareness[x_test_unawareness['marital_single'] == 1]
x_test_divorced = x_test_unawareness[x_test_unawareness['marital_divorced'] == 1]
# Drop the marital status columns for x_test_married
x_test_married.drop('marital_married', inplace=True, axis=1)
x_test_married.drop('marital_single', inplace=True, axis=1)
x_test_married.drop('marital_divorced', inplace=True, axis=1)
# Drop the marital status columns for x_test_single
x_test_single.drop('marital_married', inplace=True, axis=1)
x_test_single.drop('marital_single', inplace=True, axis=1)
x_test_single.drop('marital_divorced', inplace=True, axis=1)
# Drop the marital status columns for x_test_divorced
x_test_divorced.drop('marital_married', inplace=True, axis=1)
x_test_divorced.drop('marital_single', inplace=True, axis=1)
x_test_divorced.drop('marital_divorced', inplace=True, axis=1)

# Test on single set with SVM
svm_pred_single = svm.fit(x_train_unawareness, y_train).predict(x_test_single)
test_df = x_test_single.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(svm_pred_single, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
confusion_matrix_single = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix_single: ', confusion_matrix_single)
accuracy_svm_single = (confusion_matrix_single[0][0] + confusion_matrix_single[1][1]) / (confusion_matrix_single[0][0] + confusion_matrix_single[0][1] + confusion_matrix_single[1][0] + confusion_matrix_single[1][1])
print('accuracy_svm_single: ', accuracy_svm_single)

# Test on married set with SVM
svm_pred_married = svm.fit(x_train_unawareness, y_train).predict(x_test_married)
test_df = x_test_married.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(svm_pred_married, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
confusion_matrix_married = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix_married: ', confusion_matrix_married)
accuracy_svm_married = (confusion_matrix_married[0][0] + confusion_matrix_married[1][1]) / (confusion_matrix_married[0][0] + confusion_matrix_married[0][1] + confusion_matrix_married[1][0] + confusion_matrix_married[1][1])
print('accuracy_svm_married: ', accuracy_svm_married) 

# Test on divorced subset with SVM
svm_pred_divorced = svm.fit(x_train_unawareness, y_train).predict(x_test_divorced)
test_df = x_test_divorced.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(svm_pred_divorced, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
confusion_matrix_divorced = confusion_matrix(test_df['biY'], test_df['pred'])
print('confusion_matrix_divorced: ', confusion_matrix_divorced)
accuracy_svm_divorced = (confusion_matrix_divorced[0][0] + confusion_matrix_divorced[1][1]) / (confusion_matrix_divorced[0][0] + confusion_matrix_divorced[0][1] + confusion_matrix_divorced[1][0] + confusion_matrix_divorced[1][1])
print('accuracy_svm_divorced: ', accuracy_svm_divorced)

# Linear Regression model
linear_regression_model = linear_model.LinearRegression()
linear_regression_model.fit(x_train, y_train)
prediction = linear_regression_model.predict(x_test)
# Result
test_df = x_test.copy()
test_df['biY'] = y_test
test_df['pred'] = pd.Series(prediction, index=test_df.index)
test_df['accurate'] = (test_df['pred'] == test_df['biY'])
print("test_df['pred']: ", test_df['pred'])