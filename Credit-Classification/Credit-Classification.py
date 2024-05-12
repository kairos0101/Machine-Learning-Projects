# Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold,GridSearchCV,train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report,pair_confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

df = pd.read_csv("credit_rating.csv")
df = df.drop(['S.No', 'S.No', 'Job'], axis=1)
df['Credit classification'] = df['Credit classification'].map({' good.': 1, ' bad.': 0})
categorical_columns = ['CHK_ACCT', 'History',"Purpose of credit", 'Balance in Savings A/C', 'Employment', 'Marital status', 'Co-applicant','Real Estate','Other installment','Residence','Phone','Foreign']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
df_filtered=df

X = df_filtered.drop('Credit classification', axis=1)
y = df_filtered['Credit classification']
X_train,X_test,y_train,y_test=train_test_split(X,y)

###
pipeline_one = Pipeline([
    ('scaler', StandardScaler()),  # Scale data
    ('poly_features', PolynomialFeatures(degree=2)),  # Generate polynomial and interaction features
    ('svm', SVC(kernel='rbf', class_weight='balanced', random_state=69))])

param_grid = {'svm__C': [0.1, 1, 10], 'svm__gamma': ['scale', 'auto', 0.1, 1, 10]}
grid_search = GridSearchCV(pipeline_one, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

y_pred = grid_search.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))
print("Best parameters for Pipeline_1:", grid_search.best_params_) 
print("Best score for Pipeline_1:", grid_search.best_score_)   
p1=grid_search.best_estimator_
###

###
pipeline_two = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling
    ('poly_features', PolynomialFeatures(degree=2)),  # Create polynomial features
    ('svm', SVC(class_weight='balanced', random_state=69))
])
param_grid = {
    'svm__C': [0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'svm__gamma': [0.01, 0.1, 1, 10, 'scale', 'auto'],  # Kernel coefficient
    'svm__kernel': ['rbf', 'poly', 'sigmoid']  # Types of kernels
}
grid_search = GridSearchCV(pipeline_two, param_grid, cv=5, scoring='accuracy') #, verbose=10)
grid_search.fit(X_train, y_train)

y_pred = grid_search.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))
print("Best parametersfor Pipeline_2:", grid_search.best_params_)
print("Best score for Pipeline_2:", grid_search.best_score_) 
p2=grid_search.best_estimator_
for degree in [1, 2, 3]:
    pipeline_two.set_params(poly_features__degree=degree)
    pipeline_two.fit(X_train, y_train)
    y_pred = pipeline_two.predict(X_test)
    print(f"Results for degree {degree}:")
    print(classification_report(y_test, y_pred))


###
pipeline_three = Pipeline([
    ('scaler', StandardScaler()),  # Feature scaling is crucial for SVM
    ('poly_features', PolynomialFeatures(degree=1)),  # Using degree 1 as it performed best
    ('svm', SVC(class_weight='balanced', random_state=42))
])

# More fine-grained parameter grid
param_grid = {
    'svm__C': [0.5, 1, 2, 5,10,20,30,50],  # Explore around the best previously found C
    'svm__gamma': ['scale', 'auto', 0.01, 0.1],  # Narrow down the gamma range based on prior results
    'svm__kernel': ['rbf']  # Stick with the RBF kernel since it performed well
}

grid_search = GridSearchCV(pipeline_three, param_grid, cv=StratifiedKFold(5), scoring='accuracy')
grid_search.fit(X_train, y_train)

# Predict and evaluate
y_pred = grid_search.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))
print("Best parameters for Pipeline_3:", grid_search.best_params_) 
print("Best score for Pipeline_3:", grid_search.best_score_) 
p3=grid_search.best_estimator_
###

###
svc_l=SVC(C=1,kernel='linear')
svc_r=SVC(C=1,kernel='rbf')
svc_p=SVC(C=0.1,kernel='poly')
svc_s=SVC(C=0.1,kernel='sigmoid')
###

###
param_grid_lr = {'penalty': ['l1'], 'C': [1], 'solver': ['saga']}
grid_search = GridSearchCV(LogisticRegression(max_iter=500), param_grid_lr, cv=5) #,verbose=-1)
grid_search.fit(X_train, y_train)

y_pred = grid_search.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))
print("Best parameters for LR:", grid_search.best_params_) 
print("Best score for LR:", grid_search.best_score_) 
lr=grid_search.best_estimator_
###

###
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5) #,verbose=-1)
grid_search.fit(X_train, y_train)

y_pred = grid_search.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))
print("Best parameters for KNN:", grid_search.best_params_) 
print("Best score for KNN:", grid_search.best_score_) 
knn=grid_search.best_estimator_
###

###
param_grid_dt = {'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5) #, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

y_pred = grid_search.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))
print("Best parameters for DCC:", grid_search.best_params_) 
print("Best score for DCC:", grid_search.best_score_) 
dcc=grid_search.best_estimator_
###

###
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant','adaptive'],
    'max_iter': [500]}
grid_search = GridSearchCV(MLPClassifier(random_state=42), param_grid_mlp, cv=5) #, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

y_pred = grid_search.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))
print("Best parameters for MLPC:", grid_search.best_params_) 
print("Best score for MLPC:", grid_search.best_score_) 
mlpc=grid_search.best_estimator_
###

###
param_grid_nb = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
grid_search = GridSearchCV(GaussianNB(), param_grid_nb, cv=5) #, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

y_pred = grid_search.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))
print("Best parameters for GNB:", grid_search.best_params_)
print("Best score for GNB:", grid_search.best_score_) 
gnb=grid_search.best_estimator_
###

p1.fit(X_train,y_train)
p2.fit(X_train,y_train)
p3.fit(X_train,y_train)
svc_l.fit(X_train,y_train)
svc_r.fit(X_train,y_train)
svc_p.fit(X_train,y_train)
svc_s.fit(X_train,y_train)
lr.fit(X_train, y_train)
knn.fit(X_train,y_train)
dcc.fit(X_train,y_train)
mlpc.fit(X_train,y_train)
gnb.fit(X_train,y_train)

print("P1:      ", p1.score(X_test, y_test))
print("P2:      ", p2.score(X_test, y_test))
print("P3:      ", p3.score(X_test, y_test))
print("SVC_L:   ", svc_l.score(X_test,y_test))
print("SVC_R:   ", svc_r.score(X_test,y_test))
print("SVC_P:   ", svc_p.score(X_test,y_test))
print("SVC_S:   ", svc_s.score(X_test,y_test))
print("LR:      ", lr.score(X_test, y_test))
print("KNN:     ", knn.score(X_test,y_test))
print("DCC:     ", dcc.score(X_test,y_test))
print("MLPC:    ", mlpc.score(X_test,y_test))
print("GNB:     ", gnb.score(X_test,y_test))
