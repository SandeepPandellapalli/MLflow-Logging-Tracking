# conda create -p venv python==3.10         
# conda activate venv/
# pip install -r re.txt
# conda activate venv/

import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import mlflow
import mlflow.sklearn
from pathlib import Path
import os
import argparse


df = pd.read_csv('/Users/sandeep/Documents/MLFlow/Iris/iris.csv')
# os.mkdir("data/")
df.to_csv('data/red-wine-quality.csv', index=False)

X=df.iloc[:,0:4]
Y=df["variety"]
# X.head()

# mlflow.set_tracking_uri("file:///Users/sandeep/Documents/MLFlow/mlruns")

# print("The set tracking uri is ", mlflow.get_tracking_uri())

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

X_train.to_csv('data/X_train.csv')
X_test.to_csv('data/X_test.csv')
Y_train.to_csv('data/Y_train.csv')
Y_test.to_csv('data/Y_test.csv')


# MLFLOW Experment definiton
exp = mlflow.set_experiment(experiment_name = 'Iris_Exp_1')
mlflow.start_run()



print("Name: {}".format(exp.name))
print("Experiment_id: {}".format(exp.experiment_id))
print("Artifact Location: {}".format(exp.artifact_location))
print("Tags: {}".format(exp.tags))
print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
print("Creation timestamp: {}".format(exp.creation_time))

parser = argparse.ArgumentParser()
parser.add_argument("--C", type=float, required=False, default=1)
parser.add_argument("--max_iter", type=int, required=False, default=1000)
parser.add_argument("--penalty", type=str, required=False, default='l2')
parser.add_argument("--solver", type=str, required=False, default='lbfgs')

args = parser.parse_args()

C= args.C
max_iter= args.max_iter
penalty= args.penalty
solver= args.solver


# Start Run
tags = {
        'Model' : 'Logistic Regression',
        'Parameters' :'None',
        'Version' : '1.0.1',
    }
mlflow.set_tags(tags)


log = LogisticRegression(penalty=penalty, C=C, random_state=42, solver=solver,max_iter=max_iter)
log.fit(X_train,Y_train)
prediction=log.predict(X_test)

 # Log Hyperparameters
params = {
        'C':C,
        'solver': solver,
        'penalty': penalty,
        'max_iter': max_iter,
}
mlflow.log_params(params)

mlflow.sklearn.log_model(log, "Logistic Regression")

accuracy = metrics.accuracy_score(Y_test, prediction)
precision = metrics.precision_score(Y_test, prediction, average='macro')
recall = metrics.recall_score(Y_test, prediction, average='macro')
f1_score = metrics.f1_score(Y_test, prediction, average='macro')
# auc = metrics.roc_auc_score(Y_test, prediction)

    # Log metrics
evaluation_metrics = {
        'accuracy':accuracy,
        'precision':precision,
        'recall':recall,
        'f1_score':f1_score,
       #'auc':auc,
        }

mlflow.sklearn.log_model(log, "Logistic Regression")
mlflow.log_artifacts('data/')

mlflow.log_metrics(evaluation_metrics)

mlflow.end_run()
print('The accuracy of the Logistic Regression is ',accuracy)
print('The precision of the Logistic Regression is ',precision)
print('The recall of the Logistic Regression is ',recall)
print('The f1_score of the Logistic Regression is ',f1_score)




