import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,plot_confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
#from mlflow import log_metric, log_param, log_artifacts
import mlflow
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Import train and test dataset, merge both datasets because the datasets were seperated in a chronological order. We want randomize our datasets.
train_data = pd.read_csv("fraudTrain.csv")
test_data = pd.read_csv("fraudTest.csv")
#concatenating the two datasets
dat = pd.concat([train_data, test_data]).reset_index()
dat.drop(dat.columns[:2], axis=1, inplace=True)

# Transform column names 
dat.rename(columns={"trans_date_trans_time":"transaction_time",
                         "cc_num":"credit_card_number",
                         "amt":"amount(usd)",
                         "trans_num":"transaction_id"},
                inplace=True)


#Transform to datetime
dat["transaction_time"] = pd.to_datetime(dat["transaction_time"], infer_datetime_format=True)
dat["dob"] = pd.to_datetime(dat["dob"], infer_datetime_format=True)

# Sine column 'unix_time' is the same as transaction time but in unix format. We drop it but keep the 'transaction_time'.
dat.drop('unix_time', axis=1,inplace=True)

dat['age'] = np.round((dat['transaction_time'] - 
                      dat['dob'])/np.timedelta64(1, 'Y'))

# Sperate transaction time into year, month, day and hour
dat['year'] = dat.transaction_time.dt.year
dat['month'] = dat.transaction_time.dt.month
dat['day'] = dat.transaction_time.dt.day
dat['hour'] = dat.transaction_time.dt.hour
dat.drop('transaction_time', axis=1,inplace=True)


# change male to 1, female to 0
dat["gender"]=dat.gender.apply(lambda x: 1 if x=="M" else 0)


dat.drop(['merchant','first','last','street','city','state','job','dob','transaction_id'], axis=1,inplace=True)



# LabelEncoder
labelencoder = LabelEncoder()
dat['category'] = labelencoder.fit_transform(dat['category'])
dat_dum = pd.get_dummies(dat['category'])
dat = pd.concat([dat, dat_dum], axis="columns")


X= dat.iloc[:,dat.columns!= 'is_fraud']
y= dat.iloc[:,dat.columns== 'is_fraud']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)

# Use IP of your remote machine here
server_ip = "172.23.120.50"

# set up minio credentials and connection
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'adminadmin'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{server_ip}:9000"

# set mlflow track uri
mlflow.set_tracking_uri(f"http://{server_ip}:5000")
mlflow.set_experiment("XGBoost")


with mlflow.start_run(run_name="XGBoost_all:"):

    #xgboost    
    xgboostModel = XGBClassifier(n_estimators=500, learning_rate= 0.1, objective='binary:logistic', booster='gbtree')
    
    xgboostModel.fit(X_train, y_train)

    predicted = xgboostModel.predict(X_test)
    
    #mlflow
    param=dat.columns.to_list()
    for i in range(len(param)):
        mlflow.log_param("parameter%d"%(i+1),param[i]) 
    mlflow.log_param("Train rows", len(X_train))
    mlflow.log_param("Test rows", len(X_test))
    print("Classification report:\n", classification_report(y_test, predicted))
    mlflow.log_metric("Accuracy", accuracy_score(y_test, predicted))
    mlflow.log_metric("Recall" , recall_score(y_test, predicted))
    mlflow.log_metric("Precision", precision_score(y_test, predicted))
    mlflow.log_metric("F1", f1_score(y_test, predicted))
    mlflow.sklearn.log_model(xgboostModel,"XGBoost")





