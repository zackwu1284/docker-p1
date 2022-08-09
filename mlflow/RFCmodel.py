import pandas as pd
import numpy as np
from sklearn import preprocessing

filepath1 = 'fraudTest.csv'
data1 = pd.read_csv(filepath1)

filepath2 = 'fraudTrain.csv'
data2 = pd.read_csv(filepath2)

#concatenation of the two csv files
data = pd.concat([data1,data2],ignore_index = True)

#data = data.sample(n = 10000,random_state =50)

#encoding text columns
label_encoder = preprocessing.LabelEncoder()
encode_Category = label_encoder.fit_transform(data['category'])
encode_Gender = label_encoder.fit_transform(data['gender'])
encode_City = label_encoder.fit_transform(data['city'])
encode_State = label_encoder.fit_transform(data['state'])
encode_Job = label_encoder.fit_transform(data['job'])


#separating the trans_date_trans_time column
time = pd.DataFrame({
    'trans_time':data['trans_date_trans_time']
})
time[['date','time']] = time['trans_time'].str.split(' ',expand = True)
time[['year','month','day']]= time['date'].str.split('-', expand = True)
time[['hr','min','sec']]= time['time'].str.split(':', expand = True)

timeFrame = time.drop(labels=['trans_time','date','time'],axis = 1)
yList = timeFrame['year'].to_numpy()
moList = timeFrame['month'].to_numpy()
dList = timeFrame['day'].to_numpy()
hList = timeFrame['hr'].to_numpy()
minList = timeFrame['min'].to_numpy()
sList = timeFrame['sec'].to_numpy()

# calculating the age based on date of birth
import datetime as dt
age_df=dt.date.today().year-pd.to_datetime(data['dob']).dt.year

data['age'] = age_df


# build dataframe: explore data (2019.01-2020.12) with all 20 variables
explore_data = pd.DataFrame([
    yList, moList,dList, hList,minList, sList, data['unix_time'],encode_Category,data['amt'], encode_Gender, data['age'], encode_City, encode_State, data['zip'], data['lat'], data['long'],data['city_pop'], encode_Job, data['merch_lat'], data['merch_long'], data['is_fraud']
]).T

explore_data.columns = ['year','month','day','hr','min','sec','unix_time','encode_Category','amt','gender','age', 'city','state','zip','lat','long','city_pop','job','merch_lat','merch_long','is_fraud']
explore_data = explore_data.astype('int32')


#EDA Exploratory Data Analysis

corr = explore_data.corr()


# build dataframe: all data (2019.01-2020.12) with 15 variables
# (dropped min, sec, unix_time, merch_long, merch_lat for multicollinearity)
all_data = pd.DataFrame([
    yList, moList,dList, hList,encode_Category,data['amt'], encode_Gender, data['age'], encode_City, encode_State, data['zip'], data['lat'], data['long'],data['city_pop'], encode_Job, data['is_fraud']
]).T

all_data.columns = ['year','month','day','hr','encode_Category','amt','gender','age', 'city','state','zip','lat','long','city_pop','job','is_fraud']

all_data = all_data.astype('int32')

#here we split the dataset into train and test, and then manipulate the train dataset into a balanced new dataset
from sklearn.model_selection import train_test_split, StratifiedKFold


#in all_data set: define fraud/non-fraud
X0 = all_data.drop('is_fraud',axis = 1)
y0 = all_data['is_fraud']

# split the data into train and test set
train, test = train_test_split(all_data, test_size=0.2, random_state=42, shuffle=True)

sfold = StratifiedKFold(n_splits = 5, random_state = None, shuffle = False)

#in train set: define fraud/non-fraud
X = train.drop('is_fraud',axis = 1)
y = train['is_fraud']

num_of_fraud = train['is_fraud'].value_counts()[1]

train = train.sample(frac = 1)

#in test set: define fraud/non-fraud
X1_test = test.drop('is_fraud',axis = 1)
# print(X1_test.columns)
y1_test = test['is_fraud']

#  RandomForest model (imbalanced data + all 15 variables)

from sklearn.ensemble import RandomForestClassifier

randfor = RandomForestClassifier(n_estimators = 250,max_depth = 32)
randfor.fit(X, y)

randfor2_predict = randfor.predict(X1_test)

# classification report
from sklearn.metrics import classification_report
print(classification_report(y1_test,randfor2_predict))

# accuracy score
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

# MAE, MSE, RMSE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae2 = mean_absolute_error(y1_test, randfor2_predict)
mse2 = mean_squared_error(y1_test, randfor2_predict)
rmse2 = np.sqrt(mse2)
rsquared2 = r2_score(y1_test, randfor2_predict)

# print('\n MAE: ',mae2,'\n','MSE: ',mse2,'\n','RMSE: ',rmse2,'\n','R Squared: ',rsquared2)

#mlflow 
import mlflow
import mlflow.sklearn
import os

# Use IP of your remote machine here
server_ip = "0.0.0.0"

# set up minio credentials and connection
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'adminadmin'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"http://{server_ip}:9000"

# set mlflow track uri
mlflow.set_tracking_uri(f"http://{server_ip}:5000")
mlflow.set_experiment("random forest")
lst_param=X0.columns.to_list()
with mlflow.start_run():
    for i in range(len(lst_param)):
        mlflow.log_param("parameter%d"%(i+1),lst_param[i])
    mlflow.log_metric("Accuracy", accuracy_score(y1_test, randfor2_predict))
    mlflow.log_metric("Recall" , recall_score(y1_test, randfor2_predict))
    mlflow.log_metric("Precision", precision_score(y1_test, randfor2_predict))
    mlflow.log_metric("F1", f1_score(y1_test, randfor2_predict))
    mlflow.log_metric("mae", mae2)
    mlflow.log_metric("mse", mse2)
    mlflow.log_metric("rmse", rmse2)
    mlflow.log_metric("rsquared", rsquared2)
    mlflow.sklearn.log_model(randfor, "Random Forest")
