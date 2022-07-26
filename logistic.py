#modules for EDA steps
import pandas as pd 
#modules for data cleaning and data analysis
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import mlflow
import os

# Use IP of your remote machine here
server_ip = "0.0.0.0"

# set up minio credentials and connection
os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'adminadmin'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"

# set mlflow track uri
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test-experiment_v2")

df1=pd.read_csv("fraudTest.csv")
df2=pd.read_csv("fraudTrain.csv")
df = pd.concat([df1,df2],ignore_index=True)

df["dob"]=df["dob"].apply(lambda x:int(x.split("-")[0])) #min:1924 max:2005
bins=[1920,1935,1950,1965,1980,1995,2010]
df["dob"]=pd.cut(x=df["dob"],bins=bins)

df["trans_date_trans_time"]=pd.to_datetime(df["trans_date_trans_time"])
df["hour"]=df["trans_date_trans_time"].dt.hour

df["day"]=df["trans_date_trans_time"].dt.day_name()
df['month']=pd.to_datetime(df['trans_date_trans_time']).dt.month
df["is_fraud"]=df["is_fraud"].astype("int8")

df=df[['category','amt','zip','lat','long','city_pop','merch_lat','merch_long','dob','hour','day','month','is_fraud']]
with mlflow.start_run(run_name="logistic test:"):
    lst_param=df.columns.to_list()
    for i in range(len(lst_param)):
        mlflow.log_param("parameter%d"%(i+1),lst_param[i])   

    x=df.drop("is_fraud",axis=1)
    y=df["is_fraud"]
    x=pd.get_dummies(x,drop_first=True)

    train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3)
    del x
    del y
    method= SMOTE()
    x_resampled, y_resampled = method.fit_resample(train_x, train_y)

    model=LogisticRegression()
    model.fit(x_resampled,y_resampled)
    predicted=model.predict(test_x)

    print("Classification report:\n", classification_report(test_y, predicted))
    mlflow.log_metric("Accuracy", accuracy_score(test_y, predicted))
    mlflow.log_metric("Recall" , recall_score(test_y, predicted))
    mlflow.log_metric("Precision", precision_score(test_y, predicted))
    mlflow.log_metric("F1", f1_score(test_y, predicted))
    mlflow.sklearn.log_model(model,"logistic")