# docker-p1
Check the notes before you start.

## Introduction
- We are building a tracking server for mlflow. 
- We use docker-compose to build our environment. 
- We use MySql as our backend.
- We use minIO as our buckets.

## To start the application
`docker-compose -f docker-compose.yml up`

## Train models
 `python RFCmodel.py`

 `python logistic.py`
 
 Run Analysis.ipynb

## ML flow
`127.0.0.1:5000`

## MinIO
`127.0.0.1:9000`

## Nginx

We use 127.0.0.1:5000 to connect to  our ml flow server. If we want to do authentication in future, we could give others 127.0.0.1:80,
so they would need to enter through proxy. In other words, they would need to have passwords in order to access ml flow tracking server.

## Data
data souce :  https://www.kaggle.com/datasets/kartik2112/fraud-detection?select=fraudTest.csv


## Tableau Public

https://public.tableau.com/views/CreditCardFraudAnalysisDashboard/Dashboard1?:language=en-US&:display_count=n&:origin=viz_share_link

## Notes
- Since the CSV files are too big to upload, you will need to download the file in Kaggle, and put CSV files inside the "mlflow" folder. 
- If you want to train models on your host (out of container), you will need to prepare python packages and libraries. EX: conda install mlflow. If training models inside mlflow container, you don't need to prepare the environment, because it's written in dockerfile.
- If you want to train models inside mlflow container, you need to make sure the ip address inside you python files point to you IPV4 address. If your ip address is
localhost, you cannot be able to locate min IO inside a container. Hence, you will get an error. 

