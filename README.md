# docker-p1

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



