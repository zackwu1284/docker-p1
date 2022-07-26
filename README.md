# docker-p1

We are building a tracking server for mlflow. We use docker-compose to build our environment. 

We use MySql as our backend. We use minIO as our buckets.


## To start the application
docker-compose -f docker-copose.yml up

## Train models
python train.py

## ML flow
127.0.0.1:5000

## MinIO
127.0.0.1:9000

## Nginx

In train.py, we use 127.0.0.1:5000 to connect to ml flow server. If we want to do authentication in future, we could give others 127.0.0.1:80,
so they would need to enter through proxy. In other words, they would need to have passwords in order to access ml flow tracking server.
