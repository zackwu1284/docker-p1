# docker-p1

We are building a tracking server for mlflow. We use docker-compose to build our environment. 

We use MySql as our backend. We use minIO as our buckets.


## To start the application
docker-compose -f docker-copose.yml up

## Train models
python train.py

## ML flow
127.0.0.1:5000

## Min IO
127.0.0.1:9000

