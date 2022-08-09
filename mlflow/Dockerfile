FROM condaforge/miniforge3

RUN pip install mlflow boto3 pymysql \
    && pip install numpy>=1.21.2 \
    && pip install pandas>=1.3.3 \
    && pip install scikit-learn>=0.24.2 \
    && pip install pyodbc \
    && pip install setuptools
COPY . /test