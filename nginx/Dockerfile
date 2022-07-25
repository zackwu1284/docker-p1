FROM nginx:1.17.6-alpine

RUN rm /etc/nginx/nginx.conf
COPY nginx.conf /etc/nginx
COPY mlflow.conf /etc/nginx/sites-enabled/

COPY run.sh /

RUN apk add --no-cache --update apache2-utils
RUN rm -f /etc/nginx/conf.d/*

ENV SERVER_NAME example.com
ENV PORT 80
ENV CLIENT_MAX_BODY_SIZE 1m
ENV PROXY_READ_TIMEOUT 60s
ENV WORKER_PROCESSES auto

STOPSIGNAL SIGQUIT

RUN ["chmod", "+x", "/run.sh"]