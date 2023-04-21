FROM bde2020/spark-master:3.2.1-hadoop3.2
COPY base_models_learning/requirements.txt requirements.txt
RUN apk update
RUN apk add make automake gcc g++ subversion python3-dev
RUN pip3 install -r requirements.txt
RUN apk update && apk add tzdata
ENV TZ="Europe/Warsaw"