FROM python:3.8-slim-buster

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

RUN apt update && apt install tzdata -y
ENV TZ="Europe/Warsaw"

RUN python3 airAPI.py
CMD [ "python3", "mockAPI.py"]
