FROM python:3.8-slim-buster

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

RUN apt update && apt install tzdata -y
ENV TZ="Europe/Warsaw"

CMD [ "python3", "flask_api.py"]
