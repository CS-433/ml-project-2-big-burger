FROM ubuntu:16.04

RUN apt-get update && apt-get install -y \
	python3 python3-pip

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python", "run.py"]
