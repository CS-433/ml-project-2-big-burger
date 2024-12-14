FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
	python3 python3-pip

# Upgrade pip to the latest version
RUN pip3 install --upgrade pip

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python", "run.py"]
