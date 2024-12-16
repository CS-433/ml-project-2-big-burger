# using the april 2022 version of ubuntu
FROM ubuntu:22.04

# updating the package list and installing python3 and pip
RUN apt-get update && apt-get install -y \
	python3 python3-pip

# upgrading pip to the latest version
RUN pip3 install --upgrade pip

# our working directory inside the container
WORKDIR /app

# copying the current directory to the working directory inside the container
COPY . /app

# installing the required packages, specified in the requirements.txt file
RUN pip3 install -r requirements.txt

# can add desired arguments here to run the run script
ENTRYPOINT ["python3", "run.py"] 
