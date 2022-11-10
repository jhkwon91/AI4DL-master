FROM ubuntu:18.04

RUN apt update -y
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update   
RUN apt install -y python3.7.10 
RUN apt install vim -y
RUN apt install net-tools -y
RUN apt install iputils-ping -y
RUN apt install python-pip -y
RUN apt install python3-pip -y
RUN apt update -y



COPY . /app
WORKDIR /app
ENTRYPOINT ["python"]
CMD ["app.py"]
