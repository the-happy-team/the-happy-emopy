## docker run -it happy-backend_emopy bash
## docker build -t emopy .
## docker run -p 5000:5000 --volume=$PWD/imgs:/opt/emopy/python/imgs emopy


FROM python:3.6.10-buster

RUN apt-get update -y && apt-get install -y git-lfs

RUN mkdir -p /opt/emopy/python
WORKDIR /opt/emopy/python

ADD python/requirements.txt /opt/emopy/python/requirements.txt
RUN pip install -r requirements.txt

ADD python /opt/emopy/python

CMD ["python", "app.py"]
