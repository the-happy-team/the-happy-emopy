#!/bin/bash

docker build -t emopy .
docker run -d -p 5000:5000 --volume=$PWD/imgs:/opt/emopy/python/imgs --name emopy_run emopy
