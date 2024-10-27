#!/bin/bash

echo building docker images and starting docker containers...
docker compose up --build -d
docker compose watch