#!/usr/bin/env bash

docker rmi $(docker images -aq)
docker rm $(docker ps -aq)

echo "docker images and containers deleted"