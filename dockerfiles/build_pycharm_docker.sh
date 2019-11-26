#!/bin/sh

TAG=python_with_tensorflow
UBUNTU_MIRROR=jp.archive.ubuntu.com


fallocate -l 4G swapfile
chmod 600 swapfile
mkswap swapfile
sudo swapon swapfile
swapon -s

docker build \
	--build-arg MIRROR=$UBUNTU_MIRROR \
	-t $TAG \
	.