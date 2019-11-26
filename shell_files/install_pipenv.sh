#!/usr/bin/env bash

sudo apt update
sudo apt upgrade


sudo apt install python3.7 python3.7-dev

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

sudo apt install python3.7-distutils

python3.7 get-pip.py --user

echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc

sudo -s source ~/.bashrc

pip3.7 install --user pipenv

#ここまででpipenvが使用可能になる想定さ
#pipenv install --python 3.7 numpy pandas matplotlib jupyterlab

