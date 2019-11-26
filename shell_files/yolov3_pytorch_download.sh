#!/usr/bin/env bash

#先ずはjetsoncardをインストールする必要がある、でないと色々なライブラリのインストールなど面倒が多い。

sudo pip3 install pandas

cd ~/Documents/gitdir
git clone https://github.com/AlexeyAB/darknet.git && \
    cd darknet && \
    sed -i -e "/GPU=/s/0/1/" Makefile && \
    sed -i -e "/CUDNN=/s/0/1/" Makefile && \
    sed -i -e "/OPENCV=/s/0/1/" Makefile

#↓は実行時はあったが、論理的にはなくていいはずなので一旦除外
#wget https://pjreddie.com/media/files/yolov3.weights && \
    #wget https://pjreddie.com/media/files/yolov3-tiny.weights

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

#テスト時は一応makeだけだった（-j$(nproc)がなかった）
make -j$(nproc)



#以下のものはutilの中身などが必要なのでインストール必須
#以下はdemo実行用のスクリプトも取得するスクリプト（pycharmで動く）
mkdir ~/Documents/gitdir
cd ~/Documents/gitdir
git clone https://github.com/ayooshkathuria/pytorch-yolo-v3.git
cd pytorch-yolo-v3
wget https://pjreddie.com/media/files/yolov3.weights && \
    wget https://pjreddie.com/media/files/yolov3-tiny.weights


cp ~/Documents/gitdir/darknet/cfg/yolov3-tiny.cfg ~/Documents/gitdir/pytorch-yolo-v3/cfg

sudo apt install -y libcanberrra-gtk* v4l-utils


#以下はgreengrass用ユーザーの作成手順
sudo adduser --system ggc_user
sudo groupadd --system ggc_group
wget https://github.com/aws-samples/aws-greengrass-samples/raw/master/greengrass-dependency-checker-GGCv1.8.0.zip
unzip greengrass-dependency-checker-GGCv1.8.0.zip





#そして実行時は~/Documents/gitfiles/pytorch-yolo-v3にプログラムを置いて、実行してやるといい。


#apt-get update && apt-get install -y git


#sudo apt update
#sudo apt upgrade
#sudo apt search python3.6-*
#sudo apt install python3.6 python3.6-dev
#curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
#sudo apt install python3.6-distutils
#python3.6 get-pip.py --user

#echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc
#source ~/.bashrc
#pip3 install --user pipenv


#以下は必要ライブラリのインストール
#pipenv install --python 3.6 torch cv2
#pipenv install --python 3.6 numpy pandas matplotlib jupyterlab

#cd ~/Documents/yolov3/darknet
#./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights -c 0

#もしくは
#~/Documents/yolov3/darknet/darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights -c 0