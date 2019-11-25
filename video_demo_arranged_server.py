# -*- coding: utf-8 -*-
import socket
import numpy as np
import cv2
import time
#from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import argparse
import pickle as pkl

serverIP = "18.176.169.6"

portNo = 8000

WINDOW_NAME = 'Camera Test'

"""これが元、ちょっと変えてみる11/25
GST_STR = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)30/1 \
    ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx \
    ! videoconvert \
    ! appsink'

"""
#これで治った。丸
GST_STR = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)30/1 \
    ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx \
    ! videoconvert \
    ! appsink'

#一旦消す
#soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)#ソケットオブジェクト作成

#s.bind((serverIP, portNo))    # サーバー側PCのipと使用するポート

print("waiting for connection ...")

#soc.connect((serverIP, portNo))                   # 接続要求を待機
print("completed connection")


#cam = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)

#cam = cv2.VideoCapture(0)#カメラオブジェクト作成


def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def write(x, img):
    # スライス範囲は以上、より下なので、2要素目と3要素目を取得
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    # cv2.rectangle(画像, 左上座標, 右下座標, 色, 線の太さ)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img


def arg_parse():
    """
    Parse arguements to the detect module
    """

    # confidence nms_thresh reso を引数として使用する、デフォルト値が入っているので、未入力でも稼働可能
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.25)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="160", type=str)
    return parser.parse_args()

if __name__ == '__main__':
#while (True):
    cfgfile = "cfg/yolov3-tiny.cfg"
    weightsfile = "yolov3-tiny.weights"
    num_classes = 80

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
    print("CUDA is :",CUDA)

    num_classes = 80
    bbox_attrs = 5 + num_classes

    # ダークネットでモデル作成、configファイル使用
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])

    assert inp_dim % 32 == 0
    assert inp_dim > 32

    """新しい記載
    # いままで
    model = Model()
    if use_cuda:
        model = model.cuda()

    # 0.4以降
    device = torch.device('cuda:0')  # or 'cpu'
    model = model.to(device)
    
    """

    if CUDA:
        model.cuda()


    model.eval()

    #多分不要と思うので消した11/25
    videofile = 'video.avi'

    cap = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)#元は下
    print("cap is :", cap)
    #cap = cv2.VideoCapture(0)

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    start = time.time()

    # カメラが起動しているならば
    while cap.isOpened():
        # カメラのデータを読み込み、retが起動の可否、frameがデータ
        ret, frame = cap.read()
        if ret:
            print("ret flag is true")
            #time.sleep(0.2)
            # prep_imageメソッドを使って、numpyとかが加工しやすい形に変換
            img, orig_im, dim = prep_image(frame, inp_dim)

            im_dim = torch.FloatTensor(dim).repeat(1,2)#コメントアウトされてたところを復活させた11/25
            if CUDA:
                print("into cuda indent")
                im_dim = im_dim.cuda()
                img = img.cuda()

            # import darknetのmodel使用
            output = model(Variable(img), CUDA)

            # import utilsのwrite_resultを使用、モデルを用いたdetectionの結果を出力
            output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

            # おそらく例外処理部分と思う
            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim

            #            im_dim = im_dim.repeat(output.size(0), 1)
            output[:, [1, 3]] *= frame.shape[1]
            output[:, [2, 4]] *= frame.shape[0]

            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))

            # write関数でrectangleとoriginal_imageの重複した画像を返している。outputは判定で帰ってきたオブジェクトのdetection結果とその位置座標のリスト
            list(map(lambda x: write(x, orig_im), output))

            cv2.imshow(WINDOW_NAME, orig_im)
            print(orig_im.shape)



            #img = orig_im.tostring()  # numpy行列からバイトデータに変換
            #print(len(s))
            #soc.send(orig_im)#これでも行けた


            #ここをあとで追加する
            #soc.send(orig_im)


            #print(len(stringed_data))
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

        else:
            break



               # ソケットにデータを送信

    #time.sleep(0.5)            #フリーズするなら#を外す。

    #k = cv2.waitKey(1)         #↖
    #if k== 13 :                #←　ENTERキーで終了
        #break                  #↙

cam.releace()