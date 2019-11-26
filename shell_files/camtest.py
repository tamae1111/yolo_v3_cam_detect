import matplotlib.pyplot as plt
import numpy as np
import cv2

WINDOW_NAME = 'Camera Test'

GST_STR = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)30/1 \
    ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx \
    ! videoconvert \
    ! appsink'

#cap = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)#CAP_GSTREAMERを追加してみた
#cap = cv2.VideoCapture(GST_STR)

"""うまく行かなかったやつ、コメントアウトした箇所全部
def main():
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(GST_STR)
    spenttime = 0
    while True:
        spenttime +=1
        ret, img = cap.read()
        if ret != True:
            break
        cv2.imshow(WINDOW_NAME, img)
        print(spenttime)
"""

def main():
    cap = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)

    while True:
        ret, img = cap.read()
        if ret != True:
            break

        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(10)
        if key == 27: # ESC
            break

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()