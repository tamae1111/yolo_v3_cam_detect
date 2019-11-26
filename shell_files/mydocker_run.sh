#!/usr/bin/env bash

#今のところこの実行ファイルは未確認
function launch_docker() {
  local image_tag=$1
  # GUI不要の場合、--deviceのみでOK
  # GUI用にすべてのX接続を受け入れる
  xhost +
  docker run --privileged -it \
  -e DISPLAY=$DISPLAY \ # Xの宛先をホストと同一に
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \ # Xソケットを共有
  --device /dev/video0:/dev/video0:mwr \ # カメラデバイスを共有
  --device /dev/video1:/dev/video1:mwr \ # 複数指定も可能
  ${image_tag} /bin/bash
  }
launch_docker image_tag