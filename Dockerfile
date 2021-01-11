FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
MAINTAINER James Borg <jamesborg46@gmail.com>

RUN apt-get update && \
    apt-get install -y git-core && \
    apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3

ENV LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
