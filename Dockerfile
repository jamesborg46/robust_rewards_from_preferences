FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
MAINTAINER James Borg <jamesborg46@gmail.com>

RUN apt-get update && \
    apt-get install -y git-core
RUN git clone https://github.com/Indoril007/robust_rewards_from_preferences.git
