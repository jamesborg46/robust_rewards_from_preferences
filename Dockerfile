FROM ubuntu:16.04
MAINTAINER James Borg <jamesborg46@gmail.com>

RUN apt-get update && \
    apt-get install -y git-core
