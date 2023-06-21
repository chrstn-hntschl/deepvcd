ARG TF_VER=2.11.0-gpu
FROM tensorflow/tensorflow:${TF_VER}

LABEL maintainer="Christian Hentschel < chrstn.hntschl@gmail.com>"

ENV DEBIAN_FRONTEND noninteractive

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y \
  tzdata \
  ca-certificates \
  python3 \
  libgoogle-perftools4 \
  libyaml-cpp0.5v5\
  python3-yaml \
  python3-pil \
  python3-pip \
  vim && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists 

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN pip3 --no-cache-dir install --upgrade pip && apt-get remove --purge -y python3-pip

# installing additional stuff
RUN pip3 --no-cache-dir install keras-preprocessing==1.1.2 scikit-learn==0.24.2 tensorflow-datasets==4.5.2 protobuf==3.19.6

VOLUME [ "/data" ]

ENV LD_PRELOAD "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"

ARG UID=1000
ARG GID=1000
ARG USER=developer
ENV USER=${USER}
ENV HOME_DIR=/home/${USER}
RUN groupadd -g ${GID} developer && useradd -m -d ${HOME_DIR} -u ${UID} -g ${GID} -s /bin/bash ${USER}

USER ${USER} 

WORKDIR ${HOME_DIR}
