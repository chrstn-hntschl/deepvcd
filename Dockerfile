ARG TF_VER=2.11.0-gpu
FROM tensorflow/tensorflow:${TF_VER}

LABEL maintainer="Christian Hentschel < chrstn.hntschl@gmail.com>"

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
  tzdata \
  ca-certificates \
  python3 \
  libgoogle-perftools4 \
  libyaml-cpp0.6 \
  python3-yaml \
  python3-h5py \
  python3-pil \
  python3-pip && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists 

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN pip3 --no-cache-dir install --upgrade pip && apt-get remove --purge -y python3-pip

# installing additional stuff
RUN pip3 --no-cache-dir install keras-preprocessing==1.1.2 scikit-learn==0.24.2 tensorflow-datasets==4.9.2 protobuf==3.20.3

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
