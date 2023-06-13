help:
	@cat Makefile

DOCKER_FILE?=Dockerfile
TENSORFLOW_VER=2.11.0
USER=developer
GPUS?=cpu # all, GPU_ID or CPU for CPU version of TensorFlow
DATA?="${HOME}/tmp/data"
CHECKPOINTS?="${HOME}/tmp/checkpoints"

TAG=git
ifeq ($(strip $(GPUS)), cpu)
  TF_VER=$(TENSORFLOW_VER)
  DOCKER_FLAGS=
else
  TF_VER=$(TENSORFLOW_VER)-gpu
  DOCKER_FLAGS=--gpus '"device=$(GPUS)"'
endif


_build:
	docker build -t chrstn_hntschl/deepvcd:tf-$(TF_VER)-$(TAG) --build-arg TF_VER=$(TF_VER) --build-arg USER=$(USER) -f $(DOCKER_FILE) .

dev: _build 
	docker run $(DOCKER_FLAGS) --rm -it -v $(CURDIR):/home/$(USER)/deepvcd -v $(DATA):/data/datasets -v $(CHECKPOINTS):/data/checkpoints --workdir /home/$(USER)/deepvcd --env PYTHONPATH=/home/$(USER)/deepvcd chrstn_hntschl/deepvcd:tf-$(TF_VER)-$(TAG) /bin/bash

