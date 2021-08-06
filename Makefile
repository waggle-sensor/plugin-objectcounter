RELEASE?=0.1.0
PLATFORM?=linux/arm64
IMAGE=object-counter

all: image

image:
	docker buildx build -t "waggle/plugin-$(IMAGE):$(RELEASE)" --load .

push:
	docker buildx build -t "waggle/plugin-$(IMAGE):$(RELEASE)" --platform "$(PLATFORM)" --push .
