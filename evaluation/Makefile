DOCKER_IMAGE = epwalsh/gpt-2-repro:latest

.PHONY : docker-build
docker-build :
	docker pull $(DOCKER_IMAGE)
	docker build -t $(DOCKER_IMAGE) --cache-from $(DOCKER_IMAGE) .

.PHONY : build-and-push
build-and-push : docker-build
	docker push $(DOCKER_IMAGE)
