# Names of the code directory and the Docker image, change them to match your project
DOCKER_IMAGE_TAG := so-vits-svc
DOCKER_CONTAINER_NAME := so-vits-svc_container
CODE_DIRECTORY := so-vits-svc
TRAINING_DIRECTORY := training

DOCKER_PARAMS= -it --rm --ipc=host --name=$(DOCKER_CONTAINER_NAME)
# Specify GPU device(s) to use. Comment out this line if you don't have GPUs available
DOCKER_PARAMS+= --gpus '"device=2"' --shm-size 32G
# Mount music directory
DOCKER_PARAMS+= -v /data/music/output/mp3_128/:/workspace/mp3_128

# Run Docker container while mounting the local directory
DOCKER_RUN_MOUNT= docker run $(DOCKER_PARAMS) \
	-v /data/:/data/ \
	-v $(PWD)/$(CODE_DIRECTORY)/:/workspace/$(CODE_DIRECTORY)/ \
	-v $(PWD)/$(TRAINING_DIRECTORY)/:/workspace/$(TRAINING_DIRECTORY)/ \
$(DOCKER_IMAGE_TAG)

usage:
	@echo "Available commands:\n-----------"
	@echo "	build		Build the Docker image"
	@echo "	run 		Run the Docker image in a container, after building it. Then launch an interactive bash session in the container while mounting the current directory"
	@echo "	stop		Stop the container if it is running"
	@echo "	logs		Display the logs of the container"
	@echo "	exec		Launches a bash session in the container (only if it is already running)"
	@echo "	attach		Attach to running container"

build:
	docker build -t $(DOCKER_IMAGE_TAG) .

run: build
	$(DOCKER_RUN_MOUNT) /bin/bash

stop:
	docker stop ${DOCKER_CONTAINER_NAME} || true && docker rm ${DOCKER_CONTAINER_NAME} || true

logs:
	docker logs -f --tail 1000 $(DOCKER_CONTAINER_NAME)

exec:
	docker exec -it ${DOCKER_CONTAINER_NAME} /bin/bash

attach:
	docker attach ${DOCKER_CONTAINER_NAME}

qa-check:
	poetry run mypy $(CODE_DIRECTORY) $(TRAIN_DIRECTORY) $(DATASET_DIRECTORY) $(EVALUATION_DIRECTORY)
	poetry run ruff check --no-fix $(CODE_DIRECTORY) $(TRAIN_DIRECTORY) $(DATASET_DIRECTORY) $(EVALUATION_DIRECTORY)
	poetry run ruff format --check $(CODE_DIRECTORY) $(TRAIN_DIRECTORY) $(DATASET_DIRECTORY) $(EVALUATION_DIRECTORY)

qa-clean:
	poetry run ruff check --fix $(CODE_DIRECTORY) $(TRAIN_DIRECTORY) $(DATASET_DIRECTORY) $(EVALUATION_DIRECTORY)
	poetry run ruff format $(CODE_DIRECTORY) $(TRAIN_DIRECTORY) $(DATASET_DIRECTORY) $(EVALUATION_DIRECTORY)