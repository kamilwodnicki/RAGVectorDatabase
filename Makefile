DOCKER_COMPOSE = docker compose
SERVICE_NAME = rag-server

.PHONY: build up down shell logs

build:
	$(DOCKER_COMPOSE) build

up:
	$(DOCKER_COMPOSE) up -d

down:
	$(DOCKER_COMPOSE) down

shell:
	docker exec -it rag-server /bin/bash

logs:
	$(DOCKER_COMPOSE) logs -f $(SERVICE_NAME)
