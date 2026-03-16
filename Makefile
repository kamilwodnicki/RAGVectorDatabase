DOCKER_COMPOSE = docker-compose
SERVICE_NAME = rag-server

.PHONY: build up down shell logs clean-db

build:
	$(DOCKER_COMPOSE) build

up:
	$(DOCKER_COMPOSE) up -d

down:
	$(DOCKER_COMPOSE) down

shell:
	$(DOCKER_COMPOSE) run --rm $(SERVICE_NAME) /bin/bash

logs:
	$(DOCKER_COMPOSE) logs -f $(SERVICE_NAME)

clean-db:
	rm -rf ./vectorDatabase_MULTI/*
	@echo "Bazy danych zostały usunięte."