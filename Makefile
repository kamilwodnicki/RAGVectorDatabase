COMPOSE_PROJECT := rag
DOCKER_COMPOSE = COMPOSE_PROJECT_NAME=$(COMPOSE_PROJECT) docker compose
SERVICE_NAME = rag-server

.PHONY: build up down shell logs test test-unit test-integration test-eval prepare_pr

build:
	@$(DOCKER_COMPOSE) build

up:
	@$(DOCKER_COMPOSE) up -d --remove-orphans

down:
	@$(DOCKER_COMPOSE) down

shell:
	@docker exec -it rag-server /bin/bash

logs:
	@$(DOCKER_COMPOSE) logs -f $(SERVICE_NAME)

test-unit:
	@docker exec -it $(SERVICE_NAME) pytest -m unit

test-integration:
	@docker exec -it $(SERVICE_NAME) pytest -m integration

test-eval:
	@docker exec -it $(SERVICE_NAME) pytest -m eval

test: test-unit

prepare_pr: test-unit test-integration
