service = work

all: build up login

rerun: down up login

build:
	docker compose build --no-cache

up:
	docker compose up -d

login:
	docker compose exec ${service} /bin/bash

down:
	docker compose down
