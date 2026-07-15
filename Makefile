.PHONY: install install-dev lint test serve service-install service-status docker-build docker-up docker-down download-model

install:
	python -m pip install ".[model]"

install-dev:
	python -m pip install -e ".[model,dev]"

lint:
	ruff check src tests scripts

test:
	pytest

serve:
	grasp-anything serve

service-install:
	bash scripts/install_user_service.sh

service-status:
	systemctl --user status grasp-anything.service

download-model:
	python scripts/download_model.py

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down
