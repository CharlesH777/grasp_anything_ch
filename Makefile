.PHONY: bootstrap bootstrap-dev bootstrap-training install install-dev doctor lint test serve service-install service-status docker-build docker-up docker-down download-model

bootstrap:
	bash scripts/bootstrap.sh --service

bootstrap-dev:
	bash scripts/bootstrap.sh --dev

bootstrap-training:
	bash scripts/bootstrap.sh --training

install:
	python -m pip install ".[model]"

install-dev:
	python -m pip install -e ".[model,dev]"

doctor:
	grasp-anything doctor

lint:
	ruff check src tests scripts

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest

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
