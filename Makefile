all: install run-help

SHELL := /bin/bash

# helper for running commands inside a virtual environment:
INSIDE_ENV = source .env/bin/activate &&

install:
	python3.7 -m venv .env
	$(INSIDE_ENV) pip install -r requirements.txt
	$(INSIDE_ENV) pip install .

run-help:
	$(INSIDE_ENV) python -m image_cluster --help

run-cluster-help:
	$(INSIDE_ENV) python -m image_cluster cluster --help

run-meta-help:
	$(INSIDE_ENV) python -m image_cluster meta --help
