all: install run-help

# necessary for source command to work
SHELL := /bin/bash

# helper for running commands inside a virtual environment:
INSIDE_ENV = source .env/bin/activate &&

.env:
	python3.7 -m venv .env

install: .env
	$(INSIDE_ENV) pip install -r requirements.txt
	$(INSIDE_ENV) pip install .

run-help: .env
	$(INSIDE_ENV) python -m image_cluster --help

run-cluster-help: .env
	$(INSIDE_ENV) python -m image_cluster cluster --help

run-meta-help: .env
	$(INSIDE_ENV) python -m image_cluster meta --help
