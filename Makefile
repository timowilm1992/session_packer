.PHONY: install test clean

SHELL := /usr/bin/env bash

install:
	conda env update --file ./environment.yml

install-gpu:
	conda env update --file ./gpu_environment.yml

test:
	source activate fick && pytest

run:
	source activate fick && python -m src.main 'test/resources/dataset' 'model' 0

run-gpu:
	source activate fick && optirun python -m src.main 'test/resources/dataset' 'model' 0

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +

lint:
	find . -name '*.py' -exec autopep8 --in-place '{}' \;
