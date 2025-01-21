.ONESHELL: # Applies to every target in the file!

PYTHON_VERSION ?= $(shell python3 -c "import sys;print('{}.{}'.format(*sys.version_info[:2]))")

# name
.neuralnet:
	@echo "PYTHON_VERSION: $(PYTHON_VERSION)"
	python$(PYTHON_VERSION) -m venv .neuralnet
	. .neuralnet/bin/activate; .neuralnet/bin/pip$(PYTHON_VERSION) install --upgrade pip$(PYTHON_VERSION) ; .neuralnet/bin/pip$(PYTHON_VERSION) install -e .[dev,test] ; pre-commit install

neuralnet: .neuralnet

# setup
test: .neuralnet
	. .neuralnet/bin/activate; python3 -m ; pytest

clean: .neuralnet
	rm -rf .neuralnet
