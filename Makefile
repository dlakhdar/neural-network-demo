.ONESHELL: # Applies to every target in the file!

PYTHON_VERSION ?= $(shell python3 -c "import sys;print('{}.{}'.format(*sys.version_info[:2]))")

# name
.neuralnet:
	@echo "PYTHON_VERSION: $(PYTHON_VERSION)"
	python$(PYTHON_VERSION) -m venv .neuralnet
	. .neuralnet/bin/activate; .neuralnet/bin/pip$(PYTHON_VERSION) install --upgrade pip$(PYTHON_VERSION) ; .neuralnet/bin/pip$(PYTHON_VERSION) install -e .[dev,test]

neuralnet: .neuralnet

# setup
test: .neuralnet
	. .neuralnet/bin/activate; python3 -m ; cd test ; pytest

clean: .neuralnet
	rm -rf .neuralnet
