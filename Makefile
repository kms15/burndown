SHELL:=/bin/bash

.PHONY: check
check: venv-burndown/bin/activate
	source venv-burndown/bin/activate && \
		python3 -m pytest --cov=burndown --doctest-modules
	source venv-burndown/bin/activate && \
		python3 -m burndown test/read_test.csv -o .tmp_integration_test.png
	rm -rf .tmp_integration_test.png

.PHONY: clean
clean:
	rm -rf venv-burndown

.PHONY: black
black: venv-burndown/bin/activate
	source venv-burndown/bin/activate && \
	 black burndown test

.PHONY: container-check
container-check:
	podman run -v .:/app -it --rm debian:stable /bin/bash -c \
	 "cd app && ./install-prereqs.sh && make check"

.PHONY: testwatch
testwatch:
	ls burndown/*.py \
		test/*.py \
		Makefile requirements.txt \
		| entr time make check

venv-burndown/bin/activate: requirements.txt
	rm -rf venv-burndown
	python3 -m venv venv-burndown
	source venv-burndown/bin/activate && \
	 pip3 install --upgrade pip
	source venv-burndown/bin/activate && \
	 pip3 install -r requirements.txt

