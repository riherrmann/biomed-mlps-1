setup:
	pip3 install --requirement=requirements.txt
	python3 setup.py
main:
	python3 -m biomed.main

test:
	python3 -m pytest $(ARGS)

coverage:
	coverage run --source=./biomed -m pytest tests
	coverage report -m
