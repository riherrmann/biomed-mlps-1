setup:
	pip3 install --requirement=requirements.txt
	python3 setup.py
main:
	python3 -m biomed.main

testAll:
	python3 -m pytest tests

test:
	python3 -m pytest $(ARGS)

cov:
	coverage run -m pytest tests
	coverage report -m
