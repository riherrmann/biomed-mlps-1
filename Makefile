main:
	pipenv run python -m biomed.main

test:
	pipenv run python -m pytest tests

deps/clean:
	@pipenv --rm

deps:
	@pipenv install
	pipenv run python -m setup

deps/dev:
	@pipenv install --dev
	pipenv run python -m setup