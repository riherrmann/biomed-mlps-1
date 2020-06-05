main:
	pipenv run python -m biomed.main

deps/clean:
	@pipenv --rm

deps:
	@pipenv install