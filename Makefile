build:
	python setup.py sdist bdist_wheel

test:
	python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload:
	python -m twine upload dist/*

clean:
	rm -rf dist
