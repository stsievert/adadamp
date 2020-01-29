CURDIR := $(shell pwd)

build:
	docker image build -t adadamp:1.0 .

run:
	docker container run --detach -p 8000:8000 --name ada -v $(CURDIR):/exp adadamp:1.0

rm:
	docker container rm --force ada

login:
	docker exec -it ada /bin/bash
