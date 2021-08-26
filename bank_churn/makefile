SHELL := /bin/bash

ORG=testorg
PROJECT=far_samples
MODEL=bank_churn
PORT=5100
index=15

REGISTRY=079310353266.dkr.ecr.us-west-1.amazonaws.com

NAME=$(ORG)-$(PROJECT)-$(MODEL)

default: build

package:
	far build --org $(ORG) --project $(PROJECT) --model $(MODEL) --source .

stop:
	-docker stop -t 1 $(NAME) 
	-docker rm $(NAME) 

run:
	docker run --rm -d -p $(PORT):$(PORT) --name=$(NAME) $(NAME):latest

execute:
	far execute --org $(ORG) --project $(PROJECT) --model $(MODEL) --port $(PORT) --index $(index) 

explain:
	far explain --org $(ORG) --project $(PROJECT) --model $(MODEL) --port $(PORT) --index $(index) 

register:
	far register --org $(ORG) --project $(PROJECT) --model $(MODEL) --image=$(REGISTRY)/$(NAME):latest --source=.

ssh:
	docker exec -it $(NAME) /bin/bash

logs:
	docker logs $(NAME)

