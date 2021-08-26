SHELL := /bin/bash

ORG=myorg
PROJECT=far_samples
MODEL=sklearn_old
PORT=5100
index=15

NAME=$(ORG)-$(PROJECT)-$(MODEL)

default: build

clean:
	-rm model.pkl
	-rm model.yaml
	-rm dataset.yaml

package:
	far build --org $(ORG) --project $(PROJECT) --model $(MODEL) --source .

stop:
	-docker stop -t 1 $(NAME) 
	-docker rm $(NAME) 

train:
	far build --org $(ORG) --project $(PROJECT) --model $(MODEL) --train True --source .
	docker run --rm -p 5101:$(PORT) --name=$(NAME) --entrypoint="/app/train/runit.sh" -v `pwd`:/app/train $(NAME):latest 

run: package
	docker run -d -p $(PORT):$(PORT) --name=$(NAME) $(NAME):latest

execute:
	far execute --org $(ORG) --project $(PROJECT) --model $(MODEL) --port $(PORT) --index $(index) 

explain:
	far explain --org $(ORG) --project $(PROJECT) --model $(MODEL) --port $(PORT) --index $(index) 

ssh:
	docker exec -it $(NAME) /bin/bash

logs:
	docker logs $(NAME)

