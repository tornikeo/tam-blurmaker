#!/bin/bash
DOCKER_BUILTKIT=1 docker compose build && docker compose run -it -p 6080:6080 -v .:/workdir --rm  app python app.py --device cpu --sam_model_type vit_b --debug
