#!/bin/bash
DOCKER_BUILTKIT=1 docker compose build && docker compose run -v .:/workdir -it --rm app python cli.py