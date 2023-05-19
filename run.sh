#!/bin/bash
DOCKER_BUILTKIT=1 docker compose build && docker compose run -it -v .:/workdir --rm app python cli.py
