#!/bin/bash
DOCKER_BUILTKIT=1 docker compose build && docker compose run -it --rm app