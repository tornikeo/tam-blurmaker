#!/bin/bash
mkdir -p $2
pat="$2%06d.jpg"
echo $pat
ffmpeg -y -i $1 -q:v 0 $pat