#!/bin/bash
current_dir="$(pwd)"
docker run -it --rm -e ANALYTICS_PORT=5001 -e MONGODB_USERNAME= -e MONGODB_PASSWORD= --gpus all --runtime=nvidia -v '':/weed_data -v $current_dir/code/:/code -v $current_dir/tsne:/tsne --shm-size=6g --net=host analytics:v2 /bin/bash