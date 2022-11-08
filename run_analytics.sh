#!/bin/bash
current_dir="$(pwd)"
docker run -it --rm --gpus all --runtime=nvidia -v /home/daniel/code/cvat_stuff/weed_data:/obdb -v $current_dir/code/:/code -v $current_dir/tsne:/tsne --shm-size=6g --net=host analytics:v1 /bin/bash