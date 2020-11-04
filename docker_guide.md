# Run below to start openvino docker and map local directory (-v /home/vm/Downloads:/home) to docker directory

docker run -it -v /home/vm/Downloads:/home --rm openvino/ubuntu18_dev:latest

# attach to running container

docker exec -it <container_name> /bin/bash

# Tensorflow docker with jupyter notebook

docker run -it --rm -v /home/vm/Downloads:/tf/data -p 8888:8888 tensorflow/tensorflow:2.2.1-jupyter
