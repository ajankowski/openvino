# dockers

## docker installation (linux)
>`sudo apt install docker.io`  
`sudo groupadd docker`  
`sudo usermoc -aG docker $USER`  
`newgrp docker`  

## Openvino docker  
Run below to start openvino docker and map local directory o docker directory (-v /home/vm/Downloads:/home)

>`docker run -it -v /home/vm/Downloads:/home --rm openvino/ubuntu18_dev:latest `  

#### inside docker
>`pip install tensorflow==2.2.1`

#### inside docker
>`source /opt/intel/openvino/bin/setupvars.sh`

You can now work inside the container.

## save as a new image
>`docker commit <docker id> <new name>`

## Tensorflow docker with jupyter notebook
>`docker run -it --rm -v /home/vm/Downloads:/tf/data -p 8888:8888 tensorflow/tensorflow:2.2.1-jupyter`

## useful instructions
### attach to running container  

>`docker exec -it <container_name> /bin/bash`



### Container management  

>`docker stop <docker id>` stop running container   
`docker ps -a` list all containers (running and stopped)  
`docker images`  list all images  
`docker system prune` delete all stopped containers   
