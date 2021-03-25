# Docker Containers

## docker
`sudo apt install docker.io`
`sudo groupadd docker`
`sudo usermod -aG docker $USER`
`newgrp docker`

### Start openvino docker and map local directory (-v /home/vm/Downloads:/home) to docker directory  

`docker run -it -v /home/vm/Downloads:/home --rm openvino/ubuntu18_dev:latest `  

#### Inside docker

`pip install tensorflow==2.2.1`
`pip install notebook`

#### save as a new image
`docker commit <docker id> <new name>`

#### start notebook
set up env variables
`source /opt/intel/openvino/bin/setupvars.sh`  
  
start notebook (full path needed as below + ip paramater)
>*Why change IP address to 0.0.0.0?
Jupyter's server runs on 127.0.0.1, which is aliased by localhost. this is not a problem if you run the server from your local machine and access on the same device. This configuration may not work if you run Jupyter on VM or a remote server.*  

`/home/openvino/.local/bin/jupyter notebook --ip=0.0.0.0`

### attach to running container  

`docker exec -it <container_name> /bin/bash`

### Tensorflow docker with jupyter notebook  

`docker run -it --rm -v /home/vm/Downloads:/tf/data -p 8888:8888 tensorflow/tensorflow:2.2.1-jupyter`

### Container management  

`docker stop <docker id>`
`docker ps -a`
`docker images`
`docker system prune`

# OpenVino docker with display

Some DL Streamer samples use display. Hence, first run the following command to allow connection from Docker container to X server on host:

`xhost local:root`

Then pull and run OpenVINO container for development:

```docker run -it \
   --device /dev/dri:/dev/dri \
   -v ~/.Xauthority:/root/.Xauthority \
   -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
   -e DISPLAY=$DISPLAY \
   -v /dev/bus/usb:/dev/bus/usb \
   --rm openvino/ubuntu18_data_dev:latest /bin/bash
   ```
   
# Optionally Docker with OpenVino tutorials

`docker run -p 8888:8888 -it artyomtugaryov/openvino_workshop`

# Optionally DL workbench
`wget https://raw.githubusercontent.com/openvinotoolkit/workbench_aux/master/start_workbench.sh && bash start_workbench.sh`

`docker run -p 127.0.0.1:5665:5665 --name workbench -it openvino/workbench:latest`
