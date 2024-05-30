docker run -i -d --runtime nvidia --name ros2 \
            -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all\
            -v /tmp/.X11-unix:/tmp/.X11-unix --net=host --ipc=host --pid=host \
            -v ~/:/root/external_home \
            ros2_cu11
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ros2`

docker exec -it ros2 zsh
# -e QT_X11_NO_MITSHM=1 -e XAUTHORITY 
