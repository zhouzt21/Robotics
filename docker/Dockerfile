# from nvidia/cudagl:11.4.2-devel-ubuntu20.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

# Install language
RUN apt-get update && apt-get install -y \
  locales \
  && locale-gen en_US.UTF-8 \
  && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
  && rm -rf /var/lib/apt/lists/*
ENV LANG en_US.UTF-8

# Install timezone
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
  && export DEBIAN_FRONTEND=noninteractive \
  && apt-get update \
  && apt-get install -y tzdata \
  && dpkg-reconfigure --frontend noninteractive tzdata \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get -y upgrade \
    && rm -rf /var/lib/apt/lists/*

# Install common programs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    lsb-release \
    sudo \
    software-properties-common \
    wget \
    && rm -rf /var/lib/apt/lists/*


RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- -p autojump

RUN git clone --single-branch https://github.com/wting/autojump.git 
RUN export SHELL='zsh' && cd autojump && ./install.py && cd .. && rm -rf autojump


# Install ROS2
RUN sudo add-apt-repository universe \
  && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null \
  && apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base \
    python3-argcomplete \
  && rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=humble
ENV AMENT_PREFIX_PATH=/opt/ros/humble
ENV COLCON_PREFIX_PATH=/opt/ros/humble
ENV LD_LIBRARY_PATH=/opt/ros/humble/lib
ENV PATH=/opt/ros/humble/bin:$PATH
ENV PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages
ENV ROS_PYTHON_VERSION=3
ENV ROS_VERSION=2
ENV DEBIAN_FRONTEND=

###########################################
#  Develop image
###########################################
FROM base AS dev

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  bash-completion \
  build-essential \
  rsync \
  cmake \
  psmisc \
  gdb \
  git \
  tmux \
  fluxbox \
  x11vnc \
  openssh-client \
  python3-argcomplete \
  python3-pip \
  ros-dev-tools \
  ros-humble-ament-* \
  vim \
  && rm -rf /var/lib/apt/lists/*

RUN rosdep init || echo "rosdep already initialized"

# ARG USERNAME=ros
# ARG USER_UID=1000
# ARG USER_GID=$USER_UID

# Create a non-root user
# RUN groupadd --gid $USER_GID $USERNAME \
#   && useradd -s /bin/zsh --uid $USER_UID --gid $USER_GID -m $USERNAME \
#   # Add sudo support for the non-root user
#   && apt-get update \
#   && apt-get install -y sudo \
#   && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
#   && chmod 0440 /etc/sudoers.d/$USERNAME \
#   && rm -rf /var/lib/apt/lists/*

# # Set up autocompletion for user
# RUN apt-get update && apt-get install -y git-core bash-completion \
#   && echo "if [ -f /opt/ros/${ROS_DISTRO}/setup.zsh ]; then source /opt/ros/${ROS_DISTRO}/setup.zsh; fi" >> /home/$USERNAME/.zshrc \
#   && echo "if [ -f /usr/share/colcon_argcomplete/hook/colcon-argcomplete.zsh ]; then source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.zsh; fi" >> /home/$USERNAME/.zshrc \
#   && rm -rf /var/lib/apt/lists/* 

ENV DEBIAN_FRONTEND=
ENV AMENT_CPPCHECK_ALLOW_SLOW_VERSIONS=1

###########################################
#  Full image
###########################################
FROM dev AS full

ENV DEBIAN_FRONTEND=noninteractive
# Install the full release
RUN apt-get update && apt-get install -y --no-install-recommends \
  ros-humble-desktop \
  && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=

###########################################
#  Full+Gazebo image
###########################################
FROM full AS gazebo

ENV DEBIAN_FRONTEND=noninteractive
# Install gazebo
RUN wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null \
  && apt-get update && apt-get install -q -y --no-install-recommends \
    ros-humble-gazebo* \
  && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=

###########################################
#  Full+Gazebo+Nvidia image
###########################################

FROM gazebo AS gazebo-nvidia

################
# Expose the nvidia driver to allow opengl 
# Dependencies for glvnd and X11.
################
RUN apt-get update \
 && apt-get install -y -qq --no-install-recommends \
  libglvnd0 \
  libgl1 \
  libglx0 \
  libegl1 \
  libxext6 \
  libx11-6

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
ENV QT_X11_NO_MITSHM 1


## -----------------------------------------------------------
# Copy files.
# COPY workspace /workspace
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json

RUN pip install --no-cache torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache tensorboardX ninja cython opencv-python gymnasium tqdm gym==0.23.1 yacs==0.1.8 matplotlib transforms3d h5py moviepy coacd mujoco mujoco_py trimesh omegaconf paramiko termcolor prompt_toolkit wandb pandas yapf sorcery tensorboardX pynvml lmdb open3d

    # && pip install --no-cache stable_baselines3[extra] \
    # && pip install --no-cache --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \

RUN pip install --no-cache setuptools --upgrade && pip install --no-cache torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html 
RUN apt update && apt install -y mesa-utils ninja-build stow cmake tmux && ln -s /root/external_home/RoboLab /root/RobotLab

RUN apt install -y ros-humble-turtlebot4-desktop && sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list' && wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add - && apt-get update && apt-get install -y ignition-fortress && apt install -y ros-humble-turtlebot4-simulator ros-humble-turtlebot4-description ros-humble-turtlebot4-msgs ros-humble-turtlebot4-navigation ros-humble-turtlebot4-node

# RUN echo "if [ -f /opt/ros/${ROS_DISTRO}/setup.zsh ]; then source /opt/ros/${ROS_DISTRO}/setup.zsh; fi" >> /root/.zshrc


RUN pip3 install pygccxml pyplusplus

RUN mkdir -p /OPT
WORKDIR /OPT

# libccd
RUN git clone --single-branch -b v2.1 --depth 1 https://github.com/danfis/libccd.git && \
    cd libccd && mkdir build && cd build && \
    cmake -G "Unix Makefiles" .. && make -j && make install && \
    rm -rf /workspace/libccd

# boost (require >= 1.58)
# Reference: https://www.boost.org/doc/libs/1_76_0/more/getting_started/unix-variants.html#easy-build-and-install
# NOTE(jigu): there are compilation errors when boost.python is also built.
# To build boost.python, maybe we need to refer to https://www.boost.org/doc/libs/1_35_0/libs/python/doc/building.html#examples
RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz && \
    tar -xf boost_1_76_0.tar.gz && \
    rm boost_1_76_0.tar.gz && \
    cd boost_1_76_0 && ./bootstrap.sh --without-libraries=python && ./b2 install && \
    rm -rf /workspace/boost_1_76_0

RUN apt-get install -y castxml libyaml-cpp-dev

RUN git clone --single-branch -b 1.6.0 --depth 1 --recurse-submodules https://github.com/ompl/ompl.git && \
    cd ompl && mkdir build && cd build && \
    cmake .. && make -j && make install && \
    rm -rf /workspace/ompl

# -------------------------------------------------------------------------- #
# FCL
# -------------------------------------------------------------------------- #
# octomap (for octree collision)
RUN git clone --single-branch -b v1.9.7 --depth 1 https://github.com/OctoMap/octomap.git && \
    cd octomap/octomap && mkdir build && cd build && \
    cmake .. && make -j && make install && \
    rm -rf /workspace/octomap

# v0.6.1 does not work, use newer instead
RUN git clone --single-branch https://github.com/flexible-collision-library/fcl.git && \
    cd fcl && git checkout 7fcdc7f09bedb3d9544bfce067b01298873ad906 && mkdir build && cd build && \
    cmake .. && make -j && make install && \
    rm -rf /workspace/fcl


# -------------------------------------------------------------------------- #
# pinocchio
# -------------------------------------------------------------------------- #
RUN git clone --single-branch -b 20210000.6 --depth 1 https://github.com/coin-or/CppAD.git && \
    cd CppAD && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j && make install && \
    rm -rf /workspace/CppAD

RUN git clone --single-branch -b 0.3.2 --depth 1 https://github.com/ros/console_bridge.git && \
    cd console_bridge && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j && make install && \
    rm -rf /workspace/console_bridge

RUN git clone --single-branch -b 1.0.5 --depth 1 https://github.com/ros/urdfdom_headers.git && \
    cd urdfdom_headers && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j && make install && \
    rm -rf /workspace/urdfdom_headers

RUN git clone --single-branch -b 1.0.4 --depth 1 https://github.com/ros/urdfdom.git && \
    cd urdfdom && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j && make install && \
    rm -rf /workspace/urdfdom

RUN git clone --single-branch -b v2.5.6 --depth 1 https://github.com/stack-of-tasks/pinocchio.git && \
    cd pinocchio && git submodule update --init --recursive && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_WITH_AUTODIFF_SUPPORT=ON -DBUILD_WITH_URDF_SUPPORT=ON && make -j && make install && \
    rm -rf /workspace/pinocchio

# -------------------------------------------------------------------------- #
# Others
# -------------------------------------------------------------------------- #
RUN git clone --single-branch -b v5.0.1 --depth 1 https://github.com/assimp/assimp.git && \
    cd assimp && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DASSIMP_BUILD_TESTS=OFF && make -j && make install && \
    rm -rf /workspace/assimp

RUN git clone --single-branch -b v1.5.0 --depth 1 https://github.com/orocos/orocos_kinematics_dynamics.git && \
    cd orocos_kinematics_dynamics/orocos_kdl && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j && make install && \
    rm -rf /workspace/orocos_kinematics_dynamics


ARG SSH_PRIVATE_KEY
RUN mkdir ~/.ssh/ && echo "${SSH_PRIVATE_KEY}" > ~/.ssh/id_rsa_git && chmod 600 ~/.ssh/id_rsa_git && ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
# RUN echo "Host github.com\n\tStrictHostKeyChecking no\n\tHostName github.com\n\tIdentityFile ~/.ssh/id_rsa_git" >> ~/.ssh/config
# RUN ssh-keyscan /github.com/ >> ~/.ssh/known_hosts

# RUN cat ~/.ssh/config
# RUN GIT_SSH_COMMAND="ssh -i /root/.ssh/id_rsa_git -o IdentitiesOnly=yes"  
RUN pip install --no-cache https://github.com/haosulab/SAPIEN/releases/download/nightly/sapien-3.0.0.dev20231129-cp310-cp310-manylinux2014_x86_64.whl 
# RUN pip install --no-cache git+git@github.com:haosulab/ManiSkill3.git


RUN echo "source /opt/ros/${ROS_DISTRO}/setup.zsh" >> /root/.zshrc \
  && echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.zsh" >> /root/.zshrc \
  && echo "export GIT_SSH_COMMAND=\"ssh -i /root/.ssh/id_rsa_git -o IdentitiesOnly=yes\"" >> ~/.zshrc \
  && echo "[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh" >> ~/.zshrc \
  && echo "source /root/RobotLab/ros2_ws/install/setup.zsh" >> ~/.zshrc \
  && echo "autoload -U bashcompinit" >> /root/.zshrc \
  && echo "bashcompinit" >> /root/.zshrc \
  && echo "eval \"\$(register-python-argcomplete3 ros2)\"" >> /root/.zshrc \
  && echo "eval \"\$(register-python-argcomplete3 colcon)\"" >> /root/.zshrc


# RUN echo "source /opt/ros/${ROS_DISTRO}/setup.zsh" >> /root/.zshrc
WORKDIR /root/RobotLab
RUN git config --global safe.directory '*' 
# && cd ros_ws && colcon build --symlink-install
COPY .p10k.zsh /root/.p10k.zsh

# RUN /bin/zsh /root/.zshrc
# ENTRYPOINT ["chsh -s ~/.zshrc; zsh"]

RUN LAZYGIT_VERSION=$(curl -s "https://api.github.com/repos/jesseduffield/lazygit/releases/latest" | grep -Po '"tag_name": "v\K[^"]*') && curl -Lo lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/latest/download/lazygit_${LAZYGIT_VERSION}_Linux_x86_64.tar.gz" && tar xf lazygit.tar.gz lazygit && install lazygit /usr/local/bin

RUN chsh -s /bin/zsh
ENTRYPOINT [ "/bin/zsh" ]
