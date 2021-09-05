FROM nvidia/cuda:10.0-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
	vim \
	wget \
	unzip \
	git \
	build-essential

RUN apt-get update && \
	apt-get install -y python3-pip

RUN apt install -y curl

RUN pip3 install \
		tqdm \
		matplotlib \
		tensorflow \
		tensorboardX \
		scikit-build \
		cmake

RUN pip3 install pandas \
				boto3

RUN pip3 install -U https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl

RUN pip3 install torchvision==0.2.1

RUN pip3 install --upgrade pip
RUN pip3 install opencv-python

########## ROS Melodic insatall ##########
## NOTE: "lsb_release" -> "lsb-release"
RUN apt-get update && apt-get install -y lsb-release &&\
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt update && \
    apt install -y ros-melodic-desktop-full

########## ROS setup ##########
RUN mkdir -p /home/ros_catkin_ws/src && \
	cd /home/ros_catkin_ws/src && \
	/bin/bash -c "source /opt/ros/melodic/setup.bash; catkin_init_workspace" && \
	cd /home/ros_catkin_ws && \
	/bin/bash -c "source /opt/ros/melodic/setup.bash; catkin_make" && \
	echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc && \
	echo "source /home/ros_catkin_ws/devel/setup.bash" >> ~/.bashrc && \
	echo "export ROS_PACKAGE_PATH=\${ROS_PACKAGE_PATH}:/home/ros_catkin_ws" >> ~/.bashrc && \
	echo "export ROS_WORKSPACE=/home/ros_catkin_ws" >> ~/.bashrc
## cmk
RUN echo "function cmk(){\n	lastpwd=\$OLDPWD \n	cpath=\$(pwd) \n cd /home/ros_catkin_ws \n catkin_make \$@ \n cd \$cpath \n	OLDPWD=\$lastpwd \n}" >> ~/.bashrc
########## dnn_attitude_estimation ##########
##### NO cache #####
ARG CACHEBUST=1

RUN cd /home/ros_catkin_ws/src && \
		cd /home/ros_catkin_ws && \
		/bin/bash -c "source /opt/ros/melodic/setup.bash; catkin_make"


RUN mkdir /home/models && \
    cd /home/models && \
    wget -P models https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth && \
    wget -P models https://storage.googleapis.com/models-hao/voc-model-labels.txt

RUN cd /home/ros_catkin_ws/src

WORKDIR /home/ros_catkin_ws/src