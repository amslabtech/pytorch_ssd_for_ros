#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

from cv_bridge import CvBridge, CvBridgeError

import cv2
import PIL.Image as Image
import math
import numpy as np
import argparse
import yaml
import os
import sys

#Need in running in ROS
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.utils.misc import Timer

class ImageInfer:
    def __init__(self):
        print("Start Image Inference")

        self.bridge = CvBridge()

        self.color_img_cv = np.empty(0)
        self.inferenced_image = np.empty(0)
        
        self.subscribe_topic_name = rospy.get_param('~subscribe_topic_name', '/image0')
        self.advertise_topic_name = rospy.get_param('~advertise_topic_name', '/inferenced_image')

        self.weight_path = rospy.get_param('~weight_path', '/home/ros_catkin_ws/src/pytorch_ssd_ros/models/default/mb2-ssd-lite-mp-0_686.pth')
        self.label_path = rospy.get_param('~label_path', '/home/ros_catkin_ws/src/pytorch_ssd_ros/models/default/voc-model-labels.txt')

        self.net_type = rospy.get_param('~net_type', '/mb2')

        self.sub_color_img = rospy.Subscriber(self.subscribe_topic_name, ImageMsg, self.callbackColorImage, queue_size=1, buff_size=2**24)
        self.pub_inferenced_image = rospy.Publisher(self.advertise_topic_name, ImageMsg, queue_size=1)

        print("Net Type:", self.net_type)
        print("Model:", self.weight_path)
        print("Label:", self.label_path)

        self.class_names = [name.strip() for name in open(self.label_path).readlines()]
        self.num_classes = len(self.class_names)

        if self.net_type == 'vgg16-ssd':
            net = create_vgg_ssd(len(self.class_names), is_test=True)
        elif self.net_type == 'mb1-ssd':
            net = create_mobilenetv1_ssd(len(self.class_names), is_test=True)
        elif self.net_type == 'mb1-ssd-lite':
            net = create_mobilenetv1_ssd_lite(len(self.class_names), is_test=True)
        elif self.net_type == 'mb2-ssd-lite':
            net = create_mobilenetv2_ssd_lite(len(self.class_names), is_test=True)
        elif self.net_type == 'mb3-large-ssd-lite':
            net = create_mobilenetv3_large_ssd_lite(len(self.class_names), is_test=True)
        elif self.net_type == 'mb3-small-ssd-lite':
            net = create_mobilenetv3_small_ssd_lite(len(self.class_names), is_test=True)
        elif self.net_type == 'sq-ssd-lite':
            net = create_squeezenet_ssd_lite(len(self.class_names), is_test=True)
        else:
            print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
            sys.exit(1)
        net.load(self.weight_path)

        if self.net_type == 'vgg16-ssd':
            self.predictor = create_vgg_ssd_predictor(net, candidate_size=200)
        elif self.net_type == 'mb1-ssd':
            self.predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
        elif self.net_type == 'mb1-ssd-lite':
            self.predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
        elif self.net_type == 'mb2-ssd-lite' or self.net_type == "mb3-large-ssd-lite" or self.net_type == "mb3-small-ssd-lite":
            self.predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
        elif self.net_type == 'sq-ssd-lite':
            self.predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
        else:
            print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
            sys.exit(1)

        #self.timer = Timer()
        
    def callbackColorImage(self, msg):
        try:
            self.color_img_cv = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            print("msg.encoding = ", msg.encoding)
            print("self.color_img_cv.shape = ", self.color_img_cv.shape)

            self.inferenced_image = self.inference(self.color_img_cv)
            imgMsg = self.bridge.cv2_to_imgmsg(self.inferenced_image)
            self.pub_inferenced_image.publish(imgMsg)

        except CvBridgeError as e:
            print(e)

    def inference(self, color_img_cv):

        orig_image = color_img_cv

        image = cv2.cvtColor(color_img_cv, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = self.predictor.predict(image, 10, 0.4)
        #interval = self.timer.end()
        #print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            label = f"{self.class_names[labels[i]]}: {probs[i]:.2f}"

            #print(labels[i])
            #print(self.class_names[labels[i]])

            cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)

            cv2.putText(orig_image, label,
                        (int(box[0]+20), int(box[1]+40)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type

        #print(labels.size())
        #print(probs.size())

        return orig_image

        



def main():
    #Set Up in ROS node
    rospy.init_node('ros_infer', anonymous=True)
    image_infer = ImageInfer()
    rospy.spin()

if __name__ == '__main__':
    main()