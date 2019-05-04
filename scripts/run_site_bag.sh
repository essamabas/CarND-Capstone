#!/usr/bin/env bash


pushd /opt
wget https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip
unzip traffic_light_bag_file.zip
cd traffic_light_bag_file
rosbag play -l traffic_light_bag_file.bag

pop


popd