#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import math
import time
import numpy as np
from csv import reader

class Turtle:
    def __init__(self, name):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.linear_velocity = 0
        self.angular_velocity = 0
        self.pose_subscriber = rospy.Subscriber('/'+name+"/pose", Pose, self.poseCallback)

    def poseCallback(self, pose_message):
        self.x = pose_message.x
        self.y = pose_message.y
        self.theta = pose_message.theta
        self.linear_velocity = pose_message.linear_velocity
        self.angular_velocity = pose_message.angular_velocity

    def tp_global_to_turtle(self, x_global, y_global):
        R_t = np.transpose(np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]]))
        d = np.array([[self.x],[self.y]])
        return np.transpose(R_t.dot(np.array([[x_global],[y_global]]) - d))[0].tolist()
    
    def tp_turtle_to_global(self, x_turtle, y_turtle):
        R = np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])
        d = np.array([[self.x],[self.y]])
        return np.transpose(R.dot(np.array([[x_turtle],[y_turtle]])) + d)[0].tolist()

    def get_distance(self, x=None, y=None, turtle=None):
        if turtle is None:
            return math.sqrt((self.x - x)**2 + (self.y - y)**2)
        else:
            return math.sqrt((self.x - turtle.x)**2 + (self.y - turtle.y)**2)
class Turtle_Fuzzy_Logic(Turtle):

    def __init__(self, name, max_speed=2.0):
        super().__init__(name)
        cmd_vel_topic = '/'+name+'/cmd_vel'
        self.velocity_publisher = rospy.Publisher(cmd_vel_topic, Twist, queue_size = 10)
        self.max_speed = max_speed
        with open('/home/parallels/Documents/catkin_ws/src/turtles_fuzzy_logic/src/surface_v_ang.csv', 'r') as read_obj:
            csv_reader = reader(read_obj)
            list_z = list(csv_reader)
        self.vels_angular = [list(map(float, sublist)) for sublist in list_z]
        del self.vels_angular[1::2]

        with open('/home/parallels/Documents/catkin_ws/src/turtles_fuzzy_logic/src/surface_v_lin.csv', 'r') as read_obj:
            csv_reader = reader(read_obj)
            list_z = list(csv_reader)
        self.vels_linear = [list(map(float, sublist)) for sublist in list_z]
        del self.vels_linear[1::2]

    def set_velocity(self, linear_velocity = 0.0, angular_velocity = 0.0):
        velocity_message = Twist()
        velocity_message.linear.x = linear_velocity
        velocity_message.angular.z = angular_velocity
        self.velocity_publisher.publish(velocity_message)

    def orientate(self, x_goal=None, y_goal=None, back = False, desired_angle = None):
        # self.set_velocity()
        dt = 0.001
        desired_angle_goal = (math.atan2(y_goal-self.y, x_goal-self.x) - math.pi*back) if desired_angle is None else desired_angle
        # dtheta = desired_angle_goal-self.theta
        # dtheta = math.atan2(math.sin(dtheta),math.cos(dtheta)) #Devuelve el menor angulo para girar
        # while(abs(dtheta) > 0.5):
        time.sleep(dt)
        dtheta = desired_angle_goal-self.theta
        dtheta = math.atan2(math.sin(dtheta),math.cos(dtheta)) * 180/np.pi
        angle_indx = round((dtheta + 180)/0.5)
        angular_speed = self.vels_angular[-1][angle_indx]
        self.set_velocity(angular_velocity=angular_speed)
        # self.set_velocity()

    def go_to_goal(self, x_goal, y_goal, go_back = False, threshold = 0.1, stop = True):
        dt = 0.001
        distance_indx = round((self.get_distance(x_goal, y_goal))/0.1)
        # while(distance_indx > 1):
        time.sleep(dt)
        desired_angle_goal = math.atan2(y_goal-self.y, x_goal-self.x)
        dtheta = desired_angle_goal - (self.theta - math.pi*go_back)
        dtheta = math.atan2(math.sin(dtheta),math.cos(dtheta)) * 180/np.pi
        angle_indx = round((dtheta + 180)/0.5)
        distance_indx = round((self.get_distance(x_goal, y_goal))/0.1)

        linear_speed = (1 - 2*go_back)*self.vels_linear[distance_indx][angle_indx]
        # linear_speed = linear_speed if linear_speed < self.max_speed else self.max_speed
        angular_speed = self.vels_angular[distance_indx][angle_indx]
        self.set_velocity(linear_speed, angular_speed)
        # if stop:
        #     self.set_velocity()