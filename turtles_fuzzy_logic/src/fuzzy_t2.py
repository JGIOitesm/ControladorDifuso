#! /usr/bin/env python3

import rospy
import numpy as np
import random
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import math
import time
from std_srvs.srv import Empty

x = 0
y = 0
z = 0
theta = 0

def poseCallback(pose_message):
    global x
    global y
    global z
    global theta
    
    x = pose_message.x
    y = pose_message.y
    theta = pose_message.theta

def orientate (xgoal, ygoal):
    
    global x
    global y
    global theta
    pm_theta = 0.0
    pm_dtheta = 0.0
    
    velocity_message = Twist()
    cmd_vel_topic = '/turtle1/cmd_vel'

    while(True):
    #orientation closed-loop
        #turtle's angle from 0 to 2pi fotmat to -pi to +pi format
        if theta > math.pi:
            pm_theta = theta-(2*math.pi)
        else:
            pm_theta = theta
        #goal's angle (from 0) and angular distance from turtle's angle
        desired_angle_goal = math.atan2(ygoal-y, xgoal-x)
        pm_dtheta = desired_angle_goal - pm_theta
        #selecting turn direction (ccw or cw respectively)                      
        if abs(pm_dtheta) < math.pi:
            dtheta = desired_angle_goal - pm_theta      
        else:
            desired_angle_goal = desired_angle_goal
            dtheta = pm_theta - desired_angle_goal
        #does it meet the angular distance tolerance?    
        if (abs(pm_dtheta) < 0.005):
            print ('angle: ', round(theta*360.00/6.2831,5), 'rad reached')
            time.sleep(1)
            break
        #angular speed and publishing
        ka = 4.0 
        angular_speed = ka * dtheta
        velocity_message.linear.x = 0.0
        velocity_message.angular.z = angular_speed
        velocity_publisher.publish(velocity_message)
        
def go_to_goal (xgoal, ygoal):
            
    global x
    global y
    global theta

    velocity_message = Twist()
    cmd_vel_topic = '/turtle1/cmd_vel'
       
    while(True):
    #orientation closed-loop
       #turtle's angle from 0 to 2pi fotmat to -pi to +pi format
        if theta > math.pi:
            pm_theta = theta-(2*math.pi)
        else:
            pm_theta = theta
    #goal's angle (from 0) and angular distance from turtle's angle
        desired_angle_goal = math.atan2(ygoal-y, xgoal-x)
        pm_dtheta = desired_angle_goal - pm_theta
    #selecting turn direction (ccw or cw respectively)                      
        if abs(pm_dtheta) < math.pi:
            dtheta = desired_angle_goal - pm_theta         
        else:
            desired_angle_goal = desired_angle_goal
            dtheta = pm_theta - desired_angle_goal
    #angular speed
        ka = 4.0
        if (abs(ka * dtheta) > 0.05):  
            angular_speed = ka * dtheta
        else:
            angular_speed = 0.0

    #position closed-loop
        kv = 2.0               
        distance = abs(math.sqrt(((xgoal-x)**2)+((ygoal-y)**2)))
        linear_speed = kv * distance

        if (distance < 0.1):
            time.sleep(0.1)
            break        

        velocity_message.linear.x = linear_speed
        velocity_message.angular.z = angular_speed
        velocity_publisher.publish(velocity_message)

if __name__ == '__main__':
    try:

        rospy.init_node('turtlesim_motion_pose', anonymous = True)

        cmd_vel_topic = '/turtle2/cmd_vel'
        velocity_publisher = rospy.Publisher(cmd_vel_topic, Twist, queue_size = 10)

        position_topic = "/turtle2/pose"
        pose_subscriber = rospy.Subscriber(position_topic, Pose, poseCallback)


####CUADRADO#####
        time.sleep(1.0)
        while(True):
            
# diamond
            orientate(5.5,10.0)
            go_to_goal(5.5,10.0) 
            orientate(1.0,5.5)
            go_to_goal(1.0,5.5)
            orientate(5.5,1.0)
            go_to_goal(5.5,1.0)            
            orientate(10.0,5.5)
            go_to_goal(10.0,5.5)
            orientate(5.5,1.0)
            go_to_goal(5.5,1.0) 
            orientate(1.0,5.5)
            go_to_goal(1.0,5.5)
            orientate(5.5,10.0)
            go_to_goal(5.5,10.0)                     
            orientate(10.0,5.5)
            go_to_goal(10.0,5.5)
          
    except rospy.ROSInterruptException:        
        pass
