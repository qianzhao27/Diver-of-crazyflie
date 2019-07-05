#Author: Qian Zhao
#!/usr/bin/env python

import rospy
import math
from time import time, sleep

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from math import pow, atan2, sqrt

class Node:
    def __init__(self):
        rospy.init_node('quadrotor_controller1', anonymous=True)
        self.publisher = rospy.Publisher('drone1/cmd_vel', Twist, queue_size=10)
        self.vel = Twist()
        self.rate = rospy.Rate(10)
        self.pose = Pose()
        self.goal = Point()

    def update_pose(self, data):
        # Callback function which is called when a new message of type Pose is
        # received by the subscriber.
        self.pose.position.x = data.position.x
        self.pose.position.y = data.position.y
        self.pose.position.z = data.position.z
        self.pose.orientation.x = data.orientation.x
        
        self.vel.linear.z = self.z_vel()
        if self.perpendicular() < 0.01:
            self.vel.linear.z = 0
        
        if self.x_distance() > 0.1:
            self.vel.linear.x = 0.1
            # self.vel.linear.x = self.x_vel()
            self.vel.angular.z = self.angular_vel()
            # self.vel.angular.z = 0
        else:
            self.vel.linear.x = 0
            self.vel.angular.z = 0
        self.publisher.publish(self.vel)
        
        print(self.pose.position.x, self.pose.position.y, self.pose.position.z)
        print()

    # calculating vel by following methods
    def euclidean_distance(self):
        return sqrt(pow((self.goal.x - self.pose.position.x), 2) +
                    pow((self.goal.y - self.pose.position.y), 2) )    
    
    def perpendicular(self):
        return sqrt(pow((self.goal.z - self.pose.position.z),2))

    def x_distance(self):
        return sqrt(pow((self.goal.x - self.pose.position.x),2))

    def x_vel(self, constant = 0.8):
        return constant * self.x_distance()
   
    def linear_vel(self, constant= 0.2):
        return constant * self.euclidean_distance()

    def z_vel(self,constant= 0.5):
        return constant * (self.goal.z - self.pose.position.z)
 
    def steering_angle(self):
        if self.x_distance() > 0.1:
            return atan2(self.goal.y-self.pose.position.y, self.goal.x-self.pose.position.x)
        else:
            return 0
    
    def angular_vel(self):
        if self.steering_angle() is not 0:
            #return ((self.steering_angle()*180/math.pi) + self.pose.orientation.x)
            return self.pose.orientation.x
        else:
            return 0

def starter():
    n = Node()
    shape = 'none'
    while not rospy.is_shutdown():
            
        while shape == 'none': 
            n.vel.linear.x = 0  #set the linear motion parameters
            n.vel.linear.y = 0
            n.vel.linear.z = 0
            n.vel.angular.z = 0
            
            print('published once')
            n.publisher.publish(n.vel)         
            n.rate.sleep()
           
            
            shape = input('Do you want to fly to a goal?')
            n.goal.x = float(input("Set your x goal: "))
            n.goal.y = float(input("Set your y goal: "))
            n.goal.z = float(input("Set your z goal: "))
            
            # n.goal.x = 3.7
            # n.goal.y = 3.9
            # n.goal.z = 0.3
        subscriber = rospy.Subscriber('drone1/pose', Pose, n.update_pose)

        rospy.spin()
    

if __name__ == '__main__':
     try:
         starter()
     
     except rospy.ROSInterruptException:
         pass


