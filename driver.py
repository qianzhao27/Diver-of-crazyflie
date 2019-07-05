###!/usr/bin/env python3
### Author: Qian Zhao

import rospy
import cflib
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion

import logging
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.log import LogConfig
from cflib.positioning.motion_commander import MotionCommander

logging.basicConfig(level=logging.ERROR)
URI = 'radio://1/7/1M'


class Node:

    def __init__(self):
        
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.va = 0
        
        self.vel = Twist()
        self.pose = Pose()

        self.vel.linear.x = self.vx
        self.vel.linear.y = self.vy
        self.vel.linear.z = self.vz
        self.vel.angular.z = self.va

        cflib.crtp.init_drivers(enable_debug_driver=False)
        self.crazyflie = SyncCrazyflie(URI, cf = Crazyflie(rw_cache='./cache'))
        self.commander = MotionCommander(self.crazyflie)
        self.cf = Crazyflie()
        self.crazyflie.open_link()
        self.commander.take_off()

    def write_to_file(self, data):
        # Two loggers should yield an even number of rows of data
        # being collected in the end.
        # There is one packet missing if the array only contains
        # an even number of rows of data.
        if len(data) % 2 is not 0:
            data = data[:len(data) - 1]

        temp_df = pd.DataFrame(data)
        m, _ = temp_df.shape
        even_rows = temp_df.iloc[np.arange(0, m, 2), :]
        odd_rows = temp_df.iloc[np.arange(1, m, 2), :]

        columns = ['timestamp_start', 'timestamp_end',
                   'acc.x', 'acc.y', 'acc.z',
                   'gyro.x', 'gyro.y', 'gyro.z',
                   'stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z',
                   'stabilizer.pitch', 'stabilizer.roll', 'stabilizer.yaw']

        df = pd.DataFrame(data=np.zeros((int(m / 2), 14)), columns=columns)
        df[['gyro.x', 'gyro.y', 'gyro.z']] = np.array(even_rows[['gyro.x', 'gyro.y', 'gyro.z']])
        df[['stabilizer.pitch', 'stabilizer.roll', 'stabilizer.yaw']] = np.array(
            even_rows[['stabilizer.pitch', 'stabilizer.roll', 'stabilizer.yaw']])
        df[["acc.x", "acc.y", "acc.z"]] = np.array(odd_rows[["acc.x", "acc.y", "acc.z"]])
        df[["stateEstimate.x", "stateEstimate.y", "stateEstimate.z"]] = \
            np.array(odd_rows[["stateEstimate.x", "stateEstimate.y", "stateEstimate.z"]])
        df['timestamp_start'] = np.array(even_rows.timestamp)
        df['timestamp_end'] = np.array(odd_rows.timestamp)

        # df.to_csv("data/project2/drone2/data_set_label_"
        #    +class_label+"_packet_"+packet_num+".csv")
        df.to_csv("test.csv")

    def write(self, data):
        print(data)

    def log1(self):

        lg1 = LogConfig(name='pose_acc', period_in_ms=10)

        lg1.add_variable('stateEstimate.x', 'float')
        lg1.add_variable('stateEstimate.y', 'float')
        lg1.add_variable('stateEstimate.z', 'float')

        lg1.add_variable('acc.x', 'float')
        lg1.add_variable('acc.y', 'float')
        lg1.add_variable('acc.z', 'float')

        return lg1

    def log2(self):

        lg2 = LogConfig(name='stabilizer_gyro', period_in_ms=10)
 
        lg2.add_variable('stabilizer.roll', 'float')
        lg2.add_variable('stabilizer.pitch', 'float')
        lg2.add_variable('stabilizer.yaw', 'float')

        lg2.add_variable('gyro.x', 'float')
        lg2.add_variable('gyro.y', 'float')
        lg2.add_variable('gyro.z', 'float')

        return lg2

    def sync(self, position_pub, data):

        switch = 0
        with SyncLogger(self.crazyflie, self.log1()) as logger1, \
                SyncLogger(self.crazyflie, self.log2()) as logger2:
            end_time = time.time() + 24
            while time.time() < end_time:
                if switch == 0: 
                    logger = logger2
                elif switch == 1:
                    logger = logger1

                for log_entry in logger:
                    row = log_entry[1]
                    row["timestamp"] = log_entry[0]
                    if switch == 1:
                        x = row['stateEstimate.x']
                        y = row['stateEstimate.y']
                        z = row['stateEstimate.z']

                        self.pose.position.x = x
                        self.pose.position.y = y
                        self.pose.position.z = z

                        position_pub.publish(self.pose)
                        print('x:',x,' y:',y,' z:',z)
                        print()
                    data.append(row)
                    switch += 1
                    break

                if switch == 2:
                    switch = 0
            return None

    def shut_down(self):
        self.crazyflie.close_link()

    def cmd_vel(self, msg):
    
        self.vx = msg.linear.x
        self.vy = msg.linear.y
        self.vz = msg.linear.z
        self.va = msg.angular.z
        self.commander._set_vel_setpoint(self.vx,self.vy,self.vz,self.va)

        print(self.vx, self.vy, self.vz, self.va) 


def run():
    data =[]
    rospy.init_node('drone1')
    node = Node()
    cmdVel_subscribe = rospy.Subscriber('drone1/cmd_vel', Twist, node.cmd_vel)
    position_pub = rospy.Publisher('drone1/pose', Pose, queue_size=10)
    timer = time.time()  

    while time.time()-timer < 24:
        node.sync(position_pub, data)
    print("End")

    node.write_to_file(data)
    node.shut_down()


if __name__ == '__main__':
    run()

