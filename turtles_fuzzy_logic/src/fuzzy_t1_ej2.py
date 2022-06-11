#! /usr/bin/env python3

from Turtle_fuzzy_logic import *

if __name__ == '__main__':
    rospy.init_node('trutle_fuzzy', anonymous = True)
    turtle_main = Turtle_Fuzzy_Logic('turtle1')
    turtle_pursuited = Turtle('turtle2')
    time.sleep(3)

    while True:
        turtle_main.orientate(turtle_pursuited.x, turtle_pursuited.y)