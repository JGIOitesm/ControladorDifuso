#! /usr/bin/env python3

from Turtle_fuzzy_logic import *

if __name__ == '__main__':
    rospy.init_node('trutle_fuzzy', anonymous = True)
    turtle_main = Turtle_Fuzzy_Logic('turtle1')
    turtle_pursuited = Turtle('turtle2')
    time.sleep(3)

    while True:
        if turtle_main.get_distance(turtle=turtle_pursuited) > 1:
            turtle_main.go_to_goal(turtle_pursuited.x, turtle_pursuited.y)
        else:
            turtle_main.set_velocity()