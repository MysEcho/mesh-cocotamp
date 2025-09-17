import os
import sys
import pprint

def add_packages(packages):
    sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d)) for d in packages)

from examples.discrete_belief.dist import (
    UniformDist, DDist
)

PACKAGES = ['pddlstream']

class BeliefState():
    def __init__(self, robotNear: str, free: list, carry: list,
                 grippers:list, certain_objs: list, objects:list, 
                 locations:list, belief: tuple):
        self.robotNear = robotNear          # RobotNear ?l: which location is robot nearest to
        self.free = free                    # Free ?g: list of free grippers
        self.carry = carry                  # Carry ?o ?g: list of carrying grippers holding object
        self.grippers = grippers            # Grippers the robot has
        self.certain_objs = certain_objs    # Known objects
        self.objects = objects              # All objects in the env
        self.locations = locations          # All locations in the env
        self.belief = belief                # Belief of current state

    def print(self):
        pprint.pprint(self.belief)

def get_initial_belief(objects:list, locations: list):
    # P(obj at loc)
    dist = {}
    for object in objects:
        dist[object] = UniformDist(locations)
    return dist

def get_locations(env: dict):
    return list(env.keys())

def get_true_state():
    state = {
        'livingroom_table': ['spoon', 'fork'],
        'livingroom_shelf': ['book'],
        'kitchen_refrigerator': ['tomato', 'cabbage'],
        'kitchen_sink': [],
        'playroom_cabinet': ['baseball', 'lego']
        }
    return state

def main():
    objects = ['cabbage', 'tomato', 'spoon', 'fork', 'book', 'baseball', 'lego']
    locations = ['livingroom_table', 
                 'livingroom_shelf',
                 'kitchen_refrigerator',
                 'kitchen_sink',
                 'playroom_cabinet']
    obj_loc_belief = get_initial_belief(objects, locations)
    pprint.pprint(obj_loc_belief)


if __name__ == '__main__':
    main()
