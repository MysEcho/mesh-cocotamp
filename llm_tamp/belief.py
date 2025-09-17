from examples.pybullet.utils.pybullet_tools.pr2_problems import (
    get_bodies, create_gripper
)

from examples.pybullet.utils.pybullet_tools.pr2_primitives import (
    Pose, State, get_name, pairwise_collision,
)
from examples.pybullet.utils.pybullet_tools.utils import (
    Euler, quat_from_euler, pose_from_point_quat, get_closest_points
)
from examples.discrete_belief.dist import JDist
import numpy as np
import random
from collections import namedtuple, defaultdict

OTHER = "other"
LOCALIZED_PROB = 0.99
Interval = namedtuple('Interval', ['lower', 'upper']) # AABB
PI = np.pi
CIRCULAR_LIMITS = Interval(-PI, PI)

class BeliefTask(object):
    def __init__(self, robot, arms=tuple(), grasp_types=tuple(),
                 class_from_body={}, body_from_class={}, 
                 movables=list, surfaces=list, rooms=list, 
                 goal_localized=tuple(), goal_registered=tuple(),
                 goal_holding=tuple(), goal_on=tuple()):
        self.robot = robot
        self.arms = arms
        self.grasp_types = grasp_types
        self.class_from_body = class_from_body
        self.body_from_class = body_from_class
        self.movables = movables
        self.surfaces = surfaces
        self.rooms = rooms
        self.goal_holding = goal_holding
        self.goal_on = goal_on
        self.goal_localized = goal_localized
        self.goal_registered = goal_registered
        self.gripper = None
        self.known = set(self.surfaces + self.rooms)
    def get_bodies(self):
        return self.movables + self.surfaces + self.rooms
    def update_known(self, obj):
        self.known.add(obj)
    @property
    def fixed(self):
        movables = [self.robot] + list(self.movables)
        if self.gripper is not None:
            movables.append(self.gripper)
        return list(filter(lambda b: b not in movables, get_bodies()))
    def get_supports(self, body):
        if body in self.movables:
            return self.surfaces
        if body in self.surfaces:
            return self.rooms
        if body in self.rooms:
            return None
        raise ValueError(body)
    def get_gripper(self, arm='left'):
        if self.gripper is None:
            self.gripper = create_gripper(self.robot, arm=arm)
        return self.gripper
    
class BeliefPose(Pose):
    def __init__(self, body, value=None, support=None, init=False, prob=0.0):
        super().__init__(body, value, support, init)
        self.prob = prob

class BeliefState(State):
    def __init__(self, task: BeliefTask, ground_truth={}, b_on={}, b_in={}, particles={}, 
                 weights={}, num_particles=0, num_seen_particles={}, registered=tuple(), **kwargs):
        super(BeliefState, self).__init__(**kwargs)
        self.task = task
        self.b_on = b_on # objects surfaces
        self.b_in = b_in # objects in rooms
        self.registered = set(registered)
        self.particles = particles
        self.weights = weights
        self.ground_truth = ground_truth
        self.num_particles = num_particles
        self.num_seen_particles = num_seen_particles
        self.total_base_distance = 0

    def is_localized(self, body):
        if body in self.task.known:
            return True
        return False
    
    def get_room_body(self, surface_body):
        _, room = self.task.class_from_body[surface_body].split("|")
        room_body = self.task.body_from_class[room]
        return room_body
    
    def __repr__(self):
        indent = '  '
        surface_objs = []
        for obj, room in sorted(self.b_on.keys()):
            obj_class = self.task.class_from_body[obj]
            room_class = self.task.class_from_body[room]
            surfaces = self.b_on[(obj, room)]
            support_objs = []
            for s in sorted(surfaces.support(), key=str):
                if s is None:
                    support_objs.append(f"{indent*2}{s}: {surfaces.prob(s):.4f}")
                else:
                    support_objs.append(f"{indent*2}{self.task.class_from_body[s]}: {surfaces.prob(s):.4f}")
            support_str = ',\n'.join(support_objs)
            surface_objs.append(f"{indent}{obj_class} in {room_class}: {{\n{support_str}\n{indent}}}")
        surface_items_str = ',\n'.join(surface_objs)
        registered_str = ', '.join(map(get_name, self.registered))

        room_items = []
        for obj in sorted(self.b_in.keys()):
            obj_class = self.task.class_from_body[obj]
            rooms = self.b_in[obj]
            support_objs = []
            for r in sorted(rooms.support(), key=str):
                if r is None:
                    support_objs.append(f"{indent*2}{r}: {rooms.prob(r):.4f}")
                else:
                    support_objs.append(f"{indent*2}{self.task.class_from_body[r]}: {rooms.prob(r):.4f}")
            support_str = ',\n'.join(support_objs)
            room_items.append(f"{indent}{obj_class}: {{\n{support_str}\n{indent}}}")
        room_items_str = ',\n'.join(room_items)
        return f"{self.__class__.__name__}({{\n{surface_items_str}\n}}, {{\n{room_items_str}\n}}, [{registered_str}])"
