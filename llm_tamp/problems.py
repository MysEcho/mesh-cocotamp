import os
import time
import sys
PACKAGES = ['pddlstream']

def add_packages(packages):
    sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d)) for d in packages)
from examples.pybullet.utils.pybullet_tools.pr2_primitives import Pose
from examples.pybullet.pr2_belief.problems import set_delta_belief
from examples.pybullet.utils.pybullet_tools.pr2_problems import (
    REST_LEFT_ARM,
    create_box, 
    create_pr2, 
    get_carry_conf,
    get_other_arm, 
    set_point, 
    set_pose,
    set_arm_conf,
    set_color, 
    open_arm, 
    close_arm, 
    arm_conf
)
from examples.discrete_belief.dist import (
    DDist, UniformDist, DeltaDist, MixtureDD, JDistIndep, JDist
)
from examples.pybullet.utils.pybullet_tools.utils import (
    HideOutput, AABB,
    get_aabb, 
    Pose,
    Point,
    Euler,
    sample_aabb,
    get_center_extent,
    get_point
)
# from llm_tamp.ask_mcq import create_room_belief, create_surface_belief
from llm_tamp.ask_mcq import create_room_belief, create_surface_belief
from llm_tamp.belief import BeliefState, BeliefTask
import numpy as np
import random
from collections import namedtuple
from typing import Tuple, Dict
from itertools import product
import pprint
random.seed(42)
np.random.seed(42)

USE_DRAKE_PR2 = True
OTHER = "other"
LOCALIZED_PROB = 0.99
Interval = namedtuple('Interval', ['lower', 'upper']) # AABB
PI = np.pi
CIRCULAR_LIMITS = Interval(-PI, PI)
FLOOR_HEIGHT = 1e-3
RGBA = namedtuple('RGBA', ['red', 'green', 'blue', 'alpha'])

def get_room_sections():
    y_axes = [-6, 6]
    x_axes = [-11.25, -3.75, 3.75, 11.25]
    return [(x,y) for x, y in product(x_axes, y_axes)]

def get_room_colors():
    return [
        RGBA(0.824, 0.706, 0.549, 1),  # Tan
        RGBA(0.678, 0.847, 0.902, 1),  # Light blue
        RGBA(0.941, 0.502, 0.502, 1),  # Salmon
        RGBA(0.565, 0.933, 0.565, 1),  # Light green
        RGBA(0.855, 0.647, 0.855, 1),  # Plum
        RGBA(0.933, 0.910, 0.667, 1),  # Khaki
        RGBA(0.678, 0.847, 0.737, 1),  # Aquamarine
        RGBA(0.961, 0.871, 0.702, 1),  # Wheat
    ]

def create_floor_carpets(sections, room_names):
    body_from_class = {}
    class_from_body = {}
    room_colors = get_room_colors()
    
    for i, section in enumerate(sections):
        x, y = section
        color = room_colors[i % len(room_colors)]
        floor_carpet = create_box(w=7.5, l=12, h=FLOOR_HEIGHT, color=color, collision=True)
        floor_pose = Pose(point=Point(x=x, y=y, z=2*FLOOR_HEIGHT))
        set_pose(floor_carpet, floor_pose)
        
        if i < len(room_names):
            room_name = room_names[i]
            body_from_class[room_name] = floor_carpet
            class_from_body[floor_carpet] = room_name
            
    return body_from_class, class_from_body

def create_surface_with_objects(room, surface, section, j, width, height, mass, body_from_class, class_from_body, sampled_env):
    x, y = section
    if j == 0:
        y = y - 1.25
    elif j == 1:
        y = y + 1.25
    elif j == 2:
        y = y + 3.75
    elif j == 3:
        y = y - 3.75
    # Create surface
    surface_box = create_box(width, 4*width, height, color=(0.25, 0.25, 0.75, 1))
    pose = Pose(point=Point(x=x, y=y, z=height/2), euler=Euler(yaw=3.141592/2))
    set_pose(surface_box, pose)
    set_point(surface_box, (x, y, height / 2))
    
    body_from_class[surface+"|"+room] = surface_box
    class_from_body[surface_box] = surface+"|"+room
    
    # Create obstacles
    obstacle_box_1 = create_box(0.07, width, height, color=(1, 1, 0, 0.5))
    obstacle_box_2 = create_box(0.07, width, height, color=(1, 1, 0, 0.5))
    pose_1 = Pose(point=Point(x=x+2*width/3, y=y, z=height + height/2), euler=Euler(yaw=0))
    pose_2 = Pose(point=Point(x=x-2*width/3, y=y, z=height + height/2), euler=Euler(yaw=0))
    set_pose(obstacle_box_1, pose_1)
    set_pose(obstacle_box_2, pose_2)
    
    obstacles = {surface_box: obstacle_box_1}
    movables = []

    # Create objects
    obj_positions = [
        (x + 0.17, y + 0.17), (x + 0.17, y - 0.17),
        (x - 0.17, y + 0.17), (x - 0.17, y - 0.17),
        (x + 0.78, y + 0.17), (x + 0.78, y - 0.17),
        (x - 0.78, y + 0.17), (x - 0.78, y - 0.17)
    ]
    
    for sampled_obj, (obj_x, obj_y) in zip(sampled_env[room][surface], obj_positions):
        obj_box = create_box(0.07, 0.07, 0.07, mass=mass, color=(0.6, 0.6, 0.6, 1))
        set_point(obj_box, (obj_x, obj_y, height + 0.035))
        body_from_class[sampled_obj] = obj_box
        class_from_body[obj_box] = sampled_obj
        movables.append(obj_box)

    return surface_box, obstacles, movables

def create_sampled_env(sampled_env: dict, width: float = 0.7, height: float = 0.7):
    mass = 1
    
    # Initialize containers
    movables = []
    surfaces = []
    rooms = []
    obstacles = {}
    ground_truth = {}
    
    # Create room sections and floor carpets
    sections = get_room_sections()
    room_names = list(sampled_env.keys())
    body_from_class, class_from_body = create_floor_carpets(sections, room_names)
    
    # Create surfaces and objects
    for i, room in enumerate(sampled_env):
        rooms.append(body_from_class[room])
        ground_truth[body_from_class[room]] = []
        for j, surface in enumerate(sampled_env[room]):
            surface_box, surf_obstacles, surf_movables = create_surface_with_objects(
                room, surface, sections[i], j, width, height, mass,
                body_from_class, class_from_body, sampled_env
            )
            surfaces.append(surface_box)
            obstacles.update(surf_obstacles)
            movables.extend(surf_movables)
            ground_truth[body_from_class[room]].append(surface_box)
    return ground_truth, body_from_class, class_from_body, movables, surfaces, rooms, obstacles

def get_sampled_task(sampled_env: dict, target_objs: list, goal_surfaces: list, arm: str = "left", grasp_type: str = "top") -> BeliefTask:
    with HideOutput():
        pr2 = create_pr2(use_drake=USE_DRAKE_PR2)
    set_arm_conf(pr2, arm, get_carry_conf(arm, grasp_type))
    open_arm(pr2, arm)
    other_arm = get_other_arm(arm)
    set_arm_conf(pr2, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
    close_arm(pr2, other_arm)

    ground_truth, body_from_class, class_from_body,\
            movables, surfaces, rooms, obstacles = create_sampled_env(sampled_env)
    
    target_obj_bodies = []
    for target_obj in target_objs:
        target_obj_body = body_from_class[target_obj]
        set_color(target_obj_body, (1, 0, 0, 1))
        target_obj_bodies.append(body_from_class[target_obj])
    
    goal_surf_bodies = []
    for goal_surface in goal_surfaces:
        room, surface = goal_surface.split("|")
        goal_surf_bodies.append(body_from_class[surface+"|"+room])

    goal_on = []
    for target_obj, goal_surface in zip(target_obj_bodies, goal_surf_bodies):
        goal_on.append((target_obj, goal_surface))
    
    return ground_truth, BeliefTask(
        robot=pr2,
        arms=[arm],
        grasp_types=[grasp_type],
        class_from_body=class_from_body,
        body_from_class=body_from_class,
        movables=movables,
        surfaces=surfaces,
        rooms=rooms,
        goal_on=goal_on
    )

def room_class_to_body(task: BeliefTask, obj_class: str, belief: Dict) -> Dict:
    common_belief = {}
    for room_class, prob in belief[obj_class].items():
        common_belief[task.body_from_class[room_class]] = prob
    return common_belief

def surf_class_to_body(task: BeliefTask, obj_class: str, room_class: str, belief: Dict) -> Dict:
    common_belief = {}
    for surf_class, prob in belief[(obj_class, room_class)].items():
        common_belief[task.body_from_class[surf_class]] = prob
    return common_belief

def set_llm_belief(task: BeliefTask, b_on, b_in, room_belief, surface_belief):
    for target_obj in room_belief:
        b_in[task.body_from_class[target_obj]] = DDist(room_class_to_body(task, target_obj, room_belief))
        b_in[task.body_from_class[target_obj]].normalize()
    for target_obj, room in surface_belief:
        b_on[(task.body_from_class[target_obj], task.body_from_class[room])] = DDist(surf_class_to_body(task, target_obj, room, surface_belief))
        b_on[(task.body_from_class[target_obj], task.body_from_class[room])].normalize()

def set_surf_room_belief(task: BeliefTask, b_on, b_in, body, ground_truth):
    for room in ground_truth:
        b_on[(body, room)] = UniformDist([surf for surf in ground_truth[room]])
    b_in[body] = UniformDist([room for room in ground_truth])

def gen_particles(obj, surface, num_particles):
    surface_aabb = get_aabb(surface)
    center, extent = get_center_extent(obj)
    lower = (np.array(surface_aabb[0]) + extent/2)[:2]
    upper = (np.array(surface_aabb[1]) - extent/2)[:2]
    plane_aabb = AABB(lower, upper)
    epsilon = 1e-10
    particles = np.zeros((num_particles, 3))
    weights = np.ones(num_particles, dtype=np.float64) / num_particles
    for i in range(num_particles):
        x, y = sample_aabb(plane_aabb)
        z = (surface_aabb[1] + extent/2)[2] + epsilon
        point = np.array([x, y, z]) + (get_point(obj) - center)
        particles[i] = point
    return particles, weights

def get_initial_state(task: BeliefTask, ground_truth: dict, num_particles: int, llm: bool, **kwargs) -> BeliefState:
    # Discrete Belief
    b_on = {}
    b_in = {}

    target_objs = [goal[0] for goal in task.goal_on]
    if llm:
        room_belief = create_room_belief(target_objs, task.rooms, task.class_from_body)
        surface_belief = {}
        for room in task.rooms:
            surfaces = ground_truth[room]
            surface_belief.update(create_surface_belief(target_objs, room, surfaces, task.class_from_body))
        pprint.pprint(surface_belief)
        pprint.pprint(room_belief)
        set_llm_belief(task, b_on, b_in, room_belief, surface_belief)
    else:
        for body in target_objs:
            set_surf_room_belief(task, b_on, b_in, body, ground_truth)
        pprint.pprint(b_on)
        pprint.pprint(b_in)
    # Continuous Belief
    particles = {}
    weights = {}
    num_seen_particles = {}
    for target_obj in target_objs:
        particles[target_obj] = {}
        weights[target_obj] = {}
        num_seen_particles[target_obj] = {}
        for surface in task.surfaces:
            particles[target_obj][surface], weights[target_obj][surface] = gen_particles(target_obj, surface, num_particles)
            num_seen_particles[target_obj][surface] = 0 
    return BeliefState(task, ground_truth=ground_truth, b_on=b_on, b_in=b_in, 
                       particles=particles, weights=weights, num_particles=num_particles, 
                       num_seen_particles=num_seen_particles)

def read_problem(sampled_envs: dict, id: int, llm: False, **kwargs) -> Tuple[BeliefTask, BeliefState]:
    sampled_env = sampled_envs["env_"+str(id)]["sampled_env"]
    goal_obj = sampled_envs["env_"+str(id)]["goal_obj"]
    goal_surface = sampled_envs["env_"+str(id)]["goal_surface"]
    gt, task = get_sampled_task(sampled_env, goal_obj, goal_surface)
    start_time = time.perf_counter()
    initial = get_initial_state(task=task, ground_truth=gt, num_particles=9000, llm=llm, **kwargs)
    initial_state_time = time.perf_counter() - start_time
    return gt, task, initial, initial_state_time