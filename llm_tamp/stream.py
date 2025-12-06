import os
import sys
from collections import namedtuple

import numpy as np
from examples.discrete_belief.run import (
    INF,
    MAX_COST,
    MAX_FD_COST,
    clip_cost,
    revisit_mdp_cost,
)
from examples.pybullet.utils.pybullet_tools.pr2_primitives import (
    Conf,
    Pose,
    Trajectory,
    get_fixed_bodies,
    pairwise_collision,
    sample_placement,
)
from examples.pybullet.utils.pybullet_tools.pr2_utils import (
    HEAD_LINK_NAME,
    PR2_GROUPS,
    get_group_joints,
)
from examples.pybullet.utils.pybullet_tools.utils import (
    ConfSaver,
    Euler,
    LockRenderer,
    Ray,
    angle_between,
    compute_jacobian,
    euler_from_quat,
    get_distance,
    get_link_pose,
    invert,
    joints_from_names,
    link_from_name,
    movable_from_joints,
    point_from_pose,
    pose_from_point_quat,
    quat_from_axis_angle,
    quat_from_euler,
    quat_from_pose,
    ray_collision,
    set_joint_positions,
    set_pose,
    tform_point,
    unit_from_theta,
    unit_point,
    violates_limits,
    wrap_angle,
)

from llm_tamp.belief import BeliefPose, BeliefState, BeliefTask

Interval = namedtuple('Interval', ['lower', 'upper']) # AABB
PI = np.pi
CIRCULAR_LIMITS = Interval(-PI, PI)

def get_belief_gen(state: BeliefState):
    # stream sample-bpose
    def gen(movable, surface):
        weights = state.weights[movable][surface]
        particles = state.particles[movable][surface]
        while True:
            if weights.size == 0:
                break
            # find the maximum weight
            max_value = np.max(weights)
            # find all indices of the maximum weight
            max_indices = np.where(np.isclose(weights, max_value))[0]
            # choose one of the indices with maximum weight randomly
            index = np.random.choice(max_indices)
            # make it into BeliefPose
            theta = np.random.uniform(*CIRCULAR_LIMITS)
            rotation = Euler(yaw=theta)
            quat = quat_from_euler(rotation)
            body_pose = pose_from_point_quat(particles[index], quat)
            bp = BeliefPose(body=movable, 
                            value=body_pose, 
                            support=surface,
                            prob=weights[index])
            yield (bp,)
    return gen

# def get_stable_gen(task: BeliefTask, collisions=True, **kwargs):
#     # stream sample-pose
#     def gen(body, surface):
#         while True:
#             body_pose = sample_placement(body, surface, **kwargs)
#             if body_pose is None:
#                 break
#             p = Pose(body, body_pose, surface)
#             p.assign()
#             if not pairwise_collision(body, task.obstacles[surface]):
#                 yield (p,)
#     return gen
import random


def get_stable_gen(task, collisions=True, **kwargs):
    obstacles = task.fixed if collisions else []
    def gen(body, surface):
        # TODO: surface poses are being sampled in pr2_belief
        if surface is None:
            surfaces = task.surfaces
        else:
            surfaces = [surface]
        while True:
            surface = random.choice(surfaces) # TODO: weight by area
            body_pose = sample_placement(body, surface, **kwargs)
            if body_pose is None:
                break
            p = Pose(body, body_pose, surface)
            p.assign()
            if not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, surface}):
                yield (p,)
    # TODO: apply the acceleration technique here
    return gen

def visible_base_generator(robot, target_point, base_range=(1., 1.), theta_range=(0., 0.)):
    base_from_target = unit_from_theta(np.random.uniform(0., 2 * np.pi))
    look_distance = np.random.uniform(*base_range)
    base_xy = target_point[:2] - look_distance * base_from_target
    base_theta = np.arctan2(base_from_target[1], base_from_target[0]) + np.random.uniform(*theta_range)
    base_q = np.append(base_xy, wrap_angle(base_theta))
    return base_q

def is_optical(link_name):
    return 'optical' in link_name

def inverse_visibility(pr2, point, head_name=HEAD_LINK_NAME, head_joints=None,
                       max_iterations=100, step_size=0.5, tolerance=np.pi*1e-2, verbose=False):
    # https://github.com/PR2/pr2_controllers/blob/kinetic-devel/pr2_head_action/src/pr2_point_frame.cpp
    head_link = link_from_name(pr2, head_name)
    camera_axis = np.array([0, 0, 1]) if is_optical(head_name) else np.array([1, 0, 0])
    if head_joints is None:
        head_joints = joints_from_names(pr2, PR2_GROUPS['head'])
    head_conf = np.zeros(len(head_joints))
    with LockRenderer(lock=True):
        with ConfSaver(pr2):
            for iteration in range(max_iterations):
                set_joint_positions(pr2, head_joints, head_conf)
                world_from_head = get_link_pose(pr2, head_link)
                point_head = tform_point(invert(world_from_head), point)
                error_angle = angle_between(camera_axis, point_head)
                if abs(error_angle) <= tolerance:
                    break
                normal_head = np.cross(camera_axis, point_head)
                normal_world = tform_point((unit_point(), quat_from_pose(world_from_head)), normal_head)
                correction_quat = quat_from_axis_angle(normal_world, step_size*error_angle)
                correction_euler = euler_from_quat(correction_quat)
                _, angular = compute_jacobian(pr2, head_link)
                correction_conf = np.array([np.dot(angular[mj], correction_euler)
                                            for mj in movable_from_joints(pr2, head_joints)])
                if verbose:
                    print('Iteration: {} | Error: {:.3f} | Correction: {}'.format(
                        iteration, error_angle, correction_conf))
                head_conf += correction_conf
                if np.all(correction_conf == 0):
                    return None
            else:
                return None
    if violates_limits(pr2, head_joints, head_conf):
        return None
    return head_conf


# Output where should the robot look
def get_inverse_vis_fn(task: BeliefTask, base_range, max_attempt, collisions=True):
    # stream inverse-visibility
    robot = task.robot
    base_joints = get_group_joints(robot, 'base')
    obstacles = get_fixed_bodies(task) if collisions else []
    head_joints = get_group_joints(robot, 'head')
    def fn(o, bp, l):
        set_pose(o, bp.value)
        target_point = point_from_pose(bp.value)
        for attempt in range(max_attempt):
            base_gen = visible_base_generator(robot, target_point, base_range)
            bq = Conf(robot, base_joints, base_gen)
            bq.assign()
            # Check collision with obstacles
            if collisions and any(pairwise_collision(robot, obs) for obs in obstacles):
                continue
            head_conf = inverse_visibility(robot, target_point)
            if head_conf is None:
                print("head_conf is None")
                yield None
            link = link_from_name(robot, HEAD_LINK_NAME)
            camera_pose = get_link_pose(robot, link)
            camera_point = point_from_pose(camera_pose)
            ray = Ray(camera_point, target_point)
            ray_result = ray_collision(ray)
            if ray_result.objectUniqueId in task.known and \
                ray_result.objectUniqueId != o:
                yield None
            else:
                hq = Conf(robot, head_joints, head_conf)
                ht = Trajectory([hq])
                yield (hq, ht, bq)
    return fn

#######################################################
def move_cost_fn(bq1, bq2):
    distance = get_distance(bq1.values[:2], bq2.values[:2])
    return 1 + distance

def geometric_cost(cost, p):
    if p == 0:
        return INF
    return cost / p

def revisit_mdp_cost(success_cost, failure_cost, p):
    """
    Cost for MDP where failures remain in the same state
    Equals geometric_cost(cost, p) == revisit_mdp_cost(cost, cost, p)
    failure_cost and success_cost can be about the same
    """
    return geometric_cost(failure_cost, p)

def clip_cost(cost, max_cost=MAX_COST): # TODO: move this to downward?
    if cost == INF:
        return max_cost
    return min(cost, max_cost)

def get_detect_cost_fn(state: BeliefState, detect_cost):
    def fn(o, l):
        # if o not in state.b_on: return MAX_COST
        room_body = state.get_room_body(l)
        p_surfGroom = state.b_on[(o, room_body)].prob(l)
        p_room = state.b_in[o].prob(room_body)
        # p_part = bp.prob
        prob = p_surfGroom * p_room
        cost = revisit_mdp_cost(0, detect_cost, prob)
        return clip_cost(cost)
    return fn