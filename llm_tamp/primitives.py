from __future__ import print_function

from examples.discrete_belief.dist import DDist
from examples.pybullet.utils.pybullet_tools.pr2_primitives import (
    Command, Pose, Conf, Trajectory, SELF_COLLISIONS,
    create_trajectory,
    get_target_path
)
from examples.pybullet.utils.pybullet_tools.pr2_problems import get_fixed_bodies
from examples.pybullet.utils.pybullet_tools.pr2_utils import (
    HEAD_LINK_NAME, MAX_KINECT_DISTANCE, PR2_CAMERA_MATRIX, PR2_TOOL_FRAMES, PR2_GROUPS,
    get_visual_detections,
    inverse_visibility, 
    get_kinect_registrations, 
    get_detection_cone, 
    get_viewcone,
    get_group_joints, 
    get_group_conf, 
    set_group_conf, 
    is_visible_point,
)
from examples.pybullet.utils.pybullet_tools.utils import (
    BodySaver, LockRenderer, RED, GREEN, Ray,
    link_from_name, 
    create_mesh, 
    set_pose, 
    get_link_pose,
    wait_for_duration, 
    unit_pose,
    remove_body, 
    is_center_stable, 
    get_body_name, 
    get_name, 
    point_from_pose,
    plan_waypoints_joint_motion, 
    plan_direct_joint_motion, 
    wait_for_user, 
    apply_alpha,
    get_link_subtree, 
    child_link_from_joint, 
    draw_point, 
    draw_ray, 
    ray_collision, 
    remove_handles, 
    was_ray_hit, 
    get_pose,
    get_point,
    multiply,
    add_fixed_constraint,
    joints_from_names,
    get_min_limit,
    joint_controller_hold,
    step_simulation,
    get_aabb,
    get_aabb_top,
    get_aabb_extent,
    get_aabb_center,
    remove_fixed_constraint,
    aabb_contains_point
)
from llm_tamp.belief import BeliefState, BeliefPose

import numpy as np
import time
from collections import namedtuple
from scipy.stats import multivariate_normal

VIS_RANGE = (0.5, 1.5)
REG_RANGE = (0.5, 1.5)
ROOM_SCAN_TILT = np.pi / 6
P_LOOK_FP = 0
P_LOOK_FN = 0 # 1e-1

Interval = namedtuple('Interval', ['lower', 'upper']) # AABB
PI = np.pi
CIRCULAR_LIMITS = Interval(-PI, PI)


def plan_head_traj(task, head_conf):
    robot = task.robot
    obstacles = get_fixed_bodies(task)
    head_joints = get_group_joints(robot, 'head')
    head_path = plan_direct_joint_motion(robot, head_joints, head_conf,
                                  obstacles=obstacles, self_collisions=SELF_COLLISIONS)
    # assert(head_path is not None)
    if head_path is None:
        return None
    return create_trajectory(robot, head_joints, head_path)

def inspect_trajectory(task, trajectory):
    if not trajectory.path:
        return
    robot = trajectory.path[0].body
    obstacles = get_fixed_bodies(task)
    head_waypoints = []
    for target_point in get_target_path(trajectory):
        head_conf = inverse_visibility(robot, target_point)
        if head_conf is None:
            continue
        head_waypoints.append(head_conf)
    head_joints = get_group_joints(robot, 'head')
    head_path = plan_waypoints_joint_motion(robot, head_joints, head_waypoints,
                                            obstacles=obstacles, self_collisions=SELF_COLLISIONS)
    assert(head_path is not None)
    return create_trajectory(robot, head_joints, head_path)

def move_look_trajectory(task, trajectory, max_tilt=np.pi / 6):
    base_path = trajectory.path
    if not base_path:
        return trajectory
    obstacles = get_fixed_bodies(task)
    robot = base_path[0].body
    target_path = get_target_path(trajectory)
    waypoints = []
    index = 0
    with BodySaver(robot):
        for i, conf in enumerate(base_path):
            conf.assign()
            while index < len(target_path):
                if i < index:
                    target_point = target_path[index]
                    head_conf = inverse_visibility(robot, target_point) # TODO: this is slightly slow
                    if (head_conf is not None) and (head_conf[1] < max_tilt):
                        break
                index += 1
            else:
                head_conf = get_group_conf(robot, 'head')
            set_group_conf(robot, 'head', head_conf)
            waypoints.append(np.concatenate([conf.values, head_conf]))
    joints = tuple(base_path[0].joints) + tuple(get_group_joints(robot, 'head'))
    path = plan_waypoints_joint_motion(robot, joints, waypoints,
                                       obstacles=obstacles, self_collisions=SELF_COLLISIONS)
    return create_trajectory(robot, joints, path)

#######################################################
class Detach(Command):
    def __init__(self, robot, arm, body):
        self.robot = robot
        self.arm = arm
        self.body = body
        self.link = link_from_name(self.robot, PR2_TOOL_FRAMES.get(self.arm, self.arm))
        # TODO: pose argument to maintain same object
    def apply(self, state, **kwargs):
        del state.attachments[self.body]
        place_pose = Pose(self.body, get_pose(self.body))
        state.poses[self.body] = place_pose
        del state.grasps[self.body]
        # find which surface the pose is located
        for surface in state.task.surfaces:
            aabb = get_aabb(surface)
            top = get_aabb_top(aabb)
            extent = get_aabb_extent(aabb)
            room_body = state.get_room_body(surface)
            if (place_pose.value[0][0] < top[0] + extent[0]/2) \
                and (place_pose.value[0][0] > top[0] - extent[0]/2)\
                and (place_pose.value[0][1] < top[1] + extent[1]/2)\
                and (place_pose.value[0][1] > top[1] - extent[1]/2):
                state.b_on[(self.body, room_body)].setProb(surface, 1.0)
                state.b_in[self.body].setProb(room_body, 1.0)
                state.particles[self.body][surface] = point_from_pose(place_pose.value)
                state.weights[self.body][surface] = np.array([1.0])
            else:
                state.b_on[(self.body, room_body)].setProb(surface, 0.0)
                state.b_in[self.body].setProb(room_body, 0.0)
        yield
    def control(self, **kwargs):
        remove_fixed_constraint(self.body, self.robot, self.link)
    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, get_name(self.body))
    
class Attach(Command):
    vacuum = True
    def __init__(self, robot, arm, grasp, body):
        self.robot = robot
        self.arm = arm
        self.grasp = grasp
        self.body = body
        self.link = link_from_name(self.robot, PR2_TOOL_FRAMES.get(self.arm, self.arm))
    def assign(self):
        gripper_pose = get_link_pose(self.robot, self.link)
        body_pose = multiply(gripper_pose, self.grasp.value)
        set_pose(self.body, body_pose)
    def apply(self, state, **kwargs):
        state.attachments[self.body] = self
        state.grasps[self.body] = self.grasp
        del state.poses[self.body]
        yield
    def control(self, dt=0, **kwargs):
        if self.vacuum:
            add_fixed_constraint(self.body, self.robot, self.link)
        else:
            gripper_name = '{}_gripper'.format(self.arm)
            joints = joints_from_names(self.robot, PR2_GROUPS[gripper_name])
            values = [get_min_limit(self.robot, joint) for joint in joints] # Closed
            for _ in joint_controller_hold(self.robot, joints, values):
                step_simulation()
                time.sleep(dt)
    def __repr__(self):
        return '{}({},{},{})'.format(self.__class__.__name__, get_body_name(self.robot),
                                     self.arm, get_name(self.body))

class AttachCone(Command):
    def __init__(self, robot):
        self.robot = robot
        self.group = 'head'
        self.cone = None
    def apply(self, state, **kwargs):
        with LockRenderer():
            self.cone = get_viewcone(color=apply_alpha(RED, 0.5))
            state.poses[self.cone] = None
            cone_pose = Pose(self.cone, unit_pose())
            attach = Attach(self.robot, self.group, cone_pose, self.cone)
            attach.assign()
            wait_for_duration(1e-2)
        for _ in attach.apply(state, **kwargs):
            yield
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class DetachCone(Command):
    def __init__(self, attach):
        self.attach = attach
    def apply(self, state, **kwargs):
        cone = self.attach.cone
        detach = Detach(self.attach.robot, self.attach.group, cone)
        for _ in detach.apply(state, **kwargs):
            yield
        del state.poses[cone]
        remove_body(cone)
        wait_for_duration(1e-2)
    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

def get_cone_commands(robot):
    attach = AttachCone(robot)
    detach = DetachCone(attach)
    return attach, detach

#######################################################
class CoDetect(Command):
    _duration = 0.5

    def __init__(self, robot, item, surface, co_updater, camera_frame=HEAD_LINK_NAME):
        self.robot = robot
        self.item = item
        self.surface = surface
        self.camera_frame = camera_frame
        self.link = link_from_name(robot, self.camera_frame)
        self.co_updater = co_updater

    def detect_update(self, item, state: BeliefState, camera_pose, noise):
        """
        Update the weights of each particle based on the observation function.
        particle is [x, y, z].
        """
        state.particles[item][self.surface] = get_point(item)
        state.weights[item][self.surface] = np.array([1.0])

    def failed_update(self, item, state: BeliefState, camera_pose, noise):
        """
        Update the weights of each particle based on the observation function.
        particle is [x, y, z].
        measurement_noise is the standard deviation of the measurement noise.
        """
        # Apply kernel density estimation (Gaussian kernel centered at each particle)
        cov_matrix = np.diag(noise) ** 2  # Covariance matrix for Gaussian
        particles = state.particles[item][self.surface]
        weights = state.weights[item][self.surface]
        new_particles = np.array([np.random.multivariate_normal(mean, cov_matrix) 
                                  for mean in particles])
        camera_point = point_from_pose(camera_pose)
        new_weights = np.zeros(len(particles), dtype=np.float64)
        for i, particle in enumerate(particles):
            # Check if the particle is within the visible region or occluded
            if (is_visible_point(camera_matrix=PR2_CAMERA_MATRIX,
                                 depth=MAX_KINECT_DISTANCE,
                                 point_world=particle,
                                 camera_pose=camera_pose)):
                ray = Ray(camera_point, particle)
                ray_result = ray_collision(ray)
                for obst in state.task.known:
                    if aabb_contains_point(particle, get_aabb(obst)):
                        new_weights[i] = 1e-5
                if ray_result.objectUniqueId == -1 or ray_result.objectUniqueId == self.surface:
                    new_weights[i] = 1e-5
                else:
                    new_weights[i] = weights[i]
            else:
                    new_weights[i] = weights[i]
        # Maintain particles with weights larger then 1e-10
        valid_mask = new_weights > 1e-5
        new_particles = new_particles[valid_mask]
        new_weights = new_weights[valid_mask]
        if len(new_weights) == 0:
            # All particles have been removed due to low weights
            state.particles[item][self.surface] = np.array([])
            state.weights[item][self.surface] = np.array([])
        else:
            # Normalize the remaining weights
            new_weights /= np.sum(new_weights)
            # Update the state with filtered particles and weights
            state.particles[item][self.surface] = new_particles
            state.weights[item][self.surface] = new_weights

    def get_ray_detections(self, state: BeliefState, rays):
        detected_objs = set()
        all_detections = set()
        for ray, ray_result in rays:
            if ray_result.objectUniqueId != -1:
                all_detections.add(ray_result.objectUniqueId)
                if ray_result.objectUniqueId in state.task.movables:
                    detected_objs.add(ray_result.objectUniqueId)
        return detected_objs, all_detections
    
    def calculate_vis(self, item, state: BeliefState, camera_pose):
        """
        Generate rays from the camera to each particle.
        """
        # Room visibility
        room_body = state.get_room_body(self.surface) 
        surfaces_in_room = []
        for surface in state.task.surfaces:
            if state.get_room_body(surface) == room_body:
                surfaces_in_room.append(surface)
        total_particles = state.num_particles * len(surfaces_in_room)

        # Surface visibility
        camera_point = point_from_pose(camera_pose)
        particles = state.particles[item][self.surface]
        visibility = 0
        for i, particle in enumerate(particles):
            # Check if the particle is within the visible region or occluded
            if (is_visible_point(camera_matrix=PR2_CAMERA_MATRIX,
                                 depth=MAX_KINECT_DISTANCE,
                                 point_world=particle,
                                 camera_pose=camera_pose)):
                ray = Ray(camera_point, particle)
                ray_result = ray_collision(ray)
                if not was_ray_hit(ray_result):
                    visibility += 1
        state.num_seen_particles[item][self.surface] += visibility
        surf_visibility = state.num_seen_particles[item][self.surface] / state.num_particles 
        room_seen_particles = 0
        for room_surf in surfaces_in_room:
            room_seen_particles += state.num_seen_particles[item][room_surf]
        room_visibility = room_seen_particles / total_particles
        return surf_visibility, room_visibility
    
    def gen_visible_rays(self, camera_pose, particles):
        """
        Generate a predefined number of particles at the base of the viewcone.
        """
        camera_point = point_from_pose(camera_pose)
        count = 0
        rays = []
        for i, particle in enumerate(particles):
            # Check if the particle is within the visible region or occluded
            if (is_visible_point(camera_matrix=PR2_CAMERA_MATRIX,
                                 depth=MAX_KINECT_DISTANCE,
                                 point_world=particle,
                                 camera_pose=camera_pose)):
                ray = Ray(camera_point, particle)
                ray_result = ray_collision(ray)
                rays.append((ray, ray_result))
        return rays
        
    def apply(self, state: BeliefState, **kwargs):
        camera_pose = get_link_pose(self.robot, self.link)
        particles = state.particles[self.item][self.surface]
        weights = state.weights[self.item][self.surface]
        surf_visibility, room_visibility = self.calculate_vis(self.item, state, camera_pose)
        rays_vis = self.gen_visible_rays(camera_pose, particles)
        with LockRenderer():
            cone = get_viewcone(color=apply_alpha(RED, 0.5))
            set_pose(cone, get_link_pose(self.robot, self.link))
            # handles = []
            # for particle in particles:
            #     if (is_visible_point(camera_matrix=PR2_CAMERA_MATRIX,
            #                      depth=MAX_KINECT_DISTANCE,
            #                      point_world=particle,
            #                      camera_pose=camera_pose)):
            #         handles.extend(draw_point(particle, color=GREEN))
            # for ray, ray_result in rays_vis:
            #     handles.extend(draw_ray(ray, ray_result))
        wait_for_duration(self._duration)
        # remove_handles(handles)
        remove_body(cone)
        wait_for_duration(self._duration)
        detections, all_detections = self.get_ray_detections(state, rays_vis)
        room_body = state.get_room_body(self.surface)
        print("surface visibility: ", surf_visibility)
        print("room visibility: ", room_visibility)
        print("Detections: ", [state.task.class_from_body[obj] for obj in detections])
        for target, surface in state.task.goal_on:
            if target in detections and target not in state.task.known:
                true_pose = BeliefPose(target, get_pose(target), self.surface, False, 1.0)
                true_pose.assign()
                print("New detected item pose: ",
                        state.task.class_from_body[target], 
                        get_pose(target))
                # Correlational Update
                print("Detected b_in before update: ",
                        state.task.class_from_body[target],
                        state.b_in[target])
                print("Detected b_on before update: ",
                        state.task.class_from_body[target],
                        state.b_on[(target, room_body)])
                # Update the particle weights of the detected item
                self.detect_update(target, state, camera_pose, [0.05, 0.05, 0.001])
                self.co_updater.correlationUpdate(target_item = target,
                                                    detected_items = detections,
                                                    target_loc = self.surface,
                                                    surf_visibility = surf_visibility,
                                                    room_visibility = room_visibility,
                                                    known = state.task.known,
                                                    p_fp = 0.01, 
                                                    p_fn = 0.01,
                                                    state = state,
                                                    obs = True)
                print("Detected b_in after update: ",
                        state.task.class_from_body[target],
                        state.b_in[target])
                print("Detected b_on after update: ",
                        state.task.class_from_body[target],
                        state.b_on[(target, room_body)])
            
            elif target not in detections and target not in state.task.known:
                print("Failed detection b_in before update: ",
                        state.task.class_from_body[target],
                        state.b_in[target])
                print("Failed detection b_on before update: ",
                        state.task.class_from_body[target],
                        state.b_on[(target, room_body)])
                # Update the particle weights of the undetected item
                self.failed_update(target, state, camera_pose, [0.05, 0.05, 0.001])
                self.co_updater.correlationUpdate(target_item = target,
                                                detected_items = detections,
                                                target_loc = self.surface,
                                                surf_visibility = surf_visibility,
                                                room_visibility = room_visibility,
                                                known = state.task.known,
                                                p_fp = 0.01, 
                                                p_fn = 0.01,
                                                state = state,
                                                obs = False)
                print("Failed detection b_in after update: ",
                        state.task.class_from_body[target],
                        state.b_in[target])
                print("Failed detection b_on after update: ",
                        state.task.class_from_body[target],
                        state.b_on[(target, room_body)])
                print(len(state.particles[target][self.surface]))
        state.task.known.update(all_detections)       
        yield

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, 
                                  get_name(self.surface), 
                                  get_name(self.item), 'empty')