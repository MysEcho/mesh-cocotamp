import sys
import os
import json
import time
import pprint
PACKAGES = ['pddlstream']

def add_packages(packages):
    sys.path.extend(os.path.abspath(os.path.join(os.getcwd(), d)) for d in packages)

add_packages(PACKAGES)

from pddlstream.algorithms.meta import (
    solve, create_parser
)
from pddlstream.algorithms.search import ABSTRIPSLayer
from pddlstream.language.generator import (
    from_gen_fn, 
    from_list_fn, 
    from_fn, 
    accelerate_list_gen_fn
)
from pddlstream.utils import (
    Profiler,
    read, 
    get_file_path,
)
from pddlstream.language.constants import (
    PDDLProblem, And, Equal, Not, 
    print_solution
)
from llm_tamp.primitives import (
    CoDetect, Attach, Detach, VIS_RANGE,
    plan_head_traj, 
    move_look_trajectory 
)
# from llm_tamp.small_problems import (
#     USE_DRAKE_PR2,
#     read_problem
# )
from llm_tamp.problems import (
    USE_DRAKE_PR2,
    read_problem
)
from llm_tamp.llm_updater import LLMUpdater
from llm_tamp.belief import BeliefState
from llm_tamp.stream import (
    get_belief_gen, 
    get_stable_gen, 
    get_inverse_vis_fn,
    get_detect_cost_fn,
    move_cost_fn
)
from examples.pybullet.utils.pybullet_tools.pr2_utils import (
    LEFT_ARM,
    get_arm_joints,
    attach_viewcone, 
    get_group_joints, 
    get_group_conf,
)
from examples.pybullet.utils.pybullet_tools.utils import (
    WorldSaver, ClientSaver, HideOutput, LockRenderer, GREEN,
    set_pose, 
    get_pose, 
    connect, 
    clone_world, 
    disconnect, 
    set_client, 
    add_data_path, 
    wait_for_user,
    get_joint_positions, 
    get_configuration, 
    set_configuration,
    is_center_stable, 
    add_body_name, 
    draw_base_limits,
    wait_for_duration,
    wait_if_gui,
    draw_point
)
from examples.pybullet.utils.pybullet_tools.pr2_primitives import (
    Conf, Trajectory,
    get_ik_ir_gen, 
    get_motion_gen, 
    get_grasp_gen,
    get_base_limits
)
from examples.pybullet.utils.pybullet_tools.pr2_problems import create_pr2
from examples.discrete_belief.run import MAX_COST, MAX_FD_COST
import random
import numpy as np

def pddlstream_from_state(state: BeliefState, teleport=False) -> PDDLProblem:
    task = state.task
    robot = task.robot

    domain_pddl = read(get_file_path(__file__, 'llm_tamp/pddl/domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'llm_tamp/pddl/stream.pddl'))
    constant_map = {
        'base': 'base',
        'left': 'left',
        'right': 'right',
        'head': 'head',
    }

    base_conf = Conf(robot, get_group_joints(robot, 'base'),
                     get_group_conf(robot, 'base'))
    init = [
        ('BConf', base_conf),
        ('AtBConf', base_conf),
        Equal(('PickCost',), 0.0),
        Equal(('PlaceCost',), 0.0)
    ]
    detect_cost = 1.0
    holding_arms = set()
    holding_bodies = set()
    for attach in state.attachments.values():
        holding_arms.add(attach.arm)
        holding_bodies.add(attach.body)
        init += [('Grasp', attach.body, attach.grasp),
                 ('AtGrasp', attach.arm, attach.body, attach.grasp)]
    
    init += [('Arm', LEFT_ARM)]
    if LEFT_ARM not in holding_arms:
        init += [('HandEmpty', LEFT_ARM)]
    
    target_objs = [target[0] for target in task.goal_on]
    for target_obj in target_objs:
        if target_obj in task.known and target_obj not in holding_bodies:
            pose = state.poses[target_obj]
            init += [('Pose', target_obj, pose),
                    ('AtPose', target_obj, pose)]

    init += [('Loc', body) for body in task.surfaces]
    init += [('Item', body) for body in task.movables if body in target_objs]
    
    for surface in task.surfaces:
        for target_obj in target_objs:
            if target_obj in task.known and is_center_stable(target_obj, surface):
                if target_obj in holding_bodies:
                    continue
                pose = state.poses[target_obj]
                init += [('Supported', target_obj, pose, surface)]

    for target_obj in target_objs:
        if target_obj in task.known:
            init.append(Not(('Uncertain', target_obj)))
        else:
            init.append(('Uncertain', target_obj))
    
    goal = And(*[('Holding', a, b) for a, b in task.goal_holding] + \
           [('On', b, s) for b, s in task.goal_on] + \
           [Not(('Uncertain', b)) for b in task.goal_localized])
    
    stream_map = {
        'sample-bpose': accelerate_list_gen_fn(from_gen_fn(get_belief_gen(state)), max_attempts=2000),
        'sample-pose': from_gen_fn(get_stable_gen(task)),
        'sample-grasp': from_list_fn(get_grasp_gen(task)),
        'plan-base-motion': from_fn(get_motion_gen(task, teleport=teleport)),
        'inverse-kinematics': from_gen_fn(get_ik_ir_gen(task, max_attempts=1000, learned=True, teleport=teleport)),
        'inverse-visibility': from_gen_fn(get_inverse_vis_fn(task, VIS_RANGE, max_attempt=500)),
        'DetectCost': get_detect_cost_fn(state, detect_cost),
        'MoveCost': move_cost_fn
    }

    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)


def post_process(state, plan, llm_updater: LLMUpdater,
                 replan_obs=True, replan_base=False, look_move=False):
    if plan is None:
        return None
    robot = state.task.robot
    commands = []
    uncertain_base = False
    expecting_obs = False
    for i, (name, args) in enumerate(plan):
        if replan_obs and expecting_obs:
            break
        saved_world = WorldSaver()
        if name == 'move_base':
            c = args[-1]
            [t] = c.commands
            if look_move:
                new_commands = [move_look_trajectory(state.task, t)]
            else:
                new_commands = [t]
            if replan_base:
                uncertain_base = True
        elif name == 'pick':
            if uncertain_base:
                break
            a, b, p, g, _, c = args
            [t] = c.commands
            attach = Attach(robot, a, g, b)
            new_commands = [t, attach, t.reverse()]
        elif name == 'place':
            if uncertain_base:
                break
            a, b, p, g, _, c = args
            [t] = c.commands
            detach = Detach(robot, a, b)
            new_commands = [t, detach, t.reverse()]
        elif name == 'detect':
            l, o, obj_bp, bq, hq, ht = args
            ht0 = plan_head_traj(state.task, hq.values)
            if ht0 is None:
                new_commands = [ht, CoDetect(robot, o, l, llm_updater)]
            else:
                new_commands = [ht0, ht, CoDetect(robot, o, l, llm_updater)]
            expecting_obs = True
        else:
            raise ValueError(name)
        saved_world.restore()
        for command in new_commands:
            if isinstance(command, Trajectory) and command.path:
                command.path[-1].assign()
        commands += new_commands
    return commands


def plan_commands(state: BeliefState, llm_updater: LLMUpdater, args, profile=True, verbose=True):
    goal_obj = state.task.goal_on[0][0]
    # with LockRenderer():
    #     handles = []
    #     for surface in state.task.surfaces:
    #         particles = state.particles[goal_obj][surface]
    #         for particle in particles:
    #             handles.extend(draw_point(particle, color=GREEN))

    sim_world = connect(use_gui=args.viewer)
    task = state.task
    robot_conf = get_configuration(task.robot)
    robot_pose = get_pose(task.robot)
    with ClientSaver(sim_world):
        with HideOutput():
            robot = create_pr2(use_drake=USE_DRAKE_PR2)
        set_pose(robot, robot_pose)
        set_configuration(robot, robot_conf)
    mapping = clone_world(client=sim_world, exclude=[task.robot])
    assert all(i1 == i2 for i1, i2 in mapping.items())
    set_client(sim_world)
    saver = WorldSaver()

    pddlstream_problem = pddlstream_from_state(state, teleport=True)
    _, _, _, stream_map, init, goal = pddlstream_problem
    print('Init:', sorted(init, key=lambda f: f[0]))
    if verbose:
        goal_class = [task.class_from_body[b] + " on " + task.class_from_body[s] for b, s in task.goal_on]
        print('Goal:', goal_class)
        print('Streams:', stream_map.keys())

    hierarchy = [
        ABSTRIPSLayer(pos_pre=['AtBConf']),
    ]
    with Profiler(field='cumtime', num=10 if profile else None):
        planner = 'ff-wastar1'
        solution = solve(pddlstream_problem, algorithm=args.algorithm, unit_costs=False, 
                         hierarchy=hierarchy, debug=False, planner=planner,
                         success_cost=MAX_COST, verbose=verbose, max_failures=5)
        plan, cost, evaluations = solution
        # if MAX_COST <= cost:
        #     plan = None
        print_solution(solution)
        print('Finite cost:', cost < MAX_COST)
        commands = post_process(state, plan, llm_updater)
    saver.restore()
    disconnect()
    return commands

def apply_commands(state, commands, time_step=None, pause=False, **kwargs):
    for i, command in enumerate(commands):
        if isinstance(command, Trajectory):
            if command.__repr__() == 't(3,2)':
                state.total_base_distance += command.distance()
        for j, _ in enumerate(command.apply(state, **kwargs)):
            state.assign()
            if j == 0:
                continue
            if time_step is None:
                wait_for_duration(1e-2)
                wait_if_gui('Command {}, Step {}) Next?'.format(i, j))
            else:
                wait_for_duration(time_step)
        if pause:
            wait_if_gui()

def save_results(exp_results, use_llm, use_llm_updater, counter=0):
    if (not use_llm) and (use_llm_updater):
        with open("results/intermediate/llm_updater_%d.json"%counter, "w") as f:
            json.dump(exp_results, f, indent=4)
    
    elif (use_llm) and (use_llm_updater):
        # Save results to a JSON file
        with open("results/intermediate/mcq_llm_updater_%d.json"%counter, "w") as f:
            json.dump(exp_results, f, indent=4)


def main(time_step: float=0.01):
    parser = create_parser()
    parser.add_argument(
        "-teleport", action="store_true", help="Teleports between configurations"
    )
    parser.add_argument(
        "-viewer", action="store_true", help="Enable the viewer while planning"
    )
    args = parser.parse_args()
    print("Arguments:", args)

    with open('llm_tamp/dataset/6_4_2.json', 'r') as f:
        sampled_envs = json.load(f)

    random.seed(42)
    np.random.seed(42)

    exp_results = {}
    for use_llm, use_llm_updater in [(False, True), (True, True)]:
        for id in range(len(sampled_envs)):
            real_world = connect(use_gui=args.viewer)
            add_data_path()
            obj_gt, task, state, initial_state_time = read_problem(sampled_envs, id, llm=use_llm, p_other=0.0)
            for body in task.get_bodies():
                add_body_name(body, task.class_from_body[body])

            target_objs = task.goal_on
            llm_updater = LLMUpdater(target_objs=target_objs,
                                class_from_body=task.class_from_body,
                                body_from_class=task.body_from_class)

            robot = task.robot
            attach_viewcone(robot)
            default_base_limits = get_base_limits(robot)
            draw_base_limits(default_base_limits, color=(0, 1, 0))
            start = time.perf_counter()
            step = 0
            # wait_for_user()
            total_planning_time = 0
            total_execution_time = 0
            while True:
                step += 1
                print("\n" + 50 * "-")
                print(step)
                pprint.pprint(state)
                print("\n" + 50 * "-")
                # wait_for_user()
                with ClientSaver():
                    start_time = time.perf_counter()
                    commands = plan_commands(state, llm_updater, args)
                    total_planning_time += time.perf_counter() - start_time
                print()
                if commands is None:
                    print("Failure!")
                    result = "Failure"
                    break
                if not commands:
                    print("Success!")
                    result = "Success"
                    break
                # wait_for_user()
                start_time = time.perf_counter()
                apply_commands(state, commands, time_step=time_step)
                total_execution_time += time.perf_counter() - start_time
            end = time.perf_counter()

            exp_results[id] = {
                    "time": end - start,
                    "steps": step,
                    "result": result,
                    "total_planning_time": total_planning_time,
                    "total_execution_time": total_execution_time,
                    "total_base_distance": state.total_base_distance,
                    "initial_state_time": initial_state_time
                }
            print("Planning and Execution Time:", end - start)
            print("Total number of steps to complete the task:", step)
            save_results(exp_results, use_llm, use_llm_updater, id)
            # wait_for_user()
            disconnect()
            time.sleep(2)


if __name__ == "__main__":
    main()
