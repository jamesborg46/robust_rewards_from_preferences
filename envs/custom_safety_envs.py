import safety_gym.envs.suite as suite
import numpy as np

# ============================================================================#
#                                                                             #
#       IRL Goal Environments                                                 #
#                                                                             #
# ============================================================================#

common = {
    'continue_goal': False,
    'reward_includes_cost': True,
    'hazards_cost': 30.,
    'constrain_indicator': False,
    'reward_clip': 20,
    'reward_goal': 0,
    'task': 'goal',
    'goal_size': 0.3,
    'goal_keepout': 0.305,
    'hazards_keepout': 0.18,
    'constrain_hazards': True,
    'observe_hazards': True,
    'hazards_size': 0.4,
    'placements_extents': [-1.5, -1.5, 1.5, 1.5],
}


front = {
    'robot_locations': [(0, -1.5)],
    'robot_rot': np.pi * (1/2),
    'goal_locations': [(0, 1.5)],
    'hazards_num': 3,
    'vases_num': 0,
    'observe_vases': False,
    'hazards_placements': None,
    'hazards_locations': [(-1.3, 0.9), (1.3, 0.9), (0, 0)],
}

behind = {
    'robot_locations': [(0, -1.5)],
    'robot_rot': np.pi * (1/2),
    'goal_locations': [(0, 0)],
    'hazards_num': 3,
    'vases_num': 0,
    'observe_vases': False,
    'hazards_placements': None,
    'hazards_locations': [(-1.3, 0.9), (1.3, 0.9), (0, 1.5)],
}

three = {
    'robot_locations': [(0, -1.5)],
    'robot_rot': np.pi * (1/2),
    'goal_locations': [(0, 1.5)],
    'hazards_num': 1,
    'vases_num': 1,
    'pillars_num': 1,
    'observe_vases': True,
    'observe_pillars': True,
    'constrain_vases': True,
    'constrain_pillars': True,
    'hazards_placements': None,
    'hazards_locations': [(0, 0)],
    'vases_placements': None,
    'vases_locations': [(-1.3, 0.9)],
    'pillars_placements': None,
    'pillars_locations': [(1.3, 0.9)],
    'pillars_size': 0.2,
    'vases_size': 0.2,
    'vases_contact_cost': 30,
    'pillars_cost': 30,
}

three_reversed = {
    'robot_locations': [(0, 1.5)],
    'robot_rot': np.pi * -(1/2),
    'goal_locations': [(0, -1.5)],
    'hazards_num': 1,
    'vases_num': 1,
    'pillars_num': 1,
    'observe_vases': True,
    'observe_pillars': True,
    'constrain_vases': True,
    'constrain_pillars': True,
    'hazards_placements': None,
    'hazards_locations': [(0, 0)],
    'vases_placements': None,
    'vases_locations': [(-1.3, -0.9)],
    'pillars_placements': None,
    'pillars_locations': [(1.3, -0.9)],
    'pillars_size': 0.2,
    'vases_size': 0.2,
    'vases_contact_cost': 30,
    'pillars_cost': 30,
}

irl_goal_base = suite.bench_base.copy('IRLGoal', common)
irl_goal_base.register('Front', front)
irl_goal_base.register('Behind', behind)

# Savexp-PointIRLGoalThree-v0'
irl_goal_base.register('Three', three)

# Savexp-PointIRLGoalThreeReversed-v0'
irl_goal_base.register('ThreeReversed', three_reversed)
