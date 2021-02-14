import argparse
import os
import time

from algos import CustomDQN  # noqa: F401
from wrappers import Renderer

import dowel
from dowel import logger
import gym
import numpy as np  # noqa: F401
import torch

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.envs.wrappers import ClipReward, EpisodicLife, FireReset, \
    Grayscale, MaxAndSkip, Noop, Resize, StackFrames
from garage.experiment.deterministic import set_seed
from garage.np.exploration_policies import EpsilonGreedyPolicy  # noqa: F401
from garage.replay_buffer import PathBuffer  # noqa: F401
from garage.sampler import FragmentWorker, LocalSampler, RaySampler, \
    DefaultWorker  # noqa: F401
from garage.torch import set_gpu_mode
from garage.torch.policies import DiscreteQFArgmaxPolicy  # noqa: F401
from garage.torch.q_functions import DiscreteCNNQFunction  # noqa: F401
from garage.trainer import Trainer


def dqn(ctxt,
        **kwargs):

    kwargs['number_epochs'] += 1

    snapshot_dir = ctxt.snapshot_dir
    env = gym.make(kwargs['env_id'])

    env = Noop(env, noop_max=30)
    env = MaxAndSkip(env, skip=4)
    env = EpisodicLife(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireReset(env)
    env = Grayscale(env)
    env = Resize(env, 84, 84)
    # env = ClipReward(env)
    env = StackFrames(env, 4, axis=0)
    env = Renderer(env, directory=os.path.join(snapshot_dir, 'videos'))
    env = GymEnv(env, max_episode_length=kwargs['max_episode_length'])
    env.metadata['video.frames_per_second'] = kwargs['video_fps']
    set_seed(kwargs['seed'])
    trainer = Trainer(ctxt)

    steps_per_epoch = kwargs['steps_per_epoch']
    num_timesteps = (kwargs['number_epochs'] * steps_per_epoch  # noqa: F841
                     * kwargs['steps_per_batch'])

    replay_buffer = PathBuffer(  # noqa: F841
        capacity_in_transitions=kwargs['buffer_size'])
    qf = eval(kwargs['qf'])  # noqa: F841
    policy = eval(kwargs['policy'])  # noqa: F841
    exploration_policy = eval(kwargs['exploration_policy'])  # noqa: F841

    Sampler = RaySampler if kwargs['ray'] else LocalSampler
    sampler = Sampler(agents=exploration_policy,  # noqa: F841
                      envs=env,
                      max_episode_length=env.spec.max_episode_length,
                      worker_class=FragmentWorker,
                      n_workers=kwargs['n_workers'])

    eval_sampler = Sampler(agents=exploration_policy,  # noqa: F841
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           worker_class=DefaultWorker,
                           n_workers=kwargs['n_workers'])

    algo = eval(kwargs['algo'])  # noqa: F841

    if torch.cuda.is_available() and kwargs['use_gpu']:
        set_gpu_mode(True, gpu_id=kwargs['gpu_id'])
        algo.to()
    else:
        set_gpu_mode(False)

    trainer.setup(algo, env)

    trainer.train(n_epochs=kwargs['number_epochs'],
                  batch_size=kwargs['steps_per_batch'])
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument('--seed', type=int, required=False)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--env_id', type=str, required=True)
    parser.add_argument('--number_epochs', type=int, default=1000)
    parser.add_argument('--steps_per_batch', type=int, default=500)
    parser.add_argument('--steps_per_epoch', type=int, default=20)
    parser.add_argument('--buffer_size', type=int, default=1e6)
    parser.add_argument('--snapshot_gap', type=int, default=100)
    parser.add_argument('--video_fps', type=int, default=30)
    parser.add_argument('--max_episode_length', type=int, default=None)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--ray', action='store_true', default=False)
    parser.add_argument('--use_gpu', action='store_true', default=False)
    parser.add_argument('--gpu_id', type=int, default=0)

    # These arguments will be evaluated as code
    parser.add_argument('--qf', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--exploration_policy', type=str, required=True)
    parser.add_argument('--algo', type=str, required=True)

    torch.set_num_threads(4)

    kwargs = vars(parser.parse_args())
    args = {k: v for k, v in kwargs.items() if v is not None}

    kwargs['name'] = (
        kwargs['name'] + '_' + time.ctime().replace(' ', '_')
    )

    if '-v' not in kwargs['env_id']:
        kwargs['env_id'] += 'NoFrameskip-v4'

    experiment_dir = os.getenv('EXPERIMENT_LOGS',
                               default=os.path.join(os.getcwd(), 'experiment'))

    log_dir = os.path.join(experiment_dir,
                           kwargs['name'] + time.ctime().replace(' ', '_'))

    logger.add_output(
        dowel.WandbOutput(
            project='dqn',
            name=args['name'],
            config=kwargs,
        )
    )

    dqn = wrap_experiment(
        dqn,
        name=kwargs['name'],
        snapshot_gap=kwargs['snapshot_gap'],
        snapshot_mode='gap_overwrite',
        log_dir=log_dir,
    )
    dqn(**kwargs)
