import warnings
warnings.filterwarnings("ignore")

import argparse
import tensorflow as tf
import numpy as np

from pgrl.utils.gym_utils import GymEnv
from pgrl.policies.mlp_policy import MlpPolicy
from pgrl.algos.ppo_clip import PPO
from pgrl.algos.ppo_adpkl import KLPPO
from pgrl.algos.batch_reinforce import BatchREINFORCE
from pgrl.core.train_agent import train_agent

MUJOCO = ['Walker2d-v2',
          'Hopper-v2',
          'HalfCheetah-v2',
          'Ant-v2',
          'Reacher-v2',
          'Swimmer-v2',
          'Humanoid-v2']

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env',
                    help='environment ID',
                    type=str,
                    default='Walker2d-v2')
parser.add_argument('--seed',
                    help='RNG seed',
                    type=int,
                    default=0)
parser.add_argument('--num-timesteps',
                    type=int,
                    default=int(1e6))
parser.add_argument('--algo',
                    help='learning algorithm',
                    type=str,
                    default='KLPPO',
                    choices=['PPO',
                             'KLPPO',
                             'BatchREINFORCE'])
args = parser.parse_args()

with tf.Graph().as_default():
    seed = args.seed
    np.random.seed(seed)
    tf.set_random_seed(seed)
    env = GymEnv(args.env)
    env.env.seed(seed)
    policy_fn = MlpPolicy
    algo = eval(args.algo)(env, policy_fn, seed=seed, save_logs=True)
    train_agent(job_name='results/{}/{}/{}'.format(args.env, args.algo, seed),
                agent=algo,
                seed=seed,
                nsamples=args.num_timesteps,
                gamma=0.99,
                gae_lambda=0.95,
                num_cpu=1,
                sample_mode='samples',
                num_samples=2048,
                save_freq=10,
                evaluation_rollouts=0)

tf.get_default_session().close()
