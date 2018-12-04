import logging
logging.disable(logging.CRITICAL)

import numpy as np
import pgrl.utils.tf_utils as tfu
from pgrl.utils.gym_utils import get_environment

def do_rollout(N,
               policy,
               T=1e6,
               env=None,
               env_name=None,
               seed=None,
               stochastic=True):

    if env_name is None and env is None:
        print("No environment specified! Error will be raised")

    if env is None: env = get_environment(env_name)
    #if seed is not None: env.env.seed(seed)

    T = min(T, env.horizon)

    paths = []
    for ep in range(N):

        if seed is not None:
            seed = seed + ep
            #env.env.seed(seed)
            np.random.seed(seed)
        else:
            np.random.seed()

        observations=[]
        actions=[]
        rewards=[]
        vpreds = []
        env_infos = []

        o = env.reset()
        done = False
        t = 0

        while t < T and done != True:
            a, vpred = policy.act(stochastic, o)
            next_o, r, done, env_info = env.step(a)
            observations.append(o)
            actions.append(a)
            rewards.append(r)
            vpreds.append(vpred)
            env_infos.append(env_info)
            o = next_o
            t += 1

        path = dict(observations=np.array(observations),
                    actions=np.array(actions),
                    rewards=np.array(rewards),
                    vpreds=np.array(vpreds),
                    env_infos=env_infos, # FOR NOW
                    terminated=done)

        paths.append(path)

    del(env)
    return paths

def do_rollout_star(args_list):
    return do_rollout(*args_list)
