import gym
import numpy as np

def get_environment(env_name=None):

    if env_name is None: print("Need to specify environment name")
    return GymEnv(env_name)

class EnvSpec(object):

    def __init__(self,
                 obs_space,
                 obs_dim,
                 act_space,
                 act_dim,
                 horizon,
                 num_agents):

        self.observation_space = obs_space
        self.observation_dim = obs_dim
        self.action_space = act_space
        self.action_dim = act_dim
        self.horizon = horizon
        self.num_agents = num_agents

class GymEnv(object):

    def __init__(self, env_name):

        env = gym.make(env_name)
        self.env = env
        self.env_id = env.spec.id

        self._horizon = env.spec.timestep_limit
        self._observation_space = env.observation_space
        self._action_space = env.action_space

        try:
            self._action_dim = self.env.env.action_dim
        except AttributeError:
            if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
                self._action_dim = self.env.env.action_space.n
            elif isinstance(self.env.action_space, gym.spaces.box.Box):
                self._action_dim = self.env.env.action_space.shape[0]

        try:
            self._observation_dim = self.env.env.obs_dim
        except AttributeError:
            self._observation_dim = self.env.env.observation_space.shape[0]

        try:
            self._num_agents = self.env.env.num_agents
        except AttributeError:
            self._num_agents = 1

        self.spec = EnvSpec(self._observation_space,
                            self._observation_dim,
                            self._action_space,
                            self._action_dim,
                            self._horizon,
                            self._num_agents)

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def evaluate_policy(self,
                        policy,
                        num_episodes=5,
                        horizon=None,
                        gamma=1,
                        visual=False,
                        percentile=[],
                        get_full_dist=False,
                        mean_action=False,
                        init_state=None,
                        terminate_at_done=True,
                        save_video_location=None,
                        seed=None):
        if seed is not None:
            self.env._seed(seed)
        horizon = self._horizon if horizon is None else horizon
        mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
        ep_returns = np.zeros(num_episodes)

        if save_video_location != None:
            self.env.monitor.start(save_video_location, force=True)

        for ep in range(num_episodes):

            if init_state is not None:
                o = self.reset()
                self.env.set_state(init_state[0], init_state[1])
                o = self.env._get_obs()
            else:
                o = self.reset()

            t, done = 0, False
            while t < horizon and (done == False or terminate_at_done == False):
                if visual == True:
                    self.render()
                if mean_action:
                    a = policy.act(False, o)[0]
                else:
                    a = policy.act(True, o)[0]
                o, r, done, _ = self.step(a)
                ep_returns[ep] += (gamma ** t) * r
                t += 1

        if save_video_location != None:
            self.env.monitor.close()

        mean_eval, std = np.mean(ep_returns), np.std(ep_returns)
        min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
        base_stats = [mean_eval, std, min_eval, max_eval]

        percentile_stats = []
        full_dist = []

        for p in percentile:
            percentile_stats.append(np.percentile(ep_returns, p))

        if get_full_dist == True:
            full_dist = ep_returns

        return [base_stats, percentile_stats, full_dist]
