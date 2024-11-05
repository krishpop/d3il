from gym.envs.registration import register
from gymnasium.envs.registration import register as gymnasium_register

register(
    id="avoiding-v0",
    entry_point="gym_avoiding.envs:ObstacleAvoidanceEnv",
    max_episode_steps=150,
)

gymnasium_register(
    id="gym_avoiding/avoiding-v0",
    entry_point="gym_avoiding.envs:ObstacleAvoidanceEnv",
    max_episode_steps=150,
    kwargs={'max_steps_per_episode': 150, 'render':True, 
    'if_vision':False, "self_start": True}
)
