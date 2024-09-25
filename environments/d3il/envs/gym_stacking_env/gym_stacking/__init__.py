from gymnasium.envs.registration import register as gymnasium_register

gymnasium_register(
    id="stacking-v0",
    entry_point="gym_stacking.envs:CubeStacking_Env",
    max_episode_steps=2000,
    kwargs={'max_steps_per_episode': 1000, 'render':True, 'if_vision':False}
)
