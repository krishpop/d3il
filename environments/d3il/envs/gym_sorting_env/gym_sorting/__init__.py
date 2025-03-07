from gymnasium.envs.registration import register as gymnasium_register

gymnasium_register(
    id="gym_sorting/sorting-v0",
    entry_point="gym_sorting.envs:Sorting_Env",
    max_episode_steps=1000,
    kwargs={'max_steps_per_episode': 1000, 'render':True, 'num_boxes':2, 'if_vision':False, "action_type": "delta", "self_start":True}
)

gymnasium_register(
    id="gym_sorting/sorting_4boxes-v0",
    entry_point="gym_sorting.envs:Sorting_Env",
    max_episode_steps=1000,
    kwargs={'max_steps_per_episode': 1000, 'render':True, 'num_boxes':4, 'if_vision':False, "action_type": "delta", "self_start":True}
)

gymnasium_register(
    id="gym_sorting/sorting_6boxes-v0",
    entry_point="gym_sorting.envs:Sorting_Env",
    max_episode_steps=1000,
    kwargs={'max_steps_per_episode': 1000, 'render':True, 'num_boxes':6, 'if_vision':False, "action_type": "delta", "self_start":True}
)