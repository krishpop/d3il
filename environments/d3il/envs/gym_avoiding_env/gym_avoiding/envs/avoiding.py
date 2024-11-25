import numpy as np
import copy
import gymnasium as gym
from gymnasium.spaces import Box
from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_helper import has_collision
from environments.d3il.d3il_sim.utils.sim_path import d3il_path

from environments.d3il.d3il_sim.core import Scene
from environments.d3il.d3il_sim.gyms.gym_env_wrapper import GymEnvWrapper
from environments.d3il.d3il_sim.core.logger import ObjectLogger, CamLogger
from environments.d3il.d3il_sim.sims.mj_beta.MjRobot import MjRobot
from environments.d3il.d3il_sim.sims.mj_beta.MjFactory import MjFactory
from environments.d3il.d3il_sim.sims import MjCamera

from .objects.avoiding_objects import get_obj_list, \
    init_end_eff_pos, \
    get_obj_xy_list

obj_list = get_obj_list()


class BPCageCam(MjCamera):
    """
    Cage camera. Extends the camera base class.
    """

    def __init__(self, width: int = 1024, height: int = 1024, *args, **kwargs):
        super().__init__(
            "bp_cam",
            width,
            height,
            init_pos=[1.05, 0, 1.2],
            init_quat=[
                0.6830127,
                0.1830127,
                0.1830127,
                0.683012,
            ],  # Looking with 30 deg to the robot
            *args,
            **kwargs,
        )



class ObstacleAvoidanceManager:
    def __init__(self):
        self.index = 0
        pass

    def start(self):
        pass


class ObstacleAvoidanceEnv(GymEnvWrapper):
    def __init__(
            self,
            n_substeps: int = 35,
            max_steps_per_episode: int = 250,
            debug: bool = False,
            render: bool = False,
            self_start: bool = False,
            if_vision: bool = False,
    ):

        sim_factory = MjFactory()
        render_mode = Scene.RenderMode.HUMAN if render else Scene.RenderMode.BLIND
        scene = sim_factory.create_scene(
            object_list=obj_list, render=render_mode, dt=0.001
        )
        robot = MjRobot(
            scene,
            xml_path=d3il_path("./models/mj/robot/panda_rod_invisible.xml")
        )
        controller = robot.cartesianPosQuatTrackingController

        super().__init__(
            scene=scene,
            controller=controller,
            max_steps_per_episode=max_steps_per_episode,
            n_substeps=n_substeps,
            debug=debug,
        )

        self.if_vision = if_vision
        self.action_space = Box(
            low=np.array([-0.01, -0.01]), high=np.array([0.01, 0.01])
        )
        if self.if_vision: 
            self.observation_space = gym.spaces.Dict({
                "agent_pos": Box(low=-np.inf, high=np.inf, shape=(8,)),
                "environment_state": Box(low=-np.inf, high=np.inf, shape=(15,)),
                "pixels": gym.spaces.Dict({
                    "bp_cam": Box(low=0, high=255, shape=(96, 96, 3), dtype=np.uint8),
                    "inhand_cam": Box(low=0, high=255, shape=(96, 96, 3), dtype=np.uint8)
                })
            })
        else:
            self.observation_space = gym.spaces.Dict({
                "agent_pos": Box(low=-np.inf, high=np.inf, shape=(8,)),
                "environment_state": Box(low=-np.inf, high=np.inf, shape=(15,))
            })

        self.manager = ObstacleAvoidanceManager()

        self.bp_cam = BPCageCam()
        self.inhand_cam = robot.inhand_cam

        self.scene.add_object(self.bp_cam)

        self.log_dict = {}
        self.cam_dict = {"bp-cam": CamLogger(scene, self.bp_cam)}

        for _, v in self.cam_dict.items():
            scene.add_logger(v)

        self.obj_xy_list = get_obj_xy_list()

        self.target_min_dist = 0.06

        level_distance = 0.18
        obstacle_offset = 0.075
        self.l1_ypos = -0.1
        self.l2_ypos = -0.1 + level_distance
        self.l3_ypos = -0.1 + 2 * level_distance
        self.goal_ypos = -0.1 + 2.5 * level_distance
        self.l1_xpos = 0.5
        self.l2_top_xpos = 0.5 - obstacle_offset
        self.l2_bottom_xpos = 0.5 + obstacle_offset
        self.l3_top_xpos = 0.5 - 2 * obstacle_offset
        self.l3_mid_xpos = 0.5
        self.l3_bottom_xpos = 0.5 + 2 * obstacle_offset

        self.l1_passed = False
        self.l2_passed = False
        self.l3_passed = False

        self.mode_encoding = np.zeros(2 + 3 + 4)

        self.success = False
        if self_start:
            self.start()

    def get_environment_state(self):
        pass

    def get_observation(self) -> np.ndarray:
        robot_c_pos = self.robot_state()[:2].astype(np.float32)
        if self.if_vision:
            bp_img = self.bp_cam.get_image(depth=False)
            inhand_img = self.inhand_cam.get_image(depth=False)
            # inhand_img = self.inhand_cam.get_image(depth=False)
            return {
                "agent_pos": robot_c_pos,
                "environment_state": self.get_environment_state(),
                "pixels": {
                    "bp_cam": bp_img,
                    "inhand_cam": inhand_img
                }
            }
        else:
            return {
                "agent_pos": robot_c_pos,
                "environment_state": self.get_environment_state()
            }

    def start(self):
        self.scene.start()

        # reset view of the camera
        try:
            self.scene.viewer.cam.elevation = -55
            self.scene.viewer.cam.distance = 1.8
            self.scene.viewer.cam.lookat[0] += -0.1
            self.scene.viewer.cam.lookat[2] -= 0.2

            # self.scene.viewer.cam.elevation = -55
            # self.scene.viewer.cam.distance = 2.0
            # self.scene.viewer.cam.lookat[0] += 0
            # self.scene.viewer.cam.lookat[2] -= 0.2
            # self.scene.viewer.cam.elevation = -60
            # self.scene.viewer.cam.distance = 1.6
            # self.scene.viewer.cam.lookat[0] += 0.1
            # self.scene.viewer.cam.lookat[2] -= 0.1
        except:
            pass

        # reset the initial state of the robot
        initial_cart_position = copy.deepcopy(init_end_eff_pos)
        # initial_cart_position[2] = 0.12
        self.robot.gotoCartPosQuatController.setDesiredPos(
            [
                initial_cart_position[0],
                initial_cart_position[1],
                initial_cart_position[2],
                0,
                1,
                0,
                0,
            ]
        )
        self.robot.gotoCartPosQuatController.initController(self.robot, 1)

        self.robot.init_qpos = self.robot.gotoCartPosQuatController.trajectory[-1].copy()
        self.robot.init_tcp_pos = initial_cart_position
        self.robot.init_tcp_quat = [0, 1, 0, 0]

        self.robot.beam_to_joint_pos(self.robot.gotoCartPosQuatController.trajectory[-1])

        self.robot.gotoCartPositionAndQuat(
            desiredPos=initial_cart_position, desiredQuat=[0, 1, 0, 0], duration=0.5
        )

    def step(self, action, gripper_width=None):
        if self.self_start:
            robot_c_pos = self.robot_state()[:2]
            pred_action = action + robot_c_pos
            fixed_z = init_end_eff_pos[2]
            action = np.concatenate((pred_action, fixed_z, [0, 1, 0, 0]), axis=0)

        observation, reward, terminated, truncated, _ = super().step(action, gripper_width)
        self.check_mode()
        return observation, reward, terminated, truncated, (self.mode_encoding, self.success)

    def check_mode(self):
        r_x_pos = self.robot.current_c_pos[0]
        r_y_pos = self.robot.current_c_pos[1]
        if r_y_pos - 0.03 <= self.l1_ypos <= r_y_pos + 0.03 and (not self.l1_passed):
            if r_x_pos < self.l1_xpos:
                self.mode_encoding[0] = 1
            elif r_x_pos > self.l1_xpos:
                self.mode_encoding[1] = 1
            self.l1_passed = True

        if r_y_pos - 0.03 <= self.l2_ypos <= r_y_pos + 0.03 and (not self.l2_passed):
            if r_x_pos < self.l2_top_xpos:
                self.mode_encoding[2] = 1
            elif self.l2_top_xpos < r_x_pos < self.l2_bottom_xpos:
                self.mode_encoding[3] = 1
            elif r_x_pos > self.l2_bottom_xpos:
                self.mode_encoding[4] = 1
            self.l2_passed = True

        # if r_y_pos - 0.015 <= self.l3_ypos and (not self.l3_passed):
        if r_y_pos >= self.l3_ypos and (not self.l3_passed):
            if r_x_pos < self.l3_top_xpos:
                self.mode_encoding[5] = 1
            if self.l3_top_xpos < r_x_pos < self.l3_mid_xpos:
                self.mode_encoding[6] = 1
            elif self.l3_mid_xpos < r_x_pos < self.l3_bottom_xpos:
                self.mode_encoding[7] = 1
            elif r_x_pos > self.l3_top_xpos:
                self.mode_encoding[8] = 1
            self.l3_passed = True

    def check_failure(self):
        if has_collision('l1_obs', 'rod', self.scene.model, self.scene.data):
            return True
        elif has_collision('l2_top_obs', 'rod', self.scene.model, self.scene.data):
            return True
        elif has_collision('l2_bottom_obs', 'rod', self.scene.model, self.scene.data):
            return True
        elif has_collision('l3_top_obs', 'rod', self.scene.model, self.scene.data):
            return True
        elif has_collision('l3_mid_obs', 'rod', self.scene.model, self.scene.data):
            return True
        elif has_collision('l3_bottom_obs', 'rod', self.scene.model, self.scene.data):
            return True
        else:
            return False

    def check_success(self):
        if self.robot.current_c_pos[1] > self.goal_ypos:
            return True
        else:
            return False

    def reset_mode_encoding(self):
        self.l1_passed = False
        self.l2_passed = False
        self.l3_passed = False
        assert np.sum(self.mode_encoding) <= 3
        self.mode_encoding = np.zeros(2 + 3 + 4)

    def get_reward(self):
        ...

    def _check_early_termination(self) -> bool:

        # print(self.check_failure())

        if self.check_success() or self.check_failure():
            if self.check_success():
                self.success = True
            self.terminated = True
            return True

        return False

    def reset(self, random=True, context=None):
        self.terminated = False
        self.env_step_counter = 0
        self.episode += 1
        self.reset_mode_encoding()
        self.success = False
        obs = self._reset_env(random=random, context=context)
        return obs, {}

    def _reset_env(self, random=True, context=None):
        self.scene.reset()
        self.robot.beam_to_joint_pos(self.robot.init_qpos)
        self.scene.next_step()
        observation = self.get_observation()
        return observation

    def reward(self, x):
        def squared_exp_kernel(x, mean, scale, bandwidth):
            return scale * np.exp(np.square(np.linalg.norm(x - mean, axis=1)) / bandwidth)

        rewards = np.zeros(x.shape[0])
        for obs in self.obj_xy_list:
            rewards -= squared_exp_kernel(x, np.array(obs), 1, 1)
        # rewards += np.abs(x[:, 1]- 0.4)
        rewards -= np.abs(x[:, 0] - 0.4)
        return rewards

    def mode_decoding(self, data):
        data_decimal = data.dot(1 << np.arange(data.shape[-1]))
        _, counts = np.unique(data_decimal, return_counts=True)
        mode_dist = counts / np.sum(counts)
        entropy = - np.sum(mode_dist * (np.log(mode_dist) / np.log(24)))
        return counts, entropy
