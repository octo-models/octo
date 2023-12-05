import os
import time

import gym
import kinpy as kp
import numpy as np
import pybullet as p
import pybullet_data


class KinematicsSolver:
    def __init__(self, urdf_path: str, eef_link: str):
        with open(urdf_path, "rb") as file:
            urdf_data = file.read()
        self.chain = kp.build_serial_chain_from_urdf(urdf_data, eef_link)
        print(self.chain)

    def fk(self, joint_angles: np.ndarray):
        """forward_kinematics"""
        transformation_matrix = self.chain.forward_kinematics(joint_angles)
        return transformation_matrix

    def ik(self, target_position, target_orientation, initial_state=None):
        """inverse_kinematics"""
        # Target position and orientation (if provided)
        target = kp.Transform(
            rot=target_orientation,
            pos=target_position,
        )
        # Calculate inverse kinematics
        joint_angles = self.chain.inverse_kinematics(target, initial_state)
        return joint_angles


class WidowXSimEnv(gym.Env):
    def __init__(
        self, default_pose=np.array([0.2, 0.0, 0.15, 0.0, 1.57, 0.0, 1]), image_size=256
    ):
        """
        Define the environment
        : args default_pose: 7-dimensional vector of
                            [x, y, z, roll, pitch, yaw, gripper_state]
        : args im_size: image size
        """
        assert len(default_pose) == 7
        super(WidowXSimEnv, self).__init__()
        self.default_pose = default_pose

        # Define observation and action space
        self.observation_space = gym.spaces.Dict(
            {
                "image_0": gym.spaces.Box(
                    low=np.zeros((image_size, image_size, 3)),
                    high=255 * np.ones((image_size, image_size, 3)),
                    dtype=np.uint8,
                ),
                "image_1": gym.spaces.Box(
                    low=np.zeros((image_size, image_size, 3)),
                    high=255 * np.ones((image_size, image_size, 3)),
                    dtype=np.uint8,
                ),
                "proprio": gym.spaces.Box(
                    low=np.full((7,), -1.0), high=np.ones((7,)), dtype=np.float64
                ),
            }
        )

        self.action_space = gym.spaces.Box(
            low=np.full((7,), -1.0), high=np.ones((7,)), dtype=np.float64
        )

        # Initialize PyBullet, and hide side panel but still show obs
        self.client = p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.table = p.loadURDF("table/table.urdf", 0.5, 0.0, -0.63, 0.0, 0.0, 0.0, 1.0)

        # NOTE: Original URDF is from
        # https://github.com/avisingh599/roboverse/tree/master/roboverse/assets/interbotix_descriptions
        asset_path = os.path.dirname(os.path.realpath(__file__)) + "/assets"
        urdf_path = asset_path + "/widowx/urdf/wx250.urdf"
        eef_link = "/ee_gripper_link"
        self.arm = p.loadURDF(urdf_path, useFixedBase=True)
        print(self.arm)

        # NOTE: users can add more objects to the scene by impl p.loadURDF()
        # https://github.com/ChenEating716/pybullet-URDF-models/tree/main
        # https://github.com/bulletphysics/bullet3/tree/master/data # with collision
        p.loadURDF(
            f"{asset_path}/fork/model.urdf", 0.2, -0.08, 0.03, 0.0, 0.0, 0.0, 1.0
        )
        p.loadURDF(f"{asset_path}/bowl/model.urdf", 0.2, 0.08, 0.03, 0.0, 0.0, 0.0, 1.0)

        # Define camera parameters
        # TODO: calibrate the intrinsic and extrinsic parameters
        camera_eye_position = [0.02, -0.2, 0.38]  # x=3cm, y=20cm, z=38cm
        camera_target_position = [0.38, 0, 0]  # Pointing at the origin
        camera_up_vector = [0, 0, 1]  # Z-axis up
        self.cam_view_matrix = p.computeViewMatrix(
            camera_eye_position, camera_target_position, camera_up_vector
        )
        self.cam_proj_matrix = p.computeProjectionMatrixFOV(
            fov=58.5, aspect=1.0, nearVal=0.1, farVal=1.5
        )

        self.ksolver = KinematicsSolver(urdf_path, eef_link=eef_link)
        p.setGravity(0, 0, -10)
        self.move_eef(self.default_pose[:6], reset=True)
        self.move_gripper(self.default_pose[-1], reset=True)

    def step(self, action: np.ndarray):
        """
        step action
        : args action: 7-dimensional vector [dx, dy, dz, droll, dpitch, dyaw, gripper_state]
        : return observation, reward, done, info
        """
        print("step ", [round(a, 2) for a in action])
        p.stepSimulation()
        time.sleep(0.01)

        # TODO: orientation need static transformation, ignore for now
        action[3:6] = np.zeros(3)

        # Get image and proprioceptive data
        observation = self.get_observation()
        abs_action = observation["proprio"] + action
        self.move_eef(abs_action[:6], reset=True)  # TODO: use position control
        self.move_gripper(action[-1])

        # Define reward, done, and info (custom to your task)
        reward = 0
        done = False
        trucated = False
        info = {}
        return observation, reward, done, trucated, info

    def reset(self):
        """
        Reset the environment to an initial state
            :return observation, info
        """
        p.setGravity(0, 0, -10)
        self.move_eef(self.default_pose[:6], reset=True)
        self.move_gripper(self.default_pose[-1], reset=True)
        return self.get_observation(), {}

    def close(self):
        """Close and clean up the environment"""
        p.disconnect(self.client)

    def move_eef(self, action: np.ndarray, reset=False):
        """
        Move the endeffector to the target position and orientation
        : args action: 6-dimensional vector of absolute
                        [x, y, z, roll, pitch, yaw]
        : args reset: whether to reset the joint states else use position control
        """
        current_joints = []
        for i in range(5):
            current_joints.append(p.getJointState(self.arm, i)[0])

        joints = self.ksolver.ik(
            target_position=action[:3],
            target_orientation=action[3:6],
            initial_state=current_joints,
        )
        assert len(joints) == 5

        if reset:
            # reset joint states
            for i in range(5):
                p.resetJointState(self.arm, i, joints[i])
        else:
            # with 2 decimal places
            print("  target joints: ", [round(j, 2) for j in joints])
            # apply joint angles to the arm
            for i in range(5):
                p.setJointMotorControl2(
                    bodyIndex=self.arm,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joints[i],
                    force=10000,
                    maxVelocity=50.0,
                )

    def get_observation(self):
        """
        Return the observation
        dict of image and proprioceptive data
        """
        # Obtain the camera image
        width, height = self.observation_space["image_0"].shape[:2]
        img_arr = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=self.cam_view_matrix,
            projectionMatrix=self.cam_proj_matrix,
        )[2]

        # reshape from (height, width, 4) to (height, width, 3)
        img_arr = img_arr[:, :, :3]
        img_arr = np.array(img_arr, dtype=np.uint8)

        # get joints
        joints = []
        for i in range(5):
            joints.append(p.getJointState(self.arm, i)[0])
        print("  current joints: ", [round(j, 2) for j in joints])

        # Get proprioceptive information, end-effector [x, y, z, roll, pitch, yaw]
        # NOTE: print out from the chain
        current_state = p.getLinkState(self.arm, 11)
        pos = current_state[0]
        orn = current_state[1]
        euler = p.getEulerFromQuaternion(orn)

        proprio_data = np.array(pos + euler)
        proprio_data = np.append(proprio_data, self.gripper_state())
        print("  current proprio: ", [round(p, 2) for p in proprio_data])

        null_img = np.zeros_like(img_arr)
        observation = {
            "image_0": img_arr,
            "image_1": null_img,  # NOTE: image_1 is null
            "proprio": proprio_data,
        }
        return observation

    def move_gripper(self, grip_state: float, reset=False):
        """
        Control the gripper to open or close
        :args grip_state: 1: open, 0: close
        :args reset: whether to reset the joint states else use position control
        """
        grip_joint_indices = [9, 10]  # the joint indices of the gripper
        grip_position = [0.0, 0.0] if grip_state < 0.5 else [0.04, -0.04]

        if reset:
            # reset joint states
            for i in range(2):
                p.resetJointState(self.arm, grip_joint_indices[i], grip_position[i])
        else:
            for i in range(2):
                p.setJointMotorControl2(
                    bodyIndex=self.arm,
                    jointIndex=grip_joint_indices[i],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=grip_position[i],
                    force=10000,
                    maxVelocity=50.0,
                )

    def gripper_state(self) -> float:
        """
        Return the gripper state
            1: open, 0: close
        """
        grip_joint_indices = [9, 10]
        grip_state = []
        for i in range(2):
            grip_state.append(abs(p.getJointState(self.arm, grip_joint_indices[i])[0]))
        return 1.0 if sum(grip_state) > 0.05 else 0.0
