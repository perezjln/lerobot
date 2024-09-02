import argparse
import time

import torch

from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy


def busy_wait(seconds):

    # Significantly more accurate than `time.sleep`, and mendatory for our use case,
    # but it consumes CPU cycles.
    # TODO(rcadene): find an alternative: from python 11, time.sleep is precise
    end_time = time.perf_counter() + seconds
    while time.perf_counter() < end_time:
        pass

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Run inference with ACTPolicy')
    parser.add_argument('--inference_time', type=int, default=10, help='Duration of inference in seconds')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on')
    parser.add_argument('--modelname-or-path', type=str, default='jnm38/diffusion-pistachio-v1', help='Path to the model')
    parser.add_argument("--policy_type", type=str, choices=["diffusion", "act", "vqbet"], default="diffusion", help="Type of policy to push: diffusion or act or vqbet")
    args = parser.parse_args()

    robot_path = "lerobot/configs/robot/koch_jack.yaml"

    # push to the hub
    print("Initializing model...")
    if args.policy_type == "act":
        policy = ACTPolicy.from_pretrained(args.path_or_fileobj)

    elif args.policy_type == "vqbet":
        policy = VQBeTPolicy.from_pretrained(args.path_or_fileobj)

    elif args.policy_type == "diffusion":
        policy = DiffusionPolicy.from_pretrained(args.path_or_fileobj)

    policy.to(args.device)

    robot_cfg = init_hydra_config(robot_path)
    robot = make_robot(robot_cfg)
    robot.connect()

    for _ in range(args.inference_time_s * args.fps):

        start_time = time.perf_counter()

        # Read the follower state and access the frames from the cameras
        observation = robot.capture_observation()

        # Convert to pytorch format: channel first and float32 in [0,1]
        # with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(args.device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

        # Order the robot to move
        robot.send_action(action)

        dt_s = time.perf_counter() - start_time
        busy_wait(1 / args.fps - dt_s)
