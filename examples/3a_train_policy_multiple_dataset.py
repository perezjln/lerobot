"""This scripts demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

from pathlib import Path

import argparse
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.act.configuration_act import ACTConfig


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train Policy on multiple datasets")
    parser.add_argument("--output_dir", type=str, default="outputs/train",
                        help="Directory to store the training checkpoint")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Number of offline training steps")
    parser.add_argument("--log_freq", type=int, default=250,
                        help="Frequency of logging training progress")
    parser.add_argument("--dataset_list", type=str, default="dataset_list_small.txt",
                        help="Path to the file containing the list of datasets")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Name of the dataset if it is unique")
    parser.add_argument("--policy_type", type=str, choices=["diffusion", "act"], default="diffusion",
                        help="Type of policy to train: diffusion or act")
    args = parser.parse_args()

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    training_steps = args.steps
    log_freq = args.log_freq

    # Rest of the code...
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Set up the dataset.
    delta_timestamps = {

        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.images.laptop": [-0.1, 0.0],
        "observation.images.phone": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],

        # Load the previous action (-0.1), the next action to be executed (0.0),
        # and 14 future actions with a 0.1 seconds spacing. All these actions will be
        # used to supervise the policy.
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }

    if args.dataset_name is not None:
        dataset = LeRobotDataset(args.dataset_name, delta_timestamps=delta_timestamps)
    else:
        with open(args.dataset_list, "r") as f:
            dataset_names = f.read().splitlines()
        dataset = MultiLeRobotDataset(dataset_names, delta_timestamps=delta_timestamps)

    print(dataset.info)

    # Set up the the policy.
    # Policies are initialized with a configuration class, in this case `DiffusionConfig`.
    # For this example, no arguments need to be passed because the defaults are set up for PushT.
    # If you're doing something different, you will likely need to change at least some of the defaults.

    input_shapes={"observation.images.laptop": dataset[0]["observation.images.laptop"].shape,
                  "observation.images.phone": dataset[0]["observation.images.phone"].shape,
                  "observation.state": dataset[0]["observation.state"].shape}

    output_shapes={"action":dataset[0]["action"].shape}

    if args.policy_type == "act":

        cfg = ACTConfig(input_shapes=input_shapes,
                        output_shapes=output_shapes,                        
                        input_normalization_modes={},
                        output_normalization_modes={})
        policy = ACTPolicy(cfg, dataset_stats=dataset.stats)

    else:

        cfg = DiffusionConfig(input_shapes=input_shapes,
                                output_shapes=output_shapes,                        
                                input_normalization_modes={},
                                output_normalization_modes={})
        policy = DiffusionPolicy(cfg, dataset_stats=dataset.stats)
        
    policy.train()
    policy.to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    # Create dataloader for offline training.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=64,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

    # Save a policy checkpoint.
    policy.save_pretrained(output_directory)
