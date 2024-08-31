"""This scripts demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

from pathlib import Path

import argparse
import torch
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train Policy on multiple datasets")
    parser.add_argument("--output_dir", type=str, default="outputs-pistachio-vqbet/train", help="Directory to store the training checkpoint")
    parser.add_argument("--steps", type=int, default=80000, help="Number of offline training steps")
    parser.add_argument("--log_freq", type=int, default=250, help="Frequency of logging training progress")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--dataset_list", type=str, default="examples/dataset_list_pistachio.txt", help="Path to the file containing the list of datasets")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset if it is unique")
    parser.add_argument("--fps", type=int, default=15, help="Frame per second of the dataset")
    parser.add_argument("--policy_type", type=str, choices=["diffusion", "act", "vqbet"], default="vqbet", help="Type of policy to train: diffusion or act")
    parser.add_argument("--act_chunk_size", type=int, default=100, help="Number of actions to chunk for ACT policy")
    parser.add_argument("--act_n_action_steps", type=int, default=100, help="Number of action steps for ACT policy")
    parser.add_argument("--act_use_vae", action="store_true", help="Use VAE for ACT policy")
    parser.add_argument("--vqbet_action_pred_token", type=int, default=7, help="Number of action prediction tokens for VQBeT policy")

    args = parser.parse_args()

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Rest of the code...
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Set up the dataset.
    if args.policy_type == "diffusion":
        delta_timestamps = {

            # Load the previous image and state at -0.1 seconds before current frame,
            # then load current image and state corresponding to 0.0 second.
            "observation.images.elp0": [-0.1, 0.0],
            "observation.images.elp1": [-0.1, 0.0],
            "observation.state": [-0.1, 0.0],

            # Load the previous action (-0.1), the next action to be executed (0.0),
            # and 14 future actions with a 0.1 seconds spacing. All these actions will be
            # used to supervise the policy.
            "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
        }

    elif args.policy_type == "act":
        delta_timestamps = {
            # Load the previous action (-0.1), the next action to be executed (0.0),
            # and 14 future actions with a 0.1 seconds spacing. All these actions will be
            # used to supervise the policy.
            "action": [i / args.fps for i in range(args.act_chunk_size)]
        }
    else:
        n_obs_steps= 5
        delta_timestamps= {
            "observation.images.elp0": [i / args.fps for i in range(1 - n_obs_steps, 1)],
            "observation.state": [i / args.fps for i in range(1 - n_obs_steps, 1)],
            "action": [i / args.fps for i in range(1 - n_obs_steps, args.vqbet_action_pred_token + args.act_chunk_size - 1)]
        }
    
    if args.dataset_name is not None:
        dataset = LeRobotDataset(args.dataset_name, delta_timestamps=delta_timestamps)
    else:
        with open(args.dataset_list, "r") as f:
            dataset_names = f.read().splitlines()
        dataset = MultiLeRobotDataset(dataset_names, delta_timestamps=delta_timestamps)
    print(dataset.features.keys()) 
  
    # Set up the the policy.
    # Policies are initialized with a configuration class, in this case `DiffusionConfig`.
    # For this example, no arguments need to be passed because the defaults are set up for PushT.
    # If you're doing something different, you will likely need to change at least some of the defaults.

    if args.policy_type == "act":

        cfg = ACTConfig(input_normalization_modes={"observation.images.elp0": "mean_std",
                                                    "observation.images.elp1": "mean_std",
                                                    "observation.state": "mean_std"},
                                output_normalization_modes={"action": "min_max"},
                                chunk_size=args.act_chunk_size,
                                use_vae = args.act_use_vae,
                                n_action_steps=args.act_n_action_steps,
                                input_shapes={"observation.images.elp0": dataset[0]["observation.images.elp0"].shape,
                                              "observation.images.elp1": dataset[0]["observation.images.elp1"].shape,
                                              "observation.state": dataset[0]["observation.state"].shape},
                                    output_shapes={"action": dataset[0]["action"].shape[1:]})

        policy = ACTPolicy(cfg, dataset_stats=dataset.stats)

    elif args.policy_type == "diffusion":

        cfg = DiffusionConfig(input_normalization_modes={"observation.images.elp0": "mean_std",
                                                          "observation.images.elp1": "mean_std",
                                                          "observation.state": "mean_std"},
                               output_normalization_modes={"action": "min_max"},
                               crop_shape=None,
                               input_shapes={"observation.images.elp0": dataset[0]["observation.images.elp0"].shape[1:],
                                             "observation.images.elp1": dataset[0]["observation.images.elp1"].shape[1:],
                                             "observation.state": dataset[0]["observation.state"].shape[1:]},
                                output_shapes={"action": dataset[0]["action"].shape[1:]})
        policy = DiffusionPolicy(cfg, dataset_stats=dataset.stats)
        
    else:
        cfg = VQBeTConfig(input_normalization_modes={"observation.images.elp0": "mean_std",
                                                    "observation.state": "mean_std"},
                                output_normalization_modes={"action": "min_max"},
                                n_obs_steps=5,
                                n_action_pred_token=args.vqbet_action_pred_token,
                                action_chunk_size=5,
                                input_shapes={"observation.images.elp0": dataset[0]["observation.images.elp0"].shape[1:],
                                              "observation.state": dataset[0]["observation.state"].shape[1:]},
                                output_shapes={"action": dataset[0]["action"].shape[1:]})
        policy = VQBeTPolicy(cfg, dataset_stats=dataset.stats)    
    
    policy.train()
    policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    # Create dataloader for offline training.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=8,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    # Run training loop.
    step = 0
    done = False    
    grad_clip_norm = 10.0
    while not done:
        for batch in tqdm.tqdm(dataloader, desc="Training", unit="batch"):
            
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    grad_clip_norm,
                    error_if_nonfinite=False,
            )

            optimizer.step()
            optimizer.zero_grad()

            if step % args.log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= args.steps:
                done = True
                break

        # Save a policy checkpoint.
        # One can also push the policy into the hub.
        policy.save_pretrained(output_directory, repo_id=f"{args.policy_type}-pistachio")
