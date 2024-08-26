"""This scripts demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

import torch
import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

if __name__ == "__main__":

    # Set up the dataset.
    delta_timestamps_koch = {

        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.images.elp0": [-0.1, 0.0],
        "observation.images.elp1": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],

        # Load the previous action (-0.1), the next action to be executed (0.0),
        # and 14 future actions with a 0.1 seconds spacing. All these actions will be
        # used to supervise the policy.
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }
    #dataset_koch = LeRobotDataset("jackvial/koch_pick_and_place_pistachio_11_e20", delta_timestamps=delta_timestamps_koch)
    #print(dataset_koch.hf_dataset.features.keys())
    dataset_koch = MultiLeRobotDataset(["jackvial/koch_pick_and_place_pistachio_11_e20", 
                                        "jackvial/koch_pick_and_place_pistachio_10_e20", 
                                        "jackvial/koch_pick_and_place_pistachio_8_e100", 
                                        "jackvial/koch_pick_and_place_pistachio_5_e3"], 
                                        delta_timestamps=delta_timestamps_koch)

    # Set up the dataset.
    delta_timestamps_pusht = {
        # Load the previous image and state at -0.1 seconds before current frame,
        # then load current image and state corresponding to 0.0 second.
        "observation.image": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        # Load the previous action (-0.1), the next action to be executed (0.0),
        # and 14 future actions with a 0.1 seconds spacing. All these actions will be
        # used to supervise the policy.
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }
    dataset_pusht = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps_pusht)


    ## Print some information about the datasets
    #print("koch: ", dataset_koch.info)
    print("pusht: ", dataset_pusht.info)

    print(" --- ")
    print("keys: ", dataset_koch[0].keys())
    print("observation.images.elp0: ",  dataset_koch[0]["observation.images.elp0"].shape)
    print("observation.images.elp1: ", dataset_koch[0]["observation.images.elp1"].shape)
    print("observation.state: ", dataset_koch[0]["observation.state"].shape)
    print("action: ", dataset_koch[0]["action"].shape)

    print(" --- ")
    print("keys: ", dataset_pusht[0].keys())
    print("observation.image: ", dataset_pusht[0]["observation.image"].shape)
    print("action: ", dataset_pusht[0]["action"].shape[1:])


    ## Instantiate the models for pushT datasets
    print(" Instantiate the models for pushT datasets")
    cfg_pusht = DiffusionConfig(input_shapes={"observation.image": dataset_pusht[0]["observation.image"].shape[1:],
                                             "observation.state": dataset_pusht[0]["observation.state"].shape[1:]},
                                output_shapes={"action": dataset_pusht[0]["action"].shape[1:]},
                                crop_shape=None)
    policy_pusht = DiffusionPolicy(cfg_pusht, dataset_stats=dataset_pusht.stats)

    ## Instantiate the models for koch datasets
    print(" Instantiate the models for koch datasets")
    cfg_koch = DiffusionConfig(input_normalization_modes={"observation.images.elp0": "mean_std",
                                                          "observation.images.elp1": "mean_std",
                                                          "observation.state": "mean_std"},
                               output_normalization_modes={"action": "mean_std"},
                               crop_shape=None,
                               input_shapes={"observation.images.elp0": dataset_koch[0]["observation.images.elp0"].shape[1:],
                                             "observation.images.elp1": dataset_koch[0]["observation.images.elp1"].shape[1:],
                                             "observation.state": dataset_koch[0]["observation.state"].shape[1:]},
                                output_shapes={"action": dataset_koch[0]["action"].shape[1:]})

    policy_koch = DiffusionPolicy(cfg_koch, dataset_stats=dataset_koch.stats)

    # Rest of the code...
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")


    # Create dataloader for offline training.
    dataloader = torch.utils.data.DataLoader(
        dataset_koch,
        num_workers=0,
        batch_size=4,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    ## Forward the pushT dataset
    policy_koch.train()
    policy_koch.to(device)    
    for batch in tqdm.tqdm(dataloader, desc="Training"):            
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        output_dict = policy_koch.forward(batch)
        loss = output_dict["loss"]


    # Create dataloader for offline training.
    dataloader = torch.utils.data.DataLoader(
        dataset_pusht,
        num_workers=0,
        batch_size=4,
        shuffle=True,
        pin_memory=device != torch.device("cpu"),
        drop_last=True,
    )

    ## Forward the pushT dataset
    policy_pusht.train()
    policy_pusht.to(device)    
    for batch in tqdm.tqdm(dataloader, desc="Training"):            
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        output_dict = policy_pusht.forward(batch)
        loss = output_dict["loss"]


