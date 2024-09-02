import argparse
import json

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Push policy file to Hugging Face Hub")
    parser.add_argument("--path_or_fileobj", default="outputs-pistachio-vqbet/train/", type=str, help="Path to the policy file")
    parser.add_argument("--repo-id", default="jnm38/vqbet-pistachio-v1", type=str, help="Repository ID")
    parser.add_argument("--policy_type", type=str, choices=["diffusion", "act", "vqbet"], default="vqbet", help="Type of policy to push: diffusion or act or vqbet")
    args = parser.parse_args()

    # push to the hub
    print("Initializing model...")
    if args.policy_type == "act":
        model = ACTPolicy.from_pretrained(args.path_or_fileobj)

    elif args.policy_type == "vqbet":
        model = VQBeTPolicy.from_pretrained(args.path_or_fileobj)

    elif args.policy_type == "diffusion":
        model = DiffusionPolicy.from_pretrained(args.path_or_fileobj)

    else:
        raise ValueError(f"Policy type {args.policy_type} not supported")        
    
    print("Pushing model to the hub...")
    model.push_to_hub(args.repo_id)
    print("Done !")