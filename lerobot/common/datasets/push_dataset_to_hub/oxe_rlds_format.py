#!/usr/bin/env python

"""
For https://github.com/google-deepmind/open_x_embodiment (OXE) datasets.

Example:
    python lerobot/scripts/push_dataset_to_hub.py \
        --raw-dir /hdd/tensorflow_datasets/bridge_dataset/1.0.0/ \
        --repo-id youliangtan/sampled_bridge_data_v2 \
        --raw-format oxe_rlds \
        --episodes 3 4 5 8 9 \
        --fps 5

Exact dataset fps is specified in:
    https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit?gid=0#gid=0&range=R:R
"""

import shutil
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, save_images_concurrently
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames

np.set_printoptions(precision=2)


def get_cameras_keys(obs_keys):
    return [key for key in obs_keys if "image" in key]


def tf_to_torch(data):
    return torch.from_numpy(data.numpy())


def load_from_raw(raw_dir: Path, videos_dir: Path, fps: int, video: bool, episodes: list[int] | None = None):
    """
    Args:
        raw_dir (Path): _description_
        videos_dir (Path): _description_
        fps (int): _description_
        video (bool): _description_
        episodes (list[int] | None, optional): _description_. Defaults to None.
    """
    ds_builder = tfds.builder_from_directory(str(raw_dir))
    dataset = ds_builder.as_dataset(split="all")
    dataset_info = ds_builder.info
    print("dataset_info: ", dataset_info)

    image_keys = get_cameras_keys(dataset_info.features["steps"]["observation"].keys())

    # check if there's a 'tfds.features.Text' in step, only take 1 lang instruction
    lang_key = [
        key for key, value in dataset_info.features["steps"].items() if isinstance(value, tfds.features.Text)
    ]
    lang_key = None if len(lang_key) == 0 else lang_key[0]
    print(" - image_keys: ", image_keys)
    print(" - lang_key: ", lang_key)

    ds_length = len(dataset)
    dataset = dataset.take(ds_length)
    it = iter(dataset)

    ep_dicts = []

    # if we user specified episodes, skip the ones not in the list
    if episodes is not None:
        if ds_length == 0:
            raise ValueError("No episodes found.")
        # convert episodes index to sorted list
        episodes = sorted(episodes)

    for ep_idx in tqdm.tqdm(range(ds_length)):
        episode = next(it)

        # if we user specified episodes, skip the ones not in the list
        if episodes is not None:
            if len(episodes) == 0:
                break
            if ep_idx == episodes[0]:
                # process this episode
                print(" selecting episode idx: ", ep_idx)
                episodes.pop(0)
            else:
                continue  # skip

        steps = episode["steps"]
        num_frames = len(steps)

        # last step of demonstration is considered done
        done = torch.zeros(num_frames, dtype=torch.bool)
        done[-1] = True

        states = []
        actions = []  # TODO(YL): some actions can be a featuredict
        rewards = torch.zeros(num_frames, dtype=torch.float32)
        ep_dict = {}
        langs = []

        image_array_dict = {key: [] for key in image_keys}

        ###########################################################
        # loop through all steps in the episode
        for j, step in enumerate(steps):
            states.append(tf_to_torch(step["observation"]["state"]))
            actions.append(tf_to_torch(step["action"]))
            rewards[j] = torch.tensor(step["reward"].numpy(), dtype=torch.float32)

            if lang_key is not None:
                langs.append(str(step[lang_key]))

            for im_key in image_keys:
                if im_key not in step["observation"]:
                    continue

                img = step["observation"][im_key]
                img = np.array(img)
                image_array_dict[im_key].append(img)

        ###########################################################

        # loop through all cameras
        for im_key in image_keys:
            img_key = f"observation.images.{im_key}"
            imgs_array = image_array_dict[im_key]
            imgs_array = np.array(imgs_array)
            if video:
                # save png images in temporary directory
                tmp_imgs_dir = videos_dir / "tmp_images"
                save_images_concurrently(imgs_array, tmp_imgs_dir)

                # encode images to a mp4 video
                fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
                video_path = videos_dir / fname
                encode_video_frames(tmp_imgs_dir, video_path, fps)

                # clean temporary images directory
                shutil.rmtree(tmp_imgs_dir)

                # store the reference to the video frame
                ep_dict[img_key] = [
                    {"path": f"videos/{fname}", "timestamp": i / fps} for i in range(num_frames)
                ]
            else:
                ep_dict[img_key] = [PILImage.fromarray(x) for x in imgs_array]

        if lang_key is not None:
            ep_dict["language_instruction"] = langs

        ep_dict["observation.state"] = torch.stack(states)  # TODO better way
        ep_dict["action"] = torch.stack(actions)
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
        ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["reward"] = rewards
        ep_dict["next.done"] = done

        ep_dicts.append(ep_dict)

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def to_hf_dataset(data_dict, video) -> Dataset:
    features = {}

    keys = [key for key in data_dict if "observation.images." in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    if "observation.velocity" in data_dict:
        features["observation.velocity"] = Sequence(
            length=data_dict["observation.velocity"].shape[1], feature=Value(dtype="float32", id=None)
        )
    if "observation.effort" in data_dict:
        features["observation.effort"] = Sequence(
            length=data_dict["observation.effort"].shape[1], feature=Value(dtype="float32", id=None)
        )
    if "language_instruction" in data_dict:
        features["language_instruction"] = Value(dtype="string", id=None)

    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["reward"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
):
    """This is a test impl for rlds conversion"""
    if fps is None:
        fps = 5

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes)
    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info


if __name__ == "__main__":
    # NOTE (YL): This mainly serves as a unit test
    # austin_buds_dataset_converted_externally_to_rlds is a smaller dataset in
    # open x embodiment datasets.
    raw_dir = Path("/hdd/tensorflow_datasets/austin_buds_dataset_converted_externally_to_rlds/0.1.0/")
    videos_dir = Path("/hdd/tmp/")
    hf_dataset, episode_data_index, info = from_raw_to_lerobot_format(
        raw_dir,
        videos_dir,
        fps=50,
        video=True,
        episodes=[2, 3],
    )
    print(hf_dataset)
    print(episode_data_index)
    print(info)
