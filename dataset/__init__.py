from typing import List, Union, Optional, Tuple, Dict
from typing_extensions import TypedDict
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate, DataLoader
from torch.nn.utils.rnn import pad_sequence
from vocab import Vocab
import numpy as np
from scipy.spatial.transform import Rotation
import habitat_sim
from habitat_sim import bindings as hsim
from habitat_sim import registry as registry
from habitat_sim.agent.agent import AgentConfiguration, AgentState

#  from habitat_sim.utils.data.data_structures import ExtractorLRUCache

PathOrStr = Union[str, Path]
Sample = TypedDict(
    "Sample",
    {
        "pose": torch.Tensor,
        "image": Optional[torch.Tensor],
        "depth": Optional[torch.Tensor],
        "room_types": torch.Tensor,
        "object_labels": torch.Tensor,
    },
)


def _generate_label_map(scene, verbose=False) -> Tuple[Dict[int, str], Dict[int, str]]:
    if verbose:
        print(
            f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
        )
        print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    instance_id_to_object_name = {}
    instance_id_to_room_name = {}

    for region in scene.regions:
        for obj in region.objects:
            if not obj or not obj.category:
                continue
            obj_id = int(obj.id.split("_")[-1])
            instance_id_to_object_name[obj_id] = obj.category.name()
            instance_id_to_room_name[obj_id] = region.category.name()

    return instance_id_to_object_name, instance_id_to_room_name


class MatterportDataset(Dataset):
    """
    Provide the image, room type and object labels visible by the agent

    TODO: add camera parameter fov
    TODO: is the camera setting position correct? (h=1.5m)
    TODO: should we use a cache?
    TODO: should we count a room/object only if it appears at least on X% of the pixels?
    """

    def __init__(
        self,
        scene_filepaths: Union[PathOrStr, List[PathOrStr]],
        poses: List[Tuple[float, float, float, float]],
        img_size: Tuple[int, int] = (512, 512),
        rgb: bool = True,
        depth: bool = False,
        discard: Tuple[str] = ("misc", "",),
        data_dir: Path = Path("data/"),
    ):
        """
        @param scene_filepaths: paths to the .glb files
        @param pose: N x 5 (x, y, z, heading, elevation)
        @param dataset: location of matteport3d (symlink or real path)
        """
        self.poses = poses
        self.img_size = img_size
        self.rgb = rgb
        self.depth = depth
        self.discard = discard
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)

        if isinstance(scene_filepaths, (Path, str)):
            scene_filepaths = [scene_filepaths]
        self.scene_filepaths = scene_filepaths
        assert all(Path(f).is_file() for f in self.scene_filepaths), "Can't find scenes"

        self.sim = None
        self._is_init = False

    def _init(self):
        self.cfg = self._config_sim()
        self.sim = habitat_sim.Simulator(self.cfg)
        self._init_voc()

    def _init_voc(self):
        voc_file = self.data_dir / "voc.pth" 

        # try first to load voc
        if voc_file.is_file():
            voc = torch.load(voc_file)
            self.instance_id_to_object_name = voc["instance_id_to_object_name"]
            self.instance_id_to_room_name = voc["instance_id_to_room_name"]
            self.object_names = voc["object_names"]
            self.room_names = voc["room_names"]
            
        else:
            self.instance_id_to_object_name, self.instance_id_to_room_name = _generate_label_map(
                self.sim.semantic_scene
            )
            self.room_names = Vocab(list(self.instance_id_to_room_name.values()))
            self.object_names = Vocab(list(self.instance_id_to_object_name.values()))
            torch.save({
                "instance_id_to_object_name": self.instance_id_to_object_name,
                "instance_id_to_room_name": self.instance_id_to_room_name,
                "object_names": self.object_names,
                "room_names": self.room_names,
            }, voc_file)


    def __len__(self) -> int:
        return len(self.poses)

    def __getitem__(self, index: int) -> Sample:
        if self.sim is None:
            self._init()

        new_state = AgentState()
        new_state.position = self.poses[index][:3]

        rot = Rotation.from_euler('yz', self.poses[index][3:], degrees=False)
        new_state.rotation = rot.as_quat()
        self.sim.agents[0].set_state(new_state)

        obs = self.sim.get_sensor_observations()
        image = torch.Tensor(obs["color_sensor"]) if self.rgb else None
        depth = torch.Tensor(obs["depth_sensor"]) if self.depth else None

        content_ids = np.unique(obs["semantic_sensor"].flatten()).tolist()
        object_name = set([self.instance_id_to_object_name[cid] for cid in content_ids])
        object_name = filter( lambda x: x not in self.discard, object_name)
        object_labels = [self.object_names.word2index(n) for n in object_name]
        room_name = set([self.instance_id_to_room_name[cid] for cid in content_ids])
        room_name = filter(lambda x: x not in self.discard, room_name)
        room_types = [self.room_names.word2index(n) for n in room_name]

        return {
            "image": image,
            "depth": depth,
            "pose": torch.Tensor(self.poses[index]),
            "object_labels": torch.Tensor(object_labels),
            "room_types": torch.Tensor(room_types),
        }

    def _config_sim(self):
        settings = {
            "width": self.img_size[1],  # Spatial resolution of the observations
            "height": self.img_size[0],
            "scene": self.scene_filepaths[0],  # Scene path
            "default_agent": 0,
            "sensor_height": 1.5,  # Height of sensors in meters
            "color_sensor": self.rgb,  # RGBA sensor
            "semantic_sensor": True,  # Semantic sensor
            "depth_sensor": self.depth,  # Depth sensor
            "silent": True,
        }

        sim_cfg = hsim.SimulatorConfiguration()
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_id = settings["scene"]

        # define default sensor parameters (see src/esp/Sensor/Sensor.h)
        sensor_specs = []
        if settings["color_sensor"]:
            color_sensor_spec = habitat_sim.CameraSensorSpec()
            color_sensor_spec.uuid = "color_sensor"
            color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            color_sensor_spec.resolution = [settings["height"], settings["width"]]
            color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
            color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(color_sensor_spec)

        if settings["depth_sensor"]:
            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth_sensor"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = [settings["height"], settings["width"]]
            depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
            depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(depth_sensor_spec)

        if settings["semantic_sensor"]:
            semantic_sensor_spec = habitat_sim.CameraSensorSpec()
            semantic_sensor_spec.uuid = "semantic_sensor"
            semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
            semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
            semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(semantic_sensor_spec)

        # create agent specifications
        agent_cfg = AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])

    def close(self) -> None:
        r"""Deletes the instance of the simulator."""
        if self.sim is not None:
            self.sim.close()
            del self.sim
            self.sim = None


def collate_fn(samples: List[Sample]) -> Dict[str, torch.Tensor]:
    """ resolve optional keys on a sample """
    el0 = samples[0]
    batch = {}
    for key in el0:
        if el0[key] is None:
            continue
        seq = [el[key] for el in samples]
        if key in ("object_labels", "room_types"):
            batch[key] = pad_sequence(seq, batch_first=True, padding_value=-1)
        else:
            batch[key] = torch.stack(seq, 0)
    return batch


def display_sample(sample: Dict[str, torch.Tensor], voc_file: Path):
    import matplotlib.pyplot as plt

    # load voc
    voc = torch.load(voc_file)
    object_voc = voc["object_names"]
    room_voc = voc["room_names"]

    batch_size = sample["pose"].shape[0]

    for i in range(batch_size):
        arr = {}
        if "depth" in sample:
            arr["depth"] = sample["depth"][i].long()
        if "image" in sample:
            arr["image"] = sample["image"][i].long()

        plt.figure(figsize=(12, 8))
        for i, (key, data) in enumerate(arr.items()):
            ax = plt.subplot(1, len(arr), i + 1)
            ax.axis("off")
            ax.set_title(key)
            plt.imshow(data)
        plt.show(f"{i}.jpg")

        print(f"Object labels in sample {i}")
        for obj in sample["object_labels"][i].long().tolist():
            if obj != -1:
                print(object_voc.index2word(obj))

        print(f"Room labels in sample {i}")
        for room in sample["room_types"][i].long().tolist():
            if room != -1:
                print(room_voc.index2word(room))



if __name__ == "__main__":
    """ Testing the interface """

    dataset = MatterportDataset(
        scene_filepaths="data/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb",
        poses=torch.randn((10, 4)).tolist(),
    )

    dataloader = DataLoader(
        dataset, num_workers=4, shuffle=True, collate_fn=collate_fn, batch_size=2
    )

    sample = next(iter(dataloader))

    display_sample(sample, "data/voc.pth")
