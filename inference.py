import logging
import os
import numpy as np
from hashlib import md5
from pathlib import Path
from uuid import uuid4
import hydra
import torch
import open3d
from dotenv import load_dotenv
from plyfile import PlyData, PlyElement
from omegaconf import DictConfig, OmegaConf
from datasets.scannet200.scannet200_constants import SCANNET_COLOR_MAP_20
from datasets.utils import VoxelizeCollate
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import (
    flatten_dict,
    load_baseline_model,
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys
)
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import Dataset, DataLoader

import albumentations as A

from datasets.preprocessing.scannet_preprocessing import ScannetPreprocessing

class CustomSemSegDataset(Dataset):
    def __init__(self, data):
        self.data = data

        color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
        color_std  = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)

        self.color_map = SCANNET_COLOR_MAP_20
        self.color_map[255] = (255, 255, 255)

        self.normalize_color = A.Normalize(mean=color_mean, std=color_std)

    def __getitem__(self, index):
        points = np.load(self.data[index]["filepath"])
        
        # ???
        coordinates, color, normals, segments, labels = (
            points[:, :3],
            points[:, 3:6],
            points[:, 6:9],
            points[:, 9],
            points[:, 10:12],
        )

        # ???
        raw_coordinates = coordinates.copy()
        raw_color = color
        raw_normals = normals

        # ??? normalize color information
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])

        # ??? prepare labels and map from 0 to 20(40)
        labels = labels.astype(np.int32)
        if labels.size > 0:
            labels[:, 0] = self._remap_from_zero(labels[:, 0])
            if not self.add_instance:
                # taking only first column, which is segmentation label, not instance
                labels = labels[:, 0].flatten()[..., None]
        labels = np.hstack((labels, segments[..., None].astype(np.int32)))

        # ???
        features = color
        if True: # self.add_normals:
            features = np.hstack((features, normals))
        if True: # self.add_raw_coordinates:
            if len(features.shape) == 1:
                features = np.hstack((features[None, ...], coordinates))
            else:
                features = np.hstack((features, coordinates))

        return coordinates, features, labels, self.data[index]['raw_filepath'].split("/")[-2], \
               raw_color, raw_normals, raw_coordinates, index


    def __len__(self):
        return len(self.data)

    def _remap_from_zero(self, labels):
        labels[~np.isin(labels, list(self.label_info.keys()))] = self.ignore_label
        # remap to the range from 0
        for i, k in enumerate(self.label_info.keys()):
            labels[labels == k] = i
        return labels


def load_ply(filepath):
    with open(filepath, "rb") as f:
        plydata = PlyData.read(f)
    data = plydata.elements[0].data
    coords = np.array([data["x"], data["y"], data["z"]], dtype=np.float32).T
    feats = None
    labels = None
    if ({"red", "green", "blue"} - set(data.dtype.names)) == set():
        feats = np.array([data["red"], data["green"], data["blue"]], dtype=np.uint8).T
    if "label" in data.dtype.names:
        labels = np.array(data["label"], dtype=np.uint32)
    return coords, feats, labels


def load_ply_with_normals(filepath):
    mesh = open3d.io.read_triangle_mesh(str(filepath))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)

    coords, feats, labels = load_ply(filepath)
    assert np.allclose(coords, vertices), "different coordinates"
    feats = np.hstack((feats, normals))

    return coords, feats, labels


def prepare_data(file_path):
    save_dir = Path("/home/pelesz/Desktop/uni/adnd/Mask3D/ADND/tmp")
    

    data = {
        "filepath": str(file_path),
        "scene": "scene",
        "sub_scene": "sub_scene",
        "raw_filepath": str(file_path),
        "file_len": -1,
    }

    coords, features, _ = load_ply_with_normals(file_path)
    file_len = len(coords)
    data["file_len"] = file_len
    points = np.hstack((coords, features))
    points = np.hstack((points, np.zeros((points.shape[0],1))))

    processed_filepath = save_dir / str(file_path.name).split(".")[0]  / ".npy"
    if not processed_filepath.parent.exists():
        processed_filepath.parent.mkdir(parents=True, exist_ok=True)
    np.save(processed_filepath, points.astype(np.float32))
    data["filepath"] = str(processed_filepath)

    # segment_indexes_filepath = file_path.name.replace(".ply", ".0.010000.segs.json")
    # segments = self._read_json(file_path.parent / segment_indexes_filepath)
    # segments = np.array(segments["segIndices"])
    # # add segment id as additional feature
    # segment_ids = np.unique(segments, return_inverse=True)[1]

    collate = VoxelizeCollate(
        voxel_size=0.1,
        task="instance_segmentation",
        probing=False
    )
    dataset = CustomSemSegDataset([data])
    return DataLoader(dataset, collate_fn=collate)

    

# ADND
from torch.utils.data import Dataset, DataLoader
# ~ ADND





def get_parameters(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    loggers = []

    # cfg.general.experiment_id = "0" # str(Repo("./").commit())[:8]
    # params = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # create unique id for experiments that are run locally
    # unique_id = "_" + str(uuid4())[:4]
    # cfg.general.version = md5(str(params).encode("utf-8")).hexdigest()[:8] + unique_id

    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
    else:
        print("EXPERIMENT ALREADY EXIST")
        cfg['trainer']['resume_from_checkpoint'] = f"{cfg.general.save_dir}/last-epoch.ckpt"

    for log in cfg.logging:
        print(log)
        loggers.append(hydra.utils.instantiate(log))
        loggers[-1].log_hyperparams(
            flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        )

    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


def draw_all(pcds):
    open3d.visualization.draw_geometries_with_custom_animation(pcds,
                                            window_name='Point Cloud viewer',
                                            width=1920,
                                            height=1080,
                                            left=50,
                                            top=50)
                                            #optional_view_trajectory_json_file=O3D_VIEW_CONFIG_PATH)

class Prediction:
    def __init__(self, pred_masks, voxelated_raw_points):
        self.pred_masks = pred_masks
        self.voxelated_raw_points = voxelated_raw_points

@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def inference(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        gpus=cfg.general.gpus,
        logger=loggers,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer
    )

    # `model.model` is the Mask3D model
    file_path = Path(cfg['adnd']['input_ply'])
    dataloader = prepare_data(file_path)

    

    predictions = runner.predict(
        model=model,
        dataloaders=dataloader, # TODO
        datamodule=None,
        return_predictions=True,
        ckpt_path=None, # None -> use current weights provided by config
    )

    # scores, masks, classes, heatmap = model.get_mask_and_scores(
    #     predictions[0]['pred_logits'].cpu(),
    #     torch.stack(predictions[0]['pred_masks']).T,
    #     1,
    #     model.model.num_classes)

    # home = Path("/home/pelesz/Desktop/uni/adnd/Mask3D")
    
    # SINGLE PREDICTION
    pred_masks = predictions[0]['pred_masks'][0]
    voxelated_raw_coordinates = dataloader.collate_fn([dataloader.dataset[0]])[0].features[:, -3:]
    
    return Prediction(pred_masks, voxelated_raw_coordinates)
    
    # coordinates_predicted_instance = torch.argmax(pred_masks, axis=1)
    # colors = torch.rand(100, 3)
    # coordinate_colors = colors[coordinates_predicted_instance]

    # # COMBINED PREDICTION
    # # scannet_pred_masks = np.load(home / "ADND" / "saved" / "pred_masks.npy")
    # # combined_prediction = np.hstack((pred_masks, scannet_pred_masks))
    # # coordinates_predicted_instance = np.argmax(combined_prediction, axis=1)
    # # colors = torch.rand(200, 3)
    # # coordinate_colors = colors[coordinates_predicted_instance]


    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(voxelated_raw_coordinates)
    # pcd.colors = open3d.utility.Vector3dVector(coordinate_colors)

    # draw_all([pcd])

    # open3d.io.write_point_cloud(str(home / "ADND" / "saved" / "result.ply"), pcd)

    # home = Path("/home/pelesz/Desktop/uni/adnd/Mask3D")
    # np.save(home / "ADND" / "saved" / "voxelated_raw_coordinates.npy", voxelated_raw_coordinates)
    # np.save(home / "ADND" / "saved" / "pred_masks.npy", pred_masks)

    # return "something"
    
def show_combined_point_cloud(a: Prediction, b: Prediction, show_model_contribution=False):
    combined_prediction = np.hstack((a.pred_masks, b.pred_masks))
    coordinates_predicted_instance = np.argmax(combined_prediction, axis=1)

    if show_model_contribution:
        colors = np.array([[1, 0, 0], [0, 1, 0]])
        coordinate_colors = colors[np.where(coordinates_predicted_instance > 100, 1, 0)]
    else:
        colors = torch.rand(200, 3)
        coordinate_colors = colors[coordinates_predicted_instance]

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(a.voxelated_raw_points)
    pcd.colors = open3d.utility.Vector3dVector(coordinate_colors)

    home = Path("/home/pelesz/Desktop/uni/adnd/Mask3D")

    draw_all([pcd])
    open3d.io.write_point_cloud(str(home / "ADND" / "saved" / "result.ply"), pcd)

@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def main(cfg: DictConfig):
    s3dis_pretrained_path = "checkpoints/stpls3d/stpls3d_val.ckpt"
    scannet_pretrained_path = "checkpoints/s3dis/scannet_pretrained/area1_scannet_pretrained.ckpt"

    cfg['general']['checkpoint'] = s3dis_pretrained_path
    s3dis_pred = inference(cfg)

    cfg['general']['checkpoint'] = scannet_pretrained_path
    scannet_pred = inference(cfg)

    show_combined_point_cloud(s3dis_pred, scannet_pred, show_model_contribution=False)

    print("Exit without any fatal failure.")


if __name__ == "__main__":
    main()
