"""Dataloader for 3DMatch dataset

Modified from Predator source code by Shengyu Huang:
  https://github.com/overlappredator/OverlapPredator/blob/main/datasets/indoor.py
"""
import logging
import os
import json

# import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.se3_numpy import se3_init, se3_transform, se3_inv
from utils.pointcloud import compute_overlap

from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points
from pytorch3d.io import IO, ply_io
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.transforms import Rotate, Translate
from torchvision.transforms.functional import pil_to_tensor


from PIL import Image
import open3d as o3d
import cv2
               
                    
class T_Less(Dataset):

    def __init__(self, cfg, phase, transforms=None, obj_id_list=[5], vis_ratio = 0.7, is_visualize=False):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        if len(obj_id_list) == 0:
            obj_id_list = range(1,31)
            
        self.root = None
        if isinstance(cfg.root, str):
            if os.path.exists(f'{cfg.root}'):
                self.root = cfg.root

        if self.root is None:
            raise AssertionError(f'Dataset not found in {cfg.root}')
        else:
            self.logger.info(f'Loading data from {self.root}')

        self.cfg = cfg
        self.is_visualize = is_visualize

        # set the directories related to the T-Less dataset
        self.train_dir = os.path.join(self.root, "train_pbr")
        self.test_dir = os.path.join(
            self.root, "tless_test_primesense_bop19", "test_primesense")
        self.cad_dir = os.path.join(
            self.root, "tless_models", "models_cad")

        # each directory is related to a specific camera and scene setup
        self.selected_scenes = []
        assert phase in ['train', 'val', 'test']
        base_dir = self.train_dir if phase in ['train'] else self.test_dir
        setup_dirs = [name for name in os.listdir(
            base_dir) if os.path.isdir(os.path.join(base_dir))]
        scene_obj_list = []
        for setup_dir in setup_dirs:
            scene_gt_path = os.path.join(base_dir, setup_dir, "scene_gt.json")
            camera_gt_path = os.path.join(
                base_dir, setup_dir, "scene_camera.json")
            scene_info_path = os.path.join(base_dir, setup_dir, "scene_gt_info.json")
            with open(scene_gt_path) as scenes_f, open(camera_gt_path) as camera_f, open(scene_info_path) as scene_info_f:
                scenes_dic = json.load(scenes_f)
                cameras_dic = json.load(camera_f)
                scene_info_dic = json.load(scene_info_f)
                for scene_id, obj_list in scenes_dic.items():
                    for i, obj_dic in enumerate(obj_list):
                        for obj_id in obj_id_list:
                            if (obj_dic["obj_id"] == obj_id):
                                # remove low visible items
                                if scene_info_dic[scene_id][i]["visib_fract"] < vis_ratio:
                                    continue                                                                     
                                data_dic = {"dir": os.path.join(
                                    base_dir, setup_dir), "scene_id": int(scene_id), "obj_index": i}                                           
                                data_dic.update(obj_dic)
                                data_dic.update(cameras_dic[scene_id])
                                self.selected_scenes.append(data_dic)
                                scene_obj_list.append([int(setup_dir), int(scene_id), obj_id])
                                    
        if len(self.selected_scenes) == 0:
            raise AssertionError(
                "there is no selected obj_num {0} in the given directory.".format(obj_id_list))
        
        # remove the scenes with repeated objects
        _, indexes, count = np.unique(np.array(scene_obj_list), axis=0, return_counts=True, return_index=True)
        indexes = indexes[count == 1]
        self.selected_scenes = [self.selected_scenes[i] for i in indexes]

        self.search_voxel_size = cfg.overlap_radius
        self.transforms = transforms
        self.phase = phase

    
    def __len__(self):
        return len(self.selected_scenes)

    def __getitem__(self, ind):
        # depth_path = os.path.join(self.selected_scenes[ind]["dir"], "depth", str(
        #     self.selected_scenes[ind]["scene_id"]).zfill(6) + ".png")
        scene_pcl_path = os.path.join(self.selected_scenes[ind]["dir"], "pcl", str(
            self.selected_scenes[ind]["scene_id"]).zfill(6) + ".ply")
        obj_cad_path = os.path.join(
            self.cad_dir, "obj_" + str(self.selected_scenes[ind]["obj_id"]).zfill(6) + ".ply")

        R = np.array(self.selected_scenes[ind]["cam_R_m2c"]).reshape(3, 3)
        t = np.array([self.selected_scenes[ind]["cam_t_m2c"]]).reshape(3, 1)
        # cam_K = np.array(self.selected_scenes[ind]["cam_K"]).reshape(3, 3)
        # cam_depth_scale = 1 / self.selected_scenes[ind]["depth_scale"]

        obj_mesh = o3d.io.read_triangle_mesh(obj_cad_path)
        obj_mesh.compute_vertex_normals()
        obj_pcl = obj_mesh.sample_points_uniformly(
            number_of_points=self.cfg.obj_point_size)
        src_xyz = np.array(obj_pcl.points)

        # depth_im = o3d.io.read_image(depth_path)
        # im_width, im_height = depth_im.get_max_bound()
        # pinhole_cam = o3d.camera.PinholeCameraIntrinsic(
        #     int(im_width), int(im_height), cam_K)
        # scene_pcl = o3d.geometry.PointCloud.create_from_depth_image(
        #     depth_im, pinhole_cam, depth_scale=cam_depth_scale, depth_trunc=1000)        
        # scene_pcl = scene_pcl.farthest_point_down_sample(
        #     self.cfg.scene_point_size)
        
        scene_pcl = o3d.io.read_point_cloud(scene_pcl_path)
        tgt_xyz = np.array(scene_pcl.points, dtype=np.float32)
     
        # scale and center bothe pcls
        tgt_xyz, tgt_norm_scale, centered_t = self.pc_norm_center(tgt_xyz)
        src_xyz = src_xyz * tgt_norm_scale        
        t_scaled = t * tgt_norm_scale
        pose = se3_init(R, t_scaled - centered_t.reshape(3,1))

        src_overlap_mask, tgt_overlap_mask, src_tgt_corr = compute_overlap(
            se3_transform(pose, src_xyz),
            tgt_xyz,
            self.search_voxel_size,
        )
        
        overlap_p = len(
            src_overlap_mask[src_overlap_mask == True])/len(tgt_xyz)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # # for visualization
        if self.is_visualize:
            src_xyz = se3_transform(pose, src_xyz)
            return torch.tensor(tgt_xyz), torch.tensor(src_xyz), torch.tensor(tgt_xyz[tgt_overlap_mask]), torch.tensor(src_xyz[src_overlap_mask]), torch.tensor(tgt_xyz[src_tgt_corr[1]]), torch.tensor(src_xyz[src_tgt_corr[0]])
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        data_pair = {
            'src_xyz': torch.from_numpy(src_xyz).float(),
            'tgt_xyz': torch.from_numpy(tgt_xyz).float(),
            'src_overlap': torch.from_numpy(src_overlap_mask),
            'tgt_overlap': torch.from_numpy(tgt_overlap_mask),
            'correspondences': torch.from_numpy(src_tgt_corr),  # indices
            'pose': torch.from_numpy(pose).float(),
            'idx': ind,
            'src_path': obj_cad_path,
            'tgt_path': scene_pcl_path,
            'overlap_p': overlap_p,
        }
        if self.transforms is not None:
            self.transforms(data_pair)  # Apply data augmentation
        return data_pair

    def pc_norm_center(self, pc):
        scale = 1/np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc * scale
        
        pc_center = np.mean([(np.min(pc[:, i]), np.max(pc[:, i])) for i in range(3)], axis=1)
        pc = pc - pc_center
        return pc, scale, pc_center

    def visibility_ratio(self, data_dic):
        vis_mask_path = os.path.join(data_dic["dir"], "mask_visib", str(
                                        data_dic["scene_id"]).zfill(6) + "_" + str(
                                        data_dic["obj_index"]).zfill(6) + ".png")
        total_mask_path = os.path.join(data_dic["dir"], "mask", str(
                                        data_dic["scene_id"]).zfill(6) + "_" + str(
                                        data_dic["obj_index"]).zfill(6) + ".png")
        vis_mask = pil_to_tensor(Image.open(vis_mask_path))
        total_mask = pil_to_tensor(Image.open(total_mask_path))
        
        if len(total_mask[total_mask > 0]) == 0:
            return 0
        else:
            return len(vis_mask[vis_mask > 0]) / len(total_mask[total_mask > 0])


    def depth2pcl(self, depth_im, mask_im=None):
        height_im, width_im = depth_im.shape
        scale_x, scale_y = (
            width_im / self.camera_dic["width"]), (height_im / self.camera_dic["height"])
        cx_new, cy_new = (
            self.camera_dic["cx"] * scale_x), (self.camera_dic["cy"] * scale_y)
        # fx_new, fy_new = (
        #     self.camera_dic["fx"] * scale_x), (self.camera_dic["fy"] * scale_y)
        fx_new, fy_new = (
            self.camera_dic["fx"]), (self.camera_dic["fy"])

        z = depth_im / (1/self.camera_dic["depth_scale"])
        u, v = torch.arange(0, width_im)[None, :], torch.arange(
            0, height_im)[None, :].t()
        x = (u - cx_new) * z / fx_new
        y = (v - cy_new) * z / fy_new
        pcl = torch.stack((x, y, z), dim=-1)
        pcl = pcl.reshape(-1, 3)

        masked_pcl = None
        if mask_im is not None:
            mask_index = mask_im.flatten() > 0
            masked_pcl = pcl[mask_index]
        return pcl, masked_pcl

    def translate2gt(self, obj_pcl, R, t):
        # gt_R = Rotate(torch.tensor(self.selected_scenes[ind]["R"]).reshape(3,3))
        # gt_t = Translate(*torch.tensor(self.selected_scenes[ind]["t"]))
        # obj_pcl = self.translate2gt(obj_pcl, gt_R, gt_t)
        obj_pcl = R.inverse().compose(t).transform_points(obj_pcl)
        return obj_pcl
        
        
        
        
# ####################################################################################################################

class T_Less_pcl_generator(Dataset):

    def __init__(self, cfg, phase, transforms=None, obj_id_list=[5], vis_ratio = 0.7):
        super().__init__()
        self.logger = logging.getLogger(__name__)
                   
        self.root = None
        if isinstance(cfg.root, str):
            if os.path.exists(f'{cfg.root}'):
                self.root = cfg.root

        if self.root is None:
            raise AssertionError(f'Dataset not found in {cfg.root}')
        else:
            self.logger.info(f'Loading data from {self.root}')

        self.cfg = cfg

        # set the directories related to the T-Less dataset
        self.train_dir = os.path.join(self.root, "train_pbr")
        self.test_dir = os.path.join(
            self.root, "tless_test_primesense_bop19", "test_primesense")
        self.cad_dir = os.path.join(
            self.root, "tless_models", "models_cad")

        # each directory is related to a specific camera and scene setup
        self.selected_scenes = []
        assert phase in ['train', 'val', 'test']
        base_dir = self.train_dir if phase in ['train'] else self.test_dir
        setup_dirs = [name for name in os.listdir(
            base_dir) if os.path.isdir(os.path.join(base_dir))]
        for setup_dir in setup_dirs:
            scene_gt_path = os.path.join(base_dir, setup_dir, "scene_gt.json")
            camera_gt_path = os.path.join(
                base_dir, setup_dir, "scene_camera.json")
            scene_info_path = os.path.join(base_dir, setup_dir, "scene_gt_info.json")
            with open(scene_gt_path) as scenes_f, open(camera_gt_path) as camera_f, open(scene_info_path) as scene_info_f:
                scenes_dic = json.load(scenes_f)
                cameras_dic = json.load(camera_f)
                scene_info_dic = json.load(scene_info_f)
                for scene_id, _ in scenes_dic.items(): 
                                                                          
                    data_dic = {"dir": os.path.join(
                        base_dir, setup_dir), "scene_id": int(scene_id)} 
                    data_dic.update(cameras_dic[scene_id])
                    self.selected_scenes.append(data_dic)
                                    
        if len(self.selected_scenes) == 0:
            raise AssertionError(
                "there is no selected obj_num in the given directory")
        

        self.search_voxel_size = cfg.overlap_radius
        self.transforms = transforms
        self.phase = phase

    
    def __len__(self):
        return len(self.selected_scenes)

    def __getitem__(self, ind):
        depth_path = os.path.join(self.selected_scenes[ind]["dir"], "depth", str(
            self.selected_scenes[ind]["scene_id"]).zfill(6) + ".png")
        
       
        cam_K = np.array(self.selected_scenes[ind]["cam_K"]).reshape(3, 3)
        cam_depth_scale = 1 / self.selected_scenes[ind]["depth_scale"]

        depth_im = o3d.io.read_image(depth_path)
        im_width, im_height = depth_im.get_max_bound()
        pinhole_cam = o3d.camera.PinholeCameraIntrinsic(
            int(im_width), int(im_height), cam_K)
        scene_pcl = o3d.geometry.PointCloud.create_from_depth_image(
            depth_im, pinhole_cam, depth_scale=cam_depth_scale, depth_trunc=float('inf'))

        # scene_pcl = scene_pcl.farthest_point_down_sample(
        #     self.cfg.scene_point_size)
        pcl = torch.tensor(np.array(scene_pcl.points, dtype=np.float32))
        # pcl_path = os.path.join(self.selected_scenes[ind]["dir"], "pcl")
        # if not os.path.exists(pcl_path):
        #     os.makedirs(pcl_path)
        # o3d.io.write_point_cloud(os.path.join(pcl_path, str(self.selected_scenes[ind]["scene_id"]).zfill(6) + ".ply"), scene_pcl)
        return {"dir": self.selected_scenes[ind]["dir"], "scene_id": self.selected_scenes[ind]["scene_id"], "pcl": pcl}