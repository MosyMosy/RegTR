from data_loaders.tless import T_Less, T_Less_pcl_generator
from pytorch3d.io import IO, ply_io
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.ops import sample_farthest_points
from utils.se3_numpy import se3_transform
import torch
import os
import open3d as o3d
import time
import numpy as np
import shutil


def visualize_tless():
    class conf:
        root = "/export/livia/home/vision/Myazdanpanah/dataset/t-less"
        overlap_radius = 0.0375
        obj_point_size = 2048
        scene_point_size = 20000

    tless = T_Less(conf(), "train", is_visualize=True)
    print(len(tless))
    scene_pcl, obj_pcl, scene_overlap, obj_overlap, scene_corr, obj_corr = tless[543]

    IO().save_pointcloud(Pointclouds(scene_pcl[None]), "scene_pcl.ply")
    IO().save_pointcloud(Pointclouds(scene_overlap[None]), "scene_overlap.ply")
    IO().save_pointcloud(Pointclouds(obj_pcl[None]), "obj_pcl.ply")
    IO().save_pointcloud(Pointclouds(obj_overlap[None]), "obj_overlap.ply")
    IO().save_pointcloud(Pointclouds(scene_corr[None]), "scene_corr.ply")
    IO().save_pointcloud(Pointclouds(obj_corr[None]), "obj_corr.ply")


def generate_pcl(mode="train", batch=20, worker=20):

    class conf:
        root = "/export/livia/home/vision/Myazdanpanah/dataset/t-less"
        overlap_radius = 0.0375
        obj_point_size = 2048
        scene_point_size = 20000

    pcl_gen_dataset = T_Less_pcl_generator(conf, mode)

    def collate_fn(batch):
        return {
            'pcl': [x['pcl'] for x in batch],
            'dir': [x['dir'] for x in batch],
            'scene_id': [x['scene_id'] for x in batch]
        }
    collate = None if mode == "train" else collate_fn
    data_loader = torch.utils.data.DataLoader(
        pcl_gen_dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=worker,
        collate_fn=collate
    )

    print(len(data_loader))
    for ind, data in enumerate(data_loader):
        start_time = time.time()
        if mode == "train":
            plc_batch = sample_farthest_points(data["pcl"].cuda(), K=40960)[
                0].detach().cpu()
        else:
            plc_batch = [sample_farthest_points(pcl[None].cuda(), K=40960)[
                0][0].detach().cpu() for pcl in data["pcl"]]

        for j, pcl in enumerate(plc_batch):
            pcl_path = os.path.join(data["dir"][j], "pcl")
            if not os.path.exists(pcl_path):
                os.makedirs(pcl_path)
            path = (os.path.join(pcl_path, str(
                data["scene_id"][j]).zfill(6) + ".ply"))
            pcd = o3d.geometry.PointCloud()
            pcl = pcl.numpy()
            pcd.points = o3d.utility.Vector3dVector(pcl)
            if ~os.path.exists(path=path):
                o3d.io.write_point_cloud(path, pcd)
            # IO().save_pointcloud(data=pcl, path=path)
        end_time = time.time()
        print("{0}, {1}".format(ind, end_time - start_time))


def visulize_est():
    def info_extractor(lines, i):
        header_list = np.fromstring(lines[i], sep="\t", dtype=int)
        obj_id = int(header_list[1])
        scene_id = int(header_list[0])
        pose = np.genfromtxt(
            " ".join(lines[i+1:i+5]).splitlines(), dtype=float)
        pose = pose[:3, :4]
        return obj_id, scene_id, pose

    log_path = "../logdev/T-Less"
    dataset_path = "/export/livia/home/vision/Myazdanpanah/dataset/t-less/"
    dataset_test_path = os.path.join(
        dataset_path, "tless_test_primesense_bop19", "test_primesense")
    dataset_obj_path = os.path.join(
        dataset_path, "tless_models", "models_cad")

    setup_name_list = next(os.walk(log_path))[1]

    for setup_name in setup_name_list:
        log_file_path = os.path.join(log_path, setup_name)
        with open(os.path.join(log_file_path, "gt.log")) as gt_f, open(os.path.join(log_file_path, "est.log"))as est_f:
            gt_lines = gt_f.readlines()
            est_lines = est_f.readlines()
            for i in range(0, len(gt_lines), 5):
                obj_id, scene_id, gt_pose = info_extractor(gt_lines, i)
                _, _, est_pose = info_extractor(est_lines, i)
                scene_pcl_path = os.path.join(
                    dataset_test_path, setup_name, "pcl", str(scene_id).zfill(6) + ".ply")
                obj_mesh_path = os.path.join(
                    dataset_obj_path, "obj_" + str(obj_id).zfill(6) + ".ply")

                pcl_log_path = os.path.join(
                    log_file_path, str(scene_id) + "-" + str(obj_id))
                if not os.path.exists(pcl_log_path):
                    os.makedirs(pcl_log_path)
                # shutil.copy(scene_pcl_path, pcl_log_path)
                scene_pcl = o3d.io.read_point_cloud(scene_pcl_path)
                scene_xyz = np.array(scene_pcl.points, dtype=np.float32)

                obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_path)
                obj_mesh.compute_vertex_normals()
                obj_pcl = obj_mesh.sample_points_uniformly(
                    number_of_points=4096)
                obj_xyz = np.array(obj_pcl.points)
                
                scale = 1/np.max(np.sqrt(np.sum(scene_xyz**2, axis=1)))
                scene_xyz = scene_xyz * scale                
                scene_center = np.mean([(np.min(scene_xyz[:, i]), np.max(scene_xyz[:, i])) for i in range(3)], axis=1)
                scene_xyz = scene_xyz - scene_center
                obj_xyz = obj_xyz * scale    
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(scene_xyz)
                o3d.io.write_point_cloud(os.path.join(
                    pcl_log_path, str(scene_id).zfill(6) + ".ply"), pcd)
                    
                gt_points = se3_transform(gt_pose, obj_xyz)
                est_points = se3_transform(est_pose, obj_xyz)

                pcd.points = o3d.utility.Vector3dVector(gt_points)
                o3d.io.write_point_cloud(os.path.join(
                    pcl_log_path, "obj_gt.ply"), pcd)

                pcd.points = o3d.utility.Vector3dVector(est_points)
                o3d.io.write_point_cloud(os.path.join(
                    pcl_log_path, "obj_est.ply"), pcd)


# visulize_est()

# generate_pcl(mode = "train", batch = 100, worker=2)
# generate_pcl(mode = "test", batch = 10, worker=2)

# generate_pcl_dataset("/export/livia/home/vision/Myazdanpanah/dataset/t-less/train_pbr", 40960)
# generate_pcl_dataset("/export/livia/home/vision/Myazdanpanah/dataset/t-less/tless_test_primesense_bop19/test_primesense", 40960)
visualize_tless()
