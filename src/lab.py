from data_loaders.tless import T_Less,generate_pcl_dataset, T_Less_pcl_generator
from pytorch3d.io import IO, ply_io
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.ops import sample_farthest_points
import torch
import os
import open3d as o3d
import time

def visualize_tless():
    class conf:
            root = "/export/livia/home/vision/Myazdanpanah/dataset/t-less"
            overlap_radius = 0.0375
            obj_point_size = 2048
            scene_point_size = 20000


    tless = T_Less(conf(), "train")
    print(len(tless))
    scene_pcl, obj_pcl, scene_overlap, obj_overlap, scene_corr, obj_corr = tless[115]

    IO().save_pointcloud(Pointclouds(scene_pcl[None]), "scene_pcl.ply")
    IO().save_pointcloud(Pointclouds(scene_overlap[None]), "scene_overlap.ply")
    IO().save_pointcloud(Pointclouds(obj_pcl[None]), "obj_pcl.ply")
    IO().save_pointcloud(Pointclouds(obj_overlap[None]), "obj_overlap.ply")
    IO().save_pointcloud(Pointclouds(scene_corr[None]), "scene_corr.ply")
    IO().save_pointcloud(Pointclouds(obj_corr[None]), "obj_corr.ply")

def generate_pcl(mode = "train", batch = 20, worker=20):
    
    class conf:
            root = "/export/livia/home/vision/Myazdanpanah/dataset/t-less"
            overlap_radius = 0.0375
            obj_point_size = 2048
            scene_point_size = 20000

    pcl_gen_dataset = T_Less_pcl_generator(conf, mode)
    data_loader = torch.utils.data.DataLoader(
            pcl_gen_dataset,
            batch_size=batch,
            shuffle=False,
            num_workers=worker,
        )
    
    print(len(data_loader))
    for i, data in enumerate(data_loader):
        start_time = time.time()
        plc_batch = sample_farthest_points(data["pcl"].cuda(), K=40960)[0].detach().cpu()
                   
        for j, pcl in enumerate(plc_batch):
            pcl_path = os.path.join(data["info"]["dir"][j], "pcl")
            if not os.path.exists(pcl_path):
                os.makedirs(pcl_path)
            path=(os.path.join(pcl_path, str(data["info"]["scene_id"][j].item()).zfill(6) + ".ply"))
            pcd = o3d.geometry.PointCloud()
            pcl = pcl.numpy()
            pcd.points = o3d.utility.Vector3dVector(pcl)

            o3d.io.write_point_cloud(path, pcd)
            # IO().save_pointcloud(data=pcl, path=path)
        end_time = time.time()
        print("{0}, {1}".format(i, end_time - start_time), end="\r")

generate_pcl(mode = "train", batch = 100, worker=2)
generate_pcl(mode = "test", batch = 100, worker=2)

# generate_pcl_dataset("/export/livia/home/vision/Myazdanpanah/dataset/t-less/train_pbr", 40960)
# generate_pcl_dataset("/export/livia/home/vision/Myazdanpanah/dataset/t-less/tless_test_primesense_bop19/test_primesense", 40960)
# visualize_tless()