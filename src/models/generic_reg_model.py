"""Generic model for registration"""

import os
from abc import ABC

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from cvhelpers.torch_helpers import to_numpy
from models.generic_model import GenericModel
from models.scheduler.warmup import WarmUpScheduler
from benchmark.benchmark_predator import benchmark as benchmark_predator, benchmark_tless
import benchmark.benchmark_modelnet as benchmark_modelnet
from utils.misc import StatsMeter, metrics_to_string
from utils.se3_torch import se3_compare
from utils.se3_numpy import se3_transform

import open3d as o3d


class GenericRegModel(GenericModel, ABC):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.loss_stats_meter = StatsMeter()  # For accumulating losses
        self.reg_success_thresh_rot = cfg.reg_success_thresh_rot
        self.reg_success_thresh_trans = cfg.reg_success_thresh_trans

    def configure_optimizers(self):  # override

        scheduler_type = self.cfg.get('scheduler', None)
        if scheduler_type is None or scheduler_type in ['none', 'step']:
            base_lr = self.cfg.base_lr
        elif scheduler_type == 'warmup':
            base_lr = 0.0  # start from 0
        else:
            raise NotImplementedError

        # Create optimizer
        if self.cfg.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=base_lr,
                                               weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=base_lr,
                                              weight_decay=self.cfg.weight_decay)
        else:
            raise NotImplementedError

        # Create scheduler
        if scheduler_type == 'warmup':
            # Warmup, then smooth exponential decay
            self.scheduler = WarmUpScheduler(self.optimizer, self.cfg.scheduler_param, self.cfg.base_lr)
        elif scheduler_type == 'step':
            # Step decay
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.cfg.scheduler_param[0],
                                                             self.cfg.scheduler_param[1])
        elif scheduler_type == 'none' or scheduler_type is None:
            # No decay
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 50, 1.0)
        else:
            raise AssertionError('Invalid scheduler')

        self.logger.info(f'Using optimizer {self.optimizer} with scheduler {self.scheduler}')

    def training_step(self, batch, batch_idx):

        pred = self.forward(batch)
        losses = self.compute_loss(pred, batch)

        # Stores the losses for summary writing
        for k in losses:
            self.loss_stats_meter[k].update(losses[k])

        # visualize_registration(batch, pred)
        return pred, losses

    def train_summary_fn(self, writer: SummaryWriter, step: int,
                         data_batch, train_output, train_losses):

        losses_dict = {k: self.loss_stats_meter[k].avg for k in self.loss_stats_meter}
        self._generic_summary_function(writer, step, model=self, losses=losses_dict)
        self.loss_stats_meter.clear()

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch)
        losses = self.compute_loss(pred, batch)
        metrics = self._compute_metrics(pred, batch)

        # visualize_registration(batch, pred, metrics=metrics, iter_idx=5, b=2)

        val_outputs = (losses, metrics)

        return val_outputs

    def validation_epoch_end(self, validation_step_outputs):

        losses = [v[0] for v in validation_step_outputs]
        metrics = [v[1] for v in validation_step_outputs]

        loss_keys = set(losses[0].keys())
        losses_stacked = {k: torch.stack([l[k] for l in losses]) for k in loss_keys}

        # Computes the mean over all metrics
        avg_losses = {k: torch.mean(losses_stacked[k]) for k in loss_keys}
        avg_metrics = self._aggregate_metrics(metrics)
        return avg_metrics['reg_success_final'].item(), {'losses': avg_losses, 'metrics': avg_metrics}

    def validation_summary_fn(self, writer: SummaryWriter, step: int, val_outputs):
        """Logs data during validation. This function will be called after every
        validation run.
        The default implementation saves out the scalars from losses and metrics.

        Args:
            writer: validation writer
            step: The current step number
            val_outputs: Whatever that is returned from validation_epoch_end()

        """
        super().validation_summary_fn(writer, step, val_outputs)

        # Save histogram summaries
        metrics = val_outputs['metrics']
        for k in metrics:
            if k.endswith('hist'):
                writer.add_histogram(f'metrics/{k}', metrics[k], step)

    def test_epoch_start(self):
        if self.cfg.dataset == 'modelnet':
            self.modelnet_metrics = []
            self.modelnet_poses = []

    def test_step(self, batch, batch_idx):

        pred = self.forward(batch)
        losses = self.compute_loss(pred, batch)
        metrics = self._compute_metrics(pred, batch)

        # Dataset specific handling
        if self.cfg.dataset == '3dmatch':
            self._save_3DMatch_log(batch, pred)
        elif self.cfg.dataset == "tless":
            self._save_TLess_log(batch, pred)

        elif self.cfg.dataset == 'modelnet':
            modelnet_data = {
                'points_src': torch.stack(batch['src_xyz']),
                'points_ref': torch.stack(batch['tgt_xyz']),
                'points_raw': torch.stack(batch['tgt_raw']),
                'transform_gt': batch['pose'],
            }
            self.modelnet_metrics.append(
                benchmark_modelnet.compute_metrics(modelnet_data, pred['pose'][-1])
            )
            self.modelnet_poses.append(
                pred['pose'][-1]
            )

        else:
            raise NotImplementedError

        test_outputs = (losses, metrics)
        return test_outputs

    def test_epoch_end(self, test_step_outputs):

        losses = [v[0] for v in test_step_outputs]
        metrics = [v[1] for v in test_step_outputs]

        loss_keys = losses[0].keys()
        losses = {k: torch.stack([l[k] for l in losses]) for k in loss_keys}

        # Computes the mean over all metrics
        avg_losses = {k: torch.mean(losses[k]) for k in loss_keys}
        avg_metrics = self._aggregate_metrics(metrics)

        log_str = 'Test ended:\n'
        log_str += metrics_to_string(avg_losses, '[Losses]') + '\n'
        log_str += metrics_to_string(avg_metrics, '[Metrics]') + '\n'
        self.logger.info(log_str)

        if self.cfg.dataset =='3dmatch':
            # Evaluate 3DMatch registration recall
            results_str, mean_precision = benchmark_predator(
                os.path.join(self._log_path, self.cfg.benchmark),
                os.path.join('datasets', self.cfg.dataset, 'benchmarks', self.cfg.benchmark))
            self.logger.info('\n' + results_str)
            return mean_precision
        elif self.cfg.dataset == 'tless':
            results_str, mean_precision = benchmark_tless(
                os.path.join(self._log_path, self.cfg.benchmark),
                os.path.join(self._log_path, self.cfg.benchmark))
            self.logger.info('\n' + results_str)
            return mean_precision
            
        elif self.cfg.dataset == 'modelnet':
            metric_keys = self.modelnet_metrics[0].keys()
            metrics_cat = {k: np.concatenate([m[k] for m in self.modelnet_metrics])
                           for k in metric_keys}
            summary_metrics = benchmark_modelnet.summarize_metrics(metrics_cat)
            benchmark_modelnet.print_metrics(self.logger, summary_metrics)

            # Also save out the predicted poses, which can be evaluated using
            # RPMNet's eval.py
            poses_to_save = to_numpy(torch.stack(self.modelnet_poses, dim=0))
            np.save(os.path.join(self._log_path, 'pred_transforms.npy'), poses_to_save)

    def _compute_metrics(self, pred, batch):

        metrics = {}
        with torch.no_grad():

            pose_keys = [k for k in pred.keys() if k.startswith('pose')]
            for k in pose_keys:
                suffix = k[4:]
                pose_err = se3_compare(pred[k], batch['pose'][None, :])
                metrics[f'rot_err_deg{suffix}'] = pose_err['rot_deg']
                metrics[f'trans_err{suffix}'] = pose_err['trans']

        return metrics

    def _aggregate_metrics(self, metrics):

        if len(metrics[0]) == 0:
            return {}

        batch_dim = 1  # dim=1 is batch dimension (0 is decoder layer)
        metrics_keys = set(metrics[0].keys())
        metrics_cat = {k: torch.cat([m[k] for m in metrics], dim=batch_dim) for k in metrics_keys}
        num_instances = next(iter(metrics_cat.values())).shape[batch_dim]
        self.logger.info(f'Aggregating metrics, total number of instances: {num_instances}')
        assert all([metrics_cat[k].shape[batch_dim] == num_instances for k in metrics_keys]), \
            'Dimensionality incorrect, check whether batch dimension is consistent'

        rot_err_keys = [k for k in metrics_cat.keys() if k.startswith('rot_err_deg')]
        if len(rot_err_keys) > 0:
            num_pred = metrics_cat[rot_err_keys[0]].shape[0]

        avg_metrics = {}
        for p in range(num_pred):
            suffix = f'{p}' if p < num_pred - 1 else 'final'

            for rk in rot_err_keys:
                pose_type_suffix = rk[11:]

                avg_metrics[f'rot_err_deg{pose_type_suffix}_{suffix}'] = torch.mean(metrics_cat[rk][p])
                avg_metrics[f'rot_err{pose_type_suffix}_{suffix}_hist'] = metrics_cat[rk][p]

                tk = 'trans_err' + pose_type_suffix
                avg_metrics[f'{tk}_{suffix}'] = torch.mean(metrics_cat[tk][p])
                avg_metrics[f'{tk}_{suffix}_hist'] = metrics_cat[tk][p]

                reg_success = torch.logical_and(metrics_cat[rk][p, :] < self.reg_success_thresh_rot,
                                                metrics_cat[tk][p, :] < self.reg_success_thresh_trans)
                avg_metrics[f'reg_success{pose_type_suffix}_{suffix}'] = reg_success.float().mean()

            if 'corr_err' in metrics_cat:
                avg_metrics[f'corr_err_{suffix}_hist'] = metrics_cat['corr_err'][p].flatten()
                avg_metrics[f'corr_err_{suffix}'] = torch.mean(metrics_cat['corr_err'][p])

        return avg_metrics

    @property
    def _log_path(self):
        return self.get_trainer().log_path


    """
    Dataset specific functions
    """
    def _save_3DMatch_log(self, batch, pred):
        B = len(batch['src_xyz'])

        for b in range(B):
            scene = batch['src_path'][b].split(os.path.sep)[1]
            src_idx = int(os.path.basename(batch['src_path'][b]).split('_')[-1].replace('.pth', ''))
            tgt_idx = int(os.path.basename(batch['tgt_path'][b]).split('_')[-1].replace('.pth', ''))

            pred_pose_np = to_numpy(pred['pose'][-1][b]) if pred['pose'].ndim == 4 else \
                to_numpy(pred['pose'][b])
            if pred_pose_np.shape[0] == 3:
                pred_pose_np = np.concatenate([pred_pose_np, [[0., 0., 0., 1.]]], axis=0)

            scene_folder = os.path.join(self._log_path, self.cfg.benchmark, scene)
            os.makedirs(scene_folder, exist_ok=True)
            est_log_path = os.path.join(scene_folder, 'est.log')
            with open(est_log_path, 'a') as fid:
                # We don't know the number of frames, so just put -1
                # This will be ignored by the benchmark function in any case
                fid.write('{}\t{}\t{}\n'.format(tgt_idx, src_idx, -1))
                for i in range(4):
                    fid.write('\t'.join(map('{0:.12f}'.format, pred_pose_np[i])) + '\n')
                    
    def _save_TLess_log_(self, batch, pred):
        B = len(batch['src_xyz'])

        for b in range(B):
            scene = batch['tgt_path'][b].split(os.path.sep)[-3]
            src_idx = int(os.path.basename(batch['src_path'][b]).split(os.path.sep)[-1].replace('.ply', '').replace('obj_', ''))
            tgt_idx = int(os.path.basename(batch['tgt_path'][b]).split(os.path.sep)[-1].replace('.ply', ''))

            pred_pose_np = to_numpy(pred['pose'][-1][b]) if pred['pose'].ndim == 4 else \
                to_numpy(pred['pose'][b])
            if pred_pose_np.shape[0] == 3:
                pred_pose_np = np.concatenate([pred_pose_np, [[0., 0., 0., 1.]]], axis=0)
                
            gt_pose_np = to_numpy(batch['pose'][-1][b]) if batch['pose'].ndim == 4 else \
                to_numpy(batch['pose'][b])
            if gt_pose_np.shape[0] == 3:
                gt_pose_np = np.concatenate([gt_pose_np, [[0., 0., 0., 1.]]], axis=0)

            scene_folder = os.path.join(self._log_path, self.cfg.benchmark, scene)
            os.makedirs(scene_folder, exist_ok=True)
            est_log_path = os.path.join(scene_folder, 'est.log')
            gt_log_path = os.path.join(scene_folder, 'gt.log')
            with open(est_log_path, 'a') as fid, open(gt_log_path, 'a') as fgt:
                # We don't know the number of frames, so just put -1
                # This will be ignored by the benchmark function in any case
                fid.write('{}\t{}\t{}\n'.format(tgt_idx, src_idx, -1))
                for i in range(4):
                    fid.write('\t'.join(map('{0:.12f}'.format, pred_pose_np[i])) + '\n')
                    
                fgt.write('{}\t{}\t{}\n'.format(tgt_idx, src_idx, -1))
                for i in range(4):
                    fgt.write('\t'.join(map('{0:.12f}'.format, gt_pose_np[i])) + '\n')
                    
    
    
    def _save_TLess_log(self, batch, pred):
        B = len(batch['src_xyz'])

        for b in range(B):
            scene = batch['tgt_path'][b].split(os.path.sep)[-3]
            src_idx = int(os.path.basename(batch['src_path'][b]).split(os.path.sep)[-1].replace('.ply', '').replace('obj_', ''))
            tgt_idx = int(os.path.basename(batch['tgt_path'][b]).split(os.path.sep)[-1].replace('.ply', ''))
            log_file_path = os.path.join(self._log_path, self.cfg.benchmark, scene)

            pred_pose_np = to_numpy(pred['pose'][-1][b]) if pred['pose'].ndim == 4 else \
                to_numpy(pred['pose'][b])
            if pred_pose_np.shape[0] == 3:
                pred_pose_np = np.concatenate([pred_pose_np, [[0., 0., 0., 1.]]], axis=0)
                
            gt_pose_np = to_numpy(batch['pose'][-1][b]) if batch['pose'].ndim == 4 else \
                to_numpy(batch['pose'][b])
            if gt_pose_np.shape[0] == 3:
                gt_pose_np = np.concatenate([gt_pose_np, [[0., 0., 0., 1.]]], axis=0)

            scene_pcl_path = batch['tgt_path'][b]
            obj_mesh_path = batch['src_path'][b]
            
            pcl_log_path = os.path.join(
                log_file_path, str(tgt_idx) + "-" + str(src_idx))
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
                pcl_log_path, str(tgt_idx).zfill(6) + ".ply"), pcd)
            
            pcd.points = o3d.utility.Vector3dVector(scene_xyz[to_numpy(batch["correspondences"][b][0])])
            o3d.io.write_point_cloud(os.path.join(
                pcl_log_path, str(tgt_idx).zfill(6) +  "corres_source_gt" + ".ply"), pcd)
            
            pcd.points = o3d.utility.Vector3dVector(scene_xyz[to_numpy(batch["correspondences"][b][1])])
            o3d.io.write_point_cloud(os.path.join(
                pcl_log_path, str(tgt_idx).zfill(6) +  "corres_target_gt" + ".ply"), pcd)
            
            pcd.points = o3d.utility.Vector3dVector(to_numpy(pred["corr_source"]).reshape(-1, 3))
            o3d.io.write_point_cloud(os.path.join(
                pcl_log_path, str(tgt_idx).zfill(6) +  "corres_source_est" + ".ply"), pcd)
            
            pcd.points = o3d.utility.Vector3dVector(to_numpy(pred["corr_target"]).reshape(-1, 3))
            o3d.io.write_point_cloud(os.path.join(
                pcl_log_path, str(tgt_idx).zfill(6) +  "corres_target_est" + ".ply"), pcd)
            
            # pcd.points = o3d.utility.Vector3dVector(scene_xyz[to_numpy(batch["correspondences"][b][1])])
            # o3d.io.write_point_cloud(os.path.join(
            #     pcl_log_path, str(tgt_idx).zfill(6) +  "corres_gt" + ".ply"), pcd)
            
            # pcd.points = o3d.utility.Vector3dVector(scene_xyz[np.array(to_numpy(pred["tgt_kp"][b][1]), dtype=bool)])
            # o3d.io.write_point_cloud(os.path.join(
            #     pcl_log_path, str(tgt_idx).zfill(6) +  "corres_est" + ".ply"), pcd)
                
            gt_points = se3_transform(gt_pose_np, obj_xyz)
            est_points = se3_transform(pred_pose_np, obj_xyz)

            pcd.points = o3d.utility.Vector3dVector(gt_points)
            o3d.io.write_point_cloud(os.path.join(
                pcl_log_path, "obj_gt.ply"), pcd)

            pcd.points = o3d.utility.Vector3dVector(est_points)
            o3d.io.write_point_cloud(os.path.join(
                pcl_log_path, "obj_est.ply"), pcd)
