import os
import mmcv
import numpy as np
import torch
import pickle
import os.path as osp
from tqdm import tqdm
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from torch.utils.data import DataLoader
from models.utils import sparse2dense
from .old_metrics import Metric_mIoU
from mmcv.ops import knn


@DATASETS.register_module()
class NuScenesOccDataset(NuScenesDataset):    
    def __init__(self, *args, **kwargs):
        super().__init__(filter_empty_gt=False, *args, **kwargs)
        self.data_infos = self.load_annotations(self.ann_file)
        self.pc_range = torch.tensor([-40.0, -40.0, -1.0, 40.0, 40.0, 5.4])
        self.voxel_size = torch.tensor([0.4, 0.4, 0.4])
        self.scene_size = self.pc_range[3:] - self.pc_range[:3]
    
    def collect_cam_sweeps(self, index, into_past=150, into_future=0):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['cam_sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]['cams'])
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['cam_sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def collect_lidar_sweeps(self, index, into_past=20, into_future=0):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['lidar_sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        last_timestamp = self.data_infos[index]['timestamp']
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['lidar_sweeps'][::-1]
            if curr_sweeps[0]['timestamp'] == last_timestamp:
                curr_sweeps = curr_sweeps[1:]
            all_sweeps_next.extend(curr_sweeps)
            curr_index = curr_index + 1
            last_timestamp = all_sweeps_next[-1]['timestamp']

        return all_sweeps_prev, all_sweeps_next

    def get_data_info(self, index):
        info = self.data_infos[index]

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation_mat = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation_mat = Quaternion(lidar2ego_rotation).rotation_matrix
        ego2lidar = transform_matrix(
            lidar2ego_translation, Quaternion(lidar2ego_rotation), inverse=True)

        input_dict = dict(
            sample_idx=info['token'],
            scene_name=info['scene_name'],
            timestamp=info['timestamp'] / 1e6,
            ego2lidar=ego2lidar,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation_mat,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation_mat,
        )

        if self.modality['use_lidar']:
            lidar_sweeps_prev, lidar_sweeps_next = self.collect_lidar_sweeps(index)
            input_dict.update(dict(
                pts_filename=info['lidar_path'],
                lidar_sweeps={'prev': lidar_sweeps_prev, 'next': lidar_sweeps_next},
            ))

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []

            for _, cam_info in info['cams'].items():
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T

                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

            cam_sweeps_prev, cam_sweeps_next = self.collect_cam_sweeps(index)

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
                cam_sweeps={'prev': cam_sweeps_prev, 'next': cam_sweeps_next},
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict
    
    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        results_dict = {}
        results_dict.update(
            self.eval_miou(occ_results, runner=runner, show_dir=show_dir, **eval_kwargs))
        results_dict.update(
            self.eval_riou(occ_results, runner=runner, show_dir=show_dir, **eval_kwargs))
        return results_dict
    
    def down_gt(self,gt_points_list,gt_masks_list,gt_labels_list):
        interval = gt_points_list.new_tensor([0.1])
        exclude_values = gt_labels_list.new_tensor([17])  
        pos_list = [0, 1, 2, [0, 1], [0, 2], [1, 2], [0, 1, 2]]

        mask = ~torch.isin(gt_labels_list, exclude_values) 
        offset = gt_points_list.new_tensor([0.1,0.1,0.1])
            
        up_coords = (gt_points_list[mask]-offset).unsqueeze(1).repeat(1, 8, 1).contiguous()
        up_coords_label = gt_labels_list[mask].unsqueeze(1).repeat(1, 8).contiguous()
        up_coords_mask = gt_masks_list[mask].unsqueeze(1).repeat(1, 8).contiguous()
        for j in range(len(pos_list)):
            up_coords[:, j + 1, pos_list[j]] += interval
                
        up_coords = up_coords.reshape(-1, 3)
        up_coords_label = up_coords_label.reshape(-1)
        up_coords_mask = up_coords_mask.reshape(-1)
        new_gt_points_list = torch.cat((gt_points_list,up_coords),dim=0)
        new_gt_labels_list = torch.cat((gt_labels_list,up_coords_label),dim=0)
        new_gt_masks_list = torch.cat((gt_masks_list,up_coords_mask),dim=0)
            
        return new_gt_points_list,new_gt_masks_list,new_gt_labels_list
    
    def get_sparse_voxels(self, voxel_semantics, mask_camera):
        W, H, Z = voxel_semantics.shape
        device = voxel_semantics.device
        voxel_semantics = voxel_semantics.long()

        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * self.scene_size[0] + self.pc_range[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * self.scene_size[1] + self.pc_range[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * self.scene_size[2] + self.pc_range[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, W, Z)
        coors = torch.stack([xx, yy, zz], dim=-1) 

        gt_points, gt_masks, gt_labels = [], [], []
 
        mask = voxel_semantics != 17

        gt_points = coors[mask]
        gt_masks = mask_camera[mask] 
        gt_labels = voxel_semantics[mask]
        
        return gt_points, gt_masks, gt_labels

    def eval_miou(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        occ_gts = []
        occ_preds = []
        lidar_origins = []

        print('\nStarting Evaluation...')
        metric = Metric_mIoU(use_image_mask=True, num_classes=2)
        all_cd_loss =[]
        from tqdm import tqdm
        for i in tqdm(range(len(occ_results))):
            result_dict = occ_results[i]
            info = self.get_data_info(i)
            token = info['sample_idx']
            scene_name = info['scene_name']
            occ_root = 'data/nuscenes/gts/'
            occ_file = osp.join(occ_root, scene_name, token, 'labels.npz')
            occ_infos = np.load(occ_file)

            occ_labels = occ_infos['semantics']
            mask_lidar = occ_infos['mask_lidar'].astype(np.bool_)
            mask_camera = occ_infos['mask_camera'].astype(np.bool_)

            p,m,l = self.get_sparse_voxels(torch.tensor(occ_labels),torch.tensor(mask_camera))
            _p,_,_ = self.down_gt(p,m,l)

            all_cd_loss.append(self.calc_cd(torch.tensor(result_dict['refine_pts']), _p).detach().cpu().numpy())
            
            occ_pred, _ = sparse2dense(
                result_dict['occ_loc'],
                result_dict['sem_pred'],
                dense_shape=occ_labels.shape,
                empty_value=17)
            
            # _occ_pred = np.where(occ_pred == 17,1,0)
            # _occ_labels = np.where(occ_labels == 17,1,0)

            metric.add_batch(occ_pred, occ_labels, mask_lidar, mask_camera)
        np.save("./gtup.npy",np.asarray(all_cd_loss))
        print(f"cd loss mean:{np.asarray(all_cd_loss).mean()}")
        return {'mIoU': metric.count_miou()}
    
    # prim gtup cd loss mean:1.7633755207061768 miou 35.94
    # gtup gtup cd loss mean:1.7894799709320068 miou 35.64
    
    # prim cd loss mean:1.8695778846740723
    # gtup cd loss mean:1.9068219661712646

    # prim no fliter cd loss mean:1.2979758977890015 miou 41.48
    # gtup no fliter cd loss mean:1.3073328733444214 miou 41.44

    # prim no fliter gtup cd loss mean:1.1869158744812012 miou 41.48
    # gtup no fliter gtup cd loss mean:1.1857978105545044 miou 41.44

    def calc_cd(self, pred_pts, gt_pts):
        g = gt_pts.cuda()
        p = pred_pts.cuda()
        g_pair_p_idx = knn(1,p[None,...],g[None,...]).permute(0, 2, 1).squeeze().long()
        p_pair_g_idx = knn(1,g[None,...],p[None,...]).permute(0, 2, 1).squeeze().long()
        g_pair_p,p_pair_g = p[g_pair_p_idx],g[p_pair_g_idx]
        return torch.abs(g - g_pair_p).sum(-1).mean()+ torch.abs(p - p_pair_g).sum(-1).mean()
    
    def eval_riou(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        occ_gts = []
        occ_preds = []
        lidar_origins = []

        print('\nStarting Evaluation...')

        from .ray_metrics import main as calc_rayiou
        from .ego_pose_dataset import EgoPoseDataset

        data_loader = DataLoader(
            EgoPoseDataset(self.data_infos),
            batch_size=1,
            shuffle=False,
            num_workers=8
        )
        
        sample_tokens = [info['token'] for info in self.data_infos]

        for i, batch in enumerate(data_loader):
            token = batch[0][0]
            output_origin = batch[1]
            
            data_id = sample_tokens.index(token)
            info = self.data_infos[data_id]

            token = info['token']
            scene_name = info['scene_name']
            occ_root = 'data/nuscenes/gts/'
            occ_file = osp.join(occ_root, scene_name, token, 'labels.npz')
            occ_infos = np.load(occ_file)
            gt_semantics = occ_infos['semantics']

            occ_pred = occ_results[data_id]
            sem_pred = torch.from_numpy(occ_pred['sem_pred'])  # [B, N]
            occ_loc = torch.from_numpy(occ_pred['occ_loc'].astype(np.int64))  # [B, N, 3]
            
            occ_size = list(gt_semantics.shape)
            dense_sem_pred, _ = sparse2dense(occ_loc, sem_pred, dense_shape=occ_size, empty_value=17)
            dense_sem_pred = dense_sem_pred.squeeze(0).numpy()

            lidar_origins.append(output_origin)
            occ_gts.append(gt_semantics)
            occ_preds.append(dense_sem_pred)
        
        return calc_rayiou(occ_preds, occ_gts, lidar_origins)

    def format_results(self, occ_results,submission_prefix,**kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path=os.path.join(submission_prefix, '{}.npz'.format(sample_token))
            np.savez_compressed(save_path,occ_pred.astype(np.uint8))
        print('\nFinished.')