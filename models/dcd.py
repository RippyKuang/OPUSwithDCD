from mmdet.models import LOSSES
import torch
import torch.nn as nn


@LOSSES.register_module()
class dcdLoss(nn.Module):

    def __init__(self, alpha=40, n_lambda = 0.5, smooth=False, beta=1.0):
        super(dcdLoss, self).__init__()
        
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.loss = torch.nn.L1Loss(reduction='none') if not smooth else torch.nn.SmoothL1Loss(reduction='none',beta=beta)

    def forward(self,
                gt_pts,  
                gt_paired_pts, 
                pred_pts,  
                pred_paired_pts, 
                gt_paired_idx, 
                pred_paired_idx,
                weight_gt=None,
                non_reg=False
                ):
        num_preds, num_gts = pred_pts.shape[0], gt_pts.shape[0]
        weight_gt = weight_gt if weight_gt !=None else torch.ones(num_gts,device=gt_pts.device)

        if non_reg:
            frac_12 = max(1, num_preds / num_gts)
            frac_21 = max(1, num_gts / num_preds)
        else:
            frac_12 = num_preds / num_gts
            frac_21 = num_gts / num_preds

        nx = torch.bincount(gt_paired_idx, minlength=num_preds)[gt_paired_idx].float().detach() ** self.n_lambda
        ny = torch.bincount(pred_paired_idx, minlength=num_gts)[pred_paired_idx].float().detach() ** self.n_lambda

        gt_cost = frac_21 * torch.exp(-self.alpha*torch.norm(gt_pts-gt_paired_pts,dim=-1))/(nx+1e-6)
        loss_1 = self.loss(torch.ones(num_gts,device=gt_pts.device),gt_cost)*weight_gt
        pred_cost =frac_12 * torch.exp(-self.alpha*torch.norm(pred_pts-pred_paired_pts,dim=-1))/(ny+1e-6)
        loss_2 = self.loss(torch.ones(num_preds,device=pred_pts.device),pred_cost)

        return (loss_1.mean()+loss_2.mean())*0.5
