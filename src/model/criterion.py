import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .box_ops import generalized_box_iou
from .detector import DFINEWeightingFunction

class DFINECriterion(nn.Module):
    """
    This class computes the 4 losses for our Dense Patch-as-Predictor model:
    1. Focal Loss (Classification)
    2. L1 Loss (Bounding Box)
    3. GIoU Loss (Bounding Box)
    4. FGL Loss (D-FINE Fine-Grained Localization for edge distributions)
    """
    def __init__(self, matcher, num_classes, weight_dict, reg_max=32, alpha=0.25, gamma=2.0):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.reg_max = reg_max
        self.alpha = alpha
        self.gamma = gamma
        
        # We need the weighting function to map target residuals back to discrete bins
        self.weighting_fn = DFINEWeightingFunction(reg_max=reg_max)

    def _get_src_permutation_idx(self, indices):
        # Permute predictions following the Hungarian Matcher indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (Focal Loss) applied to ALL predictions."""
        src_logits = outputs["pred_logits"] # Shape: (Batch, Num_Predictions, Num_Classes)
        idx = indices
        
        # Get the target classes for the matched predictions
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # Create a target tensor filled with 0s (Background)
        target_classes = torch.zeros_like(src_logits)
        
        # Set the matched indices to 1.0 for their specific class
        target_classes[idx[0], idx[1], target_classes_o] = 1.0
        
        # Compute Focal Loss
        loss_ce = torchvision.ops.sigmoid_focal_loss(
            src_logits, target_classes, alpha=self.alpha, gamma=self.gamma, reduction="none"
        )
        loss_ce = loss_ce.sum() / num_boxes
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """L1 and GIoU loss applied ONLY to matched predictions."""
        # Extract only the matched predictions
        src_boxes = outputs["pred_boxes"][indices[0], indices[1]]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # 1. L1 Loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / num_boxes
        
        # 2. GIoU Loss
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
        loss_giou = loss_giou.sum() / num_boxes
        
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def loss_fgl(self, outputs, targets, indices, num_boxes):
        """D-FINE Fine-Grained Localization Loss applied ONLY to matched predictions."""
        src_edge_logits = outputs["pred_edge_logits"][indices[0], indices[1]] # (N_matched, 4, reg_max+1)
        src_centers = outputs["absolute_centers"][indices[0], indices[1]]     # (N_matched, 2)
        src_shapes = outputs["learnable_shapes"][indices[0], indices[1]]      # (N_matched, 2)
        
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        cx, cy = src_centers[:, 0], src_centers[:, 1]
        w, h = src_shapes[:, 0], src_shapes[:, 1]
        x1, y1, x2, y2 = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]
        
        # Reverse the decoding math to get the target residuals
        L_res = (cx - x1) / w - 0.5
        T_res = (cy - y1) / h - 0.5
        R_res = (x2 - cx) / w - 0.5
        B_res = (y2 - cy) / h - 0.5
        
        target_res = torch.stack([L_res, T_res, R_res, B_res], dim=-1) # (N_matched, 4)
        
        # Map the continuous target residuals to the discrete bins
        W = self.weighting_fn.w.to(target_res.device)
        target_res = target_res.clamp(W[0].item(), W[-1].item())
        
        # Find the two closest bins (left and right)
        idx_right = torch.searchsorted(W, target_res).clamp(1, self.reg_max)
        idx_left = idx_right - 1
        
        W_left = W[idx_left]
        W_right = W[idx_right]
        
        # Calculate interpolation weights (soft targets)
        w_left = (W_right - target_res) / (W_right - W_left + 1e-6)
        w_right = (target_res - W_left) / (W_right - W_left + 1e-6)
        
        # Compute Soft Cross-Entropy
        log_probs = F.log_softmax(src_edge_logits, dim=-1)
        loss_left = -torch.gather(log_probs, -1, idx_left.unsqueeze(-1)).squeeze(-1) * w_left
        loss_right = -torch.gather(log_probs, -1, idx_right.unsqueeze(-1)).squeeze(-1) * w_right
        
        loss_fgl = (loss_left + loss_right).sum() / num_boxes
        
        return {"loss_fgl": loss_fgl}

    def forward(self, outputs, targets):
        """
        outputs: dict containing:
            - "pred_logits": (B, P*K, C)
            - "pred_boxes": (B, P*K, 4)
            - "pred_edge_logits": (B, P*K, 4, reg_max+1)
            - "absolute_centers": (B, P*K, 2)
            - "learnable_shapes": (B, P*K, 2)
        targets: list of dicts containing "labels" and "boxes"
        """
        # 1. Run Hungarian Matcher
        indices = self.matcher(outputs, targets)
        
        # 2. Extract matched indices
        idx = self._get_src_permutation_idx(indices)
        
        # 3. Compute normalization factor (number of ground truth boxes)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs["pred_logits"].device)
        num_boxes = torch.clamp(num_boxes, min=1).item()
        
        # 4. Compute all losses
        losses = {}
        losses.update(self.loss_labels(outputs, targets, idx, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, idx, num_boxes))
        losses.update(self.loss_fgl(outputs, targets, idx, num_boxes))
        
        # 5. Apply weights
        return {k: v * self.weight_dict[k] for k, v in losses.items() if k in self.weight_dict}
