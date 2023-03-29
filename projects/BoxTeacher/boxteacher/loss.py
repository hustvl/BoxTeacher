import math
import torch
import torch.nn.functional as F

from detectron2.utils.events import get_event_storage


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss    


def tversky_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    fn = (target * (1 - x)).sum(dim=1)
    fp = ((1 - target) * x).sum(dim=1)
    loss = 1. - (intersection + eps) / (intersection + 0.7 * fn + 0.3 * fp + eps)
    return loss


def noisy_dice_loss(x, target, beta=1.5):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    intersection = (torch.abs(x - target) ** beta).sum(-1)
    loss = intersection / union
    return loss


def multiscale_dice_loss(x, target):
    strides = [1, 2, 4]
    kernel_sizes = [3, 3, 5]
    losses = []
    for (kernel, stride) in zip(kernel_sizes, strides):
        inp = F.max_pool2d(x, kernel_size=kernel, stride=stride, padding=kernel // 2)
        tgt = F.interpolate(
            target, (inp.shape[-2], inp.shape[-1]), mode='nearest')
        losses.append(dice_coefficient(inp, tgt))
    return sum(losses) / float(len(kernel_sizes))


def compute_avg_projection(mask_scores, gt_bitmasks, flags=None):
    mask_losses_y = dice_coefficient(
        mask_scores.mean(dim=2, keepdim=True),
        gt_bitmasks.mean(dim=2, keepdim=True)
    )
    mask_losses_x = dice_coefficient(
        mask_scores.mean(dim=3, keepdim=True),
        gt_bitmasks.mean(dim=3, keepdim=True)
    )
    return (mask_losses_x + mask_losses_y)


def compute_project_term(mask_scores, gt_bitmasks):

    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y)


def generate_grid_locations(grid_size, device):
    a = torch.arange(grid_size, device=device)
    b = torch.arange(grid_size, device=device)
    locations = torch.stack(torch.meshgrid(a, b)).view(2, -1).transpose(0, 1)
    return locations


def naive_point_sampling_loss(mask_logits, gt_bitmasks, num_points=3136):
    # 56 * 56 = 3136
    # 112 * 112 = 12544
    # generate points
    num_samples = mask_logits.size(0)
    # BxNxC
    sample_points = torch.rand(num_samples, num_points, 2).to(mask_logits.device).requires_grad_(False)
    sample_points =(2 * sample_points - 1).view(num_samples, num_points, 1, 2)
    pred_sample_points = F.grid_sample(mask_logits, sample_points, mode='bilinear', align_corners=False)
    gt_sample_points = F.grid_sample(gt_bitmasks, sample_points, mode='bilinear', align_corners=False)
    # print("pos/neg:{}".format(torch.mean(gt_sample_points)))
    # use bce loss
    get_event_storage().put_scalar("point_pos_neg", torch.mean(gt_sample_points).item())
    point_loss = F.binary_cross_entropy_with_logits(pred_sample_points, gt_sample_points, reduction='none').view(num_samples,-1)
    return point_loss.mean(1)
    
    
def dice_point_sampling_loss(mask_logits, gt_bitmasks, num_points=3136):
    # 56 * 56 = 3136
    # 112 * 112 = 12544
    # generate points
    num_samples = mask_logits.size(0)
    # BxNxC
    sample_points = torch.rand(num_samples, num_points, 2).to(mask_logits.device).requires_grad_(False)
    sample_points =(2 * sample_points - 1).view(num_samples, num_points, 1, 2)
    pred_sample_points = F.grid_sample(mask_logits.sigmoid(), sample_points, mode='bilinear', align_corners=False)
    gt_sample_points = F.grid_sample(gt_bitmasks, sample_points, mode='bilinear', align_corners=False)
    # print("pos/neg:{}".format(torch.mean(gt_sample_points)))
    # use bce loss
    get_event_storage().put_scalar("point_pos_neg", torch.mean(gt_sample_points).item())
    point_loss = dice_coefficient(pred_sample_points, gt_sample_points).view(num_samples,-1)
    return point_loss.mean(1)
    

def ratio_dice_point_sampling_loss(mask_logits, gt_bitmasks, gt_boxmasks, num_points=3136):
    # 56 * 56 = 3136
    # 112 * 112 = 12544
    # generate points
    num_samples = mask_logits.size(0)
    # BxNxC
    sample_points = torch.rand(num_samples, num_points, 2).to(mask_logits.device).requires_grad_(False)
    sample_points =(2 * sample_points - 1).view(num_samples, num_points, 1, 2)
    pred_sample_points = F.grid_sample(mask_logits.sigmoid(), sample_points, mode='bilinear', align_corners=False)
    gt_sample_points = F.grid_sample(gt_bitmasks, sample_points, mode='bilinear', align_corners=False)
    # print("pos/neg:{}".format(torch.mean(gt_sample_points)))
    # use bce loss
    get_event_storage().put_scalar("point_pos_neg", torch.mean(gt_sample_points).item())
    point_loss = dice_coefficient(pred_sample_points, gt_sample_points).view(num_samples,-1)
    return point_loss.mean(1)


def dice_point_grid_loss(mask_logits, gt_bitmasks, num_points=3136):
    # 56 * 56 = 3136
    # 112 * 112 = 12544
    # generate points
    num_samples = mask_logits.size(0)
    # BxNxC
    _num_points = int(math.sqrt(num_points))
    pred_sample_points = F.interpolate(mask_logits.sigmoid(), (_num_points, _num_points), mode='bilinear', align_corners=False)
    gt_sample_points = F.interpolate(gt_bitmasks, (_num_points, _num_points), mode='bilinear', align_corners=False)
    # sample_points = torch.rand(num_samples, num_points, 2).to(mask_logits.device).requires_grad_(False)
    # sample_points =(2 * sample_points - 1).view(num_samples, num_points, 1, 2)
    # pred_sample_points = F.grid_sample(mask_logits.sigmoid(), sample_points, mode='bilinear', align_corners=False)
    # gt_sample_points = F.grid_sample(gt_bitmasks, sample_points, mode='bilinear', align_corners=False)
    # print("pos/neg:{}".format(torch.mean(gt_sample_points)))
    # use bce loss
    # get_event_storage().put_scalar("point_pos_neg", torch.mean(gt_sample_points).item())
    point_loss = dice_coefficient(pred_sample_points, gt_sample_points).view(num_samples,-1)
    return point_loss.mean(1)

def dice_point_grid_pool_loss(mask_logits, gt_bitmasks, num_points=3136):
    # 56 * 56 = 3136
    # 112 * 112 = 12544
    # generate points
    num_samples = mask_logits.size(0)
    # BxNxC
    _num_points = int(math.sqrt(num_points))
    pred_sample_points = F.adaptive_avg_pool2d(mask_logits.sigmoid(), (_num_points, _num_points), mode='bilinear', align_corners=False)
    gt_sample_points = F.adaptive_avg_pool2d(gt_bitmasks, (_num_points, _num_points), mode='bilinear', align_corners=False)

    point_loss = dice_coefficient(pred_sample_points, gt_sample_points).view(num_samples,-1)
    return point_loss.mean(1)

def hard_mining_dice_point_sampling_loss(mask_logits, gt_bitmasks, gt_boxmasks, num_points=3136):
    # 56 * 56 = 3136
    # 112 * 112 = 12544
    # generate points
    num_samples = mask_logits.size(0)
    # BxNxC
    sample_points = torch.rand(num_samples, num_points, 2).to(mask_logits.device).requires_grad_(False)
    sample_points =(2 * sample_points - 1).view(num_samples, num_points, 1, 2)
    pred_sample_points = F.grid_sample(mask_logits.sigmoid(), sample_points, mode='bilinear', align_corners=False)
    gt_sample_points = F.grid_sample(gt_bitmasks, sample_points, mode='bilinear', align_corners=False)
    # print("pos/neg:{}".format(torch.mean(gt_sample_points)))
    # use bce loss
    get_event_storage().put_scalar("point_pos_neg", torch.mean(gt_sample_points).item())
    point_loss = dice_coefficient(pred_sample_points, gt_sample_points).view(num_samples,-1)
    return point_loss.mean(1)
    
# def positive_point_sampling_loss(mask_logits, gt_bitmasks, num_points=3136):
    
#     num_samples = mask_logits.size(0)
#     # BxNxC
#     sample_points = torch.rand(num_samples, num_points, 2).to(mask_logits.device).requires_grad_(False)
#     sample_points =(2 * sample_points - 1).view(num_samples, num_points, 1, 2)
#     pred_sample_points = F.grid_sample(mask_logits, sample_points, mode='bilinear', align_corners=False)
#     gt_sample_points = F.grid_sample(gt_bitmasks, sample_points, mode='bilinear', align_corners=False)
#     # print("pos/neg:{}".format(torch.mean(gt_sample_points)))
#     # use bce loss
#     get_event_storage().put_scalar("point_pos_neg", torch.mean(gt_sample_points).item())
#     point_loss = F.binary_cross_entropy_with_logits(pred_sample_points, gt_sample_points, reduction='none').view(num_samples,-1)
#     return point_loss.mean(1)

def standard_dice_point_loss(mask_logits, gt_bitmasks, num_points=3136):
    point_loss = dice_coefficient(mask_logits.sigmoid(), gt_bitmasks)
    return point_loss
    


def get_point_sampling_loss_fn(point_loss_type):
    if point_loss_type == "naive":
        return naive_point_sampling_loss
    elif point_loss_type == "dice":
        return dice_point_sampling_loss
    elif point_loss_type == "dice_grid":
        return dice_point_grid_loss
    elif point_loss_type == "std_dice":
        return standard_dice_point_loss
    else:
        raise NotImplementedError()
    

@torch.no_grad()
def obtain_uncertrain_mask_quality(mask_scores, gt_bitmasks, threshold=0.2):
    # using mask threshold to measure the IoU
    pred_masks = (mask_scores >= 0.5).float()
    pred_masks = pred_masks.flatten(1)
    targets = (gt_bitmasks >= 0.5).flatten(1) 
    intersection = (pred_masks * targets).sum(-1)
    union = targets.sum(-1) + pred_masks.sum(-1) - intersection
    iou = intersection / (union + 1e-6)
    uncertainty = (iou > threshold).float()
    return uncertainty
