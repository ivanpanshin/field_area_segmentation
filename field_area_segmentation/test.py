import sys
sys.path.append('../field_area_segmentation')

import os
import hydra
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.distributed import destroy_process_group, init_process_group
from field_area_segmentation.helpers import (
    build_loaders,
    build_optim,
)

import logging
from tqdm import tqdm
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import warnings
from shapely.geometry import Polygon
from imantics import Polygons, Mask
from glob import glob
import tifffile
import cv2

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def getIOU(polygon1, polygon2):
    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area

    if union == 0:
        return 0
    return intersection / union


def group_pixels_tiles(ctr, offsets, patch_size=512):
    """
    Gives each pixel in the image an instance id.
    Arguments:
        ctr: A Tensor of shape [K, 2] where K is the number of center points. The order of the second dim is (y, x).
        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of the second dim is (offset_y, offset_x).
        patch_size: The size of the patches to process the image in smaller chunks.
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
    """
    if offsets.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    offsets = offsets.squeeze(0)
    height, width = offsets.size()[1:]

    # Generate a coordinate map, where each location is the coordinate of that location
    y_coord = torch.arange(height, dtype=offsets.dtype, device=offsets.device).repeat(1, width, 1).transpose(1, 2)
    x_coord = torch.arange(width, dtype=offsets.dtype, device=offsets.device).repeat(1, height, 1)
    coord = torch.cat((y_coord, x_coord), dim=0)

    ctr_loc = coord + offsets
    ctr_loc = ctr_loc.reshape((2, height * width)).transpose(1, 0)

    # ctr: [K, 2] -> [K, 1, 2]
    # ctr_loc = [H*W, 2] -> [1, H*W, 2]
    ctr = ctr.unsqueeze(1)
    ctr_loc = ctr_loc.unsqueeze(0)

    instance_id = torch.zeros((1, height, width), dtype=torch.long, device=offsets.device)

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            y_end = min(y + patch_size, height)
            x_end = min(x + patch_size, width)

            patch_coord = coord[:, y:y_end, x:x_end]
            patch_offsets = offsets[:, y:y_end, x:x_end]
            patch_ctr_loc = patch_coord + patch_offsets
            patch_ctr_loc = patch_ctr_loc.reshape((2, -1)).transpose(1, 0).unsqueeze(0)

            patch_distance = torch.norm(ctr - patch_ctr_loc, dim=-1)
            patch_instance_id = torch.argmin(patch_distance, dim=0).reshape((1, y_end - y, x_end - x)) + 1
            instance_id[:, y:y_end, x:x_end] = patch_instance_id

    return instance_id

def find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=3, top_k=None):
    """
    Find the center points from the center heatmap.
    Arguments:
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
    Returns:
        A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).
    """
    if ctr_hmp.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    # thresholding, setting values below threshold to -1
    ctr_hmp = F.threshold(ctr_hmp, threshold, -1)

    # NMS
    nms_padding = (nms_kernel - 1) // 2
    ctr_hmp_max_pooled = F.max_pool2d(ctr_hmp, kernel_size=nms_kernel, stride=1, padding=nms_padding)
    ctr_hmp[ctr_hmp != ctr_hmp_max_pooled] = -1

    # squeeze first two dimensions
    ctr_hmp = ctr_hmp.squeeze()
    assert len(ctr_hmp.size()) == 2, 'Something is wrong with center heatmap dimension.'

    # find non-zero elements
    ctr_all = torch.nonzero(ctr_hmp > 0)
    if top_k is None:
        return ctr_all
    elif ctr_all.size(0) < top_k:
        return ctr_all
    else:
        # find top k centers.
        top_k_scores, _ = torch.topk(torch.flatten(ctr_hmp), top_k)
        return torch.nonzero(ctr_hmp > top_k_scores[-1])

def save_preds(
    total_ids,
    total_preds,
    output_path,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    total_ids_np = np.array(total_ids).reshape(-1, 1)
    combined_data = np.hstack((total_ids_np, total_preds.numpy()))
    sub = pd.DataFrame(combined_data)  

    sub.to_csv(output_path, index=False)
    logging.info(f'saved to {output_path}')

@torch.no_grad()
def test_step(
    model,
    input: torch.Tensor,
):
    with torch.cuda.amp.autocast():
        return model.module(input)


@torch.no_grad()
def test_epoch(
    cfg,
    model,
    loaders,
    device,
):
    model.eval()

    original_dataset_paths = glob(f'data/preprocessed_test/images/*tif')
    base_name_to_size = {}

    for path in original_dataset_paths:
        base_name_to_size[path.split('/')[-1]] = tifffile.imread(path).shape[:2]

    name_to_preds = {}
    for batch_index, input in enumerate(tqdm(loaders['test_0'])):
        input = input.to(device)
        preds = test_step(
            model=model,
            input=input,
        )

        for index in range(preds['masks_semantic'].shape[0]):
            masks_semantic_sigmoid_th = torch.sigmoid(preds['masks_semantic'][index]).squeeze() > 0.3
            masks_centers = (preds['masks_centers'][index]).squeeze()

            masks_centers_pp = torch.where(masks_semantic_sigmoid_th == 1, masks_centers, 0)

            instance_centers = find_instance_center(
                ctr_hmp=masks_centers_pp.unsqueeze(0),
                threshold=15,
                nms_kernel=3,
                top_k=None,
            )

            instance_segmentation_mask_pred = group_pixels_tiles(
                ctr=instance_centers,
                offsets=preds['masks_offsets'][index].unsqueeze(0),
            ).squeeze().cpu().numpy()

            instance_segmentation_mask_pred = np.where(masks_semantic_sigmoid_th.cpu().numpy() == 1,
                                                       instance_segmentation_mask_pred, 0)

            original_height, original_width = base_name_to_size[loaders['test_0'].dataset.im_ids[batch_index] + '.tif']
            instance_segmentation_mask_pred = cv2.resize(
                instance_segmentation_mask_pred,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            )

            polygons_preds = []
            for value in range(1, np.max(instance_segmentation_mask_pred) + 1):
                if np.sum(instance_segmentation_mask_pred == value) > 90:
                    polygons = Mask(instance_segmentation_mask_pred == value).polygons().segmentation
                    max_len, max_polygon = 0, None
                    for polygon in polygons:
                        if len(polygon) > max_len:
                            max_len = len(polygon)
                            max_polygon = polygon

                    if max_polygon:
                        max_polygon_points = []
                        for i in range(0, len(max_polygon), 2):
                            max_polygon_points.extend([max_polygon[i], max_polygon[i + 1]])

                        if len(max_polygon_points) >= 4:
                            max_polygon_shapely = max_polygon_points
                            polygons_preds.append(max_polygon_shapely)

            name_to_preds[loaders['test_0'].dataset.im_ids[batch_index] + '.tif'] = [{'class': 'field', 'segmentation': _} for _ in
                                                                          polygons_preds]

    json_pred_dict = {
        'images': [
            {
                'file_name': k,
                'annotations': v,
            } for (k, v) in name_to_preds.items()
        ]
    }

    with open("sub_reproduced.json", "w") as outfile:
        json.dump(json_pred_dict, outfile, indent=4, sort_keys=False)

    logging.info(f'reproduced submission saved as sub_reproduced.json')



@hydra.main(config_path="../configs", config_name="test")
def hydra_run(
    cfg: DictConfig,
) -> None:
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(cfg.trainer.trainer_hyps.model_path, map_location='cpu')['model'], strict=True)
    loaders = build_loaders(cfg=cfg)

    device = f'cuda:{int(os.environ["LOCAL_RANK"])}'

    model = model.to(device=device)
    model = DDP(model, device_ids=[device])

    test_epoch(
        cfg=cfg,
        model=model,
        loaders=loaders,
        device=device,
    )

    destroy_process_group()


if __name__ == "__main__":
    hydra_run()
