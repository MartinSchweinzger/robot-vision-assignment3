import subprocess
import os
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import re
from unidepth.models import UniDepthV1

#------------------------------------------------------------------
RUN_UNIMATCH = False
RUN_UNIDEPTH = False

# From calibration file
FOCAL_LENGTH = 721 #px
BASE_LENGTH = 0.54 #m

DIFF_IMG_MAX_ERROR = 40



#------------------------------------------------------------------
def disparity_to_depth(disparity_image, focal_length=FOCAL_LENGTH, base_length=BASE_LENGTH):
    depth_map = np.zeros_like(disparity_image, dtype=np.float32)

    for i in range(disparity_image.shape[0]):
        for j in range(disparity_image.shape[1]):
            disparity = disparity_image[i, j]
            if disparity > 0:  # Avoid division by zero
                depth_map[i, j] = (focal_length * base_length) / disparity
            else:
                depth_map[i, j] = 0  # indicate invalid depth

    return depth_map




#------------------------------------------------------------------
def run_unimatch():
    cmd = [
        "python3", "unimatch/main_stereo.py",
        "--checkpoint_dir", "/tmp",
        "--inference_dir_left", "data_ass3/Task3/rectified_images/image_2",
        "--inference_dir_right", "data_ass3/Task3/rectified_images/image_3",
        "--inference_size", "384", "1248",
        "--output_path", "fileoutput/task3/disparity_unimatch",
        "--num_scales", "2",
        "--reg_refine",
        "--num_reg_refine", "3",
        "--upsample_factor", "4",
        "--attn_type", "self_swin2d_cross_swin1d",
        "--attn_splits_list", "2", "8",
        "--corr_radius_list", "-1", "4",
        "--prop_radius_list", "-1", "1",
        "--resume", "pretrained/gmstereo-scale2-regrefine3-resumeflowthings-kitti15-04487ebf.pth",
        "--save_pfm_disp",
    ]

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print("Unimatch failed")
        exit()
    else:
        print("Unimatch completed successfully")

    return result


def _load_pfm(file):
    header = file.readline().decode('ascii').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise ValueError('Not a PFM file.')

    dims = file.readline().decode('ascii').rstrip()
    while dims.startswith('#'):
        dims = file.readline().decode('ascii').rstrip()
        
    width, height = map(int, dims.split())
    scale = float(file.readline().rstrip())

    endian = '<' if scale < 0 else '>'
    scale = abs(scale)

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = data.reshape(shape)
    data = np.flipud(data)  # PFM files are stored in bottom to top

    return data
    



def load_unimatch_results(folder_path):
    disparity_maps = []
    disparity_names = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.pfm'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as f:
                disparity = _load_pfm(f)
            disparity_maps.append(disparity)
            disparity_names.append(filename.split('.')[0])

    # Convert disparity to depth maps
    depth_maps = []
    for disparity in disparity_maps:
        depth_map = disparity_to_depth(disparity)
        depth_maps.append(depth_map)

    print(f"Loaded {len(depth_maps)} disparity maps from {folder_path}")
    return depth_maps, disparity_names


#------------------------------------------------------------------
def load_data(folder_path):
    images = []
    image_names = []

    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert("RGB")
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)
        images.append(img_tensor)
        image_names.append(filename.split('.')[0])

    print(f"Loaded {len(images)} images from {folder_path}")
    return images, image_names

def run_unidepth():
    model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
    device = torch.device("cpu")
    model = model.to(device).eval()

    images, image_names = load_data('data_ass3/Task3/rectified_images/image_2')
    print("Images loaded")

    predictions = []
    for i, image in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}...")
        pred = model.infer(image)
        predictions.append(pred)

    depth_maps = []
    os.makedirs('fileoutput/task3/disparity_unidepth', exist_ok=True)
    for i, pred in enumerate(predictions):
        depth_map = pred['depth'].squeeze().cpu().numpy()
        depth_maps.append(depth_map)
        xyz = pred['points'].squeeze().cpu().numpy()
        
        raw_output_path = os.path.join('fileoutput/task3/disparity_unidepth', f'{image_names[i]}_raw.npy')
        np.save(raw_output_path, depth_map)
        

        # Normalize depth map with percentile clipping for better visibility
        p_min, p_max = np.percentile(depth_map, (2, 98))  # Clip outliers
        pred_normalized = np.clip((depth_map - p_min) / (p_max - p_min + 1e-8), 0, 1)
        # Apply colormap (magma)
        pred_normalized = 255 - (pred_normalized*255).astype(np.uint8)  # Invert for better visualization
        depth_colored = plt.cm.magma(pred_normalized)[:, :, :3]
        depth_colored = (depth_colored * 255).astype(np.uint8)
        output_path = os.path.join('fileoutput/task3/disparity_unidepth', f'{image_names[i]}.png')
        Image.fromarray(depth_colored).save(output_path)

    return depth_maps, image_names

def load_unidepth_results(folder_path):
    depth_maps = []
    depth_names = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            depth_map = np.load(file_path)
            depth_maps.append(depth_map)
            depth_names.append(filename.split('.')[0])

    print(f"Loaded {len(depth_maps)} depth maps from {folder_path}")
    return depth_maps, depth_names


#------------------------------------------------------------------
def load_ground_truth(folder_path):
    ground_truths_raw = []
    ground_truths_meters = []
    gt_names = []
    for filename in sorted(os.listdir(folder_path)):
        gt_path = os.path.join(folder_path, filename)
        if gt_path.endswith('.png'):
            gt = Image.open(gt_path)  # Load as grayscale
            gt_array = np.array(gt)
            ground_truths_raw.append(gt_array)
            gt_names.append(filename)

            # Convert to meters
            gt_meters = gt_array.astype(np.float32) / 256.0
            ground_truths_meters.append(gt_meters)

    # Convert disparity to depth maps
    depth_maps = []
    for disparity in ground_truths_meters:
        depth_map = disparity_to_depth(disparity)
        depth_maps.append(depth_map)

    print(f"Loaded {len(depth_maps)} disparity maps from {folder_path}")
    return depth_maps, gt_names



#------------------------------------------------------------------
def calc_rms_diff(pred, gt):
    # Create a color image to visualize differences in meters
    diff_image = np.zeros((*pred.shape, 3), dtype=np.uint8)
    
    if pred.shape != gt.shape:
        print(f"Warning: Prediction shape {pred.shape} does not match GT shape {gt.shape}.")

    # Create valid pixel mask: GT != 0 (valid in GT) and pred is valid
    valid_mask = (gt != 0) & (pred < 120)

    # Calculate absolute difference in meters
    pred_masked = np.where(valid_mask, pred, 0)
    gt_masked = np.where(valid_mask, gt, 0)
    abs_diff = np.abs(pred_masked - gt_masked)

    print(f"Valid pixels: {valid_mask.sum()} / {valid_mask.size}")
    print(f"Mean absolute difference (valid pixels): {abs_diff.mean():.4f} meters")
    print(f"Max absolute difference (valid pixels): {abs_diff.max():.4f} meters")
    print(f"Min absolute difference (valid pixels): {abs_diff[valid_mask].min():.4f} meters")

    rmse = np.sqrt(np.mean(abs_diff[valid_mask] ** 2))

    # Normalize difference and apply colormap
    diff_normalized = np.clip(abs_diff / DIFF_IMG_MAX_ERROR, 0, 1)
    colormap = plt.cm.viridis(diff_normalized)
    diff_image = (colormap[..., :3] * 255).astype(np.uint8)
    diff_image[~valid_mask] = 0  

    return rmse, diff_image




#------------------------------------------------------------------
if __name__ == "__main__":
    if RUN_UNIMATCH:
        print("Running Unimatch...")
        run_unimatch()

    unimatch_results, unimatch_names = load_unimatch_results('fileoutput/task3/disparity_unimatch')
    if unimatch_results is not None:
        print("Unimatch results loaded")
    else:
        print("Failed to load Unimatch results")
        exit()


    print("Running UniDepth...")
    unidepth_results = None
    if RUN_UNIDEPTH:
        unidepth_results, unidepth_names = run_unidepth()
    else:
        unidepth_results, unidepth_names = load_unidepth_results('fileoutput/task3/disparity_unidepth')

    if unidepth_results is not None:
        print("UniDepth results loaded")
    else:
        print("Failed to load UniDepth results")
        exit()


    print("Load Ground Truth...")
    ground_truth, gt_names = load_ground_truth('data_ass3/Task3/GT_disparities/disp_noc_0')
    if ground_truth is not None:
        print("Ground truth loaded")
    else:
        print("Failed to load ground truth")
        exit()


    print(unimatch_names)
    print(unidepth_names)
    print(gt_names)
    print()
    print("Calculating RMSE and difference images...")
    for i, (pred_unimatch, pred_unidepth, gt) in enumerate(zip(unimatch_results, unidepth_results, ground_truth)):
        print(f"\nProcessing result {i+1}/{len(unimatch_results)}...")

        rmse_unimatch, diff_image_unimatch = calc_rms_diff(pred_unimatch, gt)
        rmse_unidepth, diff_image_unidepth = calc_rms_diff(pred_unidepth, gt)

        # Save difference images
        diff_output_path_unimatch = os.path.join('fileoutput/task3/diff_images', f'{unimatch_names[i].replace("_disp", "")}_unimatch_diff.png')
        diff_output_path_unidepth = os.path.join('fileoutput/task3/diff_images', f'{unidepth_names[i].replace("_raw", "")}_unidepth_diff.png')
        os.makedirs('fileoutput/task3/diff_images', exist_ok=True)
        Image.fromarray(diff_image_unimatch).save(diff_output_path_unimatch)
        Image.fromarray(diff_image_unidepth).save(diff_output_path_unidepth)

        print(f"Unimatch RMSE: {rmse_unimatch:.4f} meters")
        print(f"UniDepth RMSE: {rmse_unidepth:.4f} meters")












    



    

