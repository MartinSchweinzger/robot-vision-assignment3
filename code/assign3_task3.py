import subprocess
import os
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

from unidepth.models import UniDepthV1

#------------------------------------------------------------------

RUN_UNIMATCH = False

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

#------------------------------------------------------------------

def load_data(folder_path):
    images = []
    image_names = []

    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert("RGB")
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1) # C, H, W
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

    os.makedirs('fileoutput/task3/disparity_unidepth', exist_ok=True)
    for i, pred in enumerate(predictions):
        # Extract depth map from prediction dictionary
        depth_map = pred['depth'].squeeze().cpu().numpy()
        xyz = pred['points'].squeeze().cpu().numpy()
        #intrinsics = pred['intrinsics'].squeeze().cpu().numpy()
        
        # Save raw depth data as numpy file
        raw_output_path = os.path.join('fileoutput/task3/disparity_unidepth', f'{image_names[i]}_raw.npy')
        np.save(raw_output_path, depth_map)
        
        # Normalize to 0-255 range
        # Normalize to 0-1 range
        # Normalize depth map with percentile clipping for better visibility
        p_min, p_max = np.percentile(depth_map, (2, 98))  # Clip outliers
        pred_normalized = np.clip((depth_map - p_min) / (p_max - p_min + 1e-8), 0, 1)
        # Apply colormap (magma)
        pred_normalized = 255 - (pred_normalized*255).astype(np.uint8)  # Invert for better visualization
        depth_colored = plt.cm.magma(pred_normalized)[:, :, :3]
        depth_colored = (depth_colored * 255).astype(np.uint8)
        output_path = os.path.join('fileoutput/task3/disparity_unidepth', f'{image_names[i]}.png')
        Image.fromarray(depth_colored).save(output_path)





#------------------------------------------------------------------

if __name__ == "__main__":
    if RUN_UNIMATCH:
        print("Running Unimatch...")
        run_unimatch()


    print("Running UniDepth...")
    run_unidepth()


    

