import os
from PIL import Image
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------




def load_predictions(folder_path):
    predictions = []
    pred_names = []
    for filename in sorted(os.listdir(folder_path)):
        pred_path = os.path.join(folder_path, filename)
        if pred_path.endswith('.npy'):
            pred = np.load(pred_path)  # Load dictionary
            predictions.append(pred)
            pred_names.append(filename)

    print(f"Loaded {len(predictions)} predictions from {folder_path}")
    return predictions, pred_names


def load_ground_truth(folder_path):
    ground_truths_raw = []
    ground_truths_meters = []
    gt_names = []
    for filename in sorted(os.listdir(folder_path)):
        gt_path = os.path.join(folder_path, filename)
        if gt_path.endswith('.png'):
            gt = Image.open(gt_path).convert("L")  # Load as grayscale
            gt_array = np.array(gt)
            ground_truths_raw.append(gt_array)
            gt_names.append(filename)

            # Convert to meters
            gt_meters = gt_array.astype(np.float32) / 256.0
            ground_truths_meters.append(gt_meters)

    print(f"Loaded {len(ground_truths_raw)} ground truths from {folder_path}")
    return ground_truths_meters, gt_names



def calc_rms_diff(pred, gt):
    # Create a color image to visualize differences in meters
    diff_image = np.zeros((*pred.shape, 3), dtype=np.uint8)
    
    if pred.shape != gt.shape:
        print(f"Warning: Prediction shape {pred.shape} does not match GT shape {gt.shape}.")

    # Create valid pixel mask: GT != 0 (valid in GT) and pred is valid
    valid_mask = (gt != 0) & (pred < 120) & (pred > 0) 

    # Calculate absolute difference in meters
    pred_masked = np.where(valid_mask, pred, 0)
    gt_masked = np.where(valid_mask, gt, 0)
    abs_diff = np.abs(pred_masked - gt_masked)

    print(f"Valid pixels: {valid_mask.sum()} / {valid_mask.size}")
    print(f"Mean absolute difference (valid pixels): {abs_diff.mean():.4f} meters")
    print(f"Max absolute difference (valid pixels): {abs_diff.max():.4f} meters")
    print(f"Min absolute difference (valid pixels): {abs_diff[valid_mask].min():.4f} meters")

    rmse = np.sqrt(np.mean(abs_diff[valid_mask] ** 2))


    # Normalize difference to 0-255 range form 0-80m and apply colormap
    diff_normalized = np.clip(abs_diff / 80.0, 0, 1)
    colormap = plt.cm.viridis(diff_normalized)
    diff_image = (colormap[..., :3] * 255).astype(np.uint8)
    diff_image[~valid_mask] = 0  # Set invalid pixels to black

    return rmse, diff_image



# ------------------------------------------------------------------

if __name__ == "__main__":
    predictions, pred_names = load_predictions('fileoutput/task1')
    print("Predictions loaded")

    ground_truths, gt_names = load_ground_truth('data_ass3/Task1_2/groundtruth')
    print("Ground truths loaded")

    for i, pred, pred_name, gt, gt_name in zip(range(len(predictions)), predictions, pred_names, ground_truths, gt_names):
        print(f"Processing prediction {i+1}/{len(predictions)}...")

        rmse, diff_image = calc_rms_diff(pred, gt)
        print(f"RMSE for {pred_name} vs {gt_name}: \n {rmse:.4f} meters")


        # Save difference image (testing only)
        os.makedirs('fileoutput/debug', exist_ok=True)
        diff_output_path = os.path.join('fileoutput/debug', f'{pred_name}_diff.png')
        Image.fromarray(diff_image).save(diff_output_path)
