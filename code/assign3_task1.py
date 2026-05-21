import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt

from unidepth.models import UniDepthV1



TESTING = True  # Set to True to only process one image for testing



# ------------------------------------------------------------------


print(torch.__version__)
print(torch.cuda.is_available())  # should be False
model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
device = torch.device("cpu")
model = model.to(device).eval()



# ------------------------------------------------------------------




def load_data(folder_path):
    images = []
    image_names = []


    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert("RGB")
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1) # C, H, W
        images.append(img_tensor)
        image_names.append(filename)

    print(f"Loaded {len(images)} images from {folder_path}")
    return images, image_names

# ------------------------------------------------------------------

if __name__ == "__main__":
    images, image_names = load_data('data_ass3/Task1_2/images')
    print("Images loaded")

    predictions = []
    for i, image in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}...")
        pred = model.infer(image)
        predictions.append(pred)

        if TESTING:
            break  # only process one image for testing
    

    os.makedirs('fileoutput/task1', exist_ok=True)
    for i, pred in enumerate(predictions):
        # Extract depth map from prediction dictionary
        depth_map = pred['depth'].squeeze().cpu().numpy()
        xyz = pred['points'].squeeze().cpu().numpy()
        #intrinsics = pred['intrinsics'].squeeze().cpu().numpy()
        
        # Save raw depth data as numpy file
        raw_output_path = os.path.join('fileoutput/task1', f'{image_names[i]}_raw.npy')
        np.save(raw_output_path, depth_map)
        
        # Normalize to 0-255 range
        # Normalize to 0-1 range
        # Normalize depth map with percentile clipping for better visibility
        p_min, p_max = np.percentile(depth_map, (2, 98))  # Clip outliers
        pred_normalized = np.clip((depth_map - p_min) / (p_max - p_min + 1e-8), 0, 1)
        # Apply colormap (magma)
        #pred_normalized = 255 - (pred_normalized*255).astype(np.uint8)  # Invert for better visualization
        depth_colored = plt.cm.viridis(pred_normalized)[:, :, :3]
        depth_colored = (depth_colored * 255).astype(np.uint8)
        output_path = os.path.join('fileoutput/task1', f'{image_names[i]}.png')
        Image.fromarray(depth_colored).save(output_path)
    
    print(f"Saved {len(predictions)} predictions to fileoutput/task1")