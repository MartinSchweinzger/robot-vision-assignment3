import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T

from unidepth.models import UniDepthV1


# ------------------------------------------------------------------


print(torch.__version__)
print(torch.cuda.is_available())  # should be False
model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
device = torch.device("cpu")
model = model.to(device).eval()



# ------------------------------------------------------------------




def load_data(folder_path):
    images = []

    transform = T.Compose([
        T.Resize((518, 518)),  # adjust to model
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ])

    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        images.append(img_tensor)

    print(f"Loaded {len(images)} images from {folder_path}")
    return images













# ------------------------------------------------------------------

if __name__ == "__main__":
    images = load_data('data_ass3/Task1_2/images')
    print("Images loaded")

    predictions = []
    for image in images:
        print("Processing image...")
        pred = model.infer(image)
        predictions.append(pred)
        break  # Remove this line to process all images
    
    os.makedirs('fileoutput/task1', exist_ok=True)
    for i, pred in enumerate(predictions):
        # Extract depth map from prediction dictionary
        depth_map = pred['depth'].squeeze().cpu().numpy()
        # Normalize to 0-255 range
        pred_normalized = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        output_path = os.path.join('fileoutput/task1', f'prediction_{i:04d}.png')
        Image.fromarray(pred_normalized).save(output_path)
    
    print(f"Saved {len(predictions)} predictions to fileoutput/task1")