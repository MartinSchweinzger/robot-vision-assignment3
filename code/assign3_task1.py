import os
from PIL import Image
import numpy as np

from unidepth.models import UniDepthV1


# ------------------------------------------------------------------


model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14") 




# ------------------------------------------------------------------


def load_data(folder_path):        
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            images.append(np.array(img))
    
    print(f"Loaded {len(images)} images from {folder_path}")
    return np.array(images)













# ------------------------------------------------------------------

if __name__ == "__main__":
    images = load_data('data_ass3/Task1_2/images')
