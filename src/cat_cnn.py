import os, json, cv2, torch
from torch.utils.data import Dataset
import torchvision.transforms as T

image_dir = "images"
label_dir = "labels"

class CatLandmarksDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=224, augment=False, label_files=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment

        if label_files is None:
            self.label_files = sorted(
                f for f in os.listdir(label_dir) if f.endswith(".json")
            )
        else:
            self.label_files = list(label_files)

        self.to_tensor = T.ToTensor()
        # to improve training images
        self.augmentation_transform = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ])
    
    def __len__(self):
        return len(self.label_files)
    
    def _find_matching_image(self, label_filename):
        """The json labels match the image labels so we can match them together using this function"""
        base_name = os.path.splitext(label_filename)[0]

        possible_extensions = [
            ".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"
        ]
        
        for ext in possible_extensions:
            image_path = os.path.join(self.image_dir, base_name + ext)
            if os.path.exists(image_path):
                return image_path
        return None
    
    def __getitem__(self,index):
        label_filename = self.label_files[index]
        label_path = os.path.join(self.label_dir,label_filename)
        
        image_path = self._find_matching_image(label_filename)

        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)

        height, width, _ = image_rgb.shape

        with open(label_path, "r") as f:
            landmarks = json.load(f)["labels"]
        
        landmarks = torch.tensor(landmarks, dtype=torch.float32)

        landmarks[:,0] /= width
        landmarks[:,1] /= height

        resized_image = cv2.resize(image_rgb, (self.img_size, self.img_size))
        image_tensor = self.to_tensor(resized_image)

        if self.augment:
            image_tensor = self.augmentation_transform(image_tensor)
            
        return image_tensor, landmarks.view(-1)
        