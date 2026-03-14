import os, json, cv2, torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as T
from tools import find_matching_image

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_dir = os.path.join(base_dir, "data", "images")
label_dir = os.path.join(base_dir, "data", "labels")
split_seed = 2

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
    
    def __getitem__(self,index):
        label_filename = self.label_files[index]
        label_path = os.path.join(self.label_dir,label_filename)
        
        image_path = find_matching_image(label_filename, self.image_dir)

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

sample_label_file = [filename for filename in os.listdir(label_dir) if filename.endswith(".json")][0]

sample_label_path = os.path.join(label_dir, sample_label_file)

with open(sample_label_path, "r") as f:
    sample_labels = json.load(f)["labels"]

num_landmarks = len(sample_labels)

def make_split_indices(dataset_size,train_ratio=0.7,val_ratio=0.15,test_ratio=0.15,seed=2):
    generator = torch.Generator().manual_seed(seed)
    all_indices = torch.randperm(dataset_size, generator=generator).tolist()

    train_end=int(train_ratio * dataset_size)
    val_end = train_end + int(val_ratio * dataset_size)

    train_indices = all_indices[:train_end]
    val_indices = all_indices[train_end:val_end]
    test_indices = all_indices[val_end:]

    return train_indices, val_indices, test_indices