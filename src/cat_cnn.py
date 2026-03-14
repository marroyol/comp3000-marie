import os, json, cv2, torch
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as T
from tools import find_matching_image
import torch.nn as nn
import torchvision.models as models

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_dir = os.path.join(base_dir, "data", "images")
label_dir = os.path.join(base_dir, "data", "labels")
split_seed = 2
batch_size=16
model_name="resnet18" # available options: resnet18

class CatLandmarksDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=224, augment=False, label_files=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment

        if label_files is None:
            self.label_files = sorted(
                filename for filename in os.listdir(label_dir) if filename.endswith(".json")
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

all_label_files = sorted(filename for filename in os.listdir(label_dir) if filename.endswith(".json"))

# later when less headachy try making training dataset w/ augment=True
full_dataset = CatLandmarksDataset(image_dir=image_dir,label_dir=label_dir,augment=False,label_files=all_label_files)
dataset_size = len(all_label_files)
train_indices, val_indices, test_indices = make_split_indices(dataset_size, seed=split_seed)

train_dataset = Subset(full_dataset,train_indices)
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_indices)

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_loader =DataLoader(val_dataset,batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)

def get_model(model_name, num_landmarks, pretrained=True):
    output_features = num_landmarks * 2
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc=nn.Linear(model.fc.in_features,output_features)
        return model
    
    raise ValueError(f"{model_name} is not yet implemented!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(model_name, num_landmarks, pretrained=True).to(device)