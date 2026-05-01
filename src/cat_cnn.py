import os, json, cv2, torch
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as T
from src.tools import find_matching_image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import copy

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_dir = os.path.join(base_dir, "data", "images")
label_dir = os.path.join(base_dir, "data", "labels")
split_seed = 2
batch_size=16
model_name="resnet18" # available options: efficientnet_b0, resnet18, resnet50, mobilenet_v3_small
run_training = False
image_path = os.path.join(image_dir,"paz3.png")
model_path = os.path.join(base_dir, f"cat_model_{model_name}.pt")

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

        with open(label_path, "r") as f:
            annotation_data = json.load(f)

        
        landmarks = torch.tensor(annotation_data["labels"], dtype=torch.float32)

        x_min, y_min, x_max, y_max = annotation_data["bounding_boxes"]

        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)

        image_rgb = image_rgb[y_min:y_max, x_min:x_max]

        height, width, _ = image_rgb.shape

        landmarks[:,0] -= x_min
        landmarks[:,1] -= y_min

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

dataset_size = len(all_label_files)
train_indices, val_indices, test_indices = make_split_indices(dataset_size, seed=split_seed)
train_all_dataset = CatLandmarksDataset(image_dir, label_dir, augment=True,label_files=all_label_files)
eval_full_dataset = CatLandmarksDataset(image_dir,label_dir,augment=False,label_files=all_label_files)

train_dataset = Subset(train_all_dataset, train_indices)
val_dataset= Subset(eval_full_dataset, val_indices)
test_dataset = Subset(eval_full_dataset, test_indices)

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
    
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, output_features)
        return model

    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features,output_features)
        return model
    
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, output_features)
        return model
    
    raise ValueError(f"{model_name} is not yet implemented!")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print(f"CUDA is working! GPU being used: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available so CPU is being used, please check")

model = get_model(model_name, num_landmarks, pretrained=True).to(device)

def train_model(model, train_loader,val_loader, epochs=100):
    criterion = nn.SmoothL1Loss()
    optimiser = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.1, patience=5)
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0
    early_stopping_patience = 10
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
        "best_val_loss": None,
        "best_epoch": None
    }
    if device.type == "cuda":
        print(f"Training with CUDA. Epochs = {epochs}")

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimiser.zero_grad()

            predictions = model(images)
            loss = criterion(predictions, targets)

            loss.backward()

            optimiser.step()

            running_train_loss += loss.item() * images.size(0)

        average_train_loss = running_train_loss / len(train_loader.dataset)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                predictions = model(images)
                loss = criterion(predictions, targets)
                running_val_loss += loss.item() * images.size(0)

        average_val_loss = running_val_loss / len(val_loader.dataset)
        current_lr = optimiser.param_groups[0]["lr"]
        scheduler.step(average_val_loss)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(average_train_loss)
        history["val_loss"].append(average_val_loss)
        history["learning_rate"].append(current_lr)

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            history["best_val_loss"] = average_val_loss
            history["best_epoch"] = epoch + 1
            print("Saved the new best model state!")
        else:
            epochs_without_improvement += 1
        print(f"epoch: {epoch + 1}/{epochs}\ntrain loss: {average_train_loss:.6f}\nval loss: {average_val_loss:.6f}\nlearning rate: {current_lr:.8f}\nepochs without improvement: {epochs_without_improvement}")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n****\nEarly stop triggered! Epoch {epoch +1 }")
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history

def evaluate_model(model, test_loader):
    criterion = nn.SmoothL1Loss()
    model.eval()
    running_test_loss = 0.0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)
            loss = criterion(predictions, targets)

            running_test_loss += loss.item() * images.size(0)

        average_test_loss = running_test_loss/len(test_loader.dataset)
        print(f"test loss = {average_test_loss:.8f}")
        return average_test_loss

def predict(model, image_path, bounding_box, image_size=224):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError("error loading image into predict()")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    x_min, y_min, x_max, y_max = bounding_box
    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

    cropped_image = image_rgb[y_min:y_max, x_min:x_max]
    if cropped_image.size == 0:
        raise ValueError("bounding box generated an empty image crop")
    crop_height, crop_width, _ = cropped_image.shape

    resized_image = cv2.resize(cropped_image, (image_size, image_size))
    image_tensor = T.ToTensor()(resized_image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        predicted_landmarks = model(image_tensor).cpu().view(-1, 2)
    

    predicted_landmarks_px = predicted_landmarks.clone()
    predicted_landmarks_px[:,0] *= crop_width
    predicted_landmarks_px[:,1] *= crop_height

    predicted_landmarks_px[:,0] += x_min
    predicted_landmarks_px[:,1] += y_min
    
    return image_rgb, predicted_landmarks_px

if __name__ == "__main__":
    from src.nme import evaluate_nme, print_nme_report

    print(f"You are running model {model_name}\nSplit sizes\ntrain: {len(train_dataset)}\nval:{len(val_dataset)},test: {len(test_dataset)}")
    print(f"Split sizes train: {len(train_dataset)}   \n"
          f"val: {len(val_dataset)} \n test: {len(test_dataset)}")
    if run_training:
        model, history = train_model(model, train_loader, val_loader, epochs=50)
        torch.save(model.state_dict(), model_path)

        history_path = os.path.join(base_dir, f"training_history_{model_name}.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

    test_loss = evaluate_model(model, test_loader)
    
    test_label_files = [all_label_files[i] for i in test_indices]
    nme_results = evaluate_nme(
        model,
        label_files=test_label_files,
        image_dir=image_dir,
        label_dir=label_dir,
        device=device
    )
    print_nme_report(nme_results, model_name=model_name)