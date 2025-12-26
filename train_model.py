import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os

import copy
import json
import argparse

# Configuration
# DATA_DIR will be set via arguments
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 64  # Specific for "64 Zoo Species" task

def train_model(data_dir):
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found.")
        print("Please provide a valid dataset directory using --data_dir.")
        return

    # Data Augmentation and Normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Loading Data...")
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    print(f"Detected {len(class_names)} classes.")
    
    # Save classes.json
    with open('classes.json', 'w') as f:
        json.dump(class_names, f)
    print("Saved classes.json")

    # Load Model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Modify Classifier Head
    # MobileNetV2 classifier is a Sequential block, the last layer is [1]
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Scheduler
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Training Loop
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch}/{EPOCHS - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                exp_lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"New best model found! Acc: {best_acc:.4f}")

    print(f'Training complete. Best val Acc: {best_acc:4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'zoolens_64_species_best.pth')
    print("Best model saved as 'zoolens_64_species_best.pth'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ZooLens Model")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Path to dataset directory")
    args = parser.parse_args()
    
    train_model(args.data_dir)
