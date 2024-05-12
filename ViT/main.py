import numpy as np
import random
from tqdm import tqdm, trange
import glob
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import logging

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

from model import MyViT
from dataset import CustomDataset
logging.basicConfig(filename="training_log.txt", level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger(__name__)
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

def main():
    # Loading data

    path_dataset = "data/foundation_images"
    path_labels = "data/updated_stage_labels.csv"
    df = pd.read_csv(path_labels)
    classes = df['labels'].unique().astype(str)

    
    sequences = [] 
    for camera_folder in glob.glob(os.path.join(path_dataset, "cam_*")):
        for i in range(10):
            files = sorted(glob.glob(os.path.join(camera_folder, '*.jpg_'+str(i)+'.png')))
            sequences.extend(files)
    train_sequences, remaining_sequences = train_test_split(sequences, test_size=0.2, random_state=0)
    val_sequences, test_sequences = train_test_split(remaining_sequences, test_size=0.5, random_state=0)
    # remove images without label
    train_sequences = [train for train in train_sequences if not df.loc[df['images'] == os.path.basename(train), 'labels'].empty]
    val_sequences = [val for val in val_sequences if not df.loc[df['images'] == os.path.basename(val), 'labels'].empty]
    test_sequences = [test for test in test_sequences if not df.loc[df['images'] == os.path.basename(test), 'labels'].empty]

    train_set = CustomDataset(images_path = train_sequences, classes = classes, data_frame=df)
    val_set = CustomDataset(images_path = val_sequences, classes = classes,data_frame=df)
    test_set = CustomDataset(images_path = test_sequences, classes = classes,data_frame=df) 

    # transform = ToTensor()
    # train_set = MNIST(root='./../datasets', train=True, download=True, transform=transform)
    # test_set = MNIST(root='./../datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=8)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=8)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    # model = MyViT((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
    model = MyViT((3, 256, 256), n_patches=16, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
    N_EPOCHS = 20
    LR = 0.005

    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.squeeze(dim=-1).long().to(device)
            y_hat = model(x)
            # print("y_hat.shape, y.shape", y_hat.shape, y.shape)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)
            log.info(f"training loss of epoch {epoch+1} is {train_loss}")


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.squeeze().long().to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")

if __name__=="__main__":
    main()