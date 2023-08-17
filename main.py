import json
import torch
import optuna
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from transformers import T5Tokenizer, T5Config
from torch.utils.data import DataLoader
from core.dataset import MyDataset
from core.model import Generator, Discriminator
from utils.wrapper import train_model, save_checkpoint
import sys
sys.path.append('/home/hjjung113/styletransfer') 

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Hyperparameters
epochs = 100
batch_size = 256
t5_model_name = 't5-base'
max_length = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data ###
train = pd.read_csv('data/pandora/train.csv')
val = pd.read_csv('data/pandora/val.csv')

train = train.dropna()
val = val.dropna()

tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
train_dataset = MyDataset(train, t5_model_name, max_length)
val_dataset = MyDataset(val, t5_model_name, max_length)

data_loaders = {}
data_loaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
data_loaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

best_params = None
best_model = None
best_score = float('inf')

def objective(trial):
    global best_params
    global best_model
    global best_score
    
    # Hyperparameters tuning
    gen_lr = trial.suggest_float("gen_lr", 1e-5, 1e-3, log=True)
    disc_lr = trial.suggest_float("disc_lr", 1e-5, 1e-3, log=True)
    gen_weight_decay = trial.suggest_float("gen_weight_decay", 1e-5, 1e-3, log=True)
    disc_weight_decay = trial.suggest_float("disc_weight_decay", 1e-5, 1e-3, log=True)

    # Loss functions
    # recon_criterion = torch.nn.CrossEntropyLoss()
    recon_criterion = torch.nn.BCEWithLogitsLoss()
    gen_criterion = torch.nn.BCEWithLogitsLoss()

    # Initialize model
    # Initialize models
    generator = Generator(t5_model_name=t5_model_name).to(device)
    discriminator = Discriminator(t5_model_name=t5_model_name).to(device)

    # Optimizer & Scheduler
    # Optimizers
    for name, param in generator.named_parameters():
        if 'weight' in name and len(param.data.shape) >= 2:
            torch.nn.init.kaiming_uniform_(param.data, nonlinearity='relu')

    # Generator 내의 T5 모델 파라미터 고정
    for param in generator.t5.parameters():
        param.requires_grad = False

    # Discriminator 내의 T5 모델 파라미터 고정 (만약 Discriminator에도 T5 모델이 있다면)
    for param in discriminator.t5.parameters():
        param.requires_grad = False

    gen_optimizer = AdamW(generator.parameters(), lr=gen_lr, weight_decay=gen_weight_decay)
    disc_optimizer = AdamW(discriminator.parameters(), lr=disc_lr, weight_decay=disc_weight_decay)
    # gen_scheduler = CosineAnnealingWarmRestarts(gen_optimizer, T_0=1, T_mult=2)
    # disc_scheduler = CosineAnnealingWarmRestarts(disc_optimizer, T_0=1, T_mult=2)    
    gen_scheduler = ReduceLROnPlateau(gen_optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    disc_scheduler = ReduceLROnPlateau(disc_optimizer, mode='min', factor=0.1, patience=5, verbose=True)


    # For mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    val_loss = train_model(generator, discriminator, data_loaders, (gen_optimizer, disc_optimizer), (recon_criterion,gen_criterion), device, (gen_scheduler, disc_scheduler),scaler, num_epochs=epochs)
    
    if val_loss < best_score:
        best_score = val_loss
        best_model = {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict()
        }
        best_params = {
            "gen_lr": gen_lr,
            "disc_lr": disc_lr,
            "gen_weight_decay": gen_weight_decay,
            "disc_weight_decay": disc_weight_decay
        }
        torch.save(best_model, "model/best_model.pth")
        with open("model/best_params.json", "w") as f:
            json.dump(best_params, f)
    
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1)