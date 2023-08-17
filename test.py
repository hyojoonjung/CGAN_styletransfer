import torch
import pandas as pd
from torch.optim import AdamW
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
from core.dataset import MyDataset
from core.model import Generator, Discriminator
from utils.wrapper import test_step
import sys
sys.path.append('/home/hjjung113/styletransfer') 

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Hyperparameters
batch_size = 256
t5_model_name = 't5-base'
max_length = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data ###
test = pd.read_csv('data/pandora/test.csv')
val = pd.read_csv('data/pandora/val.csv')

test_data = test.dropna()
val = val.dropna()

# Loss functions
recon_criterion = torch.nn.BCEWithLogitsLoss()
gen_criterion = torch.nn.BCEWithLogitsLoss()

# Initialize models
generator = Generator(t5_model_name=t5_model_name).to(device)
discriminator = Discriminator(t5_model_name=t5_model_name).to(device)

checkpoint = torch.load("model/best_model.pth")
generator.load_state_dict(checkpoint["generator_state_dict"])

tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
test_dataset = MyDataset(test_data, t5_model_name, max_length)  # Assuming test_data is already defined and preprocessed.
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

test_gen_loss, test_disc_loss = test_step(generator, discriminator, test_loader, (recon_criterion, gen_criterion), device)

print("Test Generator Loss:", test_gen_loss)
print("Test Discriminator Loss:", test_disc_loss)