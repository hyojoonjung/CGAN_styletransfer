from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn as nn
import torch

import torch.nn as nn
from transformers import T5ForConditionalGeneration

class Generator(nn.Module):
    def __init__(self, t5_model_name='t5-base', noise_dim=256, label_dim=5):
        super(Generator, self).__init__()

        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 512),
            nn.BatchNorm1d(512),  # Add BatchNorm after the Linear layer
            nn.Tanh()
        )
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)

    def forward(self, noise, labels):
        input_tensor = self.noise_proj(torch.cat([noise, labels], dim=1))
        input_tensor = abs(input_tensor).long()
        generated_text = self.t5.generate(input_tensor, do_sample=True, max_length=50)
        return generated_text

class Discriminator(nn.Module):
    def __init__(self, t5_model_name='t5-base', label_dim=5):
        super(Discriminator, self).__init__()

        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.embedding_projector = nn.Sequential(
            nn.Linear(self.t5.config.d_model, self.t5.config.d_model),
            nn.BatchNorm1d(self.t5.config.d_model)  # Add BatchNorm after the Linear layer
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.t5.config.d_model + label_dim, 256),
            nn.BatchNorm1d(256),  # Add BatchNorm after the Linear layer
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, text, labels):
        output = self.t5.encoder(text)
        embeddings = self.embedding_projector(output.last_hidden_state.mean(dim=1))
        combined = torch.cat([embeddings, labels], dim=1)
        return self.classifier(combined)