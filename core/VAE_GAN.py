# model.py
import torch
from torch import nn
from transformers import T5Model, T5ForConditionalGeneration, Transformer

class Encoder(nn.Module):
    def __init__(self, t5_model_name='t5-base', z_dim=512, freeze_t5=True):
        super(Encoder, self).__init__()
        self.t5 = T5Model.from_pretrained(t5_model_name)
        self.linear_mu = nn.Linear(self.t5.config.to_dict()['d_model'], z_dim)
        self.linear_var = nn.Linear(self.t5.config.to_dict()['d_model'], z_dim)
        if freeze_t5:
            for param in self.t5.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        t5_output = self.t5(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled_output = torch.mean(t5_output, dim=1)  # Mean pooling
        mu = self.linear_mu(pooled_output)
        log_var = self.linear_var(pooled_output)
        return mu, log_var
    
class Decoder(nn.Module):
    def __init__(self, t5_model_name):
        super().__init__()

        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)

    def forward(self, z):
        # Generate text using the T5 model.
        # For simplicity, we'll just use the T5's `generate` function.
        # Depending on the specific requirements, this could be replaced with a more customized generation loop.
        generated_text = self.t5.generate(z, do_sample=True, max_length=50)

        return generated_text

class Discriminator(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super(Discriminator, self).__init__()

        self.transformer = Transformer(hidden_size=hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        transformer_output = self.transformer(x)
        logits = self.classifier(transformer_output)
        
        return logits

class VAEGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = Discriminator()

    def forward(self, input_ids, attention_mask):
        mu, log_var = self.encoder(input_ids, attention_mask)
        z = self.reparameterize(mu, log_var)
        generated_data = self.decoder(z)
        real_or_fake = self.discriminator(generated_data)
        return generated_data, real_or_fake, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) 
        eps = torch.randn_like(std)
        return mu + eps*std
