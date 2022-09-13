import torch

class ModelConfig:
    """
    Model Configuration
    """

    def __init__(self):
        self.embedding_dim = 768
        self.hidden_dim = 256
        self.content_hidden_dim = 128
        self.style_hidden_dim = 8
        self.vocab_size = 50265
        self.num_style = 5
        self.dropout = 0.25
        self.epsilon = 1e-8
        self.max_seq_len = 512

class GeneralConfig:
    """
    General Configuration
    """

    def __init__(self):
        self.content_learning_rate = 2e-5
        self.style_learning_rate = 2e-5
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_path = '/home/hjjung/ailab/VAE/model/'
        self.avg_style_emb_path = '/home/hjjung/ailab/VAE/model/'