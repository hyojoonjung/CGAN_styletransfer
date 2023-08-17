import torch

from torch.utils.data import Dataset, DataLoader
from core.dataset import MyDataset

def create_data_loader(texts, labels, tokenizer, max_len, batch_size):
    ds = MyDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

def check_parameters_in_graph(loss, model):
    graph_parameters = set([param for param in loss.grad_fn.next_functions if param[0].__class__.__name__ == 'AccumulateGrad'])
    model_parameters = set(model.parameters())
    
    missing_parameters = model_parameters - graph_parameters
    
    for param in missing_parameters:
        print("Parameter not used in loss computation:", id(param))
    
    if len(missing_parameters) == 0:
        print("All model parameters are used in the loss computation.")
    else:
        print("Some parameters are not used in the loss computation.")


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.stop = False

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            self.counter = 0