import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def binary_accuracy(preds, y):
    
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    # print(correct)
    
    acc = torch.mean(correct, dim=0)
    # print(acc)
    return acc

def pandora_label(labels):
    rounded_label = []
    for label in labels:
        b = []
        for sentiment in label:
            if sentiment > 0.5:
                b.append(1.0)
            else:
                b.append(0)
        rounded_label.append(b)

    return rounded_label
    
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs