import time
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from core.dataset import OCEANDataset, read_essay_split
from utils.config import ModelConfig, GeneralConfig
from utils.utils import count_parameters, epoch_time
from transformers import AutoModel, AutoTokenizer, AdamW

mconfig = ModelConfig()
gconfig = GeneralConfig()

tokenizer = AutoTokenizer.from_pretrained('roberta-base')

def train(model, iterator, content_optimizer, style_optimizer, style_adv_optimizer, adv_optimizer, fine_tune, transfer, best_model):
    
    ## trainer ##
    
    model.train()
    recon_losses = 0
    style_losses = 0
    s_e_losses = 0
    s_a_losses = 0
    for i, batch in enumerate(tqdm(iterator)):
        
        input_ids = batch['input_ids'].to(gconfig.device)
        labels = batch['labels'].to(gconfig.device)
        recon_loss, style_loss, s_e_loss, s_a_loss = model(input_ids, labels, fine_tune, transfer, best_model, len(iterator)-1 == i)


        target_tokenids = model.transfer_style(input_ids, labels)
        for i in range(len(target_tokenids)):
            target_tokens = tokenizer.convert_ids_to_tokens(target_tokenids[i])
            while '<pad>' in target_tokens:
                target_tokens.remove('<pad>')
        target_sentence = tokenizer.convert_tokens_to_string(target_tokens)
        print("Style transfered sentence: {}".format(target_sentence))

        content_loss = recon_loss
        style_loss = style_loss

        style_adv_optimizer.zero_grad()
        s_a_loss.backward(retain_graph=True)
        style_adv_optimizer.step()
        
        content_optimizer.zero_grad()
        content_loss.backward(retain_graph=True)
        content_optimizer.step()
        
        # if not transfer:
        style_optimizer.zero_grad()
        style_loss.backward(retain_graph=True)
        style_optimizer.step()

        adv_optimizer.zero_grad()
        s_e_loss.backward()
        adv_optimizer.step()

        recon_losses += recon_loss.item()
        style_losses += style_loss.item()
        s_e_losses += s_e_loss.item()
        s_a_losses += s_a_loss.item()
        del recon_loss, style_loss, s_e_loss, s_a_loss
        del batch

    recon_loss = recon_losses / len(iterator)
    style_loss = style_losses / len(iterator)
    s_e_loss = s_e_losses / len(iterator)
    s_a_loss = s_a_losses / len(iterator)

    return recon_loss, style_loss, s_e_loss, s_a_loss

def evaluate(model, iterator, fine_tune, transfer):
    
    ## evaluate ##
    
    model.eval()
    recon_losses = 0
    style_losses = 0
    s_e_losses = 0
    s_a_losses = 0
    with torch.no_grad():
        for batch in tqdm(iterator):
            input_ids = batch['input_ids'].to(gconfig.device)
            labels = batch['labels'].to(gconfig.device)
            recon_loss, style_loss, s_e_loss, s_a_loss = model(input_ids, labels, fine_tune, transfer, best_model=None, end_iter=None)
            recon_losses += recon_loss.item()
            style_losses += style_loss.item()
            s_e_losses += s_e_loss.item()
            s_a_losses += s_a_loss.item()
            del recon_loss, style_loss, s_e_loss, s_a_loss
            del batch
    recon_loss = recon_losses / len(iterator)
    style_loss = style_losses / len(iterator)
    s_e_loss = s_e_losses / len(iterator)
    s_a_loss = s_a_losses / len(iterator)

    return recon_loss, style_loss, s_e_loss, s_a_loss

def epoch_train(N_EPOCHS, best_valid_loss, model, train_iterator, valid_iterator, content_optimizer, style_adv_optimizer, style_optimizer, adv_optimizer, fine_tune, transfer):
    
    print(f'The model has {count_parameters(model):,} trainable parameters')

    valid_loss = 999999
    for epoch in range(N_EPOCHS):
        
        start_time = time.time()
        train_recon_loss, train_style_loss, train_s_e_loss, train_s_a_loss = train(model, train_iterator, content_optimizer, style_optimizer, style_adv_optimizer, adv_optimizer, fine_tune, transfer, valid_loss == best_valid_loss)
        valid_recon_loss, valid_style_loss, valid_s_e_loss, valid_s_a_loss = evaluate(model, valid_iterator, fine_tune, transfer)

        valid_loss = valid_recon_loss + valid_style_loss + valid_s_e_loss
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f"train: recon_loss:{train_recon_loss}, style_loss:{train_style_loss}, s_e_loss:{train_s_e_loss}, s_adv_loss:{train_s_a_loss}")
        print(f"valid: recon_loss:{valid_recon_loss}, style_loss:{valid_style_loss}, s_e_loss:{valid_s_e_loss}, s_adv_loss:{valid_s_a_loss}")

        del train_recon_loss, train_style_loss, train_s_a_loss, train_s_e_loss
        del valid_recon_loss, valid_style_loss, valid_s_a_loss, valid_s_e_loss

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            with open(gconfig.avg_style_emb_path + 'avg_style_emb.pickle', "wb") as f:
                pickle.dump(model.avg_style_emb, f)
            torch.save(model.state_dict(), gconfig.model_path + 'model.pt')
            print(f"----Epoch: {epoch+1} Model Saved!----")
    
    return valid_loss

def get_dataloader(df, batch_size, tokenizer, dataset):

    texts, labels = read_essay_split(df, dataset)

    oceandataset = OCEANDataset(texts, labels, tokenizer, dataset)

    loader = DataLoader(oceandataset, batch_size, shuffle=True)

    return loader