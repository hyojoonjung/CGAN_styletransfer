import torch
import random
import pickle
import argparse
import pandas as pd
import numpy as np
import torch.optim
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AdamW
from core.model import MultilabelStyleTransferVAE
from utils.config import ModelConfig, GeneralConfig 
from utils.wrapper import epoch_train, evaluate, get_dataloader

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
 
mconfig = ModelConfig()
gconfig = GeneralConfig()

def main(args):

    ##SET SEED##
    SEED = 1123
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ## Parameters ##
    args.fine_tune_EPOCHS = 4
    args.train_EPOCHS = 20
    args.fine_tune_batch_size = 8
    args.train_batch_size = args.fine_tune_batch_size * 4

    transformer_name = 'roberta-base'
    dataset = 'essay'

    ## Transformer ##
    transformer = AutoModel.from_pretrained(transformer_name)
    tokenizer = AutoTokenizer.from_pretrained(transformer_name)

    ## Model ##
    model = MultilabelStyleTransferVAE(transformer)
    model = model.to(gconfig.device)

    ## Get Model parameters ##
    recon_loss_params, style_loss_params, style_adv_params, adv_loss_params = model.get_params()

    ## Optimizer ##
    content_optimizer = AdamW(recon_loss_params, lr=gconfig.content_learning_rate)
    style_optimizer = AdamW(style_loss_params, lr=gconfig.style_learning_rate)
    style_adv_optimizer = AdamW(style_adv_params, lr=gconfig.style_learning_rate)
    adv_optimizer = AdamW(adv_loss_params, lr=gconfig.content_learning_rate)

    ## Learning rate scheduler ##
    # scheduler = lr_scheduler.ExponentialLR(content_optimizer, gamma=0.95)

    if args.MODE == 'fine_tune':
        train_df = pd.read_csv('data/essay/train.csv', index_col=False)
        valid_df = pd.read_csv('data/essay/valid.csv', index_col=False)

        train_loader = get_dataloader(train_df, args.fine_tune_batch_size, tokenizer, dataset)
        valid_loader = get_dataloader(valid_df, args.fine_tune_batch_size, tokenizer, dataset)

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        
        best_valid_loss = float("inf")

        print("----Fine Tune Start!----")
        best_valid_loss = epoch_train(args.fine_tune_EPOCHS, best_valid_loss, model, train_loader, valid_loader, content_optimizer, style_optimizer, style_adv_optimizer, adv_optimizer, fine_tune=True, transfer=False)
        print("----Fine Tune Completed!----")

        ## Parameters Locked ##
        for name, param in model.named_parameters():
            if name.startswith('transformers'):
                param.requires_grad = False

        train_loader = get_dataloader(train_df, args.train_batch_size, tokenizer, dataset)
        valid_loader = get_dataloader(valid_df, args.train_batch_size, tokenizer, dataset)

        print("----recon train Start!----")
        best_valid_loss = epoch_train(args.train_EPOCHS, best_valid_loss, model, train_loader, valid_loader, content_optimizer, style_optimizer, style_adv_optimizer, adv_optimizer, fine_tune=False, transfer=False)
        print("----recon train Completed!----")
        print("best_valid_loss : {}".format(best_valid_loss))
    elif args.MODE == 'transfer':
        ## Parameters Locked ##
        for name, param in model.named_parameters():
            if name.startswith('transformers'):
                param.requires_grad = False

        ## Model & avg_style_emb Load ##
        model.load_state_dict(torch.load(gconfig.model_path + 'model.pt'))
        with open(gconfig.avg_style_emb_path+ 'avg_style_emb.pickle', 'rb') as f:
            avg_style_embeddings = pickle.load(f)
        model.avg_style_emb = avg_style_embeddings

        train_df = pd.read_csv('data/essay/train.csv', index_col=False)
        valid_df = pd.read_csv('data/essay/valid.csv', index_col=False)

        train_loader = get_dataloader(train_df, args.train_batch_size, tokenizer, dataset)
        valid_loader = get_dataloader(valid_df, args.train_batch_size, tokenizer, dataset)

        best_valid_loss = 4.198076837535562
        print("----cycle train Start!----")
        best_valid_loss = epoch_train(args.train_EPOCHS, best_valid_loss, model, train_loader, valid_loader, content_optimizer, style_optimizer, style_adv_optimizer, adv_optimizer, fine_tune=False, transfer=True)
        print("----cycle train Completed!----")

    elif args.MODE == 'test':
        model.load_state_dict(torch.load(gconfig.model_path + 'model.pt'))
        with open(gconfig.avg_style_emb_path+ 'avg_style_emb.pickle', 'rb') as f:
            avg_style_embeddings = pickle.load(f)
        model.avg_style_emb = avg_style_embeddings

        test_df = pd.read_csv('data/essay/test.csv', index_col=False)
        test_loader = get_dataloader(test_df, args.batch_size, tokenizer, dataset)

        evaluate(model, test_loader, fine_tune=False, transfer=True)
    
    elif args.MODE == 'recon':
        model.load_state_dict(torch.load(gconfig.model_path + 'model.pt'))
        with open(gconfig.avg_style_emb_path+ 'avg_style_emb.pickle', 'rb') as f:
            avg_style_embeddings = pickle.load(f)
        model.avg_style_emb = avg_style_embeddings

        train_df = pd.read_csv('data/essay/train.csv', index_col=False)
        train_loader = get_dataloader(train_df, args.train_batch_size, tokenizer, dataset)

        target_sentences = []
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(gconfig.device)
            labels = batch['labels'].to(gconfig.device)
            target_tokenids = model.transfer_style(input_ids, labels)
            for i in range(len(target_tokenids)):
                target_tokens = tokenizer.convert_ids_to_tokens(target_tokenids[i])
                while '<pad>' in target_tokens:
                    target_tokens.remove('<pad>')
            target_sentence = tokenizer.convert_tokens_to_string(target_tokens)
            print("Style transfered sentence: {}".format(target_sentence))

            target_sentences.append(target_sentence)
        print("Style transfered sentence: {}".format(target_sentences))

    elif args.MODE == 'inference':
        model.load_state_dict(torch.load(gconfig.model_path + 'model.pt'))
        with open(gconfig.avg_style_emb_path+ 'avg_style_emb.pickle', 'rb') as f:
            avg_style_embeddings = pickle.load(f)
        model.avg_style_emb = avg_style_embeddings
        model.eval()

        label2index = {'neg': 0, 'pos': 1}


        source_sentence = "I am tired, but why? i didn't hear from Jeromed today yet. Put there by God I know His truth will come out in time"

        EXT_style = [input('Enter the "EXT" style | pos or neg : ')]
        NEU_style = [input('Enter the "NEU" style | pos or neg : ')]
        AGR_style = [input('Enter the "AGR" style | pos or neg : ')]
        CON_style = [input('Enter the "CON" style | pos or neg : ')]
        OPN_style = [input('Enter the "OPN" style | pos or neg : ')]

        target_style = EXT_style + NEU_style + AGR_style + CON_style + OPN_style
        encodings = tokenizer.encode_plus(source_sentence, add_special_tokens=True, max_length=mconfig.max_seq_len, padding='max_length', truncation=True)
        token_ids = torch.tensor(encodings['input_ids'], device=gconfig.device).unsqueeze(0)
        target_style_id = []
        for i in target_style:
            target_style_id.append(label2index[i])
        target_style_id = torch.tensor(target_style_id, device=gconfig.device)
        target_tokenids = model.transfer_style(token_ids, target_style_id)
        target_tokens = tokenizer.convert_ids_to_tokens(target_tokenids)
        while '<pad>' in target_tokens:
            target_tokens.remove('<pad>')
        target_sentence = tokenizer.convert_tokens_to_string(target_tokens)

        
        print("Style transfered sentence: {}".format(target_sentence))

def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', dest="MODE", type=str, required=True,
                        help='run mode: [fine_tune|train|test]')
    parser.add_argument('--epochs', dest='N_EPOCHS', type=int, default=20)
    parser.add_argument('--batch_size', dest='batch_size', type=int , default=16)

    args = parser.parse_args()
    
    return args

if __name__=='__main__':
    args = parse_argument()
    main(args)