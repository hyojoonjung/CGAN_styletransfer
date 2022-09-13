import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import ModelConfig

mconfig = ModelConfig()

##mconfig


class MultilabelStyleTransferVAE(nn.Module):
    def __init__(self, transformers):
        super().__init__()
        ## Word Embedding ##
        self.transformers = transformers

        ## Encoder ##
        self.encoder = nn.LSTM(mconfig.embedding_dim, mconfig.hidden_dim, batch_first=True, bidirectional=True)
        
        ## Content latent space ##
        self.content_emb = nn.Linear(2*mconfig.hidden_dim, mconfig.content_hidden_dim)
        
        ## Style latent space ##
        self.OPN_style_mu = nn.Linear(2*mconfig.hidden_dim, mconfig.style_hidden_dim)
        self.OPN_style_log_var = nn.Linear(2*mconfig.hidden_dim, mconfig.style_hidden_dim)
        self.CON_style_mu = nn.Linear(2*mconfig.hidden_dim, mconfig.style_hidden_dim)
        self.CON_style_log_var = nn.Linear(2*mconfig.hidden_dim, mconfig.style_hidden_dim)
        self.EXT_style_mu = nn.Linear(2*mconfig.hidden_dim, mconfig.style_hidden_dim)
        self.EXT_style_log_var = nn.Linear(2*mconfig.hidden_dim, mconfig.style_hidden_dim)
        self.AGR_style_mu = nn.Linear(2*mconfig.hidden_dim, mconfig.style_hidden_dim)
        self.AGR_style_log_var = nn.Linear(2*mconfig.hidden_dim, mconfig.style_hidden_dim)
        self.NEU_style_mu = nn.Linear(2*mconfig.hidden_dim, mconfig.style_hidden_dim)
        self.NEU_style_log_var = nn.Linear(2*mconfig.hidden_dim, mconfig.style_hidden_dim)
        ## Content Adversary ##
        self.content_adversary = nn.Linear(mconfig.num_style*mconfig.style_hidden_dim, mconfig.vocab_size)

        ## Style Adversary ##
        self.style_adversary = nn.Linear(mconfig.content_hidden_dim, mconfig.num_style)

        ## Discriminator ##
        self.OPN_style_discriminator = nn.Linear(mconfig.style_hidden_dim, 1)
        self.CON_style_discriminator = nn.Linear(mconfig.style_hidden_dim, 1)
        self.EXT_style_discriminator = nn.Linear(mconfig.style_hidden_dim, 1)
        self.AGR_style_discriminator = nn.Linear(mconfig.style_hidden_dim, 1)
        self.NEU_style_discriminator = nn.Linear(mconfig.style_hidden_dim, 1)
        
        ## Decoder ##
        self.decoder = nn.LSTM(mconfig.content_hidden_dim + mconfig.style_hidden_dim * mconfig.num_style, mconfig.embedding_dim)
        self.projector = nn.Linear(mconfig.embedding_dim, mconfig.vocab_size)
        
        ## Average label embedding ##
        self.avg_style_emb = {
            'OPN' : [torch.zeros(mconfig.max_seq_len ,mconfig.style_hidden_dim), torch.zeros(mconfig.max_seq_len, mconfig.style_hidden_dim)],
            'CON' : [torch.zeros(mconfig.max_seq_len, mconfig.style_hidden_dim), torch.zeros(mconfig.max_seq_len, mconfig.style_hidden_dim)],
            'EXT' : [torch.zeros(mconfig.max_seq_len, mconfig.style_hidden_dim), torch.zeros(mconfig.max_seq_len, mconfig.style_hidden_dim)],
            'AGR' : [torch.zeros(mconfig.max_seq_len, mconfig.style_hidden_dim), torch.zeros(mconfig.max_seq_len, mconfig.style_hidden_dim)],
            'NEU' : [torch.zeros(mconfig.max_seq_len, mconfig.style_hidden_dim), torch.zeros(mconfig.max_seq_len, mconfig.style_hidden_dim)]
        }

        self.num_all_styles = 0
        self.num_OPN_styles = 0
        self.num_CON_styles = 0
        self.num_EXT_styles = 0
        self.num_AGR_styles = 0
        self.num_NEU_styles = 0
        
        ## Dropout ##
        self.dropout = nn.Dropout(mconfig.dropout)

    def forward(self, sequences, raw_style_labels, fine_tune, transfer, best_model, end_iter):
        
        # print(sequences.shape) #[batch, seq_len]
        ## fine tune ##
        if fine_tune:
            embedded_seqs = self.dropout(self.transformers(sequences)[0])  # emb = self.transformers(sequences)[0] # [0] : last hidden state
        ## train ##
        else:
            with torch.no_grad():
                embedded_seqs = self.dropout(self.transformers(sequences)[0])
        # print(embedded_seqs.shape) # [batch, seq_len, tra_emb_size]
        sentence_emb, _ = self.encoder(embedded_seqs)
        # print(enc_output.shape) #[batch, seq_len, 2*hidden_dim]

        content_emb = self.content_emb(sentence_emb)
        # print(content_emb.shape) #[batch, seq_len, content_hidden_dim]

        if transfer:
            # rev_style_labels = 1 - raw_style_labels
            rev_style_labels = raw_style_labels

            OPN_style_emb = self.get_zero_style_emb(sentence_emb) #[batch, seq_len, style_dim]
            CON_style_emb = self.get_zero_style_emb(sentence_emb)
            EXT_style_emb = self.get_zero_style_emb(sentence_emb)
            AGR_style_emb = self.get_zero_style_emb(sentence_emb)
            NEU_style_emb = self.get_zero_style_emb(sentence_emb)

            for i in range(len(rev_style_labels)):
                if rev_style_labels[i, 0] == 0:
                    OPN_style_emb[i] = self.avg_style_emb['OPN'][0]
                else:
                    OPN_style_emb[i] = self.avg_style_emb['OPN'][1]
                if rev_style_labels[i, 1] == 0:
                    CON_style_emb[i] = self.avg_style_emb['CON'][0]
                else:
                    CON_style_emb[i] = self.avg_style_emb['CON'][1]

                if rev_style_labels[i, 2] == 0:
                    EXT_style_emb[i] = self.avg_style_emb['EXT'][0]
                else:
                    EXT_style_emb[i] = self.avg_style_emb['EXT'][1]

                if rev_style_labels[i, 3] == 0:
                    AGR_style_emb[i] = self.avg_style_emb['AGR'][0]
                else:
                    AGR_style_emb[i] = self.avg_style_emb['AGR'][1]

                if rev_style_labels[i, 4] == 0:
                    NEU_style_emb[i] = self.avg_style_emb['NEU'][0]
                else:
                    NEU_style_emb[i] = self.avg_style_emb['NEU'][1]

        else:
            OPN_mu, OPN_log_var = self.get_style_emb(sentence_emb, self.OPN_style_mu, self.OPN_style_log_var)
            CON_mu, CON_log_var = self.get_style_emb(sentence_emb, self.CON_style_mu, self.CON_style_log_var)
            EXT_mu, EXT_log_var = self.get_style_emb(sentence_emb, self.EXT_style_mu, self.EXT_style_log_var)
            AGR_mu, AGR_log_var = self.get_style_emb(sentence_emb, self.AGR_style_mu, self.AGR_style_log_var)
            NEU_mu, NEU_log_var = self.get_style_emb(sentence_emb, self.NEU_style_mu, self.NEU_style_log_var)
            # print(OPN_mu.shape) #[batch, max_seq_len, style_hidden_dim]
            
            OPN_style_emb = self.sample_prior(OPN_mu, OPN_log_var)
            CON_style_emb = self.sample_prior(CON_mu, CON_log_var)
            EXT_style_emb = self.sample_prior(EXT_mu, EXT_log_var)
            AGR_style_emb = self.sample_prior(AGR_mu, AGR_log_var)
            NEU_style_emb = self.sample_prior(NEU_mu, NEU_log_var)
            # print(OPN_style_emb.shape) #[batch, max_seq_len, style_hidden_dim]
        style_emb = torch.cat((OPN_style_emb, CON_style_emb, EXT_style_emb, AGR_style_emb, NEU_style_emb), dim=2)
        # print(style_emb.shape) #[batch, max_seq_len, 5*style_hidden_dim]

        generative_emb = torch.cat((content_emb, style_emb), dim=2)
        # print(generative_emb.shape) #[batch, max_seq_len, content_hid+5*style_hid]

        ## generate sentence ##
        reconstructed_sentences = self.generate_sentences(generative_emb, inference=True)

        ## adversarial training ##
        # content_adv_preds = self.content_adversary(style_emb)
        # content_adv_disc_loss = self.get_recon_loss(content_adv_preds, sequences)
        # content_entropy_loss = self.get_entropy_loss(content_adv_preds)

        style_adv_preds = self.style_adversary(content_emb)
        if transfer:
            style_adv_disc_loss = self.get_style_adv_disc_loss(style_adv_preds, rev_style_labels)
        else:
            style_adv_disc_loss = self.get_style_adv_disc_loss(style_adv_preds, raw_style_labels)
        style_entropy_loss = self.get_entropy_loss(style_adv_preds)
        
        ## transfer reconstruction loss##
        if transfer:
            reconstruction_loss = self.get_recon_loss(reconstructed_sentences, sequences)
            style_loss = self.style_disc_loss(OPN_style_emb[:,0,:], CON_style_emb[:,0,:], EXT_style_emb[:,0,:], AGR_style_emb[:,0,:], NEU_style_emb[:,0,:], rev_style_labels)

        ## reconstruction loss##
        else:
            reconstruction_loss = self.get_recon_loss(reconstructed_sentences, sequences)
            style_loss = self.style_disc_loss(OPN_style_emb[:,0,:], CON_style_emb[:,0,:], EXT_style_emb[:,0,:], AGR_style_emb[:,0,:], NEU_style_emb[:,0,:], raw_style_labels)
        if best_model:
            if not transfer:
                self.update_average_style_emb(OPN_style_emb, CON_style_emb, EXT_style_emb, AGR_style_emb, NEU_style_emb, raw_style_labels, end_iter)
        del sequences, embedded_seqs,sentence_emb, content_emb, style_emb
        del OPN_style_emb,  CON_style_emb,  EXT_style_emb,  AGR_style_emb,  NEU_style_emb
        if not transfer:
            del OPN_log_var, OPN_mu, CON_log_var, CON_mu, EXT_log_var, EXT_mu, AGR_log_var, AGR_mu, NEU_log_var, NEU_mu
        del generative_emb, reconstructed_sentences, style_adv_preds, 
        return reconstruction_loss, style_loss, style_entropy_loss, style_adv_disc_loss
    
    def get_params(self):
                            
        recon_loss_params = list(self.transformers.parameters()) + list(self.encoder.parameters()) + list(self.content_emb.parameters()) + \
                            list(self.decoder.parameters()) + list(self.projector.parameters())
        
        style_loss_params = list(self.transformers.parameters()) + list(self.encoder.parameters()) + \
                            list(self.OPN_style_mu.parameters()) + list(self.OPN_style_log_var.parameters()) + \
                            list(self.CON_style_mu.parameters()) + list(self.CON_style_log_var.parameters()) + \
                            list(self.EXT_style_mu.parameters()) + list(self.EXT_style_log_var.parameters()) + \
                            list(self.AGR_style_mu.parameters()) + list(self.AGR_style_log_var.parameters()) + \
                            list(self.NEU_style_mu.parameters()) + list(self.NEU_style_log_var.parameters()) + \
                            list(self.OPN_style_discriminator.parameters()) + \
                            list(self.CON_style_discriminator.parameters()) + \
                            list(self.EXT_style_discriminator.parameters()) + \
                            list(self.AGR_style_discriminator.parameters()) + \
                            list(self.NEU_style_discriminator.parameters())

        style_adv_params = self.style_adversary.parameters()

        adv_loss_params = list(self.transformers.parameters()) + list(self.encoder.parameters()) + list(self.content_emb.parameters())
        
        return recon_loss_params, style_loss_params, style_adv_params, adv_loss_params
                                
    def update_average_style_emb(self, OPN_style_emb, CON_style_emb, EXT_style_emb, AGR_style_emb, NEU_style_emb, raw_style_labels, end_iter):
        """
        Args:
            style_emb : batch of sampled style embeddings of the input sentences, [batch_size, seq_len, style_hidden_dim]
            style_labels : style labels of the corresponding input sentences, [batch_size, 5]
        """
        with torch.no_grad():
            self.avg_style_emb['OPN'][0] = self.avg_style_emb['OPN'][0].to(OPN_style_emb.device)
            self.avg_style_emb['OPN'][1] = self.avg_style_emb['OPN'][1].to(OPN_style_emb.device)
            self.avg_style_emb['CON'][0] = self.avg_style_emb['CON'][0].to(CON_style_emb.device)
            self.avg_style_emb['CON'][1] = self.avg_style_emb['CON'][1].to(CON_style_emb.device)
            self.avg_style_emb['EXT'][0] = self.avg_style_emb['EXT'][0].to(EXT_style_emb.device)
            self.avg_style_emb['EXT'][1] = self.avg_style_emb['EXT'][1].to(EXT_style_emb.device)
            self.avg_style_emb['AGR'][0] = self.avg_style_emb['AGR'][0].to(AGR_style_emb.device)
            self.avg_style_emb['AGR'][1] = self.avg_style_emb['AGR'][1].to(AGR_style_emb.device)
            self.avg_style_emb['NEU'][0] = self.avg_style_emb['NEU'][0].to(NEU_style_emb.device)
            self.avg_style_emb['NEU'][1] = self.avg_style_emb['NEU'][1].to(NEU_style_emb.device)
            
            self.num_all_styles += len(raw_style_labels)
            for i, style_labels in enumerate(raw_style_labels):
                if style_labels[0] == 0:
                    self.avg_style_emb['OPN'][0] += OPN_style_emb[i, :, :]
                else: 
                    self.num_OPN_styles += 1
                    self.avg_style_emb['OPN'][1] += OPN_style_emb[i, :, :]
                if style_labels[1] == 0:
                    self.avg_style_emb['CON'][0] += CON_style_emb[i, :, :]
                else: 
                    self.num_CON_styles += 1
                    self.avg_style_emb['CON'][1] += CON_style_emb[i, :, :]
                if style_labels[2] == 0:
                    self.avg_style_emb['EXT'][0] += EXT_style_emb[i, :, :]
                else: 
                    self.num_EXT_styles += 1
                    self.avg_style_emb['EXT'][1] += EXT_style_emb[i, :, :]
                if style_labels[3] == 0:
                    self.avg_style_emb['AGR'][0] += AGR_style_emb[i, :, :]
                else: 
                    self.num_AGR_styles += 1
                    self.avg_style_emb['AGR'][1] += AGR_style_emb[i, :, :]
                if style_labels[4] == 0:
                    self.avg_style_emb['NEU'][0] += NEU_style_emb[i, :, :]
                    # print(self.avg_style_emb['NEU'][0])
                    # print(NEU_style_emb[i,:,:])
                else: 
                    self.num_NEU_styles += 1
                    self.avg_style_emb['NEU'][1] += NEU_style_emb[i, :, :]

            if end_iter:
                self.avg_style_emb['OPN'][0] = self.avg_style_emb['OPN'][0] / (self.num_all_styles - self.num_OPN_styles)
                self.avg_style_emb['OPN'][1] = self.avg_style_emb['OPN'][1] / (self.num_OPN_styles)

                self.avg_style_emb['CON'][0] = self.avg_style_emb['CON'][0] / (self.num_all_styles - self.num_CON_styles)
                self.avg_style_emb['CON'][1] = self.avg_style_emb['CON'][1] / (self.num_CON_styles)

                self.avg_style_emb['EXT'][0] = self.avg_style_emb['EXT'][0] / (self.num_all_styles - self.num_EXT_styles)
                self.avg_style_emb['EXT'][1] = self.avg_style_emb['EXT'][1] / (self.num_EXT_styles)

                self.avg_style_emb['AGR'][0] = self.avg_style_emb['AGR'][0] / (self.num_all_styles - self.num_AGR_styles)
                self.avg_style_emb['AGR'][1] = self.avg_style_emb['AGR'][1] / (self.num_AGR_styles)

                self.avg_style_emb['NEU'][0] = self.avg_style_emb['NEU'][0] / (self.num_all_styles - self.num_NEU_styles)
                self.avg_style_emb['NEU'][1] = self.avg_style_emb['NEU'][1] / (self.num_NEU_styles)
            # print(self.avg_style_emb)
    def get_style_emb(self, sentence_emb, style_mu, style_log_var):
        """
        """
        mu = style_mu(sentence_emb)
        log_var = style_log_var(sentence_emb)

        return mu, log_var

    def sample_prior(self, mu, log_var):
        """
        reparameterization trick
        """
        epsilon = torch.randn(mu.size(2), device=mu.device)
        return mu + epsilon * torch.exp(log_var)
    
    def get_zero_style_emb(self, sentence_emb):
        zero_style_emb = torch.zeros(sentence_emb.size(0), mconfig.max_seq_len, mconfig.style_hidden_dim, device=sentence_emb.device)

        return zero_style_emb

    def style_disc_loss(self, OPN_style_emb, CON_style_emb, EXT_style_emb, AGR_style_emb, NEU_style_emb, style_labels):
        OPN_pred = self.OPN_style_discriminator(OPN_style_emb)
        CON_pred = self.CON_style_discriminator(CON_style_emb)
        EXT_pred = self.EXT_style_discriminator(EXT_style_emb)
        AGR_pred = self.AGR_style_discriminator(AGR_style_emb)
        NEU_pred = self.NEU_style_discriminator(NEU_style_emb)
        
        criterion = nn.BCEWithLogitsLoss()
        OPN_loss = criterion(OPN_pred, style_labels[:,0].unsqueeze(1))
        CON_loss = criterion(CON_pred, style_labels[:,1].unsqueeze(1))
        EXT_loss = criterion(EXT_pred, style_labels[:,2].unsqueeze(1))
        AGR_loss = criterion(AGR_pred, style_labels[:,3].unsqueeze(1))
        NEU_loss = criterion(NEU_pred, style_labels[:,4].unsqueeze(1))

        # style_loss = torch.cat((OPN_loss, CON_loss, EXT_loss, AGR_loss, NEU_loss), dim=1)
        style_loss = OPN_loss + CON_loss + EXT_loss + AGR_loss + NEU_loss
        return style_loss

    def get_style_adv_disc_loss(self, style_disc_preds, style_labels):
        
        style_disc_preds = torch.mean(style_disc_preds, dim=1)
        style_disc_loss = nn.BCEWithLogitsLoss()(style_disc_preds, style_labels)
        return style_disc_loss
    
    def get_kl_loss(self, mu, log_var):
        """
        Args:
            mu : batch of means of the gaussian distribution
            log_var : batch of log variance

        Returns:
            kl loss
        """
        kl_loss = torch.mean((-0.5*torch.sum(1+log_var - log_var.exp()-mu.pow(2), dim=1)))

        return kl_loss

    def get_entropy_loss(self, preds):
        """
        negative entropy loss
        """
        preds = F.relu(preds)

        return torch.mean(torch.sum(preds * torch.log(preds + mconfig.epsilon), dim=2))

    def generate_sentences(self, generative_emb, inference):
        """
            Input:
                generative emb : [batch, seq_len, generative_emb_size]
        """
        if inference:
            output_sentences = torch.zeros(generative_emb.size(0), mconfig.max_seq_len, device=generative_emb.device) #[batch, seq_len, vocab_size]
            # hidden_states = torch.zeros(generative_emb.size(0), 1, 1, mconfig.embedding_dim)

            for idx in range(mconfig.max_seq_len):
                words = generative_emb[:, idx, :]
                # print(words.shape) #[batch, genertaive_emb]
                dec_output, _ = self.decoder(words.unsqueeze(1))
                # print(dec_output.shape) #[batch, 1, embedding_dim]
                next_word_logits = self.projector(dec_output).squeeze(1)
                # print(next_word_logits.shape) #[batch, vocab_size]
                # output_sentences[:,idx,:] = next_word_logits.argmax(1)
                output_sentences[:,idx] = next_word_logits.argmax(1)

                # print(output_sentences.shape) #[batch, seq_len, 1]

        else:
            output_sentences = torch.zeros(generative_emb.size(0), mconfig.max_seq_len, mconfig.vocab_size, device=generative_emb.device) #[batch, seq_len, vocab_size]
            # hidden_states = torch.zeros(generative_emb.size(0), 1, 1, mconfig.embedding_dim)
            for idx in range(mconfig.max_seq_len):
                words = generative_emb[:, idx, :]
                # print(words.shape) #[batch, genertaive_emb]
                dec_output, _ = self.decoder(words.unsqueeze(1))
                # print(dec_output.shape) #[batch, 1, embedding_dim]
                next_word_logits = self.projector(dec_output).squeeze(1)
                # print(next_word_logits.shape) #[batch, vocab_size]
                output_sentences[:,idx,:] = next_word_logits
            # print(output_sentences.shape) #[batch, seq_len, vocab]
        return output_sentences
    
    def get_recon_loss(self, reconstructed_sentences, input_sentences):
        
        # print(reconstructed_sentences.shape) #[batch, seq_len, vocab_size]
        # print(input_sentences.shape) #[batch, seq_len]

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        recon_loss = criterion(reconstructed_sentences.view(-1, mconfig.vocab_size), input_sentences.view(-1))
        return recon_loss

    def transfer_style(self, sequence, style_labels):
        """
        Args:
            sequence : token [random_seq_length]
            style : target style
        """
        embedded_seqs = self.transformers(sequence)[0]

        sentence_emb, _ = self.encoder(embedded_seqs)

        content_emb = self.content_emb(sentence_emb)
        
        OPN_mu, OPN_log_var = self.get_style_emb(sentence_emb, self.OPN_style_mu, self.OPN_style_log_var)
        CON_mu, CON_log_var = self.get_style_emb(sentence_emb, self.CON_style_mu, self.CON_style_log_var)
        EXT_mu, EXT_log_var = self.get_style_emb(sentence_emb, self.EXT_style_mu, self.EXT_style_log_var)
        AGR_mu, AGR_log_var = self.get_style_emb(sentence_emb, self.AGR_style_mu, self.AGR_style_log_var)
        NEU_mu, NEU_log_var = self.get_style_emb(sentence_emb, self.NEU_style_mu, self.NEU_style_log_var)
        # print(OPN_mu.shape) #[batch, max_seq_len, style_hidden_dim]
            
        OPN_style_emb = self.sample_prior(OPN_mu, OPN_log_var)
        CON_style_emb = self.sample_prior(CON_mu, CON_log_var)
        EXT_style_emb = self.sample_prior(EXT_mu, EXT_log_var)
        AGR_style_emb = self.sample_prior(AGR_mu, AGR_log_var)
        NEU_style_emb = self.sample_prior(NEU_mu, NEU_log_var)
        # OPN_style_emb = self.get_zero_style_emb(sentence_emb) #[1, max_seq_len, style_dim]
        # CON_style_emb = self.get_zero_style_emb(sentence_emb)
        # EXT_style_emb = self.get_zero_style_emb(sentence_emb)
        # AGR_style_emb = self.get_zero_style_emb(sentence_emb)
        # NEU_style_emb = self.get_zero_style_emb(sentence_emb)

        # rev_style_labels = style_labels
        # for i in range(len(rev_style_labels)):
        #     if rev_style_labels[i, 0] == 0:
        #         OPN_style_emb[i] = self.avg_style_emb['OPN'][0]
        #     else:
        #         OPN_style_emb[i] = self.avg_style_emb['OPN'][1]
        #     if rev_style_labels[i, 1] == 0:
        #         CON_style_emb[i] = self.avg_style_emb['CON'][0]
        #     else:
        #         CON_style_emb[i] = self.avg_style_emb['CON'][1]

        #     if rev_style_labels[i, 2] == 0:
        #         EXT_style_emb[i] = self.avg_style_emb['EXT'][0]
        #     else:
        #         EXT_style_emb[i] = self.avg_style_emb['EXT'][1]

        #     if rev_style_labels[i, 3] == 0:
        #         AGR_style_emb[i] = self.avg_style_emb['AGR'][0]
        #     else:
        #         AGR_style_emb[i] = self.avg_style_emb['AGR'][1]

        #     if rev_style_labels[i, 4] == 0:
        #         NEU_style_emb[i] = self.avg_style_emb['NEU'][0]
        #     else:
        #         NEU_style_emb[i] = self.avg_style_emb['NEU'][1]

        # if style_labels[0] == 0:
        #     OPN_style_emb = self.avg_style_emb['OPN'][0]
        # else:
        #     OPN_style_emb = self.avg_style_emb['OPN'][1]
        # if style_labels[1] == 0:
        #     CON_style_emb = self.avg_style_emb['CON'][0]
        # else:
        #     CON_style_emb = self.avg_style_emb['CON'][1]
        # if style_labels[2] == 0:
        #     EXT_style_emb = self.avg_style_emb['EXT'][0]
        # else:
        #     EXT_style_emb = self.avg_style_emb['EXT'][1]

        # if style_labels[3] == 0:
        #     AGR_style_emb = self.avg_style_emb['AGR'][0]
        # else:
        #     AGR_style_emb = self.avg_style_emb['AGR'][1]

        # if style_labels[4] == 0:
        #     NEU_style_emb = self.avg_style_emb['NEU'][0]
        # else:
        #     NEU_style_emb = self.avg_style_emb['NEU'][1]
        
        # OPN_style_emb = OPN_style_emb.unsqueeze(0)
        # CON_style_emb = CON_style_emb.unsqueeze(0)
        # EXT_style_emb = EXT_style_emb.unsqueeze(0)
        # AGR_style_emb = AGR_style_emb.unsqueeze(0)
        # NEU_style_emb = NEU_style_emb.unsqueeze(0)

        # print(OPN_style_emb.shape)
        style_emb = torch.cat((OPN_style_emb, CON_style_emb, EXT_style_emb, AGR_style_emb, NEU_style_emb), dim=2)

        generative_emb = torch.cat((content_emb, style_emb), dim=2)

        # reconstructed_sentences = self.generate_sentences(generative_emb, inference=True).squeeze(0) #[1, max_seq_len]
        reconstructed_sentences = self.generate_sentences(generative_emb, inference=True) #[batch, max_seq_len]

        return reconstructed_sentences