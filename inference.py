import torch
from core.model import Generator, Discriminator
from transformers import T5Tokenizer, T5Config

def transfer_sentence(generator, sentence, label, tokenizer, max_length, device):
    generator.eval()
    with torch.no_grad():
        encoded_sentence = tokenizer(sentence, return_tensors="pt", max_length=max_length, padding='max_length', truncation=True).input_ids.to(device)
        label_tensor = torch.tensor(label).to(device)
        # label_tensor = label_tensor.unsqueeze(0)
        # encoded_sentence = encoded_sentence.squeeze(0)
        print(encoded_sentence.shape)
        print(label_tensor.shape)
        generated_text_ids = generator(encoded_sentence, label_tensor)
        decoded_text = tokenizer.decode(generated_text_ids[0], skip_special_tokens=True)
    return decoded_text

sentence = "Your input sentence here."
#'OPN','CON','EXT','AGR','NEU'
label = [0.1, 0.2, 0.3, 0.4, 0.5]

t5_model_name = 't5-base'
max_length = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

generator = Generator(t5_model_name=t5_model_name).to(device)
discriminator = Discriminator(t5_model_name=t5_model_name).to(device)

# checkpoint = torch.load("model/best_model.pth")
checkpoint = torch.load("checkpoint_epoch_8.pth")

generator.load_state_dict(checkpoint["generator_state_dict"])

transformed_sentence = transfer_sentence(generator, sentence, label, tokenizer, max_length, device)
print(transformed_sentence)
