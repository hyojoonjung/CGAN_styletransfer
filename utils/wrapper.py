import torch
import logging
from tqdm import tqdm
from utils.utils import EarlyStopping
from torch.utils.tensorboard import SummaryWriter

# Set up logging
def train_step(generator, discriminator, gen_optimizer, disc_optimizer, data_loader, criterions, gen_scheduler, disc_scheduler, scaler, epoch, device, writer):
    recon_criterion, gen_criterion = criterions

    generator.train()
    discriminator.train()

    total_gen_loss = 0.0
    total_disc_loss = 0.0
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Training")

    for batch_idx, data in pbar:
        text, labels = data['text'].to(device), data['labels'].to(device)
        noise = torch.randn(text.size(0), 256, device=device).float()
        
        # Discriminator training
        disc_optimizer.zero_grad()

        # Without AMP
        generated_text = generator(noise, labels)
        real_output = discriminator(text, labels)
        fake_output = discriminator(generated_text.detach(), labels)
        real_loss = recon_criterion(real_output, torch.ones_like(real_output))
        fake_loss = recon_criterion(fake_output, torch.zeros_like(fake_output))
        disc_loss = real_loss + fake_loss

        disc_loss.backward()  # backward for discriminator
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        disc_optimizer.step()  # update for discriminator

        # Generator training
        gen_optimizer.zero_grad()

        generated_text_for_gen_loss = generator(noise, labels) 
        output = discriminator(generated_text_for_gen_loss, labels)
        gen_loss = gen_criterion(output, torch.ones_like(output))

        gen_loss.backward()  # backward for generator
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        gen_optimizer.step()  # update for generator


        total_gen_loss += gen_loss.item()
        total_disc_loss += disc_loss.item()

        gen_scheduler.step(epoch + batch_idx / len(data_loader))
        disc_scheduler.step(epoch + batch_idx / len(data_loader))
        pbar.set_postfix({'Gen Loss': gen_loss.item(), 'Disc Loss': disc_loss.item()})
        
        # 기록
        global_step = epoch * len(data_loader) + batch_idx
        writer.add_scalar('Val: Batch/Gen Loss', gen_loss.item(), global_step)
        writer.add_scalar('Val: Batch/Disc Loss', disc_loss.item(), global_step)

    return total_gen_loss / len(data_loader), total_disc_loss / len(data_loader)

def test_step(generator, discriminator, data_loader, criterions, device, epoch, writer):
    recon_criterion, gen_criterion = criterions

    generator.eval()
    discriminator.eval()

    total_gen_loss = 0.0
    total_disc_loss = 0.0

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Testing")

    with torch.no_grad():  # disable gradient computation for evaluation
        for batch_idx, data in pbar:
            text, labels = data['text'].to(device), data['labels'].to(device)
            noise = torch.randn(text.size(0), 256, device=device)

            # Discriminator evaluation
            generated_text = generator(noise, labels)
            real_output = discriminator(text, labels)
            fake_output = discriminator(generated_text, labels)
            real_loss = recon_criterion(real_output, torch.ones_like(real_output))
            fake_loss = recon_criterion(fake_output, torch.zeros_like(fake_output))
            disc_loss = real_loss + fake_loss

            # Generator evaluation
            output = discriminator(generated_text, labels)
            gen_loss = gen_criterion(output, torch.ones_like(output))

            total_gen_loss += gen_loss.item()
            total_disc_loss += disc_loss.item()
            
            # 기록
            pbar.set_postfix({'Gen Loss': gen_loss.item(), 'Disc Loss': disc_loss.item()})
            global_step = epoch * len(data_loader) + batch_idx
            writer.add_scalar('Test: Batch/Gen Loss', gen_loss.item(), global_step)
            writer.add_scalar('Test: Batch/Disc Loss', disc_loss.item(), global_step)
            
    return total_gen_loss / len(data_loader), total_disc_loss / len(data_loader)


def save_checkpoint(state, filename="model/checkpoint.pth"):
    logging.info("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, generator, discriminator, gen_optimizer, disc_optimizer, gen_scheduler, disc_scheduler, scaler):
    logging.info("=> Loading checkpoint")
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    if checkpoint["gen_optimizer_state_dict"]:        
        gen_optimizer.load_state_dict(checkpoint["gen_optimizer_state_dict"])
    elif checkpoint["disc_optimizer_state_dict"]:
        disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state_dict"])
    elif checkpoint["gen_scheduler_state_dict"]:
        gen_scheduler.load_state_dict(checkpoint["gen_scheduler_state_dict"])
    elif checkpoint["disc_scheduler_state_dict"]:
        disc_scheduler.load_state_dict(checkpoint["disc_scheduler_state_dict"])
    elif checkpoint["scaler_state_dict"]:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])


def train_model(generator, discriminator, data_loaders, optimizers, criterions, device, schedulers, scaler, num_epochs=5):
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    writer = SummaryWriter()

    best_gen_loss = float('inf')
    gen_optimizer, disc_optimizer = optimizers
    gen_scheduler, disc_scheduler = schedulers
    early_stopping = EarlyStopping(patience=10)

    for epoch in range(num_epochs):
        
        train_gen_loss, train_disc_loss = train_step(generator, discriminator, gen_optimizer, disc_optimizer, data_loaders['train'], criterions, gen_scheduler, disc_scheduler, scaler, epoch, device, writer)
        val_gen_loss, val_disc_loss = test_step(generator, discriminator, data_loaders['val'], criterions, device, epoch, writer)
        
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] | Train Gen Loss: {train_gen_loss:.4f} | Train Disc Loss: {train_disc_loss:.4f} | Val Gen Loss: {val_gen_loss:.4f} | Val Disc Loss: {val_disc_loss:.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Gen Loss: {train_gen_loss:.4f} | Train Disc Loss: {train_disc_loss:.4f} | "
            f"Val Gen Loss: {val_gen_loss:.4f} | Val Disc Loss: {val_disc_loss:.4f}")
        
        # Save the model if it's the best one so far
        if val_gen_loss < best_gen_loss:
            best_gen_loss = val_gen_loss
            save_checkpoint({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'gen_optimizer_state_dict': gen_optimizer.state_dict(),
                'disc_optimizer_state_dict': disc_optimizer.state_dict(),
                'gen_scheduler_state_dict': gen_scheduler.state_dict(),
                'disc_scheduler_state_dict': disc_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }, f"checkpoint_epoch_{epoch}.pth")
        # Early stopping check
        
        early_stopping(val_gen_loss)
        if early_stopping.stop:
            print("Early stopping")
            return val_gen_loss+val_disc_loss
        
    return val_gen_loss+val_disc_loss

