import os
from os.path import join

import torch
from torch.nn.utils import clip_grad_norm_

from utils import cal_loss, cal_epsilon, create_mask
from utils import PAD_TOKEN

from tqdm import tqdm

#DEBUG:
import code
import torch
from torch.profiler import profile, record_function, ProfilerActivity


class Trainer(object):
    def __init__(self, optimizer, model, lr_scheduler,
                 train_loader, val_loader, batch_size,
                 args, device, use_cuda=True, init_epoch=1,
                 last_epoch=15, writer=None):

        self.optimizer = optimizer
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.args = args

        self.device = device

        self.step = 0
        self.epoch = init_epoch
        self.total_step = (init_epoch-1)*len(train_loader)
        self.last_epoch = last_epoch
        self.best_val_loss = 1e18
        
        self.writer = writer


    def save_model(self, model_name):
        if not os.path.isdir(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        save_path = join(self.args.save_dir, model_name+'.pt')
        print("Saving checkpoint to {}".format(save_path))

        # torch.save(self.model, model_path)

        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_sche': self.lr_scheduler.state_dict(),
            'epoch': self.epoch,
            'args': self.args
        }, save_path)



class TransformerTrainer(Trainer):
    def __init__(self, optimizer, model, lr_scheduler,
                 train_loader, val_loader, args,
                 device, use_cuda=True, init_epoch=1,
                 last_epoch=15, writer=None):

        super().__init__(optimizer, model, lr_scheduler,
            train_loader, val_loader, args.batch_size, args,
            device, use_cuda, init_epoch, last_epoch,
            writer)

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    def train(self):
        mes = "Epoch {}, step:{}/{} {:.2f}%, Loss:{:.4f}, Perplexity:{:.4f}"
        while self.epoch <= self.last_epoch:
            print('Epoch: {}'.format(self.epoch))
            self.model.train() # switch to training mode
            losses = 0.0
            for imgs, tgt in tqdm(self.train_loader):
                step_loss = self.train_step(imgs, tgt)
                losses += step_loss
                # log message
                if self.step % self.args.print_freq == 0:
                    avg_loss = losses / self.args.print_freq
                    print(mes.format(
                        self.epoch, self.step, len(self.train_loader),
                        100 * self.step / len(self.train_loader),
                        avg_loss,
                        2**avg_loss
                    ))
                    losses = 0.0
                    
            # Calculate val loss
            val_loss = self.validate()
            self.lr_scheduler.step(val_loss)

            if self.writer is not None:
                loss = losses/self.step
                self.writer.add_scalar("loss", loss, self.epoch)
                self.writer.add_scalar("perplexity", 2**loss, self.epoch)
                self.writer.add_scalar("val_loss", val_loss, self.epoch)
                self.writer.add_scalar("val_perplexity", 2**val_loss, self.epoch)
            
            #self.save_model('ckpt-{}-{:.4f}'.format(self.epoch, val_loss))
            self.epoch += 1
            self.step = 0

    def train_step(self, imgs, tgt):
        #with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        self.optimizer.zero_grad()
        
        imgs = imgs.to(self.device)
        imgs.requires_grad = False
        input_tgt, output_tgt = tgt # input tgt with without end token, output target without start token
        
        input_tgt = input_tgt.to(self.device)
        output_tgt = output_tgt.to(self.device)

        output_tgt.requires_grad = False
        input_tgt.requires_grad = False

        # input_tgt = tgt[:, :-1] # cannot use the last token for prediction
        # output_tgt = tgt[:, 1:] # cannot use the start token or calculation loss

        input_target_mask, input_target_padding_mask = create_mask(input_tgt)
        input_target_mask = input_target_mask.to(self.device)
        input_target_padding_mask = input_target_padding_mask.to(self.device)

        input_target_padding_mask.requires_grad = False
        input_target_mask.requires_grad = False

        logits = self.model(imgs, input_tgt, input_target_mask, input_target_padding_mask)

        # calculate loss
        loss = self.calculate_loss(logits, output_tgt)
        self.step += 1
        self.total_step += 1
        loss.backward()
        output_loss = loss.item()
        del loss, logits, input_target_padding_mask, input_target_mask, imgs
        clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.optimizer.step()
    
        #print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        return output_loss              

    def calculate_loss(self, logits, target):
        return self.loss_fn(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))

    def validate(self):
        self.model.eval()
        val_total_loss = 0.0
        mes = "Epoch {}, validation average loss:{:.4f}, Perplexity:{:.4f}"
        with torch.no_grad():
            for imgs, tgt in self.val_loader:
                imgs = imgs.to(self.device)
                input_tgt, output_tgt = tgt
                input_tgt = input_tgt.to(self.device)
                output_tgt = output_tgt.to(self.device)
                input_target_mask, input_target_padding_mask = create_mask(input_tgt)
                input_target_mask = input_target_mask.to(self.device)
                input_target_padding_mask = input_target_padding_mask.to(self.device)
                logits = self.model(imgs, input_tgt, input_target_mask, input_target_padding_mask)
                loss = self.calculate_loss(logits, output_tgt)
                val_total_loss += loss.item()
            
            avg_loss = val_total_loss / len(self.val_loader)
            print(mes.format(
                self.epoch, avg_loss, 2**avg_loss
            ))
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_model('best_ckpt')
        
        return avg_loss 



class LSTMTrainer(Trainer):
    def __init__(self, optimizer, model, lr_scheduler,
                 train_loader, val_loader, args,
                 device, use_cuda=True, init_epoch=1,
                 last_epoch=15, writer=None):

        super().__init__(optimizer, model, lr_scheduler,
            train_loader, val_loader, args.batch_size, args,
            device, use_cuda, init_epoch, last_epoch,
            writer)
        

    def train(self):
        mes = "Epoch {}, step:{}/{} {:.2f}%, Loss:{:.4f}, Perplexity:{:.4f}"
        while self.epoch <= self.last_epoch:
            print('Epoch: {}'.format(self.epoch))
            self.model.train()
            losses = 0.0
            for imgs, tgt4training, tgt4cal_loss in tqdm(self.train_loader):
                if imgs.shape[0] == self.batch_size:
                    step_loss = self.train_step(imgs, tgt4training, tgt4cal_loss)
                    losses += step_loss
                    # log message
                    if self.step % self.args.print_freq == 0:
                        avg_loss = losses / self.args.print_freq
                        print(mes.format(
                            self.epoch, self.step, len(self.train_loader),
                            100 * self.step / len(self.train_loader),
                            avg_loss,
                            2**avg_loss
                        ))
                        losses = 0.0
  
            # Calculate val loss
            val_loss = self.validate()
            self.lr_scheduler.step(val_loss)

            if self.writer is not None:
                loss = losses/self.step
                self.writer.add_scalar("loss", loss, self.epoch)
                self.writer.add_scalar("perplexity", 2**loss, self.epoch)
                self.writer.add_scalar("val_loss", val_loss, self.epoch)
                self.writer.add_scalar("val_perplexity", 2**val_loss, self.epoch)
            
            #self.save_model('ckpt-{}-{:.4f}'.format(self.epoch, val_loss))
            self.epoch += 1
            self.step = 0

    def train_step(self, imgs, tgt4training, tgt4cal_loss):
        self.optimizer.zero_grad()
        imgs = imgs.to(self.device)
        
        tgt4training = tgt4training.to(self.device)
        tgt4cal_loss = tgt4cal_loss.to(self.device)
        epsilon = cal_epsilon(
            self.args.decay_k, self.total_step, self.args.sample_method)
        logits = self.model(imgs, tgt4training, epsilon)

        # calculate loss
        loss = cal_loss(logits, tgt4cal_loss)
        self.step += 1
        self.total_step += 1
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.optimizer.step()

        # gc.collect()
        # torch.cuda.empty_cache()

        return loss.item()

    def validate(self):
        self.model.eval()
        val_total_loss = 0.0
        mes = "Epoch {}, validation average loss:{:.4f}, Perplexity:{:.4f}"
        with torch.no_grad():
            for imgs, tgt4training, tgt4cal_loss in self.val_loader:
                if imgs.shape[0] == self.batch_size:
                    imgs = imgs.to(self.device)
                    tgt4training = tgt4training.to(self.device)
                    tgt4cal_loss = tgt4cal_loss.to(self.device)

                    epsilon = cal_epsilon(
                        self.args.decay_k, self.total_step, self.args.sample_method)
                    logits = self.model(imgs, tgt4training, epsilon)
                    loss = cal_loss(logits, tgt4cal_loss)
                    val_total_loss += loss
            avg_loss = val_total_loss / len(self.val_loader)
            print(mes.format(
                self.epoch, avg_loss, 2**avg_loss
            ))
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_model('best_ckpt')
        return avg_loss