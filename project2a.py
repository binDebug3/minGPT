import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from mingpt.trainer import Trainer
import numpy as np
set_seed(42)

# dataset class
class RedPajama(Dataset):
    def __init__(self, data, max_length=1024):
        # super().__init__()
        model_architecture = "gpt2_tokenizer"
        self.data = data
        self.max_length = max_length
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_architecture)
        self.tokenizer.pad_token_id = 50256
        self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        # return the number of data points
        return len(self.data)
    
    def __getitem__(self, idx):
        # load and tokenize the text
        text = self.data[idx]["text"]
        tokens = self.tokenizer.encode(text, 
                                       add_special_tokens=True, 
                                       max_length=self.max_length,
                                       truncation=True,
                                       return_tensors="pt",
                                       padding=True)
        
        # split and shift tokens after excluding the CLS token
        targets = tokens[:, 1:].clone()
        tokens = tokens[:, :-1].clone()

        return tokens, targets


# train model
def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")


if __name__ == '__main__':

    # load in the dataset
    rp_dataset = load_dataset(
        "json", data_files="/lustre/scratch/usr/dw87/pile_data_10.jsonl", cache_dir='pile_dataset')
    rp_dataset = rp_dataset['train']
    print('Loaded Dataset')
    data = RedPajama(rp_dataset)
    print('Instatiated Dataset Class')

    # set up model configurations
    model_config = GPT.get_default_config()
    # model_config.model_type = 'gpt-nano'
    model_config.model_type = 'gpt2'
    model_config.vocab_size = data.vocab_size
    model_config.block_size = data.max_length - 1
    model_config.checkpoint = None

    # set up model with configurations
    model = GPT(model_config)


    # set up trainer configurations
    # max_iters = 50000
    max_iters = 1000
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    train_config.max_iters = max_iters + model.iter_num if model_config.checkpoint else max_iters
    train_config.num_workers = 0
    train_config.checkpoint_iters = 1000
    train_config.batch_size = 1
    # train_config.checkpoint_name = path

    # set up trainer with configurations
    trainer = Trainer(train_config, model, data)
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()


    # plot the loss of every hundred elements
    loss_list = [a.detach().cpu() for a in trainer.saved_loss]
    length = 100
    new_losses = np.mean(np.array(loss_list).reshape(-1, length), axis=1)

    plt.plot(np.arange(len(new_losses)), new_losses)
    plt.title('Mingpt Loss')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.savefig('loss_mingpt.png')
