import copy
import time
import torch
import torch.nn as nn
import math
from _models import TransformerModel
from _data import get_data
from utils import train, evaluate

train_data, val_data, test_data, vocab = get_data()

device = 'cuda'
ntokens = len(vocab)  # size of vocabulary
emsize = 100  # embedding dimension
d_hid = 100  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 10  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
bptt = 35
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 1e-4 # learning rate
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    
best_val_loss = float('inf')
epochs = 120

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model)
    val_loss = evaluate(model, val_data)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss

#         torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 }, 'exps/lm/saved/n_%s.pth'%(epoch))
