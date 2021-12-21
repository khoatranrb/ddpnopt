from _models import ResNet, get_cfg
from _loaders import get_data
from _utils import calculate_topk_accuracy, train, evaluate, epoch_time
from torch import nn, optim
from tqdm.auto import tqdm
import time
import torch

ROOT = '/content/data'
bs = 24
train_iterator, valid_iterator, n_classes = get_data(ROOT, bs)

model = ResNet(get_cfg(), n_classes)

LR = 1e-3

optimizer = optim.Adam(model.parameters(), lr = LR)

device = 'cuda'

criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

best_valid_loss = float('inf')
EPOCHS = 100
for epoch in range(1,EPOCHS+1):
    start_time = time.monotonic()
    
    train_loss, train_acc_1, train_acc_5 = train(model, train_iterator, optimizer, criterion, device)
    valid_loss, valid_acc_1, valid_acc_5 = evaluate(model, valid_iterator, criterion, device)
        
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
#         torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 },  '/content/drive/MyDrive/UETLAB/ddp/ddpnopt0/exps/conv/saved/n_%s.pth'%(epoch))

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc @1: {train_acc_1*100:6.2f}% | ' \
          f'Train Acc @5: {train_acc_5*100:6.2f}%')
    print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1*100:6.2f}% | ' \
          f'Valid Acc @5: {valid_acc_5*100:6.2f}%')
