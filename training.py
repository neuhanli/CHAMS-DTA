import os
import sys
import torch.nn as nn


from utils import *
from create_data import create_dataset
import time

from model import ModelNet

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        DrugData = data[0].to(device)
        TargetData = data[1].to(device)

        optimizer.zero_grad()
        output, _, _ = model(DrugData, TargetData)
        loss = loss_fn(output, DrugData.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * TRAIN_BATCH_SIZE,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

    # predicting function
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output, _, _ = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
        loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
        print('Test epoch: Loss: {:.6f}'.format(loss.item()))
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


datasets = ['davis']
modeling = ModelNet



model_st = modeling.__name__

print("dataset:", datasets)
print("modeling:", modeling)

# determine the device in the following line
# torch.device( "cpu")
cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LR = 0.0001

LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset)

    start_time = time.time()
    train_data_file = "./data/" + dataset + "_traindata_"+ str(TRAIN_BATCH_SIZE) +".data"
    test_data_file = "./data/" + dataset + "_testdata_"+ str(TEST_BATCH_SIZE) +".data"
    if not (os.path.isfile(train_data_file) and os.path.isfile(test_data_file)):
        train_data, test_data = create_dataset(dataset)
        torch.save(train_data, train_data_file)  # save train data
        torch.save(test_data, test_data_file)  # save test data
    else:
        train_data = torch.load(train_data_file)
        test_data = torch.load(test_data_file)

    print('load dataset successfully')


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)
    # 查看数据集的大小
    print("Number of samples in train_data:", len(train_loader))
    print(train_loader)

    checkpoint_path = 'checkpoint_davis.pt'
    startEpoch = 0

    print('complete dataloader loading')
    end_time = time.time()
    all_time = end_time-start_time
    print('The data preparation took a total of ',all_time,' seconds')
    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_mse = 1000
    best_ci = 0
    rm2 = 0
    best_epoch = -1
    model_file_name = 'model_' + model_st + '_' + dataset + '_final1'+  '.model'
    result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        startEpoch = checkpoint['epoch']

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    for epoch in range(startEpoch, NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch+1)
        G, P = predicting(model, device, test_loader)
        ret = [mse(G, P), ci(G, P), get_rm2(G, P)]
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)
        if ret[0] < best_mse:
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, 'w') as f:
                f.write(','.join(map(str, ret)))
            best_epoch = epoch+1
            best_mse = ret[0]
            best_ci = ret[1]
            rm2 = ret[2]
            print('mse improved at epoch ', best_epoch, '; best_mse,best_ci,rm2:', best_mse, best_ci, rm2, model_st, dataset)
        elif (epoch - best_epoch)<500:
            print(ret[0], 'No improvement since epoch ', best_epoch, '; best_mse,best_ci,rm2:', best_mse, best_ci, rm2, model_st, dataset)
        else:
            print('early stop  ''; best_mse,best_ci,rm2:', best_mse, best_ci, rm2, model_st, dataset)
            break
