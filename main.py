import torch
import torch.nn as nn
import torch.functional as F 
from tqdm import tqdm
from model.mscred import MSCRED
from utils.data import load_data
from utils.matrix_generator import *
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import pickle
import sys

LEARNING_RATE = float(sys.argv[1])
TEST_NAME = str(sys.argv[2])

print("__________________ TEST", LEARNING_RATE)


def train(dataLoader, model, optimizer, epochs, device):
    model = model.to(device)
    train_losses = []
    epoch_train_losses = []
    print("------training on {}-------".format(device))
    for epoch in range(epochs):
        train_l_sum,n = 0.0, 0
        for x in tqdm(dataLoader):
            x = x.to(device)
            x = x.squeeze()
            l = torch.mean((model(x)-x[-1].unsqueeze(0))**2)
            train_losses.append(l)
            train_l_sum += l
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            n += 1
            #print("[Epoch %d/%d][Batch %d/%d] [loss: %f]" % (epoch+1, epochs, n, len(dataLoader), l.item()))
        epoch_train_losses.append(train_l_sum/n)     
        print("[Epoch %d/%d] [loss: %f]" % (epoch+1, epochs, train_l_sum/n))
    return train_losses, epoch_train_losses

def test(dataLoader, model, data_path="data/", index=800):
    print("------Testing-------")
    loss_list = []
    reconstructed_data_path = data_path + "reconstructed_data/"
    if not os.path.exists(reconstructed_data_path):
        os.mkdir(reconstructed_data_path)
        
    with torch.no_grad():
        for x in dataLoader:
            x = x.to(device)
            x = x.squeeze()
            reconstructed_matrix = model(x) 
            path_temp = os.path.join(reconstructed_data_path, 'reconstructed_data_' + str(index) + ".npy")
            np.save(path_temp, reconstructed_matrix.cpu().detach().numpy())
            l = torch.mean((reconstructed_matrix - x[-1].unsqueeze(0))**2)
            loss_list.append(l)
            print("[test_index %d] [loss: %f]" % (index, l.item()))
            index += 1
    return loss_list


if __name__ == '__main__':
    steps = [5]
    for step in steps:

        # Get Execution Code
        date_tag = datetime.now()
        date_tag = date_tag.strftime('%Y%m%d_%H%M')
        exec_code = 'experiments/'+TEST_NAME+'_mscred_'+date_tag+'/'
        if not os.path.exists(exec_code):
            os.mkdir(exec_code)

        print('TESTING EXPERIMENT MSCRED', date_tag)
        print('----------------------------------------------')
        learning_rate   = LEARNING_RATE #0.0002
        n_epochs        = 10
        step_max        = step
        win_size        = [10, 30, 60]


        # Generate Signature Matrices
        s_matrix = SignatureMatrix(step_max, win_size, exec_code)
        s_matrix.generate_signature_matrix_node()
        s_matrix.generate_train_test_data()
        matrices_path = s_matrix.matrix_data_path


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device is", device)

        dataLoader = load_data(matrices_path)
        mscred = MSCRED(3, 256)

        optimizer = torch.optim.Adam(mscred.parameters(), lr = learning_rate)
        train_loss, epoch_train_loss = train(dataLoader["train"], mscred, optimizer, n_epochs, device)

        checkpoint_path = exec_code + "checkpoints/"
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        train_check = checkpoint_path + "train_checkpoint.p"
        model_check = checkpoint_path + "model2.pth"
        
        pickle.dump( mscred.state_dict(), open( train_check, "wb" ) )

        state_dict = pickle.load( open( train_check, "rb" ) )
        
        torch.save(state_dict, model_check)

        # # 测试阶段
        mscred.load_state_dict(torch.load(model_check))
        mscred.to(device)
        test_loss = test(dataLoader["test"], mscred, matrices_path)

        stats = {
        'trloss': train_loss,
        'etrloss': epoch_train_loss,
        'tsloss': test_loss
        }
        stats_path = exec_code + "stats.p"
        pickle.dump( stats, open( stats_path, "wb" ) )


        torch.cuda.empty_cache()