import torch
import torch.nn as nn
import torch.functional as F 
from tqdm import tqdm
from model.mscred import MSCRED
from utils.data import load_data
from utils.matrix_generator import SignatureMatrix
import matplotlib.pyplot as plt
import numpy as np
import os

def train(dataLoader, model, optimizer, epochs, device):
    model = model.to(device)
    print("------training on {}-------".format(device))
    for epoch in range(epochs):
        train_l_sum,n = 0.0, 0
        for x in tqdm(dataLoader):
            x = x.to(device)
            x = x.squeeze()
            l = torch.mean((model(x)-x[-1].unsqueeze(0))**2)
            train_l_sum += l
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            n += 1
            #print("[Epoch %d/%d][Batch %d/%d] [loss: %f]" % (epoch+1, epochs, n, len(dataLoader), l.item()))
            
        print("[Epoch %d/%d] [loss: %f]" % (epoch+1, epochs, train_l_sum/n))

def test(dataLoader, model, data_path="data/", index=800):
    print("------Testing-------")
    loss_list = []
    reconstructed_data_path = data_path+"matrix_data/reconstructed_data/"
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
    steps = [5,10,20]
    for step in steps:

        print('TESTING WITH STEM MAX= ', step)
        print('----------------------------------------------')
        learning_rate   = 0.0002
        n_epochs        = 1
        step_max        = step
        win_size        = [10, 30, 60]

        data_name = 'data_'+str(step_max)+'/'
        if not os.path.exists(data_name):
            os.mkdir(data_name)

        # Generate Signature Matrices
        SignatureMatrix(step_max, win_size, data_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device is", device)

        dataLoader = load_data(data_name)
        mscred = MSCRED(3, 256)

        optimizer = torch.optim.Adam(mscred.parameters(), lr = learning_rate)
        train(dataLoader["train"], mscred, optimizer, n_epochs, device)

        checkpoint_path = 'checkpoints'
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        #torch.save(mscred.state_dict(), "checkpoints/model2.pth")
        import pickle
        pickle.dump( mscred.state_dict(), open( "checkpoint-4epochs.p", "wb" ) )

        state_dict = pickle.load( open( "checkpoint-4epochs.p", "rb" ) )
        
        torch.save(state_dict, "checkpoints/model2.pth")

        # # 测试阶段
        mscred.load_state_dict(torch.load("checkpoints/model2.pth"))
        mscred.to(device)
        test(dataLoader["test"], mscred, data_name)


        torch.cuda.empty_cache()