from torch.utils.data import Dataset, DataLoader
from encoder_decoder import E_D_Net
from DataBuilder import MyDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch
import argparse
import numpy as np
from skimage import io

def train(epoch, input_dir):
    device = (torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    my_images = MyDataset(input_dir)
    my_loader = DataLoader(dataset = my_images, batch_size = 3, shuffle = True, num_workers = 0)#0)
    start_channel = 1
    net_layers = 3
    my_net = E_D_Net(start_channel, net_layers).to(device)
    # print(my_net)
    # for i, para in enumerate(my_net.parameters()):
    #     print(para.shape)
        # print(list(my_net.parameters())[0].dtype)
    loss = torch.nn.MSELoss()
    optimizer = Adam(my_net.parameters(),lr = 1e-4)
    scheduler = StepLR(optimizer, step_size = 5, gamma = 1.0)
    for i in range(epoch):
        for batch_number, (in_batch, gt_batch) in enumerate(my_loader):     
            in_batch = in_batch.to(device)
            gt_batch = gt_batch.to(device)
            optimizer.zero_grad()
            out_batch = my_net(in_batch)
            l = loss(gt_batch,out_batch)
            print('Epoch-{} lr: {} loss: {}'.format(i, optimizer.param_groups[0]['lr'],l))
            l.backward()
            optimizer.step()
            # print(output.shape)
        scheduler.step()
    torch.save(my_net.state_dict(),input_dir+'/model_state_{}.pth'.format(epoch))

def predict(epoch,input_dir):
    device = (torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    start_channel = 1
    net_layers = 3
    my_net = E_D_Net(start_channel, net_layers).to(device)
    my_net.load_state_dict(torch.load(input_dir+'/model_state_{}.pth'.format(epoch)))
    dataset = MyDataset(input_dir)
    my_first_img = dataset[0][0][None,:].to(device)
    my_predict = my_net(my_first_img)
    my_predict_img_torch = my_predict.to('cpu')[0][0]
    my_predict_img_np = (my_predict_img_torch.detach().numpy()* 255).astype(np.uint8)
    print(my_predict_img_np.shape)
    io.imsave(input_dir+'/test_result.png',my_predict_img_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type = str)
    myargs = parser.parse_args()
    # myargs = parser.parse_args(["/Users/xudongma/PhD/phd/prostate_data/NHS_data/MR/MRI-DVD1/DICOM/ST000000/SE000000/png"])
    input_dir = myargs.input_dir
    epoch = 3
    train(epoch, input_dir)
    predict(epoch, input_dir)