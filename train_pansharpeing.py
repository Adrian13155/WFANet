import os
from datetime import datetime
import torch.nn.functional as F
import argparse
from Dataset import MatDataset
import torch
from torch import nn
import torch.utils.data as data
from torch.utils.data import DataLoader 
from torch.optim.lr_scheduler import CosineAnnealingLR 
from tqdm import tqdm
import logging 
from datetime import datetime
from net_torch import HWViT
from skimage.metrics import peak_signal_noise_ratio as PSNR


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def main(opt):

    now = datetime.now()

    # 格式化为字符串，包含月、日、小时和分钟
    formatted_time = now.strftime("%m-%d_%H:%M")
    save_dir = os.path.join(opt.save_dir, f'{formatted_time}_{opt.exp_name}For{opt.satellite}')
    os.makedirs(save_dir,exist_ok=True)
    lr = opt.learning_rate

    total_iteration = opt.total_iteration
    num_epoch = opt.num_epochs
    batch_size =  opt.batch_size

    # train
    dataset_folder = opt.pan_root


    dataset_path = os.path.join(dataset_folder,f'{opt.satellite}/train')
    train_dataset = MatDataset(dataset_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=True, drop_last=False)
    del train_dataset

    val_dataset_path = os.path.join(dataset_folder,f'{opt.satellite}/test')
    val_dataset = MatDataset(val_dataset_path)

    channel = 0
    if opt.satellite == "WV2":
        channel = 8
    else:
        channel = 4
    model = HWViT(L_up_channel=channel, pan_channel=1, ms_target_channel=32,
              pan_target_channel=32, head_channel=8, dropout=0.085).cuda()
    

    logger = get_logger(os.path.join(save_dir,f'run_{opt.exp_name}For{opt.satellite}.log'))
    logger.info(opt)
    logger.info(f"model params: {sum(p.numel() for p in model.parameters() )/1e6} M")
    logger.info(f"Network Structure: {str(model)}")

    optimizer_G = torch.optim.Adam([{'params': (p for name, p in model.named_parameters() if 'bias' not in name), 'weight_decay': 0.00000001},
     {'params': (p for name, p in model.named_parameters() if 'bias' in name)}], lr=lr) 
    lr_scheduler_G = CosineAnnealingLR(optimizer_G, total_iteration, eta_min=1.0e-6)
    L1 = nn.L1Loss(reduction='mean').cuda() 

    best_psnr = 0
    best_index = 0
    for epoch in range(opt.epoch_start,num_epoch):

        pbar = tqdm(train_dataloader)
        model.train()

        for index, data in enumerate(pbar):
            optimizer_G.zero_grad() 
            inp_ms, inp_pan, inp_gt = data
            inp_ms = inp_ms.type(torch.FloatTensor).cuda().permute(0,3,1,2)
            inp_lms = F.interpolate(inp_ms , scale_factor = 4, mode = 'bilinear')
            inp_pan = inp_pan.type(torch.FloatTensor).cuda().unsqueeze(1)
            inp_gt = inp_gt.type(torch.FloatTensor).cuda().permute(0,3,1,2) 

            restored  = model(pan=inp_pan, ms=inp_ms, lms=inp_lms)

            loss_l1 = L1(restored, inp_gt)
            loss_G = loss_l1 
            loss_G.backward()
            optimizer_G.step()
            torch.cuda.empty_cache() 
            lr_scheduler_G.step()
        
        current_lr = optimizer_G.param_groups[0]['lr']
        pbar.set_description("Epoch:{}   loss_G:{:6}  lr:{:.6f}".format(epoch, loss_G.item(), current_lr))
        pbar.update()
        
        if epoch % 10== 0:
            model.eval()
            with torch.no_grad():
                psnr = 0
                count = 0
                val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

                for index, datas in enumerate(tqdm(val_dataloader)):
                    count += 1
                    inp_ms, inp_pan, inp_gt = datas[0], datas[1], datas[2]
                    inp_ms = inp_ms.type(torch.FloatTensor).cuda().permute(0,3,1,2)
                    inp_lms = F.interpolate(inp_ms , scale_factor = 4, mode = 'bilinear')
                    inp_pan = inp_pan.type(torch.FloatTensor).cuda().unsqueeze(1)
                    inp_gt = inp_gt.type(torch.FloatTensor).permute(0,3,1,2)
                    output = model(pan=inp_pan, ms=inp_ms, lms=inp_lms)

                    netOutput_np = output.cpu().numpy()[0]
                    gtLabel_np = inp_gt.numpy()[0]
                    psnrValue = PSNR(gtLabel_np, netOutput_np)
                    psnr += psnrValue     

                psnr = psnr / count
                torch.cuda.empty_cache()   
            if psnr > best_psnr:
                best_psnr = psnr
                best_index = epoch
                torch.save(model.state_dict(), os.path.join(save_dir,'Best.pth'))
            ## record
            logger.info('Epoch:[{}]\t PSNR = {:.4f}\t BEST_PSNR = {:.4f}\t BEST_epoch = {}'.format(
                        epoch, psnr, best_psnr, best_index))
            print('Epoch:[{}]\t PSNR = {:.4f}\t BEST_PSNR = {:.4f}\t BEST_epoch = {}'.format(
                        epoch, psnr, best_psnr, best_index))
            
            torch.save(model.state_dict(), os.path.join(save_dir,f'epoch={epoch}.pth'))
            
def get_opt():
    parser = argparse.ArgumentParser(description='Hyper-parameters for network')
    parser.add_argument('--exp_name', type=str, default='WDANet(2025AAAI)', help='experiment name')
    parser.add_argument('-learning_rate', help='Set the learning rate', default=9e-4, type=float)
    parser.add_argument('-batch_size', help='批量大小', default=4, type=int)
    parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
    parser.add_argument('-num_epochs', help='', default=1000, type=int)
    parser.add_argument('-pan_root', help='数据集路径', default='/PublicData/xmm/NBU_dataset0730', type=str)
    parser.add_argument('-satellite', help='GF1/QB/WV2/WV4', default='GF1', type=str)
    parser.add_argument('-save_dir', help='日志保存路径', default='/PublicData/xmm/project/WFANet/experiment', type=str)
    parser.add_argument('-gpu_id', help='gpu下标', default=0, type=int)
    parser.add_argument('-total_iteration', help='', default=2e5, type=int)
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    opt = get_opt()
    torch.cuda.set_device(opt.gpu_id)
    main(opt)
