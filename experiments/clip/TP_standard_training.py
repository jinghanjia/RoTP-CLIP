from random import betavariate
from sched import scheduler
from tqdm import tqdm
import argparse
from functools import partial
import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import clip
from PIL import Image
import io

import sys
sys.path.append(".")
from datatools.prepare_data import prepare_clip_data
from tools.misc import *
from tools.gen_text_embedding import get_saparate_text_embedding
from models.adv_program import VisualPrompt
from cfg import *
from models.Custom_CLIP import clip_img_preprocessing,CustomCLIP
from algorithms.attack import attack_pgd


def train(visual_prompt,model,loaders,epoch,optimizer,scheduler,args):
    pbar = tqdm(loaders['train'], total=len(loaders['train']), desc=f"Epo {epoch} Training", ncols=160)    
    losses = AverageMeter()
    total_losses = AverageMeter()
    TAs = AverageMeter()
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        images = clip_img_preprocessing(x)
        if args.visual_prompter is not None:
            prompted_images = visual_prompt(images)
        else:
            prompted_images = images
        optimizer.zero_grad()
        output = model(prompted_images)
        loss = F.cross_entropy(output,y)
        total_loss = loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        output = output.float()
        loss = loss.float()
        total_loss = total_loss.float()
        TA = accuracy(output.data, y)[0]
        losses.update(loss.item(), images.size(0))
        total_losses.update(total_loss.item(), images.size(0))
        TAs.update(TA.item(), images.size(0))
        pbar.set_postfix_str(f"TA {TAs.avg:.2f}%, loss {losses.avg:.2f}, total_loss {total_losses.avg:.2f}")
    return TAs.avg

def test(visual_prompt,model,loaders,epoch,args):
    pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Epo {epoch} Testing", ncols=160)    
    losses = AverageMeter()
    TAs = AverageMeter()
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        images = clip_img_preprocessing(x)
        if args.visual_prompter is not None:
            prompted_images = visual_prompt(images)
        else:
            prompted_images = images
        output = model(prompted_images)
        loss = F.cross_entropy(output,y)
        output = output.float()
        loss = loss.float()
    
        TA = accuracy(output.data, y)[0]
        losses.update(loss.item(), images.size(0))
        TAs.update(TA.item(), images.size(0))
        pbar.set_postfix_str(f"TA {TAs.avg:.2f}%, loss {losses.avg:.2f}")
    return TAs.avg



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--dataset', choices=["cifar10", "cifar100", "abide", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], default='cifar10')
    
    
    ################# attack settings ###################
    p.add_argument('--n-restarts', type=int, default=1)
    p.add_argument('--attack-iters', type=int, default=2)
    p.add_argument('--attack-eps', default=1., type=float,
                        help='attack constraint for training (default: 8/255)')
    p.add_argument('--attack_lr', default=2., type=float,
                        help='attack learning rate (default: 2./255). Note this parameter is for training only. The attack lr is always set to attack_eps / 4 when evaluating.')    
    p.add_argument('--lp', type=str, choices=['l2', 'linf'], default='linf')
    p.add_argument('--visual-prompter', type=str,default=None)
    p.add_argument('--attack-methods', type=str, choices=['pgd', 'cw'],default='pgd')

    ################# text prompting settings ###################
    p.add_argument('--cnt-prompt', type=int, default=1)

    ################# text prompting settings ###################
    p.add_argument('--batch_size', type=int,
                        default=32, help='batch size')
    p.add_argument('--lr', default=0.002, type=float,
                        help='initial learning rate')
    p.add_argument('--momentum', default=0.9, type=float, help='momentum')    
    p.add_argument('--weight_decay', default=5e-4,
                        type=float, help='weight decay')
    p.add_argument('--epochs', type=int,
                        default=50, help='epochs') 
    p.add_argument('--lamb', type=float,default=0.1)
    p.add_argument('--template', action='store_true',help='whether using a template')
    args = p.parse_args()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)
    


    exp = f"training-clip"
    save_path = os.path.join(results_path, exp, gen_folder_name(args))
    # Make Dir
    os.makedirs(save_path, exist_ok=True)
    logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    model, _ = clip.load("ViT-B/32")
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    if args.template:
        prefix = DEFAULT_TEMPLATE
    else:
        prefix = ""
    loaders, class_names = prepare_clip_data(dataset=args.dataset, data_path=data_path, preprocess=preprocess)
    model = CustomCLIP(args,class_names,model,prefix=prefix)   

    if args.visual_prompter is not None:
        visual_prompt = VisualPrompt(224, 30).to(device)
        visual_prompt = None
    else:
        visual_prompt = None
    

    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    optimizer = torch.optim.SGD(model.prompt_learner.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs) * len(loaders['train']))
    epoch = 0
    Best_SA = 0
    while epoch < args.epochs:
        epoch+=1
        train_SA = train(visual_prompt,model,loaders,epoch,optimizer,scheduler,args)
        test_SA = test(visual_prompt,model,loaders,epoch,args)
        if test_SA > Best_SA:
            Best_SA = test_SA
    print(f"best SA is {Best_SA}")

