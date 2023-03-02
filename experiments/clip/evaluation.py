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
from models.clip_new import clip_img_preprocessing
from algorithms.attack import network,attack_pgd

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--dataset', choices=["cifar10", "cifar100", "abide", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], default='cifar10')
    p.add_argument('--n-restarts', type=int, default=1)
    p.add_argument('--attack-iters', type=int, default=10)
    p.add_argument('--attack-eps', default=8., type=float,
                        help='attack constraint for training (default: 8/255)')
    p.add_argument('--attack_lr', default=2., type=float,
                        help='attack learning rate (default: 2./255). Note this parameter is for training only. The attack lr is always set to attack_eps / 4 when evaluating.')    
    p.add_argument('--lp', type=str, choices=['l2', 'linf'], default='l2')
    p.add_argument('--visual-prompter', type=str,default=None)
    p.add_argument('--attack-methods', type=str, choices=['pgd', 'cw'],default='pgd')
    args = p.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)
    


    exp = f"adv-test-clip"
    save_path = os.path.join(results_path, exp, gen_folder_name(args))
    # Make Dir
    os.makedirs(save_path, exist_ok=True)
    logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    if args.attack_methods== "pgd":
        if args.attack_iters == 2:
            args.attack_lr = args.attack_eps * 0.5 / 255
        elif args.attack_iters == 10:
            args.attack_lr = 2.0 / 255
        else:
            args.attack_lr = args.attack_lr / 255

    model, _ = clip.load("ViT-B/32")
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    templates = [DEFAULT_TEMPLATE]
    loaders, class_names = prepare_clip_data(dataset=args.dataset, data_path=data_path, preprocess=preprocess)
    txt_emb = torch.cat(get_saparate_text_embedding(class_names, templates, model))
    emb_names = np.array([f"T{i//len(class_names)} {class_names[i%len(class_names)]}" for i in range(txt_emb.size(0))])

    if args.visual_prompter is not None:
        visual_prompt = VisualPrompt(224, 30).to(device)
        visual_prompt = None
    else:
        visual_prompt = None

    pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f" Testing", ncols=100)    
    losses = AverageMeter()
    adv_losses = AverageMeter()
    RAs = AverageMeter()
    TAs = AverageMeter()
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        delta = attack_pgd(visual_prompt,model,txt_emb,x,y,eps=args.attack_eps,alpha=args.attack_lr,attack_iters=args.attack_iters,n_restarts=args.n_restarts)
        adv_images = clip_img_preprocessing(x + delta)
        images = clip_img_preprocessing(x)
        if args.visual_prompter is not None:
            prompted_adv_images = visual_prompt(adv_images)
            prompted_images = visual_prompt(images)
        else:
            prompted_adv_images = adv_images
            prompted_images = images
        adv_output = network(model,prompted_adv_images,txt_emb)
        output = network(model,prompted_images,txt_emb)
        adv_loss = F.cross_entropy(adv_output,y)
        loss = F.cross_entropy(output,y)
        adv_output = adv_output.float()
        output = output.float()
        adv_loss = adv_loss.float()
        loss = loss.float()

        TA = accuracy(output.data, y)[0]
        RA = accuracy(adv_output.data, y)[0]
        losses.update(loss.item(), adv_images.size(0))
        adv_losses.update(adv_loss.item(), adv_images.size(0))
        TAs.update(TA.item(), adv_images.size(0))
        RAs.update(RA.item(), adv_images.size(0))
        pbar.set_postfix_str(f"TA {TAs.avg:.2f}%, RA {RAs.avg:.2f}%, loss {losses.avg:.2f}, adv_loss {adv_losses.avg:.2f}")
