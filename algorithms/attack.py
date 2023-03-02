import torch
import torch.nn.functional as F
from models.clip_new import clip_img_preprocessing
lower_limit, upper_limit = 0, 1

def sign(grad):
    grad_sign = torch.sign(grad)
    return grad_sign

def network(model,x,text_emb):
    x_emb = model.encode_image(x)
    x_emb = x_emb / x_emb.norm(dim=-1,keepdim=True)
    logits = model.logit_scale.exp() * x_emb @ text_emb.t()
    return logits

def clamp(X, l, u, cuda=True):
    if type(l) is not torch.Tensor:
        if cuda:
            l = torch.cuda.FloatTensor(1).fill_(l)
        else:
            l = torch.FloatTensor(1).fill_(l)
    if type(u) is not torch.Tensor:
        if cuda:
            u = torch.cuda.FloatTensor(1).fill_(u)
        else:
            u = torch.FloatTensor(1).fill_(u)
    return torch.max(torch.min(X, u), l)

def attack_pgd(prompter, model,text_embedding, X, y,eps,alpha,attack_iters,n_restarts,rs = True, verbose=False,
               linf_proj=True, l2_proj=False, l2_grad_update=False, cuda=True):
    if n_restarts > 1 and not rs:
        raise ValueError('no random step and n_restarts > 1!')
    max_loss = torch.zeros(y.shape[0])
    max_delta = torch.zeros_like(X)
    if cuda:
        max_loss, max_delta = max_loss.cuda(), max_delta.cuda()
    for i_restart in range(n_restarts):
        delta = torch.zeros_like(X)
        if cuda:
            delta = delta.cuda()
        if attack_iters == 0:
            return delta.detach()
        if rs:
            delta.uniform_(-eps, eps)

        delta.requires_grad = True
        for _ in range(attack_iters):
            ##### image transformations from clip #####
            _images = clip_img_preprocessing(X + delta)
            #### visual prompting ####
            if prompter is not None:
                prompted_images = prompter(_images)
            else:
                prompted_images = _images

            output = network(model,prompted_images,text_embedding)
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()

            ### sign or l2 update ####
            if not l2_grad_update:
                delta.data = delta + alpha * sign(grad)
            else:
                delta.data = delta + alpha * grad / (grad ** 2).sum([1, 2, 3], keepdim=True) ** 0.5

            ##### linf or l2 constraint #####
            if linf_proj:
                delta.data = clamp(delta.data, -eps, eps, cuda)
            if l2_proj:
                delta_norms = (delta.data ** 2).sum([1, 2, 3], keepdim=True) ** 0.5
                delta.data = eps * delta.data / torch.max(eps * torch.ones_like(delta_norms), delta_norms)
            ##### clamp data from 0 to 1 ####
            delta.data = clamp(X + delta.data, 0, 1, cuda) - X
            delta.grad.zero_()

        with torch.no_grad():
            ##### image transformations from clip #####
            _images = clip_img_preprocessing(X + delta)
            #### visual prompting ####
            if prompter is not None:
                prompted_images = prompter(_images)
            else:
                prompted_images = _images

            output = network(model,prompted_images,text_embedding)
            all_loss = F.cross_entropy(output, y, reduction='none')  # .detach()  # prevents a memory leak
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]

            max_loss = torch.max(max_loss, all_loss)
            if verbose:  # and n_restarts > 1:
                print('Restart #{}: best loss {:.3f}'.format(i_restart, max_loss.mean()))
    max_delta = clamp(X + max_delta, 0, 1, cuda) - X
    return max_delta    
