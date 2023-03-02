import clip
import torch
import torch.nn as nn
from models.text_prompter import Text_Prompter,TextEncoder
IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)

mu = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).cuda()
std = torch.tensor(IMAGENET_STD).view(3, 1, 1).cuda()

def normalize(X):
    return (X - mu) / std

def clip_img_preprocessing(X):
    img_size = 224
    X = torch.nn.functional.interpolate(X, size=(img_size, img_size), mode='bicubic')
    X = normalize(X)
    return X

def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()
    return logits_per_x1, logits_per_x2

class CustomCLIP(nn.Module):
    def __init__(self, args, classnames, clip_model,prefix):
        super().__init__()
        self.prompt_learner = Text_Prompter(args, classnames, clip_model,prefix)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits