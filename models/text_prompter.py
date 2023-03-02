import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
class Text_Prompter(nn.Module):
    def __init__(self,args,classnames,clip_model,prefix) -> None:
        device = next(clip_model.parameters()).device
        super().__init__()
        self.n_cls = len(classnames)
        n_cnt_prompt = args.cnt_prompt
        dtype = clip_model.dtype
        clip_width = clip_model.ln_final.weight.shape[0]
        prompt_embedding = torch.empty(n_cnt_prompt, clip_width, dtype=dtype).to(device)
        nn.init.normal_(prompt_embedding, std=0.02)
        self.prompt = nn.Parameter(prompt_embedding)  # to be optimized
        templates = [prefix.format(classname) for classname in classnames]
        tokenized_templates = torch.cat([clip.tokenize(p) for p in templates]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_templates).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, -1:, :])  # EOS                
        self.register_buffer("token_templates",embedding[:, 1:-n_cnt_prompt-1, :])   # templates     
        prompt_prefix = " ".join(["X"] * n_cnt_prompt)

        tokenized_prompts = [prompt_prefix +' '+ template for template in templates]
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in tokenized_prompts])


    def forward(self):

        prompt = self.prompt
        if prompt.dim() == 2:
            prompt = prompt.unsqueeze(0).expand(self.n_cls, -1, -1)
        ### start and end token
        prefix = self.token_prefix
        suffix = self.token_suffix
        token_templates = self.token_templates
        final_prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                prompt,     # (n_cls, n_cnt_prompt, dim)
                token_templates, #(n_cls,n_texts+len(class_name),dim)
                suffix,  # (n_cls, 1, dim)
            ],
            dim=1,
        )    
        return final_prompts
    
