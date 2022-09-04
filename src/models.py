from turtle import forward
import torch
from torch import nn, optim
import numpy as np
import re
from  warnings import warn

from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModelForSequenceClassification

# https://github.com/huggingface/transformers/issues/9919
from torch.utils.checkpoint import checkpoint

import configs as cfg
from dataset import get_special_token_ids


def put_to_ndim(x, ndim):
    for _ in range(ndim - 1):
        x = x[:, None]
    return x


def get_model_info(model, tokenizer=None):
    classifier_name = None
    for k in ["fc", "classification_head", "classifier"]:
        if hasattr(model, k):
            classifier_name = k
            break
    # assert classifier_name is not None, "Unable to get classifier"

    in_features = None
    if classifier_name is not None:
        try:
            in_features =  getattr(model, classifier_name).in_features
        except AttributeError:
            pass

        setattr(model, classifier_name, nn.Identity())

    if tokenizer is not None:
        sp_ids = get_special_token_ids(tokenizer, add_custom_tokens=False)
        
        if not sp_ids["bos_token_id"] is None:
            x = [sp_ids["bos_token_id"]]
        elif not sp_ids["cls_token_id"] is None:
            x = [sp_ids["cls_token_id"]]

        x += [10, 11, 12, 13, 14]

        if not sp_ids["eos_token_id"] is None:
            x += [sp_ids["eos_token_id"]]
        elif not sp_ids["cls_token_id"] is None:
            x += [sp_ids["cls_token_id"]]

        x = torch.tensor([x], dtype=torch.long)
    else:
        x = torch.arange(10, dtype=torch.long)[None]

    out = model(x)
    keys = out.keys()

    out_name = None
    for k in ["encoder_last_hidden_state", "last_hidden_state", "logits"]:
        if k in keys:
            out_name = k
            break
    assert out_name is not None, "Can't get out_name"
    
    if in_features is None:
        in_features = out[out_name].shape[-1]

    info = {
        "out_name": out_name,
        "in_features": in_features,
        "classifier_name": classifier_name,
    }

    return info


def load_model(model_class=None, model=None, checkpoint_path=None, verbose=False, remove=None, match=None, strict=False, **kwargs):
    DEVICE = torch.device("cpu")
    if model is None:
        assert model_class is not None

        model = model_class(**kwargs)

    model = model.to(DEVICE)
    
    if checkpoint_path is not None:
        weights_dict = torch.load(checkpoint_path, map_location=DEVICE)

        for key in list(weights_dict):
            if match and not re.search(match, key):
                weights_dict.pop(key)
            elif remove:
                key2 = re.sub(remove, "", key)
                weights_dict[key2] = weights_dict.pop(key)

        model.load_state_dict(weights_dict, strict=strict)

        if verbose:
            print(f"Weights loaded from: '{checkpoint_path}'")

    model = model.eval()
    return model

def get_model(model_name=None, task="token_classification", num_targets=None, config=None, tokenizer=None,
        from_pretrained=True, use_token_types=False):
    num_targets =  num_targets or cfg.NUM_TARGETS or 1
    task = task.lower()

    model_name = model_name or getattr(config, "_model_name", "") or getattr(
                        config, "name_or_path", "") or getattr(config, "_name_or_path", "")
        
    if "token" in task:
        model_instance = AutoModelForTokenClassification
    elif "sequence" in task:
        model_instance = AutoModelForSequenceClassification

    if not from_pretrained:  
        assert config is not None

        if use_token_types:
            assert config.type_vocab_size == cfg.NUM_D

        if hasattr(model_instance, "from_config"):
            model = model_instance.from_config(config)
        else:
            model = model_instance.from_pretrained(model_name,  config=config)

        # tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if config is None:
            if use_token_types:
                config = AutoConfig.from_pretrained(model_name, type_vocab_size=cfg.NUM_D)
            else:
                config = AutoConfig.from_pretrained(model_name)
        elif use_token_types:
            assert config.type_vocab_size == cfg.NUM_D

        model = model_instance.from_pretrained(model_name, config=config)


    info = get_model_info(model, tokenizer=tokenizer)
    
    return info, config, tokenizer, model


class MultiSampleDropout(nn.Module):
    def  __init__(self, n_drops=None, p_drops=None):
        super().__init__()

        self.q_0 = 0.10
        self.q_1 = 0.50 - self.q_0

        self.n_drops = n_drops or cfg.N_DROPS
        self.p_drops = (p_drops or cfg.P_DROPS) or self.gen_dropout_probas()

        self.drop_modules = nn.ModuleList([nn.Dropout(p_drop) for p_drop in self.p_drops])

       
    def gen_dropout_probas(self):
        assert self.n_drops >= 0

        if self.n_drops == 0:
            return []
        elif self.n_drops == 1:
            return [self.q_0]
        else:
            return [ self.q_0 + self.q_1 * n / (self.n_drops -1) for n in range(self.n_drops)]
    
    def forward(self, x):
        if not self.training or not self.n_drops:
            return x[:, None]

        res = []
        for drop_module in self.drop_modules:
            res.append(drop_module(x))
        res = torch.stack(res, dim=1)
        return res


class AttentionHead(nn.Module):
    def __init__(self, h_size, hidden_dim=512):
        super().__init__()
        self.W = nn.Linear(h_size, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        
    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector


class BaseModel(nn.Module):
    def __init__(self, model_name=None, num_targets=None, config=None, checkpoint_path=None, use_mixup=None,
        forward_type=None, task=None, out_name=None, from_pretrained=True, load_strict=True,
        vocab_size_before_load=None, resize_before_load=True, use_token_types=True, use_layer_norm=False, use_gradient_checkpointing=False):
        super().__init__()

        assert use_mixup is not None
        assert forward_type is not None

        assert forward_type in ["forward_1span", "forward_full_oof", "forward_avgpool"]

        task = task or "token_classification"

        self.model_name = model_name or cfg.MODEL_NAME
        self.task = task
        self.vocab_size_before_load = vocab_size_before_load
        self.use_token_types = use_token_types
        self.use_layer_norm = use_layer_norm

        self.use_mixup = use_mixup
        self.forward_type = forward_type
        
        
        model_info, config, tokenizer, model = get_model(model_name=model_name, task=self.task, config=config,
                    num_targets=1, from_pretrained=from_pretrained, use_token_types=use_token_types)

        self.model_info = model_info
        self.out_name =  self.model_info["out_name"]
        self.in_features =  self.model_info["in_features"]
        self.classifier_name =  self.model_info["classifier_name"]
        
        
        assert self.out_name is not None, "Can't get `out_name` automatically"
        assert self.in_features is not None, "Can't get in_features"

        self.num_targets = cfg.NUM_TARGETS if num_targets is None else num_targets
        
        if "sequence" in task:
            setattr(model, self.classifier_name, nn.Identity())

        
        if use_gradient_checkpointing:
            # gradient checkpointing
            model.gradient_checkpointing_enable()
        
        self.config = config
        self.tokenizer = tokenizer
        self.model = model

        if checkpoint_path is not None:
            if resize_before_load:
                vocab_size_0 = vocab_size_before_load or (len(tokenizer) if tokenizer is not None else None)
                if vocab_size_0:
                    model.resize_token_embeddings(vocab_size_0)

            self.model = load_model(model=self.model, checkpoint_path=checkpoint_path, verbose=True,
            remove=r"^model\.", match=r"^model\.", strict=load_strict)

        if tokenizer is not None:
            tokenizer.add_special_tokens({"additional_special_tokens": cfg.SPECIAL_TOKENS})
            model.resize_token_embeddings(len(tokenizer))
            config.vocab_size = len(tokenizer)

        
        if self.use_layer_norm:
            in_features = self.in_features

            self.layer_norm = nn.LayerNorm(in_features)


        self.fc = nn.Linear(self.in_features, self.num_targets)


        

class Model(BaseModel):
    _different_lr_s = [r"^(?!model).*$"]

    def __init__(self, model_name=None, num_targets=None, config=None, checkpoint_path=None, use_mixup=None, forward_type=None,
        task=None, from_pretrained=True, load_strict=True, vocab_size_before_load=None, resize_before_load=True, use_token_types=True,
        use_layer_norm=False, use_gradient_checkpointing=False):

        super().__init__(model_name=model_name, num_targets=num_targets, config=config, checkpoint_path=checkpoint_path,
        use_mixup=use_mixup, forward_type=forward_type, task=task, from_pretrained=from_pretrained, load_strict=load_strict,
        vocab_size_before_load=vocab_size_before_load, resize_before_load=resize_before_load, use_token_types=use_token_types,
        use_layer_norm=use_layer_norm, use_gradient_checkpointing=use_gradient_checkpointing,
        )

        self.ms_dropout = MultiSampleDropout()

    @staticmethod
    def pool_with_mask(x, mask, dim=1):
        assert x.ndim == mask.ndim 

        return torch.sum( x * (mask != 0), dim=dim) / (1e-3 + torch.sum( (mask != 0).float(), dim=dim))

    def forward_features(self, *, input_ids, attention_mask, span_ids=None, token_type_ids=None, index=None, alpha=None):
        assert index is None, alpha is None
        
        if self.use_token_types:
            assert token_type_ids is not None
            x = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[self.out_name]
        else:
            x = self.model(input_ids=input_ids, attention_mask=attention_mask)[self.out_name]
        
        if self.use_layer_norm:
            x = self.layer_norm(x)
            
        return x
    
    def forward_features_v2(self, *, input_ids, attention_mask, span_ids=None, token_type_ids=None, index=None, alpha=None, ret_model_output=False):
        assert index is None, alpha is None
        
        if self.use_token_types:
            assert token_type_ids is not None
            x_model = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[self.out_name]
        else:
            x_model = self.model(input_ids=input_ids, attention_mask=attention_mask)[self.out_name]
        
        x = x_model
        all_span_types = sorted(torch.unique(span_ids[span_ids >= 0]).cpu().numpy())
        sample_ids = []
        x_all = []
        mask_all = []
        for span_type in all_span_types:
            span_type_mask = (span_ids == span_type)
            span_type_sel = span_type_mask.any(-1)
            span_type_index = torch.nonzero(span_type_sel).squeeze(-1)

            x_all.append(x[span_type_index])
            mask_all.append(span_type_mask[span_type_index])

            sample_ids.extend([( idx, torch.nonzero(span_type_mask[idx]).squeeze(-1)[0].item() ) for idx in span_type_index.cpu().numpy()])
        
        x = torch.cat(x_all)
        mask = torch.cat(mask_all)

        x = self.pool_with_mask(x, mask=mask[:, :, None])

        if self.use_layer_norm:
            x = self.layer_norm(x)

        idx = np.array([el[0] for el in sample_ids])
        idy = np.array([el[1] for el in sample_ids])
            
        return ((idx, idy), x, x_model) if ret_model_output else ((idx, idy), x,)


    def forward_full_oof(self, *, input_ids, attention_mask, span_ids=None, token_type_ids=None, index=None, alpha=None):
        assert index is None, alpha is None

        x = self.forward_features(input_ids=input_ids, attention_mask=attention_mask, span_ids=span_ids,
                    token_type_ids=token_type_ids, index=index, alpha=alpha)

        x = self.fc(x)

        return x


    def forward_1span(self, *, input_ids, attention_mask, span_ids=None, token_type_ids=None, index=None, alpha=None):
        assert index is None, alpha is None

        x = self.forward_features(input_ids=input_ids, attention_mask=attention_mask, span_ids=span_ids,
                    token_type_ids=token_type_ids, index=index, alpha=alpha)


        d_span_token_ids = torch.tensor(list(cfg.D_SPAN_TOKEN_IDS.values()), device=input_ids.device, dtype=input_ids.dtype)
        bools = (span_ids >= 0) & torch.isin(input_ids, d_span_token_ids, assume_unique=False)
        i, j = torch.nonzero(bools).T

        x = x[i, j]
        x = self.ms_dropout(x)
        x = self.fc(x)

        i, j = i.cpu().numpy(), j.cpu().numpy()

        return (i, j), x

    
    def forward_avgpool(self, *, input_ids, attention_mask, span_ids=None, token_type_ids=None, index=None, alpha=None):
        assert index is None, alpha is None

        sample_ids, x = self.forward_features_v2(input_ids=input_ids, attention_mask=attention_mask, span_ids=span_ids,
                    token_type_ids=token_type_ids, index=index, alpha=alpha)

        x = self.ms_dropout(x)
        x = self.fc(x)

        return sample_ids, x

    
    def forward(self, *, input_ids, attention_mask, span_ids=None, token_type_ids=None, index=None, alpha=None):

        forward = getattr(self, self.forward_type)
        return forward(input_ids=input_ids, attention_mask=attention_mask, span_ids=span_ids,
                    token_type_ids=token_type_ids, index=index, alpha=alpha)


class GenericLoss(nn.Module):
    def __init__(self, num_iters, loss, apply_neg=False):
        super().__init__()

        self.num_iters = num_iters
        self._loss = loss
        self.apply_neg = apply_neg

        self.num_calls = 0
    
    def get_loss(self, num_calls=None):
        raise NotImplementedError

    def forward(self, out, target, weights=None, is_filtered=False):

        use_weights =  cfg.USE_SAMPLE_WEIGHTS or (weights is not None)

        if not is_filtered:
            nb_msd = out.size(1)
            target = torch.stack([target]*nb_msd,  dim=1)

            if use_weights:
                nrepeats = ( 1, *(target.shape[1:-1] if cfg.IS_PSL else target.shape[1:]) )
                new_shape = (len(weights), *nrepeats[1:])
                weights = weights.repeat_interleave(np.prod(nrepeats)).view(new_shape)

            bools = (target >= 0)
            if cfg.IS_PSL:
                bools = bools.all(-1)
            target = target[bools]
            out = out[bools]

            if use_weights:
                weights = weights[bools]
            
                assert len(weights) == len(target) == len(out)

        out = out.view(-1, out.shape[-1])
        target = target.view(-1, target.shape[-1]) if cfg.IS_PSL else target.view(-1)

        if use_weights:
            weights = weights.view(-1)

            assert len(out) == len(target) == len(weights), (len(out), len(target), len(weights))


        loss = self.get_loss()
        l = loss(out, target)

        if cfg.USE_SAMPLE_WEIGHTS:
            l = torch.mean(weights * l)


        if self.apply_neg:
            l = -l

        self.num_calls += 1

        return l


class CrossEntropyLoss(GenericLoss):
    def __init__(self, num_iters):
        super().__init__(
            num_iters=num_iters,
            loss=nn.CrossEntropyLoss(reduction="none") if cfg.USE_SAMPLE_WEIGHTS else nn.CrossEntropyLoss(),
            apply_neg=False,
        )

    def get_loss(self, num_calls=None):
        return self._loss


LOSS_MAP = {
    "ce": CrossEntropyLoss,
}


def get_loss(num_iters):
    return LOSS_MAP[cfg.LOSS_NAME](num_iters)

class DecreaseIncreaseSinLR:
    def __init__(self, optimizer, T_max, eta_min=1e-7, warmup_ratio=0.20, gamma=3):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.eta_max = [pgroup["lr"] for pgroup in optimizer.param_groups]
        self.gamma = gamma or cfg.SCHEDULER_GAMMA
        self.T_cur = -1
        self.warmup_ratio = warmup_ratio or cfg.WARMUP_RATIO
        
        self.T_max1 = self.T_max * self.warmup_ratio
        self.T_max2 = self.T_max -  self.T_max1
        
        self.step()
    
    def iter_q(self):
        self.T_cur += 1
        if self.T_cur <= self.T_max1:
            q = np.sin( 0.5 * self.T_cur * np.pi / self.T_max1 )
        else:
            q = np.sin( 0.5* np.pi + 0.5*(self.T_cur - self.T_max1) * np.pi / self.T_max2 )
        
        q = q ** self.gamma
        return q
    
    def iter_lr(self):
        q = self.iter_q()
        
        lr = []
        
        for eta_max in self.eta_max:
            lr.append(self.eta_min + q * (eta_max - self.eta_min))
        
        return lr

    def step(self):
        for pgroup, lr in zip(self.optimizer.param_groups, self.iter_lr()):
            pgroup["lr"] = lr
            
    def get_lr(self):
        lr = []
        for pgroup in self.optimizer.param_groups:
            lr.append(pgroup["lr"])
        return lr