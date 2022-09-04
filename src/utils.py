import os
import torch
import numpy as np, random
import re
from pathlib import Path

import configs  as cfg

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def slugify(s):
    return re.sub(r"[^\w\-_]", "_", s)


def get_config_as_param(configs):
    config_ok_types = (
        bool,
        int,
        float,
        dict,
        str,
        tuple,
        list,
        np.ndarray,
        Path,
        torch.device
    )

    config_dict = {
        key: getattr(configs, key) for key in configs.__dir__()
    }
    
    config_dict = {
        key : val for key, val in config_dict.items() if isinstance(val, config_ok_types) and not key.startswith("__")
    }
    
    return config_dict


def copy_param_to_configs(param, cfg):
    # cfg = param["configs"]
    for attr, val in param.items():
        cfg_attr = None
        if hasattr(cfg, attr):
            cfg_attr = attr
        elif hasattr(cfg, attr.upper()):
            cfg_attr = attr.upper()

        if cfg_attr is not None:
            setattr(cfg, cfg_attr, val)


def get_special_token_ids(tokenizer, add_custom_tokens=True):
    pad_token_id=tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    mask_token_id=tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.unk_token_id

    cls_token_id = tokenizer.cls_token_id

    sep_token_id = tokenizer.sep_token_id
    if sep_token_id is None:
        sep_token_id = tokenizer.eos_token_id

    assert pad_token_id >= 0
    assert mask_token_id >= 0
    assert sep_token_id >= 0


    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    if bos_token_id is None:
        bos_token_id = cls_token_id
        if bos_token_id is None:
            bos_token_id = sep_token_id
    
    if eos_token_id is None:
        eos_token_id = sep_token_id
        if eos_token_id is None:
            eos_token_id = cls_token_id

    special_token_ids = dict(
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
        sep_token_id=sep_token_id,
        cls_token_id=cls_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )

    if add_custom_tokens:
        for special_token in cfg.SPECIAL_TOKENS:
            special_token_id = tokenizer(special_token , add_special_tokens=False)["input_ids"]
            assert len(special_token_id) == 1, (special_token, special_token_id)
            special_token_id = special_token_id[0]

            special_token_ids[special_token] = special_token_id

        cfg.D_SPAN_TOKEN_IDS = {special_token: special_token_ids[special_token] for special_token in cfg.D_SPAN_TOKENS}

    return special_token_ids