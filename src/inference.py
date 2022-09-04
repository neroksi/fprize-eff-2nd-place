
import pandas as pd, numpy as np

import torch
from  torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoConfig, AutoModel

from tqdm.auto import tqdm
from warnings import warn

import pickle
from pathlib import Path
import gc
from functools import partial

from utils import slugify, get_config_as_param, copy_param_to_configs as uitls_copy_param_to_configs
import configs as cfg
from models import Model
from dataset import get_special_token_ids, gen_uuid_data, InferenceDataset, collate_fn_train, WorkerInitFn

copy_param_to_configs = partial(uitls_copy_param_to_configs, cfg=cfg)


def get_max_pads(mask):
    bools = (mask[..., 1:] == 0) & (mask[..., :-1] != 0)
    max_pads = bools.to(torch.uint8).argmax(-1)

    if (max_pads == 0).any(): # There at least one full batch, no padding
        max_pads = mask.shape[-1]
    else:
        max_pads = 1 + max_pads.max().item()
    return max_pads

def load_tokenizer_and_config(config_path, tokenizer_path, is_pickle=True):
    if is_pickle:
        with open(config_path, "rb") as f:
            config = pickle.load(f)

        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
    else:
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.add_special_tokens({"additional_special_tokens": cfg.SPECIAL_TOKENS})
        config = AutoConfig.from_pretrained(config_path, trim_offsets=False, vocab_size=len(tokenizer))

    return config, tokenizer


def load_net(checkpoint_path, param):
    # other_model_params = param.get("other_model_params", {})
    param["config"].type_vocab_size = cfg.NUM_D
    net = Model(config=param["config"], use_mixup=param["use_mixup"], forward_type=param["forward_type"],
            use_token_types=param["use_token_types"], use_layer_norm=param["use_layer_norm"], from_pretrained=False,)
    net.model.resize_token_embeddings(len(param["tokenizer"]))
    
    net = net.to(cfg.DEVICE)
    
    if checkpoint_path is not None:
        net.load_state_dict(torch.load(checkpoint_path, map_location=cfg.DEVICE))
    net = net.eval()
    return net


def get_params(
    model_name,
    num_targets=None,
    maxlen=None,
    is_pickle=True,
    tokenizer_path=None,
    config_path=None,
    model_paths=None,
    models=None,
    **kwargs,
    
):

    model_name_slug = slugify(model_name)
    
    config_path = config_path or Path(f"../input/fprize-kkiller-tools/{model_name_slug}-config.pkl")
    tokenizer_path = tokenizer_path or Path(f"../input/fprize-kkiller-tools/{model_name_slug}-tokenizer.pkl")

    config, tokenizer = load_tokenizer_and_config(config_path=config_path, tokenizer_path=tokenizer_path, is_pickle=is_pickle)
    
    params = {
        "model_name": model_name,
        "model_name_slug": model_name_slug,
        "maxlen": maxlen,
        "num_targets": num_targets or cfg.NUM_TARGETS,
        "config_path": config_path,
        "tokenizer_path": tokenizer_path,
        "is_pickle": is_pickle,
        "config": config,
        "tokenizer": tokenizer,
        "model_paths": model_paths,
        "models": models,
    }
    
    params.update(kwargs)
         
    return params


@torch.no_grad()
def _predict(nets, input_ids, attention_mask, span_ids, token_type_ids, agg="mean", delete_pad=False):

    if isinstance(nets, torch.nn.Module):
        nets = [nets]
    
    assert len(nets) == 1, "Can't predict for model than 1 model when softmax/sigmoid is not applied!!!"
    
    if delete_pad:
        max_pads = get_max_pads(attention_mask)

        input_ids, attention_mask, span_ids = input_ids[..., :max_pads], attention_mask[..., :max_pads], span_ids[..., :max_pads]
    
    pred = None
    for net in nets:

        idx, o = net(input_ids=input_ids, attention_mask=attention_mask, span_ids=span_ids, token_type_ids=token_type_ids)
        
        if  agg == "max":
            pred = o if pred is None else torch.max(o, pred)
        elif agg == "mean":
            pred = o if pred is None else pred.add_(o)
        else:
            raise ValueError(f"Unknow value `{agg}` for `agg`")

    if agg == "mean":
        pred /= len(nets)

    return idx, (pred, o)
    

@torch.no_grad()
def predict_eval(net, test_data, bar=False, ret_out=False, ret_target=True, delete_pad=True, loss_func=None, ret_preds=True, agg_func=None):

    preds_list = []

    if ret_out:
        out_list = []

    if ret_target:
        target_list = []

    index_list = []

    test_data = tqdm(test_data) if bar else test_data

    if loss_func is not None:
        loss = 0.

    i_inp = -1
    for  inp in  test_data:

        i_inp += 1

        batch_ids  = inp["uuids"]
        span_ids = inp["span_ids"]
        input_ids = inp["input_ids"]
        token_type_ids = inp["token_type_ids"]
        attention_mask = inp["masks"]
        weights = inp.get("weights")
        target = inp.get("target")

        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids)
            attention_mask = torch.from_numpy(attention_mask)
            span_ids = torch.from_numpy(span_ids)
            token_type_ids = torch.from_numpy(token_type_ids)


        span_ids = span_ids.to(cfg.DEVICE)#.long()
        token_type_ids = token_type_ids.to(cfg.DEVICE)
        input_ids = input_ids.to(cfg.DEVICE)#.long()
        attention_mask = attention_mask.to(cfg.DEVICE)#.long()
            
        
        
        (idx, idy), (preds, out) = _predict(net, input_ids=input_ids, attention_mask=attention_mask,
                span_ids=span_ids, token_type_ids=token_type_ids, delete_pad=delete_pad)

        all_ids = [(batch_ids[i][0], batch_ids[i][1][j]) for  i, j in zip(idx, idy)]

        if ret_preds:
            if cfg.LOSS_NAME in cfg.SIGMOID_COMPATIBLE_LOSSES:
                preds = preds.sigmoid()
            elif cfg.LOSS_NAME in cfg.SOFTMAX_COMPATIBLE_LOSSES:
                preds = preds.softmax(-1)
                
            preds = preds.squeeze(1).cpu().numpy()

            assert np.allclose(preds.sum(-1), 1.0)


        if target is not None and (ret_target or loss_func is not None):
            

            if weights is not None:
                weights = weights[idx]

                if isinstance(weights, np.ndarray):
                    weights = torch.from_numpy(weights)
                    
                weights = weights.to(cfg.DEVICE)
            
            target = target[idx, idy]
            if isinstance(target, np.ndarray):
                target = torch.from_numpy(target)
            
            target_v2 = target.cpu().numpy()

            if not (target_v2 >= -1e-3).all():
                print(target_v2[(target_v2 < 0)][:10].round(2))
                
                raise ValueError("NEGATIVE VALUES\n\n" + str(all_ids))

            target = target.to(cfg.DEVICE)

            if loss_func is not None:
                loss += loss_func(out, target=target, weights=weights, is_filtered=True).item()

            if ret_target:
                target_list.append(target_v2)

        if ret_out:
            out_list.append(out.cpu())

        index_list.extend(all_ids)

        if ret_preds:
            preds_list.append(preds)

    
    batch_ids = index_list

    preds = np.concatenate(preds_list) if ret_preds else None

    out = torch.cat(out_list) if ret_out else None
    loss = loss / (i_inp + 1) if loss_func is not None else None
    target = np.concatenate(target_list).round().astype(np.int32) if (target is not None and ret_target) else None

    res = {"out": out, "target": target, "preds": preds, "batch_ids": batch_ids, "loss_v1": loss}
    return res



@torch.no_grad()
def predict_full_oof(net, test_data, bar=False,):

    if not isinstance(net, torch.nn.Module):
        assert len(net) == 0
        net = net[0]

    preds_list = []

    test_data = tqdm(test_data) if bar else test_data

    i_inp = -1
    for  inp in  test_data:

        i_inp += 1


        batch_ids  = inp["uuids"]
        span_ids = inp["span_ids"]
        input_ids = inp["input_ids"]
        token_type_ids = inp["token_type_ids"]
        attention_mask = inp["masks"]

        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids)
            attention_mask = torch.from_numpy(attention_mask)
            span_ids = torch.from_numpy(span_ids)
            token_type_ids = torch.from_numpy(token_type_ids)


        span_ids = span_ids.to(cfg.DEVICE)#.long()
        token_type_ids = token_type_ids.to(cfg.DEVICE)
        input_ids = input_ids.to(cfg.DEVICE)#.long()
        attention_mask = attention_mask.to(cfg.DEVICE)#.long()
            
        
        
        preds = net(input_ids=input_ids, attention_mask=attention_mask, span_ids=span_ids, token_type_ids=token_type_ids)

        if cfg.LOSS_NAME in cfg.SIGMOID_COMPATIBLE_LOSSES:
            preds = preds.sigmoid()
        elif cfg.LOSS_NAME in cfg.SOFTMAX_COMPATIBLE_LOSSES:
            preds = preds.softmax(-1)
            
        preds = preds.cpu().numpy()

        bools = (span_ids.cpu().numpy() >= 0)

        pred_tuples = [( batch_id_0, [ batch_id_1_bb for batch_id_1_bb, bb in zip(batch_id_1, b) if bb ], pred[b] )
                    for (batch_id_0, batch_id_1), b, pred in zip(batch_ids, bools, preds)]


        preds_list.extend(pred_tuples)

    return {"preds": preds_list}



def load_data(uuids, param, df, essays, bar_for_data=False, verbose=False):
    assert df is not None
    assert essays is not None

    copy_param_to_configs(param)
    
    tokenizer = param["tokenizer"]

    df = df[df["id"].isin(uuids)]

    df_iter = df.groupby("id")
    if bar_for_data:
        df_iter = tqdm(df_iter, total=df["id"].nunique())

    data = {uuid: gen_uuid_data(df_uuid, tokenizer=tokenizer, essay=essays[uuid], compute_target=False,) for uuid, df_uuid  in df_iter}

    if verbose:
        print("DECODE:", tokenizer.decode(data[uuids[0]][1], skip_special_tokens=False))
        print("DECODE:", tokenizer.decode(data[uuids[1]][1], skip_special_tokens=False))
    
    return data

@torch.no_grad()
def predict_from_param(uuids, param, data=None, df=None, essays=None, reduce="mean", bar=True, bar_for_models=False,
            bar_for_data=False, verbose=False, agg_func=None, do_full_oof=False):

    assert df is not None

    param["use_mixup"] = False
    
    copy_param_to_configs(param)
    
    tokenizer = param["tokenizer"]

    
    special_token_ids = get_special_token_ids(tokenizer)

    if data is None:
        data = load_data(
            uuids=uuids,
            param=param,
            df=df,
            essays=essays,
            bar_for_data=bar_for_data,
            verbose=verbose,
        )

    test_data = InferenceDataset(uuids=uuids, data=data, special_token_ids=special_token_ids,
        max_num_spans=param.get("max_num_spans"), maxlen=param["maxlen"], is_train=False)

    test_loader = DataLoader(test_data, batch_size=param["batch_size"], num_workers=param["num_workers"], shuffle=False,
    collate_fn=partial(collate_fn_train, pad_token_id=special_token_ids["pad_token_id"], is_psl=cfg.IS_PSL),
    worker_init_fn=WorkerInitFn(get_config_as_param(cfg)),
    )

    if reduce == "mean":
        preds = None
    else:
        assert reduce is None
        results = []
    
    models_iter = param["model_paths"]
    if bar:
        models_iter = tqdm(models_iter)

    for model_path in models_iter:
        model = load_net(model_path, param)
        if do_full_oof:
            res = predict_full_oof(model, test_data=test_loader, bar=bar_for_models)
        else:
            res = predict_eval([model], test_loader, bar=bar_for_models, ret_preds=True, delete_pad=True, ret_out=False, ret_target=False)
        
        if reduce == "mean":
            if preds is None:
                preds = res["preds"]
            else:
                preds += res["preds"]
        else:
            results.append(res["preds"])

        del model

        gc.collect()
        torch.cuda.empty_cache()

    if do_full_oof:
        return preds
    
    elif (reduce == "mean"):
        preds /= len(param["model_paths"])
    else:
        return results
    
    sub = pd.DataFrame(res["batch_ids"], columns=["id", "discourse_id"])
    sub[[cfg.ID2CLASS[i] for i in range(preds.shape[1])]] = preds

    temp = df.loc[df["id"].isin(uuids), ["id", "discourse_id"]].copy()
    sub = temp.merge(sub, on=["id", "discourse_id"], how="left")
    
    return sub