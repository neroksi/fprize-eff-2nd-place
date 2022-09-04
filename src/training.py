
import shutil
import numpy as np, pandas as pd
from pathlib import Path

import torch
from torch import nn, optim
from  torch.utils.data import DataLoader
from  torch.nn.utils import clip_grad_norm_
from torch.cuda import amp

from transformers import AdamW, get_cosine_schedule_with_warmup

from matplotlib import pyplot as plt

import os, random, gc
import re, time, json, pickle

from sklearn.model_selection import KFold, GroupKFold

from tqdm.notebook import tqdm

from collections import defaultdict
from warnings import warn
from functools import partial
from copy import deepcopy

from dataset import Dataset, collate_fn_train, WorkerInitFn
from models  import get_loss, Model, load_model as models_load_model, DecreaseIncreaseSinLR
import configs as  cfg
from utils import slugify, seed_everything, get_config_as_param
from inference import predict_eval, get_max_pads


average = "macro"
multi_class = "ovo" # "ovr"
zero_division = 0



def disable_tokenizer_parallelism():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_cv_score(model_path):
    log_paths = sorted(Path(model_path).glob("*.json"))
    scores = {}
    for path in log_paths:
        with open(path) as f:
            log = json.load(f)
            fold = int(str(path).split("_fold")[1][0])
            scores[f"fold_{fold}"] = max([l["f1_val"] for l in log])
            
    scores["mean_f1"] = sum(scores.values())/len(scores)
    
    return scores


def one_step(inp, net, criterion, optimizer, scheduler=None, scaler=None):

    span_ids = inp["span_ids"].to(cfg.DEVICE)
    token_type_ids = inp["token_type_ids"].to(cfg.DEVICE)
    input_ids = inp["input_ids"].to(cfg.DEVICE)
    attention_mask = inp["masks"].to(cfg.DEVICE)
    weights = inp["weights"].to(cfg.DEVICE) if cfg.USE_SAMPLE_WEIGHTS else None
    target = inp["target"].to(cfg.DEVICE)


    if cfg.REMOVE_LEADING_PADS:
        max_pads = get_max_pads(attention_mask)

        span_ids = span_ids[..., :max_pads]
        token_type_ids = token_type_ids[..., :max_pads]
        input_ids = input_ids[..., :max_pads]
        attention_mask = attention_mask[..., :max_pads]
        target = target[..., :max_pads, :] if cfg.IS_PSL else target[..., :max_pads]
        
    assert not net.use_mixup

    optimizer.zero_grad(set_to_none=True)

    use_amp = scaler is not None

    with amp.autocast(use_amp):
        (i, j), out = net(span_ids=span_ids, token_type_ids=token_type_ids, input_ids=input_ids, attention_mask=attention_mask)

        loss = criterion(
            out,
            target=target[i, j],
            weights=weights[i] if cfg.USE_SAMPLE_WEIGHTS else None,
        )

    if use_amp:
        scaler.scale(loss).backward()

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        clip_grad_norm_(net.parameters(), cfg.CLIP_GRAD_NORM)

        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    with torch.no_grad():
      
        l = loss.item()
        if np.isnan(l):
            l = 0.

    if  scheduler is not None:
        scheduler.step()

    metrics = {
        "loss": l,
    }

    return metrics


@torch.no_grad()
def evaluate(net, criterion, val_loader):
    net.eval()

    val_loader = tqdm(val_loader, leave = False, total=len(val_loader))

    res = predict_eval(net, val_loader, ret_out=False, ret_target=True, loss_func=criterion)

    preds = res["preds"]
    target_v2 = res["target"]

    oof = {
        "batch_ids": res["batch_ids"],
        "out": preds,
        "target": target_v2,
    }

    metrics = {
    }

    target_v2 = torch.from_numpy(target_v2.argmax(-1) if cfg.IS_PSL else target_v2.astype(np.int64) )
    metrics.update(
        loss_v1=res["loss_v1"],
        loss_v2=torch.nn.NLLLoss()( torch.from_numpy(np.log(np.clip(preds, 1e-5, 1-1e-5))), target_v2 ).item()
    )

    return oof, metrics


def finalize_epoch(*, net, criterion, val_loader, icount, train_metrics, **kwargs):

    metrics = deepcopy(train_metrics)
    div_factor = max(1, icount)
    for key in list(metrics):
         metrics[key] /= div_factor

    oof, metrics_val = evaluate(net, criterion, val_loader)

    net.train() # Important !!!

    for key, val in metrics_val.items():
        metrics[f"{key}_val"] = val

    
    print_and_save(
        oof=oof,
        metrics=metrics,
        net=net,
        criterion=criterion,
        **kwargs,
    )

    return oof, metrics

def one_epoch(*, net, criterion, optimizer, scheduler, train_loader, val_loader, schedule_each_step=False,
    scaler=None, save_each=None, **kwargs):
    net.train()
    icount = 0
    num_iter = len(train_loader)
    train_loader = tqdm(train_loader, leave = False)
    epoch_bar = train_loader

    metrics_format_dict = dict(
        loss="{loss:.6f}",
    )

    metrics = defaultdict(int)

    if save_each is not None:
        save_step = save_each*num_iter
        save_points = np.arange(save_step, num_iter, save_step).round().astype(int)
        # print("SAVE POINTS:", num_iter, save_step, save_points)
    else:
        save_points = []

    for step, inp in  enumerate(epoch_bar):

        _metrics = one_step(inp, net, criterion, optimizer, scaler=scaler)

        if schedule_each_step:
            scheduler.step()

        for key, val in _metrics.items():
            metrics[key] += val

        icount += 1
          
        if hasattr(epoch_bar, "set_postfix") and not icount%10:
            metrics_normalized = {key: val / icount for key, val in metrics.items()}
            metrics_formated = {
                key: val.format(**metrics_normalized) for key,val in metrics_format_dict.items()
            }
            
            epoch_bar.set_postfix(**metrics_formated)

        if step in save_points:
            oof, metrics_temp = finalize_epoch(
                net=net,
                criterion=criterion,
                val_loader=val_loader,
                icount=icount,
                train_metrics=metrics,
                optimizer=optimizer,
                scheduler=scheduler,
                **kwargs,
            )

            icount = 0
            # metrics = {k: 0 for k in metrics}
            metrics = defaultdict(int)

            torch.cuda.empty_cache()

        
    if not schedule_each_step:
        scheduler.step()

    metrics = dict(metrics)

    oof, metrics = finalize_epoch(
        net=net,
        criterion=criterion,
        val_loader=val_loader,
        icount=icount,
        train_metrics=metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        **kwargs,
    )
    torch.cuda.empty_cache()
    return oof, metrics



class AutoSave:
    def __init__(self, top_k=1, metric="f1", mode="max", root=None, name="ckpt"):
        self.top_k = top_k
        self.logs = []
        self.metric = metric
        self.mode = mode
        self.root = Path(root or cfg.MODEL_ROOT)
        assert self.root.exists()
        self.name = name

        self.top_models = []
        self.top_metrics = []

        self.slug_regex = r"[^\w_\-\.]"
        self.oof_suffix = "oof"

        self._log_path = None

        self.mult = 1 if mode == "max" else -1

    
    def slugify(self, s):
        return re.sub(self.slug_regex, "", s)

    def log(self, model, metrics, oof=None):
        metric = self.mult * metrics[self.metric]
        rank = self.rank(metric)

        self.top_metrics.insert(rank+1, metric)
        if len(self.top_metrics) > self.top_k:
            self.top_metrics.pop(0)

        self.logs.append(metrics)
        self.save(model, metrics[self.metric], rank, metrics["epoch"], oof=oof)

    
    def oof_path_from_model_path(self, model_path):
        oof_name = model_path.parent / "{}_{}.pkl".format(
            model_path.stem, self.oof_suffix
        )
        return oof_name
        

    def save(self, model, metric, rank, epoch, oof=None):
        t = time.strftime("%Y%m%d%H%M%S")
        name = "{}_epoch_{:02d}_{}_{:.04f}_{}".format(self.name, epoch, self.metric, metric, t)
        name = self.slugify(name) + ".pth"
        path = self.root.joinpath(name)

        old_model = None
        self.top_models.insert(rank+1, name)
        if len(self.top_models) > self.top_k:
            old_model = self.root.joinpath(self.top_models[0])
            self.top_models.pop(0)      

        torch.save(model.state_dict(), path.as_posix())


        if oof is not None:
            with  self.oof_path_from_model_path(path).open(mode="wb") as f:
                pickle.dump(oof, f)

        if old_model is not None:
            old_model.unlink()
            old_oof = self.oof_path_from_model_path(old_model)
            if old_oof.exists():
                old_oof.unlink()

        self.to_json()

    def copy(self, root, rank=None):
        if not len(self.top_models):
            warn("Nothing to copy!")
            return
        rank = -1 if rank is None else rank

        root = Path(root)

        name = self.top_models[rank]
        old_path = self.root.joinpath(name)
        new_path = root.joinpath(name)

        shutil.copy(str(old_path), str(new_path))

        old_oof_path = self.oof_path_from_model_path(old_path)

        if old_oof_path.exists():
            new_oof_path = self.oof_path_from_model_path(new_path)

            shutil.copy(str(old_oof_path), str(new_oof_path))
        
        new_log_path = root.joinpath(self.log_path.name)

        shutil.copy(str(self.log_path), str(new_log_path))


    def rank(self, val):
        r = -1
        for top_val in self.top_metrics:
            if val <= top_val:
                return r
            r += 1

        return r

    @property
    def log_path(self):
        if self._log_path is  None:
            t = time.strftime("%Y%m%d%H%M%S")
            name = "{}_{}_logs".format(self.name, t)
            name =  self.slugify(name) + ".json"
            self._log_path = self.root.joinpath(name)

        return self._log_path

    
    def to_json(self):
        with self.log_path.open(mode="w") as f:
            data = {
                "logs": self.logs,
                "params": get_config_as_param(cfg),
            }
            json.dump(data, f, indent=2, default=str)


def fetch_optimizer(net):
    return optim.AdamW(get_param_groups(net), lr=cfg.OPTIMIZER_LR, weight_decay=cfg.OPTIMIZER_WEIGHT_DECAY)


def fetch_scheduler(optimizer, num_train_steps):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_train_steps),
        num_training_steps=num_train_steps,
        num_cycles=1,
        last_epoch=-1,
    )
    return scheduler
    

def print_and_save(*, fold, oof, metrics, epoch, net, optimizer, criterion, scheduler, val_set, train_set, epochs_bar, do_save, saver):

    metrics["epoch"] = epoch
    metrics["learning_rates"] = [pgroup["lr"] for pgroup in optimizer.param_groups]

    oof["val_set"] = val_set
    oof["train_set"] = train_set
    oof["fold"] = fold
    oof["criterion"] = criterion.__class__.__name__
    oof["model"] = net.__class__.__name__
    oof["scheduler"] = scheduler.__class__.__name__

    metrics_format_dict = dict(
        loss="({loss:.6f}, {loss_v1_val:.6f})",
    )

    metrics_print_format = "[{epoch:02d}] loss: {loss}"

    for key in metrics:
        if (key.startswith("l_") or key.startswith("loss_v2_")  ) and key.endswith("_val"):
            key2 = key
            metrics_format_dict[key] = f"{{{key}:.3f}}"
            metrics_print_format += f" {key2}: {{{key}}}"

    metrics_formated = {key: val.format(**metrics) for key, val in metrics_format_dict.items()}

    epochs_bar.set_postfix(**metrics_formated)

    print(
        metrics_print_format.format(epoch=epoch, **metrics_formated)
    )

    if do_save:
        saver.log(net, metrics, oof=oof)


def get_param_groups(net, remove_regex=None):

    params_1, params_2, params_2_names = [], [], []
    for param_name, param in net.named_parameters():
        if not param.requires_grad:
            continue

        param_name = param_name.lower()
        
        clean_param_name = param_name if remove_regex is None else re.sub(remove_regex, "", param_name)

        use_different_lr = False
        for layer_regex in net._different_lr_s:
            if re.search(layer_regex, clean_param_name):
                use_different_lr = True
                break

        if  use_different_lr:
            # print(param_name)
            params_2.append(param)
            params_2_names.append(param_name)
        else:
            params_1.append(param)

    if len(params_2):
        params = [
            {
                "params": params_1,
                "lr":  cfg.OPTIMIZER_LR,
            },

            {
                "params": params_2,
                "lr":  cfg.OPTIMIZER_LR * cfg.INTER_GROUP_LR_SCALE,
            },
        ]

        print("LR is different from Main_LR for these params: {}".format(params_2_names))
        print("Main LR: {}   LR: {}".format(cfg.OPTIMIZER_LR, params[1]["lr"]))
    else:
        params = params_1

    return params


class FoldSamplesSelector:
    def __init__(self, true_uuid_fold_maps, psl_uuid_fold_maps, include_psl_in_val=False):
        self.true_uuid_fold_maps = true_uuid_fold_maps
        self.psl_uuid_fold_maps = psl_uuid_fold_maps

        self.include_psl_in_val = include_psl_in_val
        
    
    @property
    def uuids(self):
        uuids = self.true_uuid_fold_maps.copy()
        uuids.update(self.psl_uuid_fold_maps)
        return uuids
        
    def is_uuid_and_fold_ok_for_train(self, uuid, active_fold):
        try:
            return self.true_uuid_fold_maps[uuid] != active_fold
        except KeyError:
            return self.psl_uuid_fold_maps[uuid] == active_fold
    
    def is_uuid_and_fold_ok_for_val(self, uuid, active_fold,):
        try:
            return self.true_uuid_fold_maps[uuid] == active_fold
        except KeyError:
            if not self.include_psl_in_val:
                return False
            else:
                return self.psl_uuid_fold_maps[uuid] != active_fold

    
    def select_samples(self, active_fold, uuids, is_train=True):
        f = self.is_uuid_and_fold_ok_for_train if is_train else self.is_uuid_and_fold_ok_for_val

        uuids_idx = [i  for i, uuid in enumerate(uuids) if f(uuid=uuid, active_fold=active_fold)]
        return uuids_idx


def one_fold(*, uuids, data, model_name, fold, train_set, val_set, use_mixup, forward_type, task=None, epochs=20, save=True, save_root=None,
    checkpoint_paths=None, best_ckpt_dest_root=None, model_config=None, early_stop_epochs=None, save_top_k=1, save_each=None,
    other_model_params=None, **data_kwargs):

    model_name_slug = slugify(model_name)

    save_root = Path(save_root) or cfg.MODEL_ROOT

    saver = AutoSave(root=save_root, name=f"{cfg.SAVE_PREFIX}_{model_name_slug}_fold{fold}",
    metric=cfg.SAVE_METRIC_NAME, top_k=save_top_k, mode=cfg.SAVE_MODE)
    
    checkpoint_path=(checkpoint_paths or {}).get(f"fold_{fold}")
    other_model_params = {} if other_model_params is None else other_model_params
    net = Model(model_name, checkpoint_path=checkpoint_path, use_mixup=use_mixup,
            forward_type=forward_type, task=task, config=deepcopy(model_config), **other_model_params)

    # display(net)

    try:
        net = models_load_model(model=net, checkpoint_path=checkpoint_path, verbose=True)
    except Exception as e:
        print("Second Load ERRORRRRRRRRRR:\n", e)

    net = net.to(cfg.DEVICE)

    val_maxlen = data_kwargs.pop("val_maxlen")
    try:
        val_max_num_spans = data_kwargs.pop("val_max_num_spans")
    except KeyError:
        val_max_num_spans = None

    train_data = Dataset(uuids=uuids[train_set] , data=data, is_train=True, **data_kwargs)
    train_loader = DataLoader(train_data, batch_size=cfg.TRAIN_BATCH_SIZE, num_workers=cfg.TRAIN_NUM_WORKERS,
    shuffle=True, pin_memory=True, drop_last=True,
    collate_fn=partial(collate_fn_train, pad_token_id=data_kwargs["special_token_ids"]["pad_token_id"], is_psl=cfg.IS_PSL),
    worker_init_fn=WorkerInitFn(get_config_as_param(cfg)),
    )

    data_kwargs["maxlen"] = val_maxlen
    data["max_num_spans"] = val_max_num_spans
    val_data = Dataset(uuids=uuids[val_set] , data=data, is_train=False,**data_kwargs)
    val_loader = DataLoader(val_data, batch_size=cfg.VAL_BATCH_SIZE, num_workers=cfg.VAL_NUM_WORKERS, shuffle=False,
    collate_fn=partial(collate_fn_train, pad_token_id=data_kwargs["special_token_ids"]["pad_token_id"], is_psl=cfg.IS_PSL),
    worker_init_fn=WorkerInitFn(get_config_as_param(cfg)),
    )

    num_iters = len(train_loader)*epochs
    criterion = get_loss(num_iters= int(cfg.CLASS_WEIGHTS_USE_TIME_RATIO*num_iters))
    optimizer = fetch_optimizer(net)

    scheduler = DecreaseIncreaseSinLR(optimizer, eta_min=cfg.SCHEDULER_ETA_MIN, T_max=num_iters)
    schedule_each_step = True

    if cfg.USE_AMP and ("cuda" in str(cfg.DEVICE)):
        scaler = amp.GradScaler()
        warn("amp and fp16 are enabled !")
    else:
        scaler = None

    epochs_bar = tqdm(list(range(epochs if early_stop_epochs is None else min(epochs, early_stop_epochs))), leave=False)

    for epoch in epochs_bar:

        epochs_bar.set_description(f"--> [EPOCH {epoch:02d}]")
        net.train()

        oof, metrics = one_epoch(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_loader=train_loader,
            val_loader=val_loader,
            schedule_each_step=schedule_each_step,
            save_each=save_each,


            # For print and save
            fold=fold,
            epoch=epoch,
            val_set=val_set,
            train_set=train_set,
            epochs_bar=epochs_bar,
            do_save=save,
            saver=saver,
        )

    if best_ckpt_dest_root is not None:
        saver.copy(best_ckpt_dest_root)


def train(*, uuids, data, model_name, use_mixup, forward_type, task=None, epochs=20, early_stop_epochs=None, save=True, n_splits=5, seed=None,save_root=None,
    suffix="", folds=None, checkpoint_paths=None, best_ckpt_dest_root=None, model_config=None, exclude_from_val=None, save_top_k=1,
    other_model_params=None, save_each=None, fold_selector: FoldSamplesSelector=None, **data_kwargs):
    gc.collect()
    torch.cuda.empty_cache()

    seed = cfg.SEED if seed is None else seed
    
    model_name_slug = slugify(model_name)
    save_root = save_root or cfg.MODEL_ROOT/f"{model_name_slug}{suffix}"
    save_root.mkdir(exist_ok=True, parents=True)

    if best_ckpt_dest_root is not None:
        best_ckpt_dest_root = Path(best_ckpt_dest_root) / save_root.name
        best_ckpt_dest_root.mkdir(exist_ok=True, parents=True)

    seed_everything(seed)

    if isinstance(uuids, dict):
        fold_bar = []
        for fold in range(n_splits):
            if fold_selector is None:
                train_set = [i for i, fold_i in enumerate(uuids.values()) if fold_i != fold]
                val_set = [i for i, fold_i in enumerate(uuids.values()) if fold_i == fold]
            else:
                train_set = fold_selector.select_samples(active_fold=fold, uuids=uuids.keys(), is_train=True)
                val_set = fold_selector.select_samples(active_fold=fold, uuids=uuids.keys(), is_train=False)

            train_set = np.array(train_set, dtype=np.int64)
            val_set = np.array(val_set, dtype=np.int64)

            fold_bar.append((train_set, val_set))

        uuids = np.array(list(uuids.keys()))
        # print(fold_bar)
    else:
        kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
        fold_bar = kf.split(np.arange(len(uuids)))

    fold_bar = tqdm(enumerate(fold_bar), total=n_splits)

    if exclude_from_val is not None:
        temp = dict(zip(uuids, range(len(uuids))))
        exclude_from_val = [temp.get(uuid, -1) for uuid in exclude_from_val]
        exclude_from_val = set(exclude_from_val)
    
    for fold, (train_set, val_set) in fold_bar:
        if folds and not fold in folds:
          continue
        
        print(f"\n############################### [FOLD {fold}  SEED {seed}]")
        fold_bar.set_description(f"[FOLD {fold}  SEED {seed}]")

        if exclude_from_val is not None:
            val_set = sorted(set(val_set).difference(exclude_from_val))

        if not len(val_set):
            warn("Empty VAL_SET, will set TRAIN_SET as val")
            val_set = [idx for idx in train_set if np.random.random() < 0.15] # Keep 10%
          
        one_fold(
            uuids=uuids, data=data, model_name=model_name, use_mixup=use_mixup, forward_type=forward_type, task=task, fold=fold,
            train_set=train_set, val_set=val_set, epochs=epochs, save=save, save_root=save_root, checkpoint_paths=checkpoint_paths,
            best_ckpt_dest_root=best_ckpt_dest_root, model_config=model_config, early_stop_epochs=early_stop_epochs, save_top_k=save_top_k,
            other_model_params=other_model_params, save_each=save_each, **data_kwargs,
        )
      
        gc.collect()
        torch.cuda.empty_cache()