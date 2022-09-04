import pandas as pd, numpy as np
import torch
from pathlib import Path
import re
from collections import Counter
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold

from sklearn.model_selection import GroupKFold

from torch.utils.data import Dataset as TorchDataset
from warnings import warn
import unicodedata
from string import punctuation

split_regex = ["(\\s+)", "([{}]+)".format(re.escape(punctuation)), "(\\d+)"]

import configs as cfg
from utils import copy_param_to_configs
from essay_utils import *


def clean_text(text):
    text = unicodedata.normalize("NFKD",  text)
    text = text.replace("\n", cfg.NEW_LINE_TOKEN)
    text = text.strip()
    return text

def read_from_id(text_id, root=None, apply_clean=True):
    root = Path(root or cfg.TRAIN_ROOT)
    text = (root / text_id).with_suffix(".txt").read_text(encoding="utf-8")
    if apply_clean:
        text = clean_text(text)
    return text

def read_train_csv(train_csv_path=None, n_splits=5, nrows=None, apply_clean=True, is_test=False):

    train_csv_path = train_csv_path or cfg.TRAIN_CSV_PATH

    df_train = pd.read_csv(
        train_csv_path,
        nrows=nrows,
        encoding="utf-8",
    )

    if apply_clean:
        cols = ['discourse_text']
        
        for col in cols:
            df_train[col] = df_train[col].apply(clean_text)

    df_train["id"] = df_train["essay_id"]
    
    if not is_test:
        df_train = build_folds(df_train, group_col="id", strate_col="discourse_effectiveness", fold_col_name=cfg.FOLD_COL_NAME, n_splits=n_splits, seed=cfg.SEED)

    return  df_train

def read_old_df(df, old_data_root=None):
    old_data_root = old_data_root or cfg.DATA_ROOT / "feedback-prize-2021"
    df_old = pd.read_csv(
        old_data_root / "train.csv",
        usecols=["id", "discourse_id", "discourse_text", "discourse_type", ],
        dtype={"discourse_id": str}
    )
    
    df_old["discourse_text"] = df_old["discourse_text"].apply(clean_text)
    
    df_old["essay_id"] = df_old["id"]
    df_old["train_root"] = str(old_data_root / "train")

    df_old = build_folds(df_old, group_col="id", strate_col="discourse_type",
                                 fold_col_name=cfg.FOLD_COL_NAME, n_splits=5, seed=cfg.SEED)
    
    bools = df_old["essay_id"].isin(df["essay_id"].unique())
    df_old.loc[bools, cfg.FOLD_COL_NAME] = -1

    print("df_old.shape", df_old.shape)

    df = pd.concat([df, df_old[~bools]], axis=0, sort=False)
    df.reset_index(drop=True, inplace=True)
    
    return df

def build_simple_kfold(df, fold_col_name=None, n_splits=5, seed=2222):
    fold_col_name = fold_col_name or f"{n_splits}fold"
    df[fold_col_name] = -1
    
    kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
    splits = kf.split(np.arange(len(df)))
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        df.loc[df.index[val_idx], fold_col_name] = fold
    
    return df


def build_folds(df, group_col="id", strate_col="categories", fold_col_name="fold", n_splits=5, seed=2222):
    
    df2 = df[[group_col, strate_col]].drop_duplicates([group_col]).copy(deep=False)
    df2[fold_col_name] = -1
    
    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    splits = skf.split(np.arange(len(df2)), y=df2[strate_col])
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        df2.loc[df2.index[val_idx], fold_col_name] = fold
    
    df = df.merge(df2[[group_col, fold_col_name]], on = group_col, how="left")
    
    return df


def load_oof_dfs(oof_df_paths):
    cols = ["Ineffective", "Adequate", "Effective"]

    df_oof = None
    for oof_df_path in oof_df_paths:
        temp =  pd.read_pickle(oof_df_path)
        temp = temp[temp[cols].notnull().all(1)]
        
        if df_oof is None:
            df_oof = temp
        else:
            df_oof = pd.concat([df_oof, temp], axis=0, sort=False)

    df_oof[cols] = df_oof[cols].values * df_oof["weight"].values[:, None]
    df_oof = df_oof.groupby(["id", "discourse_id", "discourse_num", cfg.FOLD_COL_NAME])[cols].sum().reset_index()
    df_oof[cols] = df_oof[cols].values / df_oof[cols].sum(1).values[:, None]

    return df_oof



class WorkerInitFn:
    def __init__(self, config_params) -> None:
        self.config_params = config_params

    def __call__(self, worker_id: int):
        copy_param_to_configs(self.config_params, cfg=cfg)


class Dataset(TorchDataset):
    def __init__(self, uuids, data, maxlen=None, special_token_ids=None,  max_num_spans=None, is_train=True,):
        self.uuids = uuids
        self.data = data
        self.maxlen = maxlen or cfg.MAXLEN
        self.is_train = is_train
        self.special_token_ids = special_token_ids
        self.pad_token_id = self.special_token_ids["pad_token_id"]
        self.mask_token_id = self.special_token_ids["mask_token_id"]

        self.max_num_spans = max_num_spans or cfg.MAX_NUM_SPANS

    def __len__(self):
        return len(self.uuids)

    def mask_input_ids(self, input_ids):

        p_mask_size = np.random.uniform(cfg.P_MASK_SIZE_LOW, cfg.P_MASK_SIZE_HIGH)
        
        n_masked = int(p_mask_size * len(input_ids))
        
        if n_masked > 0:
            index = np.random.choice(len(input_ids), size=n_masked, replace=False)

            for i in index:
                input_ids[i] = self.mask_token_id

        return input_ids
    
    def truncate(self, essay_id, d):

        compute_target = d["span_data"][0].get("target") is not None

        span_start = span_end = None
        
        r = retrieve_span(uuid_data=d, special_ids=self.special_token_ids, maxlen=self.maxlen,
            span_start=span_start, span_end=span_end,
            mask_func=self.mask_input_ids if self.is_train else None, compute_target=compute_target)

        r["uuids"] = (essay_id, r["discourse_ids"])
        r["masks"] = [1]* len(r["input_ids"])

        res = [r]

        return res

    
    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        return self.truncate(uuid, self.data[uuid])


class InferenceDataset(Dataset):
    def __init__(self, uuids, data, maxlen, special_token_ids, max_num_spans=None, is_train=False,):
        super().__init__(uuids=uuids, data=data, maxlen=maxlen, max_num_spans=max_num_spans, 
                is_train=is_train, special_token_ids=special_token_ids)


def collect_as_list(dict_list, key):
    return [dd[key] for d in dict_list for dd  in d]

def copy_to_array(a, pos, v):
    if a.ndim > 1:
        a[pos, :len(v)] = v
    else:
        a[pos] = v
    return a

def collate_fn_train(inputs, pad_token_id, is_psl):

    uuids = collect_as_list(inputs, "uuids")
    span_ids = collect_as_list(inputs, "span_ids")
    token_type_ids = collect_as_list(inputs, "token_type_ids")
    input_ids = collect_as_list(inputs, "input_ids")
    masks = collect_as_list(inputs, "masks")

    if cfg.USE_SAMPLE_WEIGHTS:
        weights = collect_as_list(inputs, "weights")

    has_target = inputs[0][0].get("target") is not None
    if has_target:
        target = collect_as_list(inputs, "target")


    batch_size = len(input_ids)
    batch_maxlen = max([len(x) for x in input_ids])

    shape = (batch_size, batch_maxlen)

    new_span_ids = np.full(shape, -1, dtype=np.int64)
    new_token_type_ids = np.full(shape, cfg.DISCOURSE_PAD_ID, dtype=np.int64)
    new_input_ids = np.full(shape, pad_token_id, dtype=np.int64)
    new_masks = np.zeros(shape, dtype=np.int64)

    if cfg.USE_SAMPLE_WEIGHTS:
        new_weights = np.zeros(batch_size, dtype=np.float32)

    if cfg.USE_SAMPLE_WEIGHTS:
        new_weights = np.zeros(batch_size, dtype=np.float32)
    if has_target:
        if is_psl:
            new_target = np.full((*shape, 3), cfg.PYTORCH_CE_IGNORE_INDEX, dtype=np.float32)
        else:
            new_target = np.full(shape, cfg.PYTORCH_CE_IGNORE_INDEX, dtype=np.int64)

    for pos in range(batch_size):

        copy_to_array(new_span_ids, pos, span_ids[pos])
        copy_to_array(new_token_type_ids, pos, token_type_ids[pos])
        copy_to_array(new_input_ids, pos, input_ids[pos])
        copy_to_array(new_masks, pos, masks[pos])

        if cfg.USE_SAMPLE_WEIGHTS:
            copy_to_array(new_weights, pos, weights[pos])
        
        if has_target:
            copy_to_array(new_target, pos, target[pos])

    new_span_ids = torch.from_numpy(new_span_ids)
    new_token_type_ids = torch.from_numpy(new_token_type_ids)
    new_input_ids = torch.from_numpy(new_input_ids)
    new_masks = torch.from_numpy(new_masks)
    
    if cfg.USE_SAMPLE_WEIGHTS:
        new_weights = torch.from_numpy(new_weights)
    if has_target:
        new_target = torch.from_numpy(new_target)

    res = {
        "uuids": uuids,
        "span_ids": new_span_ids,
        "token_type_ids": new_token_type_ids,
        "input_ids": new_input_ids,
        "masks": new_masks,
        "weights": new_weights if cfg.USE_SAMPLE_WEIGHTS else None,
        "target": new_target if has_target else None
    }

    return res