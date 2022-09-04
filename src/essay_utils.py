import pandas as pd, numpy as np
import torch
from pathlib import Path
import re
from collections import Counter
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
except ModuleNotFoundError:
    pass

from sklearn.model_selection import GroupKFold

from torch.utils.data import Dataset as TorchDataset
from warnings import warn
import unicodedata
from string import punctuation
import bisect
from itertools import chain as chain_it

import pickle

split_regex = ["(\\s+)", "([{}]+)".format(re.escape(punctuation)), "(\\d+)"]

import configs as cfg
from utils import copy_param_to_configs, get_special_token_ids
from size_alocation import size_allocator


def prepare_special_token(token):
    token = token.lower()
    token = re.sub("[^\\sa-z]", "", token)
    token = re.sub("\\s+", "-", token)
    return token


def split_upper_lower_case(s):
    ul_regex = r"([a-z][A-Z])"
    
    res = re.split(ul_regex, s)
    
    new_res = []
    t = ""
    for r in res:
        if re.search(ul_regex, r):
            t += r[0]
            new_res.append(t)
            t = r[1]
        else:
            t += r
    
    if t:
        new_res.append(t)
    return new_res

def split(s):
    res = [s]
    for rgx in split_regex:
        new_res = []
        for r in res:
            new_res.extend( re.split(rgx, r) )

        res = [r for r  in new_res if r]

    new_res = []
    for r in res:
        new_res.extend( split_upper_lower_case(r) )
    res = [r for r  in new_res if r]

    valid_word = r"\S"
    new_res = []
    t = ""
    # print(res)
    for r in res:
        t += r
        if re.search(valid_word, t):
            new_res.append(t)
            t = ""

    else:
        new_res[-1] += t

    return new_res


def get_sart_end(discourse, essay, i=0, j=None, offset=0, icount=0,):
    
    if icount > 2000:
        raise ValueError("RECURSION ERRROOOOORRRRR")
        
    if isinstance(discourse, str):
        discourse = split(discourse.strip())
        
    if j is None:
        j = len(discourse)
        
    if i >= j or not len(essay):
        return []
        
    res = []
    
        
    text = "".join(discourse[i:j]).strip()
    start = essay.find(text)
    if start >= 0:
        end = start + len(text)
        res.append(((i, j), (start+offset, end+offset)))
        
        offset += end
        i = j
        j = len(discourse)
        essay = essay[end:]
    else:
        if j > i+1:
            j = j-1
        elif i < len(discourse) - 1:
            i = i+1
            j = len(discourse)
        else:
            return res
    
    icount = icount + 1
    
    res.extend(get_sart_end(discourse, essay, i=i, j=j, icount=icount, offset=offset))

    return res

def unify_spans(spans):
    assert len(spans)
    
    start, end = spans[0][1]

    if len(spans) == 1:
        return (start, end)
    
    for (_, _), (start_1, end_1) in spans[1:]:
        start = min(start, start_1)
        end = max(end, end_1)
    
    return (start, end)
    

def char_span_to_word_span(char_span, char_offsets):
   
    start, end = char_span
    
    assert start >= 0
    assert start < end
    
    n = len(char_offsets)
    
    
    word_start_1 = bisect.bisect_left(char_offsets, start)
    word_start_2 = bisect.bisect_right(char_offsets, start)

    word_start = None

    if word_start is None:
        e1 = abs(start - (char_offsets[word_start_1-1] if word_start_1 > 0 else 0))
        e2 = abs(start - (char_offsets[word_start_2-1] if word_start_2 > 0 else 0))
    #     print()
    #     print( e1, e2, word_start_1, word_start_2)
        
        word_start = word_start_1 if e1 < e2 else word_start_2
        
    word_end_1 = bisect.bisect_left(char_offsets, end)
    word_end_2 = bisect.bisect_right(char_offsets, end)
    
    e1 = abs(end - (char_offsets[word_end_1] if word_end_1 < n else char_offsets[-1]))
    e2 = abs(end - (char_offsets[word_end_2] if word_end_2 < n else char_offsets[-1]))
#     print(e1, e2, word_end_1, word_end_2)
    
    word_end = word_end_1 if e1 < e2 else word_end_2
    word_end = min(word_end+1, n)
    
    word_span = (word_start, word_end)

    # print(char_span, word_span)
    
    return word_span


def get_essay_signature(span_data, special_ids):

    assert len(span_data)

    signature = [special_ids[cfg.ESSAY_START_TOKEN]]

    for span in span_data:
        if not span["is_real"]:
            continue

        signature.append(span["start_special_token_id"])
        signature.append(span["span_token_id"])
        signature.append(span["end_special_token_id"])
    
    signature.append(special_ids[cfg.ESSAY_END_TOKEN])

    return signature


def make_contiguous_spans(span_data, special_ids, inputs_size, compute_target=True):
    assert len(span_data)

    null_d_start_token_id = special_ids[cfg.NULL_DISCOURSE_START_TOKEN]
    null_d_end_token_id = special_ids[cfg.NULL_DISCOURSE_END_TOKEN]

    target = get_dummy_target() if compute_target else None
    
    # null_span_num = cfg.MAX_OVER_ALL_NUM_SPANS - 1
    null_span_num = -1
    # null_span_token_id = special_ids[cfg.D_SPAN_TOKEN_FORMAT.format(null_span_num)]

    start_0 = 0
    new_span_data = []

    for span in span_data:
        start, end = span["start"], span["end"]
        if start > start_0:

            new_span_data.append(
                {
                    "span_num": null_span_num,
                    "discourse_id": "", # Fake discourse_id
                    "is_real": False,
                    # "span_token_id": null_span_token_id,
                    "start": start_0,
                    "end": start,
                    "num_real_tokens": start - start_0,
                    "start_special_token_id": null_d_start_token_id,
                    "end_special_token_id": null_d_end_token_id,
                    "discourse_type": cfg.NULL_DISCOURSE_TYPE,
                    "token_type_start_id": cfg.DISCOURSE_OUT_ID,
                    "token_type_in_id": cfg.DISCOURSE_OUT_ID,
                    "token_type_end_id": cfg.DISCOURSE_OUT_ID,
                    "target": target,
                }
            )

        new_span_data.append(span)
        
        start_0 =  end

    
    if start_0 < inputs_size:
        start = inputs_size

        new_span_data.append(
            {
                "span_num": null_span_num,
                "discourse_id": "", # Fake discourse_id
                "is_real": False,
                # "span_token_id": null_span_token_id,
                "start": start_0,
                "end": start,
                "num_real_tokens": start - start_0,
                "start_special_token_id": null_d_start_token_id,
                "end_special_token_id": null_d_end_token_id,
                "discourse_type": cfg.NULL_DISCOURSE_TYPE,
                "token_type_start_id": cfg.DISCOURSE_OUT_ID,
                "token_type_in_id": cfg.DISCOURSE_OUT_ID,
                "token_type_end_id": cfg.DISCOURSE_OUT_ID,
                "target": target,
            }
        )

    return new_span_data

    
def get_dummy_target():
    if cfg.IS_PSL:
        dummy_target = [cfg.PYTORCH_CE_IGNORE_INDEX] * cfg.NUM_TARGETS
    else:
        dummy_target = cfg.PYTORCH_CE_IGNORE_INDEX
    return dummy_target


def add_target(final_span_ids, df_uuid):

    final_span_ids = np.array(final_span_ids)
    assert final_span_ids.ndim == 1

    dummy_target = get_dummy_target()
    target = np.array([dummy_target] * len(final_span_ids))

    idx = final_span_ids[(final_span_ids >= 0)]
    assert idx.max() < len(df_uuid)

    cols = cfg.TARGET_NAMES if cfg.IS_PSL else  "discourse_effectiveness"
    target[idx] = df_uuid[cols].values[idx]

    return target


def gen_uuid_data(df_uuid, essay, tokenizer, compute_target=True, check_unique_id=True):
    if check_unique_id:
        assert df_uuid["id"].nunique() == 1

    inputs = tokenizer(essay, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = inputs["input_ids"]
    offsets = inputs["offset_mapping"]
    char_ends = [offset[1] for offset in offsets]
    
    special_ids = get_special_token_ids(tokenizer)

    span_data = []

    for span_num, (idx, row) in enumerate(df_uuid.iterrows()):
        char_start, char_end = unify_spans(get_sart_end(row["discourse_text"], essay=essay))
        start, end = char_span_to_word_span((char_start, char_end), char_ends)

        assert end >= start

        discourse_type = prepare_special_token(row["discourse_type"])

        start_special_token_id = special_ids[cfg.START_SPECIAL_TOKEN_FORMAT.format(discourse_type)]
        end_special_token_id = special_ids[cfg.END_SPECIAL_TOKEN_FORMAT.format(discourse_type)]

        if compute_target:
            if cfg.IS_PSL:
                target = row[cfg.TARGET_NAMES].values.astype(np.float32)
            else:
                target = cfg.CLASS2ID[row["discourse_effectiveness"]]
        else:
            target = None

        token_type_in_id = cfg.D2ID[ cfg.DISCOURSE_IFORMAT.format(row["discourse_type"]) ]
        token_type_start_id = cfg.D2ID[ cfg.DISCOURSE_BFORMAT.format(row["discourse_type"]) ]
        token_type_end_id = cfg.D2ID[ cfg.DISCOURSE_EFORMAT.format(row["discourse_type"]) ]


        span_data.append(
            {
                "span_num": span_num,
                "discourse_id": row["discourse_id"],
                "is_real": True,
                "span_token_id": special_ids[cfg.D_SPAN_TOKEN_FORMAT.format(min(span_num, cfg.MAX_OVER_ALL_NUM_SPANS - 1))],
                "start": start,
                "end": end,
                "num_real_tokens": end - start,
                "start_special_token_id": start_special_token_id,
                "end_special_token_id": end_special_token_id,
                "discourse_type": row["discourse_type"],
                "token_type_start_id": token_type_start_id,
                "token_type_in_id": token_type_in_id,
                "token_type_end_id": token_type_end_id,
                "target": target,
                
            }
        )


    span_data = make_contiguous_spans(span_data, special_ids=special_ids, inputs_size=len(offsets), compute_target=compute_target)
    non_null_span_pos = [i for i, span in enumerate(span_data) if span["is_real"]]
    signature = get_essay_signature(span_data, special_ids=special_ids)

    res = {
        "inputs_size": len(offsets),
        "is_true_obs":  (df_uuid["is_true_obs"].mean() > 0.5),
        "non_null_span_pos": non_null_span_pos,
        "signature": signature,
        "input_ids": input_ids,
        "offsets": offsets,
        "span_data": span_data,
    }
    
    return res


def retrieve_span(uuid_data, special_ids, maxlen=None, min_num_tokens=10, span_start=None, span_end=None, mask_func=None, compute_target=False):
    maxlen = maxlen or cfg.MAXLEN

    span_data = uuid_data["span_data"]
    if span_start is not None:
        span_data = span_data[span_start:]
    if span_end is not None:
        span_data = span_data[:span_end]

    # NN = len(uuid_data["non_null_span_pos"])
    NN = len([span for span in span_data if span["is_real"]])

    max_real_seq = (
        maxlen
        - 2 * NN # Counting for <> & </>
        - 1 * NN # Counting for <span::{}>
        - ( 2 + 3 * NN) # Counting for signature
        - 3 # Counting <BOS> <SEP> & <EOS>
    )

    assert max_real_seq > 0

    num_tokens = np.array([span["num_real_tokens"] for span in span_data])

    if len(uuid_data["input_ids"]) > max_real_seq:
        
        # num_tokens = num_tokens / num_tokens.sum()
        # num_tokens = np.maximum(min_num_tokens, np.round(max_real_seq*num_tokens).astype(int) )

        num_tokens = size_allocator(total_size=max_real_seq, sizes=num_tokens, min_size=min_num_tokens)
    
    if compute_target:
        dummy_target = get_dummy_target()
    else:
        dummy_target = None

    span_ids = [-1]
    discourse_ids = [""]
    input_ids = [special_ids["bos_token_id"]]
    token_type_ids = [cfg.DISCOURSE_OUT_ID]
    target = [dummy_target] if compute_target else None

    for num_token, span in zip(num_tokens, span_data):
        start, end = span["start"], span["end"]
        inp = uuid_data["input_ids"][start:end].copy()

        if len(inp) > num_token:
            inp = inp[:num_token]
            if mask_func is not None:
                inp = mask_func(inp)
            inp.append(special_ids[cfg.TRUNCATION_TOKEN])

        else:
            if mask_func is not None:
                inp = mask_func(inp)

        if span["is_real"]:
            inp.insert(0, span["span_token_id"])
            inp.insert(0, span["start_special_token_id"])
            inp.append(span["end_special_token_id"])

        input_ids.extend(inp)

        token_type_ids.append(span["token_type_start_id"])
        if len(inp) > 1:
            token_type_ids.extend([span["token_type_in_id"]] * (len(inp) -1) )

            if len(inp) > 2:
                token_type_ids[-1] = span["token_type_end_id"]
        
        span_ids.extend([span["span_num"]] * len(inp))
        discourse_ids.extend([span["discourse_id"]]* len(inp))
        if compute_target:
            target.extend([span["target"]]*len(inp))

    N = len(input_ids)

    input_ids.append(special_ids["sep_token_id"])
    input_ids.extend(uuid_data["signature"])
    input_ids.append(special_ids["eos_token_id"])

    token_type_ids.extend([cfg.DISCOURSE_OUT_ID] * (len(input_ids) - N))
    span_ids.extend([-1] * (len(input_ids) - N))
    discourse_ids.extend([""] * (len(input_ids) - N))
    if compute_target:
        target.extend([dummy_target] * (len(input_ids) - N))
    

    assert len(span_ids) == len(input_ids) == len(token_type_ids) == len(discourse_ids), (
                len(span_ids), len(input_ids), len(token_type_ids), len(discourse_ids))
    if compute_target:
        assert len(span_ids) == len(target)

    if cfg.USE_SAMPLE_WEIGHTS:
        weights = 2  * ( cfg.TRUE_OBS_WEIGHT_RATIO if uuid_data["is_true_obs"] else (1-cfg.TRUE_OBS_WEIGHT_RATIO) )
    else:
        weights = None

    res = {
        "span_ids": span_ids,
        "discourse_ids": discourse_ids,
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "target": target,
        "weights": weights,
    }
    
    return res