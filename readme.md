# How to run
The training kernel is at notebooks/training.ipyng and inference is under notebooks/inference.ipynb.

## Working directory and paths configuration

To run the notebook at *notebooks/training.ipyng*, one needs to:  
 

- Create a folder called *data* which represents **DATA_ROOT** configs variable.  
    You should put the competition's data under *data/feedback-prize-effectiveness* and the 2021 competition's data under
    *data/feedback-prize-2021*.  

By the way, *you* can put these files anywhere you want but in that case you should set these params (see notebook):
- **TRAIN_CSV_PATH**
- **TRAIN_ROOT**

You could also need to set many other path related variables like: *MODEL_ROOT*, *OOF_DFs_ROOT* ... (see notebook).


## Ressources
You could need these ressources. Many of them are available here:
    https://www.kaggle.com/code/kneroma/kkiller-fpzeff-all-data-kernel/data?scriptVersionId=104898683

- Pickled essays data (for fast running) -> **TRAIN_ESSAYS_PICKLE_PATH** :
    data/essays_220728.pkl

- OOFs Pickled DataFrames -> **OOF_DFs_ROOT** :
    data/oof_dfs_v2/*

- Pickled model inputs -> **DATA_ROOT / f"full_inputs_{slugify(configs.MODEL_NAME)}_220819.pkl"** :
    - microsoft/deberta-v3-large : data/full_inputs_microsoft_deberta-v3-large_220819.pkl
    - microsoft/deberta-large : data/full_inputs_microsoft_deberta-large_220819.pkl
    - microsoft/deberta-xlarge : 
    - microsoft/deberta-v2-xlarge : 

- Pretrained weights -> **checkpoints_store** :
    - microsoft/deberta-v3-large : data/fprize_microsoft_deberta-v3-large_fold0_epoch_04_iov_v2_val_0.7346_20220625092726.pth
    - microsoft/deberta-large : data/fprize_microsoft_deberta-large_fold0_epoch_05_iov_v2_val_0.7418_20220625133915.pth
    - microsoft/deberta-xlarge : 
    - microsoft/deberta-v2-xlarge :


## Efficiency track

The training for efficiency track is almost the same. YWe just have to discard the folding-related things, which is done in the *notebooks/training-efficiency.ipynb* file.
The inference kernel is strictly the same for efficiency and non-efficiency.