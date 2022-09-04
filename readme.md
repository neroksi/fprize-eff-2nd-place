# How to run

## Working directory and paths configuration
To run the notebook at *fprize/fprize_efgprint/notebooks/Shujun - Fpz Eff PSL eFgP.ipynb*, one needs to:  

- Create a folder called *fprize* which will  our packet's root such that `this file` would
    appear at fprize/fprize_efgprint/readme.md  

- Create another folder called *data* at the same level as *fprize*. In `fprize/fprize_efgprint/src/configs.py`, this
    *data* folder is represented by **DATA_ROOT** configs variable.  
    You should put the competition's data under *data/feedback-prize-effectiveness* and the 2021 competition's data under
    *data/feedback-prize-2021*.  

By the way, *you* can put these files anywhere you want but  in that case you should set these params (see notebook):
- **TRAIN_CSV_PATH**
- **TRAIN_ROOT**

You could also need to set many other path related variables like: *MODEL_ROOT*, *OOF_DFs_ROOT* ... (see notebook).


## Downloads
You could need these ressources.

- Pickled essays data (for fast running) -> **TRAIN_ESSAYS_PICKLE_PATH** :
    https://drive.google.com/file/d/19BhZzaz2HUDuXFKmJL6HHT7dd2hgdanh/view?usp=sharing

- OOFs Pickled DataFrames -> **OOF_DFs_ROOT** :
    https://drive.google.com/drive/folders/1k2GIFHZQ0po1rLlR_bDmrmKzPvcBdIKp?usp=sharing

- Pickled model inputs -> **DATA_ROOT / f"full_inputs_{slugify(configs.MODEL_NAME)}_220819.pkl"** :
    - microsoft/deberta-v3-large : https://drive.google.com/file/d/1QoO-77c1QNQYKf9HcLii9Q54AjJmRAIr/view?usp=sharing
    - microsoft/deberta-large : https://drive.google.com/file/d/1-5YddlRlPkf136XABHOaxNr3CV0xnvIA/view?usp=sharing
    - microsoft/deberta-xlarge : https://drive.google.com/file/d/1WtmvjGBw2Mw72CVKjH6tgLSzZ8THvA3h/view?usp=sharing
    - microsoft/deberta-v2-xlarge : https://drive.google.com/file/d/1jSZ3nkUpIYA9uTzKw8Okt5O0cPcjNm2J/view?usp=sharing

- Pretrained weights -> **checkpoints_store** :
    - microsoft/deberta-v3-large : https://drive.google.com/file/d/12OkutZm3DT4eNPVWCjXQ-RENwUZUzm6q/view?usp=sharing
    - microsoft/deberta-large : https://drive.google.com/file/d/139F_pH1cWj_6qbve4cB1fnu98oFSP_mV/view?usp=sharing
    - microsoft/deberta-xlarge : https://drive.google.com/file/d/14Z9MarE8dJi7xBGQ3o3YeTdvLQtPblOw/view?usp=sharing
    - microsoft/deberta-v2-xlarge : https://drive.google.com/file/d/13rt3MKXM1naSFYu9waLGqJKXHYBDXo8e/view?usp=sharing