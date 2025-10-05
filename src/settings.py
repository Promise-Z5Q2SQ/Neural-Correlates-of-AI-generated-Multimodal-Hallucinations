import os
import json
import mne
import sys
import numpy as np

pre_frontal = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4']
frontal = ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8']
central = ['CZ', 'FCZ', 'C1', 'C2', 'C3', 'C4', 'FC1', 'FC2', 'FC3', 'FC4']
l_temporal = ['FT7', 'FC5', 'T7', 'C5', 'TP7', 'CP5', 'P7', 'P5']
r_temporal = ['FT8', 'FC6', 'T8', 'C6', 'TP8', 'CP6', 'P8', 'P6']
parietal = ['CPZ', 'CP1', 'CP3', 'CP2', 'CP4', 'PZ', 'P1', 'P3', 'P2', 'P4']
occipital = ['POZ', 'PO3', 'PO5', 'PO7', 'PO4', 'PO6', 'PO8', 'O1', 'O2', 'OZ', 'CB1', 'CB2']
all_channels = pre_frontal + frontal + central + l_temporal + r_temporal + parietal + occipital

eeg_file_name = [""]
record_file_name = [""]

FREQ_BANDS = {
    "delta": [0.5, 4],
    "theta": [4, 8],
    "alpha": [8, 13],
    "beta": [13, 30],
    "gamma": [30, 80]
}

EEG_RAW_DIR = '../data/raw_data/'
EEG_PROCESSED_DIR = '../data/processed_data/'
RECORD_DIR = '../record/'


def load_record(_select, _type):
    assert _type in ['sen', 'word'], "Type must be 'sentence' or 'word'"
    if _type == 'sen':
        _type = 'sentence'

    record_dicts = []
    with open(os.path.join(RECORD_DIR, f"{record_file_name[_select]}_{_type}.txt"), 'r') as f:
        for _ in range(4):
            next(f)
        for lineno, line in enumerate(f, start=5):
            line = line.strip().replace('\'', '\"')
            line = line.replace('nan', '\"none\"')
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {lineno} 行 JSON 解析失败：{e.msg}")
            record_dicts.append(obj)
    total_len = 120 if _type == 'sentence' else 60
    assert len(record_dicts) == total_len, f"解析得到的 dict 数量为 {len(record_dicts)}，预期为 {total_len}"

    for record in record_dicts:
        hallu_index = []
        if _type == 'sentence':
            if record["hallu_word"] != "none":
                if record["id"] == 294:
                    record["hallu_word"] = "坐在、双腿"
                elif record["id"] == 64:
                    record["hallu_word"] = "右"
                elif record["id"] == 300:
                    record["hallu_word"] = "同一、方向"
                elif record["id"] == 393:
                    record["hallu_word"] = "关闭、空白"
                elif record["id"] == 85:
                    record["hallu_word"] = "手举、手指"
                elif record["id"] == 196:
                    record["hallu_word"] = "上"
                elif record["id"] == 395:
                    record["hallu_word"] = "飞行"
                elif record["id"] == 82:
                    record["hallu_word"] = "举、过、手握着"
                elif record["id"] == 221:
                    record["hallu_word"] = "黑色、脚去"
                elif record["id"] == 422:
                    record["hallu_word"] = "三、把"
                for hallu_word in record["hallu_word"].split("、"):
                    hallu_index.append(record["text"].index(hallu_word))
            record["hallu_index"] = hallu_index
        else:
            if record["hallu_word"] != "none":
                hallu_index.append(record["entities"].index(record["hallu_word"]))
            record["hallu_index"] = hallu_index

        if record["judgement"] == 'n' and record["hallu_word"] == "none":
            record["result"] = True
        elif record["judgement"] == 'n' and record["hallu_word"] != "none":
            record["result"] = False
        elif record["judgement"] == 'y' and record["hallu_word"] == "none":
            record["result"] = False
        else:
            record["result"] = True
    return record_dicts


def abs_threshold(epochs, threshold):
    data = epochs.copy().pick(occipital).get_data()
    # channels and times are last two dimension in MNE ndarrays,
    # and we collapse across them to get a (n_epochs,) shaped array
    rej = np.any(np.abs(data) > threshold, axis=(-1, -2))
    return rej


def load_evoked_for_condition(subj_idx: int, _type: str, cond: str, threshold=50e-6):
    fpath = os.path.join(EEG_PROCESSED_DIR, f"{subj_idx}_{_type}_all-epo.fif")
    epochs = mne.read_epochs(fpath, preload=True)[cond]
    print(f"Subject {subj_idx}, condition {cond}, loaded {len(epochs)} epochs")
    if len(epochs) == 0:
        return None
    if threshold is not None:
        bad_epoch_mask = abs_threshold(epochs, threshold)
        epochs.drop(bad_epoch_mask, reason="absolute threshold")
    print(f"Subject {subj_idx}, condition {cond}, loaded {len(epochs)} epochs after absolute threshold")
    return epochs.average()
