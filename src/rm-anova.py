import pandas as pd
import pingouin as pg
from settings import *

_type = "sen"
subjects = list(range(27))
time_windows_ms = [[50, 120], [120, 280], [280, 550], [550, 750]]
ROI = occipital  # ROI：若只分析部分电极，填入通道名列表；None 表示使用全部 EEG 通道


def mean_amp_in_window(evoked: mne.Evoked, tmin_s: float, tmax_s: float, roi=None) -> float:
    """返回 evoked 在 [tmin_s, tmax_s] 内、指定电极集合上的平均波幅（通道×时间双均值）"""
    e = evoked.copy()
    if roi is not None:
        e = e.pick(roi)
    # MNE 的 crop 包含端点（tmin <= t <= tmax）
    e = e.crop(tmin=tmin_s, tmax=tmax_s)
    return float(e.data.mean())


evoked_hallu_list = []
evoked_no_hallu_list = []
evoked_hallu_wrong_list =[]
for subj in subjects:
    evoked_1 = load_evoked_for_condition(subj_idx=subj, _type=_type, cond="hallu")
    evoked_2 = load_evoked_for_condition(subj_idx=subj, _type=_type, cond="no_hallu")
    evoked_3 = load_evoked_for_condition(subj_idx=subj, _type=_type, cond="hallu_wrong", threshold=None)
    if evoked_3 is not None:
        evoked_hallu_list.append(evoked_1)
        evoked_no_hallu_list.append(evoked_2)
        evoked_hallu_wrong_list.append(evoked_3)

for (start_ms, end_ms) in time_windows_ms:
    tmin, tmax = start_ms / 1000.0, end_ms / 1000.0
    rows = []
    for idx, (evk_h, evk_n, evk_w) in enumerate(zip(evoked_hallu_list, evoked_no_hallu_list, evoked_hallu_wrong_list)):
        amp_h = mean_amp_in_window(evk_h, tmin, tmax, ROI)
        amp_n = mean_amp_in_window(evk_n, tmin, tmax, ROI)
        amp_w = mean_amp_in_window(evk_w, tmin, tmax, ROI)
        # rows.append({"Subject": str(idx), "Condition": "hallu", "Amplitude": amp_h})
        rows.append({"Subject": str(idx), "Condition": "no_hallu", "Amplitude": amp_n})
        rows.append({"Subject": str(idx), "Condition": "hallu_wrong", "Amplitude": amp_w})

    df = pd.DataFrame(rows)

    print(f"\n=== One-way RM-ANOVA @ Window {start_ms}-{end_ms} ms ===")
    aov = pg.rm_anova(
        data=df,
        dv="Amplitude",
        within="Condition",
        subject="Subject",
        detailed=True,
        correction="auto"
    )
    with pd.option_context("display.max_columns", None):
        print(aov)
    aov.to_csv(f"../output/anova-n-w/{ROI[0]}-{start_ms}-{end_ms}-aov.csv")

