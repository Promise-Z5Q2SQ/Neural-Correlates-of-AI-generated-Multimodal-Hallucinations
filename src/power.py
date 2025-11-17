import statsmodels.stats.power as smp
from settings import *


def mean_amp_in_window(evoked: mne.Evoked, tmin_s: float, tmax_s: float, roi=None) -> float:
    """返回 evoked 在 [tmin_s, tmax_s] 内、指定电极集合上的平均波幅（通道×时间双均值）"""
    e = evoked.copy()
    if roi is not None:
        e = e.pick(roi)
    e = e.crop(tmin=tmin_s, tmax=tmax_s)
    return float(e.data.mean())


subjects = list(range(27))
_type = "sen"
ROI = central  # ROI：若只分析部分电极，填入通道名列表；None 表示使用全部 EEG 通道

evoked_hallu_list = []
evoked_no_hallu_list = []
evoked_hallu_wrong_list = []
for subj in subjects:
    evoked_1 = load_evoked_for_condition(subj_idx=subj, _type=_type, cond="hallu")
    evoked_2 = load_evoked_for_condition(subj_idx=subj, _type=_type, cond="no_hallu")
    evoked_3 = load_evoked_for_condition(subj_idx=subj, _type=_type, cond="hallu_wrong", threshold=None)
    evoked_hallu_list.append(evoked_1)
    evoked_no_hallu_list.append(evoked_2)
    # if evoked_3 is not None:
    #     evoked_hallu_list.append(evoked_1)
    #     evoked_no_hallu_list.append(evoked_2)
    #     evoked_hallu_wrong_list.append(evoked_3)

data_a = np.array([mean_amp_in_window(evk, 0.05, 0.7, ROI) for evk in evoked_hallu_list[0:6]])
data_b = np.array([mean_amp_in_window(evk, 0.05, 0.7, ROI) for evk in evoked_no_hallu_list[0:6]])

diff = data_b - data_a
mean_diff = diff.mean()
sd_diff = diff.std(ddof=1)
cohens_d = mean_diff / sd_diff

analysis = smp.TTestPower()
alpha = 0.05
target_power = 0.80

n_required = analysis.solve_power(effect_size=cohens_d,
                                  alpha=alpha,
                                  power=target_power,
                                  alternative='two-sided')

print(f"Estimated required number of pairs (n) ≈ {np.ceil(n_required):.0f}")

data_a = np.array([mean_amp_in_window(evk, 0.05, 0.7, ROI)for evk in evoked_hallu_list])
data_b = np.array([mean_amp_in_window(evk, 0.05, 0.7, ROI) for evk in evoked_no_hallu_list])

diff = data_b - data_a
mean_diff = diff.mean()
sd_diff = diff.std(ddof=1)
cohens_d = mean_diff / sd_diff

power_current = analysis.solve_power(effect_size=cohens_d,
                                     nobs=27,
                                     alpha=alpha,
                                     alternative='two-sided')

print(f"Achieved power with n=27 ≈ {power_current:.4f}")
