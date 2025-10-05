from settings import *

if __name__ == '__main__':
    evoked_hallu_list = []
    evoked_no_hallu_list = []
    evoked_hallu_wrong_list = []
    _type = "sen"
    for select in np.arange(0, 27):
        evoked_1 = load_evoked_for_condition(subj_idx=select, _type=_type, cond="hallu", threshold=None)
        evoked_2 = load_evoked_for_condition(subj_idx=select, _type=_type, cond="no_hallu", threshold=None)
        evoked_3 = load_evoked_for_condition(subj_idx=select, _type=_type, cond="hallu_wrong", threshold=None)
        evoked_hallu_list.append(evoked_1)
        evoked_no_hallu_list.append(evoked_2)
        if evoked_3 is not None:
            evoked_hallu_wrong_list.append(evoked_3)

    ga_hallu = mne.grand_average(evoked_hallu_list).savgol_filter(h_freq=8)
    ga_no_hallu = mne.grand_average(evoked_no_hallu_list).savgol_filter(h_freq=8)
    ga_hallu_wrong = mne.grand_average(evoked_hallu_wrong_list).savgol_filter(h_freq=4)
    mne.viz.plot_compare_evokeds(evokeds={'Hallu': ga_hallu, 'NoHallu': ga_no_hallu, 'HalluWrong': ga_hallu_wrong},
                                 combine="mean", picks=central)
    mne.viz.plot_compare_evokeds(
        evokeds=mne.grand_average(evoked_hallu_list + evoked_no_hallu_list + evoked_hallu_wrong_list))
