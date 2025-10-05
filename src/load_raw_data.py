import mne

from settings import *
from utilities import *

if __name__ == '__main__':
    cnt_correct_list = []
    evoked_hallu_list = []
    evoked_no_hallu_list = []
    _type = "sen"
    for select in np.arange(0, 27):
        raw = mne.io.read_raw_curry(os.path.join(EEG_RAW_DIR, f"{eeg_file_name[select]}_{_type}.cdt"), preload=True)
        raw.pick(all_channels)
        montage = mne.channels.read_dig_fif('../data/mode/montage.fif')
        montage.ch_names = json.load(open("../data/mode/montage_ch_names.json"))
        raw.set_montage(montage)
        # raw.plot_sensors(ch_type='eeg', show_names=True)
        raw.set_eeg_reference('average')
        raw = raw.notch_filter(freqs=50)
        raw = raw.filter(l_freq=0.5, h_freq=30)
        print(raw)
        print(raw.info)
        print(raw.info['ch_names'])

        events, event_id = mne.events_from_annotations(raw)
        if select == 6 and _type == "sen":
            events = events[:1482]
        if select == 6 and _type == "word":
            events = events[1482:]
        print(events.shape)
        print(event_id)
        record_dict = load_record(_select=select, _type=_type)
        cnt_correct = 0
        for record_item in record_dict:
            if record_item["result"]:
                cnt_correct += 1
        print(cnt_correct, cnt_correct / len(record_dict))
        cnt_correct_list.append(cnt_correct)

        # check trigger order
        expected_order = []
        for record_item in record_dict:
            list_len = len(record_item["text"]) if _type == "sen" else len(record_item["entities"])
            expected_order.extend([1, 2] + [3] * list_len + [4])
        ev_ids = events[:, 2]
        for i in range(len(expected_order)):
            assert expected_order[i] == ev_ids[i], f"Time {events[i]}: {expected_order[i]} != {ev_ids[i]}"
        assert ev_ids.shape[0] == len(expected_order), f"事件数量不匹配：实际 {len(ev_ids)}，期望 {len(expected_order)}"

        # get hallu word event
        cnt = 0
        for record_item in record_dict:
            list_len = len(record_item["text"]) if _type == "sen" else len(record_item["entities"])
            if not record_item["result"]:
                events[cnt + 2: cnt + 2 + list_len, 2] = 6
                for hallu_index in record_item["hallu_index"]:
                    events[cnt + hallu_index + 2, 2] = 7
            else:
                for hallu_index in record_item["hallu_index"]:
                    events[cnt + hallu_index + 2, 2] = 5
            cnt += list_len + 3
        # with np.printoptions(threshold=sys.maxsize):
        #     print(events)

        # hallu_epochs = mne.Epochs(raw, events, 5, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), preload=True)
        # no_hallu_epochs = mne.Epochs(raw, events, 3, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), preload=True)
        # hallu_epochs.save(os.path.join(EEG_PROCESSED_DIR, f"{select}_{_type}_hallu-epo.fif"), overwrite=True)
        # no_hallu_epochs.save(os.path.join(EEG_PROCESSED_DIR, f"{select}_{_type}_no_hallu-epo.fif"), overwrite=True)
        all_epochs = mne.Epochs(raw, events, {
            "image": 1,
            "+": 2,
            "no_hallu": 3,
            "judge": 4,
            "hallu": 5,
            "no_hallu_wrong": 6,
            "hallu_wrong": 7
        }, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), on_missing='warn', preload=True)
        all_epochs.save(os.path.join(EEG_PROCESSED_DIR, f"{select}_{_type}_all-epo.fif"), overwrite=True)

    print(cnt_correct_list)
