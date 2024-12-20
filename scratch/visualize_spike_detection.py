"""
Temp code to look at b1 vs b2 spikes

"""
# Imports
import matplotlib.pyplot as plt
from mosquito.process_abf import load_processed_data

# ---------------------------------------
# MAIN
# ---------------------------------------
# Run script
if __name__ == "__main__":
    # loading separate data for b1 and b2
    data_folder1 = 65
    axo_num1 = 4
    data_folder2 = 65  # 46
    axo_num2 = 4  # 6

    # get data files
    try:
        data1 = load_processed_data(data_folder1, axo_num1, data_suffix='_spikes')
    except ValueError:
        data1 = load_processed_data(data_folder1, axo_num1)

    try:
        data2 = load_processed_data(data_folder2, axo_num2, data_suffix='_spikes')
    except (ValueError, ModuleNotFoundError):
        data2 = load_processed_data(data_folder2, axo_num2)

    data_list = [data1, data2]
    data_labels = ['b1', 'b2']

    # plot emg and spike detection
    fig, ax_list = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax_list.ravel()

    tmin = 0
    tmax = 160

    for ith, data in enumerate(data_list):
        t = data['time']
        emg_filt = data['emg_filt']
        emg = data['emg']
        try:
            good_spike_idx = data['good_spike_idx']
        except KeyError:
            good_spike_idx = data['spike_idx']

        mask = (t >= tmin) & (t <= tmax)

        if type(good_spike_idx) is list:
            for idx, sig in zip(good_spike_idx, emg_filt):  # zip(good_spike_idx, emg_filt):
                mask_spikes = (t[idx] >= tmin) & (t[idx] <= tmax)

                ax_list[ith].plot(t[mask], sig[mask])
                ax_list[ith].plot(t[idx][mask_spikes], sig[idx][mask_spikes], 'x')

        else:
            mask_spikes = (t[good_spike_idx] >= tmin) & (t[good_spike_idx] <= tmax)

            ax_list[ith].plot(t[mask], emg_filt[mask])
            ax_list[ith].plot(t[good_spike_idx][mask_spikes], emg_filt[good_spike_idx][mask_spikes], 'rx')


    fig.tight_layout()
    plt.show()
