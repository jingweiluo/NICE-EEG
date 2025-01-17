# import numpy as np
# import os
# import mne
# import argparse
# from sklearn.utils import shuffle

# parser = argparse.ArgumentParser()
# parser.add_argument('--sub', default=1, type=int)
# parser.add_argument('--n_ses', default=4, type=int)
# parser.add_argument('--sfreq', default=250, type=int)
# parser.add_argument('--mvnn_dim', default='epochs', type=str)
# parser.add_argument('--project_dir', default='/data1/labram_data/Things-EEG2/', type=str)
# args = parser.parse_args()

# data_part = "training" # "test", "training"
# seed = 42

# chan_order = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
#                 'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
#                 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
#                 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
#                 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
#                 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
#                 'O1', 'Oz', 'O2']

# epoched_data = []
# img_conditions = []

# for s in range(args.n_ses):
#     eeg_dir = os.path.join('Raw_data', 'sub-'+
#         format(args.sub,'02'), 'ses-'+format(s+1,'02'), 'raw_eeg_'+
#         data_part+'.npy')
#     eeg_data = np.load(os.path.join(args.project_dir, eeg_dir),
#         allow_pickle=True).item()

#     ch_names = eeg_data['ch_names']
#     sfreq = eeg_data['sfreq']
#     ch_types = eeg_data['ch_types']
#     eeg_data = eeg_data['raw_eeg_data']
#     # Convert to MNE raw format
#     info = mne.create_info(ch_names, sfreq, ch_types)
#     raw = mne.io.RawArray(eeg_data, info)
#     del eeg_data

#     events = mne.find_events(raw, stim_channel='stim')
#     raw.pick_channels(chan_order, ordered=True)
#     idx_target = np.where(events[:,2] == 99999)[0]
#     events = np.delete(events, idx_target, 0)
#     ### Epoching, baseline correction and resampling ###
#     # * [0, 1.0]
#     epochs = mne.Epochs(raw, events, tmin=-.2, tmax=1.0, baseline=(None,0),
#         preload=True)
#     # epochs = mne.Epochs(raw, events, tmin=-.2, tmax=.8, baseline=(None,0),
#     # 	preload=True)
#     del raw
#     # Resampling
#     if args.sfreq < 1000:
#         epochs.resample(args.sfreq)
#     ch_names = epochs.info['ch_names']
#     times = epochs.times

#     ### Sort the data ###
#     data = epochs.get_data()
#     events = epochs.events[:,2]
#     img_cond = np.unique(events)
#     # print(events.shape, img_cond.shape)
#     del epochs
#     # Select only a maximum number of EEG repetitions
#     if data_part == 'test':
#         max_rep = 20
#     else:
#         max_rep = 2
#     # Sorted data matrix of shape:
#     # Image conditions × EEG repetitions × EEG channels × EEG time points
#     sorted_data = np.zeros((len(img_cond),max_rep,data.shape[1],
#         data.shape[2]))
#     for i in range(len(img_cond)):
#         # Find the indices of the selected image condition
#         idx = np.where(events == img_cond[i])[0]
#         # Randomly select only the max number of EEG repetitions
#         idx = shuffle(idx, random_state=seed, n_samples=max_rep)
#         sorted_data[i] = data[idx]
#     del data
#     epoched_data.append(sorted_data[:, :, :, 50:])
#     img_conditions.append(img_cond)
#     del sorted_data

# ### Output ###
# # return epoched_data, img_conditions, ch_names, times
# print(len(epoched_data), epoched_data[0].shape, img_conditions[0].shape)
# # Image conditions × EEG repetitions × EEG channels × EEG time points
# # (4, 200, 20, 63, 250)


# # whitened_test = epoched_data
# # ### Merge and save the test data ###
# # for s in range(args.n_ses):
# #     if s == 0:
# #         merged_test = whitened_test[s]
# #     else:
# #         merged_test = np.append(merged_test, whitened_test[s], 1)
# # del whitened_test
# # # Shuffle the repetitions of different sessions
# # idx = shuffle(np.arange(0, merged_test.shape[1]), random_state=seed)
# # merged_test = merged_test[:,idx]
# # print('xxx', merged_test.shape)

# ### Merge and save the training data ###
# whitened_train = epoched_data
# img_conditions_train = img_conditions
# for s in range(args.n_ses):
#     if s == 0:
#         white_data = whitened_train[s]
#         img_cond = img_conditions_train[s]
#     else:
#         white_data = np.append(white_data, whitened_train[s], 0)
#         img_cond = np.append(img_cond, img_conditions_train[s], 0)
# del whitened_train, img_conditions_train
# # Data matrix of shape:
# # Image conditions × EGG repetitions × EEG channels × EEG time points
# merged_train = np.zeros((len(np.unique(img_cond)), white_data.shape[1]*2,
#     white_data.shape[2],white_data.shape[3]))
# for i in range(len(np.unique(img_cond))):
#     # Find the indices of the selected category
#     idx = np.where(img_cond == i+1)[0]
#     for r in range(len(idx)):
#         if r == 0:
#             ordered_data = white_data[idx[r]]
#         else:
#             ordered_data = np.append(ordered_data, white_data[idx[r]], 0)
#     merged_train[i] = ordered_data
# # Shuffle the repetitions of different sessions
# idx = shuffle(np.arange(0, merged_train.shape[1]), random_state=seed)
# merged_train = merged_train[:,idx]
# print('xxx', merged_train.shape)


import numpy as np
train_img_feature = np.load('./dnn_feature/' + 'clip_feature_maps_training.npy', allow_pickle=True)
test_img_feature = np.load('./dnn_feature/' + 'clip_feature_maps_test.npy', allow_pickle=True)
test_center = np.load('./dnn_feature/' + 'center_clip.npy', allow_pickle=True)

train_img_feature = np.squeeze(train_img_feature)
test_img_feature = np.squeeze(test_img_feature)

print(train_img_feature.shape, test_img_feature.shape, test_center.shape)