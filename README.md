# Decoding Nature Images from EEG for Object Recognition

## Datasets
many thanks for sharing good datasets!
1. [Things-EEG2](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub)
2. [Things-MEG](https://elifesciences.org/articles/82580) (updating)

## Pre-processing
### Script path
- ./preprocessing/
### Data Path 
- raw data: ./Data/Things-EEG2/Raw_data/
- proprocessed eeg data: ./Data/Things-EEG2/Preprocessed_data_250Hz/
### Steps
1. pre-processing EEG data of each subject
   - modify `preprocessing_utils.py` as you need.
     - choose channels
     - epoching
     - baseline correction
     - resample to 250 Hz
     - sort by condition
     - **Multivariate Noise Normalization**
   - `python preprocessing.py` for each subject. 

2. get the center images of each test condition (for testing, contrast with EEG features)
   - get images from original Things dataset but discard the images used in EEG test sessions.
  
## Get the Features from Pre-Trained Models
### Script path
- ./dnn_feature_extraction/
### Data Path (follow the original dataset setting)
- raw image: ./Data/Things-EEG2/Image_set/image_set/
- preprocessed eeg data: ./Data/Things-EEG2/Preprocessed_data/
- features of each images: ./Data/Things-EEG2/DNN_feature_maps/full_feature_maps/model/pretrained-True/
- features been packaged: ./Data/Things-EEG2/DNN_feature_maps/pca_feature_maps/model/pretrained-True/
- features of condition centers: ./Data/Things-EEG2/Image_set/
### Steps
1. obtain feature maps with each pre-trained model with `obtain_feature_maps_xxx.py` (clip, vit, resnet...)
2. package all the feature maps into one .npy file with `feature_maps_xxx.py`
3. obtain feature maps of center images with `center_fea_xxx.py`
   - save feature maps of each center image into `center_all_image_xxx.npy`
   - save feature maps of each condition into `center_xxx.npy` (used in training)

## Training and Testing
### Script path
- ./nice_stand.py

## Visualization - updating
### Script path
- ./visualization/
### Steps

## Milestones
1. nice_v0.50 NICE (image contraste eeg)
2. nice+_v0.55 NICE++ （image contraste eeg guided by text - 'a photo of a xxx'）

<!-- ## Citation
```
@misc{song2023decoding,
  title = {Decoding {{Natural Images}} from {{EEG}} for {{Object Recognition}}},
  author = {Song, Yonghao and Liu, Bingchuan and Li, Xiang and Shi, Nanlin and Wang, Yijun and Gao, Xiaorong},
  year = {2023},
  month = nov,
  number = {arXiv:2308.13234},
  eprint = {2308.13234},
  primaryclass = {cs, eess, q-bio},
  publisher = {{arXiv}},
  doi = {10.48550/arXiv.2308.13234},
  archiveprefix = {arxiv}
}
```
## Acknowledgement

## References

## License -->

