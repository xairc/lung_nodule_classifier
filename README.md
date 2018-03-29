# lung_nodule_classifier
lung nodule classifier for 9 attribute

# Prerequisites
python 3.5.2      
pytorch 0.2.0

# lidc 9 attribute
 - malignancy (range: 1~5)
 - sphericity (range: 1~5)
 - margin (range: 1~5)
 - spiculation (range: 1~5)
 - texture (range: 1~5)
 - calcification (range: 1~6)
 - internal structure (range: 1~4)
 - lobulation (range: 1~5)
 - subtlety (range: 1~5)

# Result
|                    |               |               |               |               |               |               |               |               |               |             | 
|--------------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|-------------| 
|                    | MAE under 0.1 | MAE under 0.2 | MAE under 0.3 | MAE under 0.4 | MAE under 0.5 | MAE under 0.6 | MAE under 0.7 | MAE under 0.8 | MAE under 0.9 | MAE under 1 | 
| malignancy         | 15.24%        | 24.76%        | 42.86%        | 60.00%        | 71.43%        | 80.95%        | 86.67%        | 92.38%        | 93.33%        | 94.29%      | 
| sphericity         | 17.14%        | 29.52%        | 50.48%        | 60.00%        | 67.62%        | 72.38%        | 80.95%        | 84.76%        | 89.52%        | 92.38%      | 
| margin             | 19.05%        | 38.10%        | 48.57%        | 60.95%        | 67.62%        | 77.14%        | 82.86%        | 85.71%        | 86.67%        | 89.52%      | 
| spiculation        | 21.90%        | 37.14%        | 63.81%        | 72.38%        | 76.19%        | 81.90%        | 83.81%        | 87.62%        | 88.57%        | 91.43%      | 
| texture            | 22.86%        | 59.05%        | 76.19%        | 80.95%        | 84.76%        | 84.76%        | 86.67%        | 89.52%        | 91.43%        | 92.38%      | 
| calcification      | 89.52%        | 89.52%        | 92.38%        | 93.33%        | 93.33%        | 93.33%        | 97.14%        | 99.05%        | 99.05%        | 100.00%     | 
| internal_structure | 100.00%       | 100.00%       | 100.00%       | 100.00%       | 100.00%       | 100.00%       | 100.00%       | 100.00%       | 100.00%       | 100.00%     | 
| lobulation         | 19.05%        | 31.43%        | 52.38%        | 67.62%        | 75.24%        | 79.05%        | 85.71%        | 87.62%        | 91.43%        | 92.38%      | 
| subtlety           | 15.24%        | 32.38%        | 50.48%        | 57.14%        | 67.62%        | 71.43%        | 80.95%        | 86.67%        | 86.67%        | 88.57%      | 


# Data Download
 - https://luna16.grand-challenge.org/download/
   - download data and candidates_V2.csv
 - https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI
   - download Radiologist Annotations/Segmentations (XML)

# Training
 - set luna and lidc path: set config_training.py
 - export path: source export_path.sh
 - train, val idx npy make: python preprocess/make_validate_npy.py
 - preprocess data make: python preprocess/prepare.py
 - train: sh train.sh
 
# Reference Code
https://github.com/lfz/DSB2017    
https://github.com/juliandewit/kaggle_ndsb2017  
