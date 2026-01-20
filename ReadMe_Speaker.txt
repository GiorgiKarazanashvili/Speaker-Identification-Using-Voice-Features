# Speaker Identification Using Voice Features

CS 438/638 Machine Learning - Fall 2025  
Giorgi Karazanashvili

## Dataset

- 213 audio samples from 7 speakers
- 5 diverse speakers (Dani, Giorgi, Niko, Shani, Khoi) for training
- 2 brothers (Murad, Teymur) as test subjects

## Features

64 acoustic features extracted using `librosa`:
- MFCCs (52): 13 coefficients × 4 statistics (mean, std, min, max)
- Spectral (8): Centroid, rolloff, bandwidth, zero-crossing rate × 2 statistics
- Pitch (4): Fundamental frequency × 4 statistics

## Project Structure

```
speaker_identification.py    # Main code 
all_speaker_features.csv     # Extracted features from audio
features_processed.csv       # Scaled features (62 after initial correlation removal >0.95)
train.csv, val.csv, test.csv # Data splits
train_selected.csv           # Lasso-selected features (43)
train_corr_reduced.csv       # Correlation-reduced features (29)
logreg_best_model.pkl        # Trained Lasso model
logreg_corr_best_model.pkl   # Trained Correlation model
*.png                        # Generated plots
```

## Requirements

```
pip install numpy pandas librosa scikit-learn matplotlib seaborn
```

## How to Run

1. Place audio files in a directory (WAV format, named `speaker-01.wav`, `speaker-02.wav`, etc.)
2. Update `audio_dir` in the script
3. Run cells sequentially

## Methods

### Preprocessing
- StandardScaler normalization
- Removed 2 features with correlation > 0.95

### Feature Selection
1. Lasso (L1): C=0.25 → 43 features
2. Correlation-based: |r| > 0.7 threshold → 29 features

### Model
- Logistic Regression with L2 regularization
- Best C=0.1 for both models

## Results

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| Lasso (43 features) | 100% | 100% | 100% |
| Correlation (29 features) | 97.4% | 98.4% | 97.1% |

### Brother Classification
- Neither model confused one brother for the other
- Single error: Murad → Giorgi 

## Output Files

- `speaker_distribution.png` - Sample counts per speaker
- `before_scaling_boxplot.png` / `after_scaling_boxplot.png` - Scaling visualization
- `correlation_heatmap.png` - Feature correlations
- `logreg_tuning.png` / `logreg_corr_tuning.png` - Hyperparameter tuning
- `learning_curves.png` - Training vs validation performance
- `confusion_matrices.png` - Final test results
- `roc_curves.png` - ROC curves for all classes
- `error_venn_diagram.png` - Model error comparison

## Conclusion

Voice features learned from diverse speakers can effectively distinguish between brothers with similar voices. The Lasso model achieved perfect classification, while the correlation-reduced model made only one error (not a brother confusion).