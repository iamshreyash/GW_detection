# GW_detection
# Gravitational Wave Detection Using Convolutional Neural Networks (CNNs)

## Project Overview
This project develops a Convolutional Neural Network (CNN) model to detect gravitational waves in LIGO data, specifically focusing on the GW150914 event—the first direct observation of gravitational waves. The model performs binary classification to distinguish between noise and signal (gravitational wave) segments in the time-series strain data from the LIGO Hanford detector (`H-H1_LOSC_16_V1-1126259446-32.hdf5`).

### Objectives
- Preprocess LIGO strain data using whitening, bandpass filtering, and normalization.
- Augment the dataset with simulated noise and gravitational wave signals to increase training data.
- Train a CNN model to classify 1-second segments as noise (0) or signal (1).
- Evaluate the model’s performance using accuracy, precision, recall, F1-score, and confusion matrix.

### Dataset
- **Source**: LIGO Open Science Center (LOSC) dataset for GW150914 (`H-H1_LOSC_16_V1-1126259446-32.hdf5`).
- **Size**: 32 seconds of strain data at 16384 Hz (524,288 samples).
- **Segmentation**: Split into 1-second windows (~32 real segments).
- **Augmentation**: Added 500 noise and 500 signal segments (total ~1032 segments), later increased to 1000 each (total ~2032 segments).
- **Labels**: Signal (1) if segment contains GW150914 event (within 0.2 seconds of GPS time 1126259442.4), Noise (0) otherwise.

### Methodology
1. **Preprocessing**:
   - Whitening: Remove frequency-dependent noise using Power Spectral Density (PSD).
   - Bandpass Filtering: Focus on 20–500 Hz frequency band.
   - Normalization: Standardize data to zero mean and unit variance.
2. **Data Augmentation**:
   - Noise: Generated using the PSD of the original data.
   - Signals: Simulated using PyCBC’s `get_td_waveform` (SEOBNRv4 approximant, masses 36 & 29 solar masses, SNR=20).
3. **Model**:
   - 1D CNN: Two convolutional layers (16 & 32 filters, kernel size 8), max-pooling, dropout (0.5), L2 regularization.
   - Optimizer: Adam (learning rate=0.0001, clipnorm=1.0).
   - Loss: Binary cross-entropy.
   - Class Weights: `{0: 1.0, 1: 2.0}` to prioritize signal detection.
4. **Training**:
   - Epochs: 20 (later increased to 100 in updated script).
   - Batch Size: 16.
   - Validation Split: 20% test set.
5. **Evaluation**:
   - Metrics: Accuracy, precision, recall, F1-score.
   - Visualizations: Confusion matrix, training history plots (accuracy & loss).

### Results
- **Initial Performance**: Validation accuracy ~40–57%, high loss (~10⁹), indicating numerical instability.
- **Optimized Performance** (1D CNN):
  - Validation Accuracy: 94.69%.
  - Noise: Precision=1.00, Recall=0.89, F1=0.94.
  - Signal: Precision=0.90, Recall=1.00, F1=0.95.
  - Observations: Good signal detection (no false negatives), but some false positives for noise. Training loss showed spikes.
- **Future Scope**: Switching to a 2D CNN with spectrogram inputs could improve accuracy to ~95–98% by capturing frequency-time patterns.

### Dependencies
- Python 3.x
- Libraries: `h5py`, `numpy`, `scipy`, `tensorflow`, `pycbc==2.8.2`, `cryptography==43.0.3`, `matplotlib`, `scikit-learn`
- Install dependencies:
  ```bash
  pip install pycbc==2.8.2 cryptography==43.0.3 tensorflow h5py scipy numpy matplotlib scikit-learn
