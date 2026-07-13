# Marine Acoustics: Killer Whale Call Detection

## Overview

This project uses machine learning and deep learning to automatically detect killer whale, also known as orca, calls in underwater audio recordings.

Hydrophone recordings often contain waves, vessel noise, weather sounds, and calls from other marine animals. Manually reviewing long recordings is slow and difficult. This project processes the recordings into short audio segments and classifies each segment as either:

| Label | Meaning |
|---|---|
| `1` | Killer whale call detected |
| `0` | No killer whale call detected |

The repository contains two classification approaches:

1. **Perch audio embeddings with logistic regression**
2. **Mel spectrogram images with a ResNet18 CNN**

## Objectives

The main objectives are to:

1. Process large underwater audio recordings automatically.
2. Resample recordings to a consistent sampling rate.
3. divide recordings into fixed length segments.
4. Detect killer whale calls despite background ocean noise.
5. Compare embedding based and spectrogram based classification methods.
6. Report the timestamp and confidence of detected calls.

## System Workflow

```text
Hydrophone Recording
        |
        v
Audio Resampling to 32 kHz
        |
        v
Five Second Audio Segmentation
        |
        +-------------------------------+
        |                               |
        v                               v
Perch Audio Embeddings          Mel Spectrogram Images
        |                               |
        v                               v
Standard Scaler                 ResNet18 CNN
        |                               |
        v                               v
Logistic Regression             Binary Classification
        |                               |
        +---------------+---------------+
                        |
                        v
             Orca or Non Orca Result
```

## Project Structure

```text
Marine_acoustics-main/
|
|-- Sampling_and_Processing.py   # Metadata processing, segmentation and spectrogram generation
|-- embeddings_extractor.py      # Perch embedding extraction and logistic regression training
|-- cnn.py                       # ResNet18 training and evaluation using spectrograms
|-- main.py                      # Interactive inference pipeline for new recordings
|-- graph.py                     # Model comparison and evaluation graphs
|-- train.csv                    # Timestamp level training labels and recording metadata
`-- README.md                    # Project documentation
```

## Main Components

### 1. Audio and Metadata Processing

`Sampling_and_Processing.py` provides functions to:

1. Read the labels from `train.csv`.
2. Separate positive and negative timestamps.
3. Merge nearby labeled intervals.
4. Normalize or split intervals into four second windows.
5. Export five second audio clips by adding one second of context.
6. Resample audio to `32,000 Hz`.
7. Convert audio clips into mel spectrogram images.

Important functions include:

| Function | Purpose |
|---|---|
| `sort_csv()` | Filters positive or negative examples from `train.csv` |
| `init_metadata()` | Groups consecutive timestamps into intervals |
| `process_metadata()` | Normalizes or splices the intervals |
| `chop_and_resample()` | Extracts clips and resamples them to 32 kHz |
| `convert_audio_to_spectrograms()` | Creates mel spectrogram PNG files |
| `splice_video()` | Generates overlapping windows for a new recording |

### 2. Perch Embedding Model

`embeddings_extractor.py` uses the Google Perch 2.0 model from TensorFlow Hub to convert each audio segment into a numerical embedding.

The embeddings are standardized using `StandardScaler` and classified using logistic regression.

This approach is useful because the pretrained embedding model can transfer acoustic patterns learned from bioacoustic recordings to killer whale detection.

### 3. CNN Spectrogram Model

`cnn.py` trains a pretrained ResNet18 model using spectrogram images.

The final classification layer is replaced according to the number of folders detected by PyTorch `ImageFolder`. For binary classification, the spectrogram dataset should contain two class folders.

### 4. Inference Pipeline

`main.py` performs inference on a new WAV recording by:

1. Loading the saved logistic regression model and scaler.
2. Loading the Perch embedding model.
3. Dividing the recording into overlapping windows.
4. Extracting an embedding from every audio segment.
5. Predicting whether the segment contains an orca call.
6. Printing the timestamp and confidence for positive detections.

### 5. Evaluation Graphs

`graph.py` generates:

1. A radar chart for class specific precision, recall and F1 score.
2. A normalized confusion matrix breakdown.
3. A single point ROC comparison using false positive rate and true positive rate.

## Dataset Format

The included `train.csv` contains timestamp level annotations. Important columns include:

| Column | Description |
|---|---|
| `wav_filename` | Original audio recording name |
| `start_time_s` | Start time of the labeled one second interval |
| `duration_s` | Duration of the labeled interval |
| `location` | Recording location |
| `date` | Recording date |
| `Id` | Combined WAV filename and timestamp identifier |
| `Expected` | Binary target label, `1.0` for orca and `0.0` for non orca |

The raw WAV recordings are not included in this repository and must be placed in the configured input folders.

## Requirements

Install Python and the required packages:

```bash
pip install numpy pandas librosa soundfile matplotlib tqdm natsort scikit-learn joblib tensorflow tensorflow-hub torch torchvision
```

A virtual environment is recommended:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy pandas librosa soundfile matplotlib tqdm natsort scikit-learn joblib tensorflow tensorflow-hub torch torchvision
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas librosa soundfile matplotlib tqdm natsort scikit-learn joblib tensorflow tensorflow-hub torch torchvision
```

## Suggested Folder Structure

Create the following folders before processing the dataset:

```text
data/
|
|-- raw_audio/
|-- clips/
|   |-- positive/
|   `-- negative/
|-- spectrograms/
|   |-- positive/
|   `-- negative/
`-- inference/
    |-- input/
    `-- output/
```

## Data Preparation

The following example extracts positive metadata and creates resampled clips:

```python
from Sampling_and_Processing import (
    sort_csv,
    init_metadata,
    process_metadata,
    chop_and_resample,
)

positive_rows = sort_csv(1.0)
positive_metadata = init_metadata({}, positive_rows, merge_window=2.0)
positive_metadata = process_metadata(positive_metadata, method=0.0)

chop_and_resample(
    positive_metadata,
    input_dir="data/raw_audio",
    output_dir="data/clips/positive",
)
```

For negative clips, use:

```python
negative_rows = sort_csv(0.0)
negative_metadata = init_metadata({}, negative_rows, merge_window=2.0)
negative_metadata = process_metadata(negative_metadata, method=1.0)

chop_and_resample(
    negative_metadata,
    input_dir="data/raw_audio",
    output_dir="data/clips/negative",
)
```

## Training the Perch Embedding Classifier

```python
import joblib

from embeddings_extractor import (
    load_perch,
    extract_embeddings,
    train_Lregression_model,
)

inference_function = load_perch()

embeddings, labels = extract_embeddings(
    positive_path="data/clips/positive",
    negative_path="data/clips/negative",
    inference_function=inference_function,
)

classifier, scaler = train_Lregression_model(embeddings, labels)

joblib.dump(classifier, "LR_model.joblib")
joblib.dump(scaler, "scaler.joblib")
```

The generated files are required by `main.py`:

```text
LR_model.joblib
scaler.joblib
```

## Training the CNN Model

First, convert the positive and negative clips into spectrograms:

```python
from Sampling_and_Processing import convert_audio_to_spectrograms

convert_audio_to_spectrograms(
    "data/clips/positive",
    "data/spectrograms/positive",
)

convert_audio_to_spectrograms(
    "data/clips/negative",
    "data/spectrograms/negative",
)
```

Train and evaluate ResNet18:

```python
from cnn import train_and_test_resnet

model = train_and_test_resnet(
    data_dir="data/spectrograms",
    epochs=5,
    batch_size=32,
    test_split=0.2,
)
```

The class names are automatically taken from the subfolder names inside `data/spectrograms`.

## Running Inference

Before running `main.py`, update the hardcoded paths:

```python
path = "data/inference/input/"
o_path = "data/inference/output/"
```

Also make sure that the folder passed to `start_pipeline()` is the same folder in which the segmented WAV files are saved. For example:

```python
start_pipeline(
    inference_dict,
    inference_function,
    classifier,
    scaler,
    spliced_audio_folder=o_path,
)
```

Then run:

```bash
python main.py
```

Enter the filename of the WAV recording when prompted. Type `exit` to stop the program.

Example detection output:

```text
ORCA detected with confidence of 91.4%!
Video: sample_recording.wav
Timestamp: 25.0s - 29.0s
```

## Recorded Evaluation Values

The following confusion matrix values are currently hardcoded in `graph.py` for visualization:

| Model | True Negative | False Positive | False Negative | True Positive | Approximate Accuracy |
|---|---:|---:|---:|---:|---:|
| Perch embeddings | 78 | 18 | 26 | 81 | 78.33% |
| ResNet18 CNN | 139 | 48 | 22 | 195 | 82.67% |

These values are not automatically read from newly trained models. Update `graph.py` after running new experiments.

Run the graph script using:

```bash
python graph.py
```

## Important Configuration Notes

1. `train.csv` must be available in the current working directory when `sort_csv()` is called.
2. Raw audio recordings must match the filenames listed in `train.csv`.
3. The first Perch model load requires an internet connection to download the model.
4. The repository does not currently include raw audio files, trained model files, scaler files or CNN weights.
5. The paths in `main.py` are currently machine specific and must be changed.
6. The segmented file location used during inference must match `spliced_audio_folder` in `start_pipeline()`.
7. GPU acceleration is used automatically by the CNN when CUDA or Apple MPS is available.

## Limitations

1. Ocean noise and sounds from other animals may produce false positives.
2. The quality of the classifier depends on the balance and diversity of the training data.
3. The current inference program uses fixed local paths.
4. Model persistence is implemented for logistic regression, but CNN weight saving and loading must still be added.
5. Evaluation charts currently use manually entered confusion matrix values.
6. The current train and test split is random and may include clips from the same original recording in both sets.

## Future Improvements

1. Move all paths and parameters into a configuration file.
2. Add command line arguments for training and inference.
3. Save and load CNN model weights.
4. Split data by original recording to prevent data leakage.
5. Add cross validation and class balancing.
6. Apply noise reduction and audio augmentation.
7. Generate evaluation plots directly from model predictions.
8. Export detections to CSV with timestamps and confidence scores.
9. Build a web interface for uploading and analyzing recordings.
10. Support real time hydrophone stream analysis.

## References

1. William A. Watkins Collection of Marine Mammal Sound Recordings and Data, New Bedford Whaling Museum.
2. Woods Hole Oceanographic Institution, Historic Marine Mammal Sound Archive.
3. Google Research, Perch bioacoustics model.
4. TensorFlow Hub.
5. PyTorch ResNet18.
6. Librosa audio processing library.

## Project Status

This project is an academic prototype for marine bioacoustic research. It demonstrates preprocessing, feature extraction, machine learning classification, deep learning classification and timestamp based inference for killer whale call detection.
