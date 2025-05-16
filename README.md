# The Simpsons Characters Classification

## 📖 Project Overview

This project implements a Convolutional Neural Network (CNN) to classify images of top three most frequent Simpson characters 

Key features include:

* Filtering dataset to the three most common characters
* Data augmentation (rotation, shift, shear, zoom, flip)
* Training/validation/test split
* Deep CNN built with Keras Sequential API
* Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
* Visualizations of training/validation accuracy and loss
* Exporting model to TensorFlow SavedModel, TF-Lite, and TFJS formats

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* `pip` package manager

### Python Dependencies

All required packages and versions are listed in `requirements.txt`. To install:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```text
tensorflow==2.18.0
tensorflowjs==4.22.0
matplotlib==3.9.2
numpy==1.26.4
gdown
kaggle
```

> **Note:** Ensure `kaggle.json` (your Kaggle API credentials) is placed in the project root.

---

## ⚙️ Project Structure

```
├── README.md
├── requirements.txt
├── simpsons_dataset/           # Downloaded Kaggle dataset
│   └── simpsons_dataset/
├── simpsons_dataset_filtered/  # Filtered top-3 classes
│   ├── character_1/
│   ├── character_2/
│   └── character_3/
├── simpsons_classification.py  # Main training script
├── best_simpsons_model.keras    # Checkpointed best model
├── simpsons_model.keras         # Final Keras model
├── saved_model_simpsons/        # TensorFlow SavedModel format
├── tflite/                      # TF-Lite model + labels
│   ├── simpsons_model.tflite
│   └── labels.txt
├── tfjs_simpsons_model/         # TFJS format
└── training_metrics.png         # Training plot image
```

---

## 🛠️ Usage

1. **Download dataset**
   The script automatically downloads and unzips the dataset from Kaggle.

2. **Filter top-3 classes**
   It selects the three most frequent Simpson characters and prepares a filtered directory.

3. **Train the model**

   ```bash
   python simpsons_classification.py
   ```

   * Training will run with data augmentation and callbacks.
   * Best model saved to `best_simpsons_model.keras`.

4. **Evaluate**
   After training, the script prints test-set accuracy.

5. **Exports**

   * Keras `.keras` model
   * TensorFlow SavedModel in `saved_model_simpsons/`
   * TF-Lite model in `tflite/` with labels
   * TensorFlow\.js format in `tfjs_simpsons_model/`

---

## 📈 Results

* **Training/Validation Accuracy & Loss:** See `training_metrics.png`
* **Test Accuracy:** (Printed at end of script; aim ≥ 85%)

---

## 🧩 Customization

* **Dataset Selection:** Modify `data_path` variable to point to any dataset with ≥1000 images.
* **Model Architecture:** Edit layers in the Sequential model block.
* **Hyperparameters:** Tweak learning rate, batch size, epochs, or augmentations in `ImageDataGenerator`.

---
Submission

Proyek Klasifikasi Gambar
