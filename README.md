# 🎙️ Hotword Detection System

This project provides a complete pipeline for training, evaluating, and deploying a hotword (keyword) detection model based on MFCC features and a CNN classifier. It includes both offline analysis and real-time audio stream detection.

---

## 📁 Project Structure

```
project/
├── model.py           # CLI tool for training and analyzing audio files
├── service.py         # Real-time audio stream recording and hotword detection
├── weights/           # Directory for storing trained model weights
├── dataset/
│   ├── hotword_augmented/  # WAV files with hotword (label 1)
│   └── not_hotword/        # WAV files without hotword (label 0)
├── output/            # Recorded audio segments from streams
└── logs/              # Log files
```

---

## ⚙️ Installation

Ensure you have the following installed:

- Python 3.8+
- FFmpeg
- Required Python libraries:

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
torch
librosa
scikit-learn
numpy
soundfile
```

---

## 🧪 Training the Model (Offline)

```bash
python model.py train --epochs 30 --save_path weights/best_weights.pth
```

**Arguments:**

- `--epochs`: Number of training epochs (default: 20)
- `--reload_every`: Reload dataset every N epochs (default: 5)
- `--save_path`: Path to save model weights
- `--test_size`: Proportion of data used for validation (default: 0.2)

> ⚠️ Ensure you have training data in:
>
> - `dataset/hotword_augmented/`
> - `dataset/not_hotword/`

---

## 🔍 Analyze Audio File

```bash
python model.py analyze path/to/audio.wav --model_path weights/best_weights.pth
```

- The script slides a 1s window through the audio file to detect hotwords.
- Detected segments are saved to the `detected_segments/` directory.

**Arguments:**

- `filepath`: Path to input `.wav` file
- `--model_path`: Path to trained weights (default: `weights/best_weights.pth`)
- `--threshold`: Detection threshold (default: 0.9)

---

## 🌐 Run Real-Time Detection Service

```bash
python service.py run
```

Features:

- 📡 Records audio from an internet radio stream
- 🤖 Performs real-time hotword detection
- 💾 Saves full stream to `output/`
- 📝 Logs activity to `logs/`

**Default settings in ****\`\`****:**

- `AUDIO_URL = "https://radio.kotah.ru/exam"`
- `SAMPLE_RATE = 16000`
- `CHUNK_DURATION = 1.0` (seconds)
- `threshold = 0.9`

---

## 🧪 Examples

```bash
# Train the model
python model.py train --epochs 25

# Analyze an audio file
python model.py analyze audio.wav

# Run real-time detection service
python service.py run
```

---

## 📌 Notes

- Logs are automatically saved to `logs/recording_*.log` and `logs/realtime_*.log`
- Full stream audio recordings are stored in `output/`
- Detected segments during analysis are stored in `detected_segments/`

---

## ✅ License

MIT License
