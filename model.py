import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
import argparse
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 1.0
N_MFCC = 13
WINDOW_SIZE = int(SAMPLE_RATE * DURATION)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AudioDataset(Dataset):
    def __init__(self, files, labels, augment=False):
        self.files = files
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        label = self.labels[idx]
        audio, sr = librosa.load(file, sr=SAMPLE_RATE)

        if len(audio) < WINDOW_SIZE:
            audio = np.pad(audio, (0, WINDOW_SIZE - len(audio)))
        else:
            audio = audio[:WINDOW_SIZE]

        if self.augment:
            if random.random() < 0.5:
                audio += 0.005 * np.random.randn(len(audio))
            if random.random() < 0.5:
                n_steps = random.uniform(-2, 2)
                audio = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=n_steps)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        mfcc = torch.tensor(mfcc).unsqueeze(0).float()  # (1, 13, time)
        return mfcc, torch.tensor(label).float()


class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Без Sigmoid!
        )

    def forward(self, x):
        return self.net(x)


def load_files(test_size):
    pos_files = [f"dataset/hotword_augmented/{f}" for f in os.listdir("dataset/hotword_augmented") if f.endswith(".wav")]
    neg_files = [f"dataset/not_hotword/{f}" for f in os.listdir("dataset/not_hotword") if f.endswith(".wav")]
    all_files = pos_files + neg_files
    labels = [1] * len(pos_files) + [0] * len(neg_files)
    return train_test_split(all_files, labels, test_size=test_size)


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y_hat = model(x).cpu()
            probs = torch.sigmoid(y_hat).numpy()
            preds += list((probs > 0.5).astype(int).flatten())
            targets += list(y.numpy().astype(int).flatten())
    p, r, f1, _ = precision_recall_fscore_support(targets, preds, average="binary")
    print(f"[VAL] Precision: {p:.2f}, Recall: {r:.2f}, F1: {f1:.2f}")


def create_dataloaders(test_size):
    train_files, test_files, train_labels, test_labels = load_files(test_size)
    train_dataset = AudioDataset(train_files, train_labels, augment=True)
    test_dataset = AudioDataset(test_files, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=8)
    return train_loader, val_loader


def train_model(model, epochs=20, reload_every=5, save_path="weights.pth", test_size=0.2):
    model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(epochs):
        if epoch % reload_every == 0:
            train_loader, val_loader = create_dataloaders(test_size)

        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f"Epoch {epoch + 1} — Loss: {loss.item():.4f}")
        evaluate(model, val_loader)

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model


def load_model(path="weights.pth"):
    model = ImprovedCNN()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {path}")
    return model


def detect_hotword(model, filepath, threshold=0.9, save_segments=True, output_dir="detected_segments"):
    import soundfile as sf  # Внутренний импорт для совместимости

    audio, _ = librosa.load(filepath, sr=SAMPLE_RATE)
    model.eval()
    step = int(SAMPLE_RATE * 0.5)
    found = False
    last_detection_time = -float("inf")

    if save_segments and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    segment_count = 0
    for i in range(0, len(audio) - WINDOW_SIZE, step):
        timestamp = i / SAMPLE_RATE
        if timestamp - last_detection_time < 1.0:
            continue

        window = audio[i:i + WINDOW_SIZE]
        mfcc = librosa.feature.mfcc(y=window, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        mfcc_tensor = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            logits = model(mfcc_tensor)
            prob = torch.sigmoid(logits).item()

        if prob > threshold:
            found = True
            last_detection_time = timestamp
            print(f"[+] Hotword detected at {timestamp:.2f}s (prob={prob:.2f})")

            if save_segments:
                pre_start = max(0, i - WINDOW_SIZE)
                post_end = min(len(audio), i + 2 * WINDOW_SIZE)
                segment = audio[pre_start:post_end]
                filename = os.path.join(output_dir, f"hotword_detected_{segment_count}.wav")
                sf.write(filename, segment, SAMPLE_RATE)
                print(f"    Saved segment ({(post_end - pre_start) / SAMPLE_RATE:.2f}s) to: {filename}")
                segment_count += 1

    if not found:
        print("[-] Hotword not detected.")



def main():
    parser = argparse.ArgumentParser(description="Hotword detection CLI")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    train_parser.add_argument("--reload_every", type=int, default=5, help="Reload dataloaders every N epochs")
    train_parser.add_argument("--save_path", type=str, default="weights/learned_weights.pth", help="Path to save the model")
    train_parser.add_argument("--test_size", type=float, default=0.2, help="Test set size ratio")

    detect_parser = subparsers.add_parser("analyze", help="Analyze audio file for hotword")
    detect_parser.add_argument("filepath", type=str, help="Path to audio file")
    detect_parser.add_argument("--model_path", type=str, default="weights/best_weights.pth", help="Path to model weights")
    detect_parser.add_argument("--threshold", type=float, default=0.9, help="Detection threshold")

    args = parser.parse_args()

    if args.command == "train":
        model = ImprovedCNN()
        train_model(
            model,
            epochs=args.epochs,
            reload_every=args.reload_every,
            save_path=args.save_path,
            test_size=args.test_size
        )
    elif args.command == "analyze":
        if not os.path.exists(args.model_path):
            print(f"Model file {args.model_path} not found, please train first.")
            return
        model = load_model(args.model_path)
        detect_hotword(model, args.filepath, threshold=args.threshold)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
