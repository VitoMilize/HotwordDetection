import argparse
import logging
import os
import subprocess
import threading
from datetime import datetime

import librosa
import numpy as np
import torch

from model import load_model

AUDIO_URL = "https://radio.kotah.ru/exam"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FILE = f"output/stream_record_{timestamp}.wav"
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.0
WINDOW_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_logger(name: str, logfile: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler(logfile, encoding='utf-8')
    ch = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


record_logger = create_logger("recording", f"logs/recording_{timestamp}.log")
realtime_logger = create_logger("realtime", f"logs/realtime_{timestamp}.log")

# Загружаем модель
model = load_model("weights/best_weights.pth")
model.to(DEVICE)


def record_stream(audio_url=AUDIO_URL, output_file=OUTPUT_FILE, duration=10800):
    """Запись потока в файл .wav"""
    record_logger.info("Начинается запись потока...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    command = [
        "ffmpeg", "-y",
        "-i", audio_url,
        "-t", str(duration),
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-f", "wav",
        output_file
    ]

    try:
        subprocess.run(command, check=True)
        record_logger.info(f"Поток записан в {output_file}")
    except subprocess.CalledProcessError as e:
        record_logger.error(f"Ошибка записи потока: {e}")


def realtime_detection(audio_url=AUDIO_URL):
    """Онлайн детекция ключевого слова из потока"""
    realtime_logger.info("Подключение к потоку через ffmpeg...")

    # ffmpeg будет перекодировать в WAV 16kHz mono и писать в stdout
    command = [
        "ffmpeg",
        "-i", audio_url,
        "-f", "s16le",  # RAW PCM
        "-acodec", "pcm_s16le",
        "-ac", "1",  # mono
        "-ar", str(SAMPLE_RATE),
        "-"
    ]

    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

        bytes_per_sample = 2  # 16-bit PCM
        chunk_size = int(SAMPLE_RATE * CHUNK_DURATION * bytes_per_sample)

        buffer = b""
        while True:
            chunk = process.stdout.read(chunk_size)
            if not chunk:
                break
            buffer += chunk

            if len(buffer) >= chunk_size:
                try:
                    # Преобразуем в numpy массив float32
                    audio = np.frombuffer(buffer[:chunk_size], dtype=np.int16).astype(np.float32) / 32768.0
                    buffer = buffer[chunk_size:]

                    if len(audio) >= WINDOW_SIZE:
                        mfcc = librosa.feature.mfcc(y=audio[:WINDOW_SIZE], sr=SAMPLE_RATE, n_mfcc=13)
                        x = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
                        with torch.no_grad():
                            prob = torch.sigmoid(model(x)).item()
                        if prob > 0.9:
                            realtime_logger.info(f"Обнаружено ключевое слово! Вероятность: {prob:.2f}")
                except Exception as e:
                    realtime_logger.error(f"Ошибка обработки аудио: {e}")
                    buffer = b""
    except Exception as e:
        realtime_logger.error(f"Ошибка запуска ffmpeg: {e}")



def run_service():
    record_thread = threading.Thread(target=record_stream, kwargs={"duration": 10800})
    realtime_thread = threading.Thread(target=realtime_detection)

    record_thread.start()
    realtime_thread.start()

    record_thread.join()
    realtime_thread.join()


def main():
    parser = argparse.ArgumentParser(description="🎙️ Сервис для записи и анализа аудио")
    subparsers = parser.add_subparsers(dest="command", help="Команды")

    subparsers.add_parser("run", help="Запустить сервис (запись + детекция в реальном времени)")

    args = parser.parse_args()

    if args.command == "run":
        run_service()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
