from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset
import os
import csv
import numpy as np

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# create output directory
if not os.path.exists("pitt_audio_and_text_data_tts"):
    os.makedirs("pitt_audio_and_text_data_tts")

with open("pitt_audio_and_text_data/metadata.csv", "r") as f:
    csvDictReader = csv.DictReader(f)
    for row in csvDictReader:
        try:
            print(row["file_name"])
            outputPath = os.path.join("pitt_audio_and_text_data_tts", row["file_name"][:-3] + "wav")
            # if the file already exists, skip
            if os.path.exists(outputPath):
                continue
            print("processing: ", row["file_name"])
            # split the text into sentences
            sentences = row["transcription"].split(".")
            audio_segments = []
            # process each sentence
            for text in sentences:
                if (text == ' '):
                    continue
                text += '. '
                print(text)
                inputs = processor(text=text, return_tensors="pt")
                speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
                audio_segments.append(speech.numpy())
            concatenated_audio = np.concatenate(audio_segments, axis=0)
            print(concatenated_audio.shape)
            sf.write(outputPath, concatenated_audio, samplerate=16000)
        except Exception as e:
            print("error: {} for file: {}".format(e, row["file_name"]))
            continue