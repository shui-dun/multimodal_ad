import numpy as np
import torch
from datasets import Audio
from datasets import load_dataset, DatasetDict, concatenate_datasets
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification
import torch.nn as nn
import itertools


class Config:
    logPath = "./pitt_clippo_like.log"

    device = torch.device("cuda:0")

    sampling_rate = 16000

    finetune_audioModel_batch = 4

    finetune_audioModel_accumulation_steps = 5

    finetune_audioModel_lr = 1.5e-5

    finetune_audioModel_epoch = 100

    audioModelName = "microsoft/wavlm-base"

    truncateAudio = 80

    audioDataPath1 = "pitt_audio_and_text_data"
    audioDataPath2 = "pitt_audio_and_text_data_tts"

    add_val_set = False

    t_of_contrastive_loss = 0.2

    # gpu index
    device_ids = [0, 1, 2, 3]


def writelog(s):
    with open(Config.logPath, "a+") as f:
        f.write(s + '\n')


class KFold:

    def __init__(self, k=10):
        self._k = k

        self._load_audio_dataset()

    def _load_audio_dataset(self):
        dataset1 = load_dataset(Config.audioDataPath1)["train"]
        dataset2 = load_dataset(Config.audioDataPath2)["train"]

        def merge_datasets(example1, index):
            example2 = dataset2[index]
            return {
                "audio1": example1["audio"],
                "audio2": example2["audio"],
                "label": example1["label"],  # or example2["label"], as they are the same
                "transcription1": example1["transcription"],
                "transcription2": example2["transcription"],
            }

        pitt_data = dataset1.map(
            merge_datasets, with_indices=True, batched=True)

        pitt_data = pitt_data.shuffle()

        pitt_data = pitt_data.remove_columns(['audio', 'transcription'])

        pitt_data = pitt_data.cast_column(
            "audio1", Audio(sampling_rate=Config.sampling_rate))

        pitt_data = pitt_data.cast_column(
            "audio2", Audio(sampling_rate=Config.sampling_rate))

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            Config.audioModelName)

        def preprocess_function1(examples):
            audio_arrays = [x["array"] for x in examples["audio1"]]
            inputs = feature_extractor(
                audio_arrays, sampling_rate=feature_extractor.sampling_rate,
                max_length=Config.sampling_rate * Config.truncateAudio, truncation=True, padding='longest'
            )
            return inputs

        encoded_pitt_data = pitt_data.map(
            preprocess_function1, remove_columns="audio1", batched=True)

        encoded_pitt_data = encoded_pitt_data.rename_column(
            "input_values", "input_values1")
        encoded_pitt_data = encoded_pitt_data.rename_column(
            "attention_mask", "attention_mask1")

        def preprocess_function2(examples):
            audio_arrays = [x["array"] for x in examples["audio2"]]
            inputs = feature_extractor(
                audio_arrays, sampling_rate=feature_extractor.sampling_rate,
                max_length=Config.sampling_rate * Config.truncateAudio, truncation=True, padding='longest'
            )
            return inputs

        encoded_pitt_data = encoded_pitt_data.map(
            preprocess_function2, remove_columns="audio2", batched=True)

        encoded_pitt_data = encoded_pitt_data.rename_column(
            "input_values", "input_values2")
        encoded_pitt_data = encoded_pitt_data.rename_column(
            "attention_mask", "attention_mask2")

        self.encoded_pitt_data = DatasetDict({'train': encoded_pitt_data})

        self.encoded_pitt_data.set_format("torch")

    def _statistics(self, data):
        ans = dict()
        for d in data:
            # if ans.get(d[1]) is None:
            label = int(d['label'])
            if ans.get(label) is None:
                ans[label] = 1
            else:
                ans[label] += 1
        return ans

    def writeIth(self, i):
        # self._cleanDir()
        perLen = len(self.encoded_pitt_data["train"]) // self._k
        startInd = i * perLen
        endInd = (i + 1) * perLen
        self.trainAndValInds = list(
            range(startInd)) + list(range(endInd, len(self.encoded_pitt_data["train"])))

        if Config.add_val_set:
            self.trainInds = self.trainAndValInds[:int(
                len(self.trainAndValInds) * 0.9)]
            self.valInds = self.trainAndValInds[int(
                len(self.trainAndValInds) * 0.9):]
            self.trainSet = self.encoded_pitt_data["train"].select(
                self.trainInds)
            self.valSet = self.encoded_pitt_data["train"].select(self.valInds)
        else:
            self.trainSet = self.encoded_pitt_data["train"].select(
                self.trainAndValInds)

        self.testInds = list(range(startInd, endInd))
        self.testSet = self.encoded_pitt_data["train"].select(self.testInds)

        trainStat = self._statistics(self.trainSet)
        testStat = self._statistics(self.testSet)
        stat = "statistics: train: {}, test: {}".format(trainStat, testStat)
        print(stat)
        if Config.add_val_set:
            self.kFold_encoded_pitt_data = DatasetDict(
                {'train': self.trainSet, 'test': self.testSet, 'val': self.valSet})
        else:
            self.kFold_encoded_pitt_data = DatasetDict(
                {'train': self.trainSet, 'test': self.testSet})

def contrastive_loss(feature1, feature2, t=Config.t_of_contrastive_loss):
    import torch.nn.functional as F
    normalized_feature1 = F.normalize(feature1, dim=1)
    normalized_feature2 = F.normalize(feature2, dim=1)
    logits = torch.matmul(normalized_feature1,
                          normalized_feature2.T) * torch.tensor(t).exp()
    n = logits.size(0)
    labels = torch.arange(n).to(logits.device)
    loss_1 = F.cross_entropy(logits, labels, reduction='mean')
    loss_2 = F.cross_entropy(logits.T, labels, reduction='mean')
    loss = (loss_1 + loss_2) / 2
    return loss


def finetuneAudioModel(encoded_pitt_data):
    label2id = {"HC": "0", "AD": "1"}
    id2label = {"0": "HC", "1": "AD"}
    num_labels = len(label2id)
    model = AutoModelForAudioClassification.from_pretrained(
        Config.audioModelName, num_labels=num_labels, label2id=label2id, id2label=id2label
    )

    def evalate(dataloader, model, device):
        ground_truth = np.array([])
        preds = []

        model.eval()
        for i, batch in enumerate(dataloader):
            input_values = batch["input_values1"].to(device)
            attention_mask = batch["attention_mask1"].to(device)
            labels = batch["label"].to(device)

            with torch.no_grad():
                outputs = model(input_values=input_values,
                                attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                # print(logits)
                pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()
                labels = labels.cpu().numpy()
            if ground_truth.size == 0:
                ground_truth = labels
                preds = pred
            else:
                ground_truth = np.concatenate([ground_truth, labels])
                preds = np.concatenate([preds, pred])
        acc = accuracy_score(y_true=ground_truth, y_pred=preds)
        return acc

    model = nn.DataParallel(model, device_ids=Config.device_ids)

    model.to(Config.device)

    train_dataloader = DataLoader(
        encoded_pitt_data["train"], batch_size=Config.finetune_audioModel_batch, shuffle=True)
    test_dataloader = DataLoader(
        encoded_pitt_data["test"], batch_size=Config.finetune_audioModel_batch)
    optimizer = AdamW(model.parameters(), lr=Config.finetune_audioModel_lr)

    optimizer = nn.DataParallel(optimizer, device_ids=Config.device_ids)

    acc = 0

    for epoch in range(Config.finetune_audioModel_epoch):
        model.train()
        for i, batch in enumerate(train_dataloader):
            input_values1 = batch["input_values1"].to(Config.device)
            attention_mask1 = batch["attention_mask1"].to(Config.device)
            input_values2 = batch["input_values2"].to(Config.device)
            attention_mask2 = batch["attention_mask2"].to(Config.device)
            labels = batch["label"].to(Config.device)
            outputs1 = model(input_values=input_values1, attention_mask=attention_mask1,
                             labels=labels, output_hidden_states=True)
            outputs2 = model(input_values=input_values2, attention_mask=attention_mask2,
                             labels=labels, output_hidden_states=True)
            loss1 = outputs1.loss
            loss1 = loss1.mean()
            loss2 = outputs2.loss
            loss2 = loss2.mean()
            outputs1_last_hidden_state = outputs1.hidden_states[-1][:, 0, :]
            outputs2_last_hidden_state = outputs2.hidden_states[-1][:, 0, :]

            cons_loss = contrastive_loss(
                outputs1_last_hidden_state, outputs2_last_hidden_state)

            loss = loss1 + loss2 + cons_loss
            loss = loss / Config.finetune_audioModel_accumulation_steps

            loss.backward()

            if (i + 1) % Config.finetune_audioModel_accumulation_steps == 0 or (i + 1) == len(train_dataloader):                
                optimizer.module.step()
                
                optimizer.module.zero_grad()

        acc = evalate(test_dataloader, model, Config.device)
        print("finetune_audioModel: acc: {}".format(acc))

    print("finetune_audioModel: final_test_acc: {}".format(acc))
    writelog("finetune_audioModel: final_test_acc: {}".format(acc))


def train_and_test():
    k = 10
    kFold = KFold(k)
    for i in range(k):
        kFold.writeIth(i)
        finetuneAudioModel(kFold.kFold_encoded_pitt_data)


for turn in range(0, 5):
    train_and_test()
