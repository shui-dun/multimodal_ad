import os
import time
import numpy as np
import torch
import shutil
import torch.backends.cudnn as cudnn
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from graph4nlp.pytorch.modules.prediction.classification.graph_classification.feedforward_nn import FeedForwardNNLayer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from graph4nlp.pytorch.modules.evaluation.accuracy import Accuracy
from graph4nlp.pytorch.modules.graph_construction import (
    NodeEmbeddingBasedGraphConstruction,
    NodeEmbeddingBasedRefinedGraphConstruction,
)
from graph4nlp.pytorch.modules.graph_embedding_initialization.embedding_construction import (
    WordEmbedding,
)
from graph4nlp.pytorch.modules.graph_embedding_initialization.graph_embedding_initialization import (
    GraphEmbeddingInitialization,
)
from graph4nlp.pytorch.modules.graph_embedding_learning import GAT, GGNN, GraphSAGE
from graph4nlp.pytorch.modules.loss.general_loss import GeneralLoss
from graph4nlp.pytorch.modules.prediction.classification.graph_classification import FeedForwardNN, AvgPooling
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.generic_utils import EarlyStopping, grid, to_cuda
from graph4nlp.pytorch.modules.utils.logger import Logger
from graph4nlp.pytorch.data.data import from_batch
from graph4nlp.pytorch.data.multimodal_dataset import *

import torch
from torch import nn
import os
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import shutil
from datasets import load_dataset, Dataset, DatasetDict
import json
import csv
import random
import os
import shutil
import os
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor
import evaluate
import numpy as np
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import DataLoader
import json
import numpy as np
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from transformers import EarlyStoppingCallback, IntervalStrategy



class Config:
    """Configuration class for model hyperparameters."""
    the_name = "pitt_multimodal"

    root_dir = "./pitt_data"

    gpu = 2

    batch_size = 1

    accumulation_steps = 20

    # Graph embedding initializer strategy
    emb_strategy = 'w2v_bert_bilstm'
    # emb_strategy = 'w2v_bilstm'

    epochs = 20

    lr = 0.001

    # hidden size of Graph embedding initializer, gnn, classifier
    num_hidden = 300

    # cuda config
    no_cuda = False

    graph_construction_args = {
        'graph_name': 'dependency',
        # 'graph_name': 'node_emb', # dynamic graph
        # 'graph_name': 'node_emb_refined', # fused graph
        # intrinsic graph
        'dynamic_init_graph_name': 'dependency',
        'root_dir': root_dir,
        'thread_number': 10,
        'port': 9012,
        'timeout': 15000,
        'edge_strategy': 'homogeneous',
        'merge_strategy': 'tailhead',
    }

    # embedding initializer args
    num_rnn_layers = 1
    no_fix_word_emb = False
    no_fix_bert_emb = False
    bert_model_name = "bert-base-uncased"
    pretrained_word_emb_name = '840B'
    rnn_dropout = 0.1
    word_dropout = 0.4

    # GNN setting
    # for ablating GNN
    delete_gnn_layers = False
    # GNN type
    # gnn = 'ggnn'
    gnn = 'graphsage'
    gnn_direction_option = 'bi_fuse'
    gnn_dropout = 0.4
    # gnn layer number
    gnn_num_layers = 2

    # ReduceLROnPlateau setting
    scheduler_patience = 2
    lr_reduce_factor = 0.5

    # DataLoader num_workers
    num_workers = 0

    out_dir_prefix = 'out/pitt_data/'

    out_dir_suffix = 'default'

    # EarlyStopping's patience
    stopper_patience = 10

    seed = 123456

    # dynamic graph setting
    # Specify similarity metric function type
    gl_metric_type = "weighted_cosine"
    # Specify the number of heads for multi-head similarity metric function
    gl_num_heads = 1
    # Specify the top k value for knn neighborhood graph sparsificaiton
    gl_top_k = None
    # Specify the epsilon value (i.e., between 0 and 1) for epsilon neighborhood graph sparsificaiton
    gl_epsilon = 0.95
    # Specify the smoothness ratio (i.e., between 0 and 1) for graph regularization on smoothness
    gl_smoothness_ratio = 0.1  # kernel: 0.5, weighted_cosine: 0.4
    # Specify the connectivity ratio (i.e., between 0 and 1) for graph regularization on connectivity
    gl_connectivity_ratio = 0.1  # kernel: 0.1, weighted_cosine: 0.1
    # Specify the sparsity ratio (i.e., between 0 and 1) for graph regularization on sparsity
    gl_sparsity_ratio = 0.3  # kernel: 0.3, weighted_cosine: 0
    # The dimension of hidden layers
    gl_num_hidden = 300
    # Specify the fusion value (between 0 and 1) for combining initial and learned adjacency matrices.
    init_adj_alpha = 0.5
    
        
    # GAT setting
    gat_attn_dropout = 0.2
    gat_negative_slope = 1
    gat_num_heads = 3
    gat_num_out_heads = 3
    gat_residual = True
    
    # graphsage setting
    graphsage_aggreagte_type = 'lstm'


    sampling_rate = 16000

    audioModelName = "microsoft/wavlm-base"

    truncateAudio = 80

    audioDataPath = "pitt_audio_and_text_data"

    fuse_method = "concat"
    # fuse_method = "crossnet"

class CrossNet(nn.Module):
    def __init__(self, in_features, layer_num=2, seed=1024):
        # def __init__(self, in_features, device, layer_num=2, seed=1024):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.kernels = torch.nn.ParameterList(
            [nn.Parameter(nn.init.xavier_normal_(torch.empty(in_features, 1))) for i in range(self.layer_num)])
        self.bias = torch.nn.ParameterList(
            [nn.Parameter(nn.init.zeros_(torch.empty(in_features, 1))) for i in range(self.layer_num)])
        # self.to(device)

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
            dot_ = torch.matmul(x_0, xl_w)
            x_l = dot_ + self.bias[i] + x_l
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


class MulmodalClassifier(nn.Module):
    def __init__(self, vocab, label_model):
        super(MulmodalClassifier, self).__init__()
        self.vocab_model = vocab
        self.label_model = label_model
        self.graph_name = Config.graph_construction_args["graph_name"]

        assert not (
                self.graph_name in ("node_emb", "node_emb_refined") and Config.gnn == "gat"
        ), "dynamic graph construction does not support GAT"

        # The embedding construction module aims to learn the initial node/edge embeddings for the input graph before being consumed by the subsequent GNN model.
        embedding_style = {
            #  both single-token (i.e., containing single token) and multi-token (i.e., containing multiple tokens) items (i.e., node/edge)
            "single_token_item": True if self.graph_name != "ie" else False,
            # w2v_bilstm strategy means we first use word2vec embeddings to initialize each item, and then apply a BiLSTM encoder to encode the whole graph (assuming the node order reserves the sequential order in raw text)
            # the `w2v_bert_bilstm` strategy in addition applies the BERT encoder to the whole graph (i.e., sequential text), the concatenation of the BERT embedding and word2vec embedding instead of word2vec embedding will be fed into the BiLSTM encoder.
            "emb_strategy": Config.emb_strategy,
            "num_rnn_layers": Config.num_rnn_layers,
            "bert_model_name": Config.bert_model_name,
            "bert_lower_case": True,
        }

        self.graph_initializer = GraphEmbeddingInitialization(
            word_vocab=self.vocab_model.in_word_vocab,
            embedding_style=embedding_style,
            hidden_size=Config.num_hidden,
            word_dropout=Config.word_dropout,
            rnn_dropout=Config.rnn_dropout,
            fix_word_emb=not Config.no_fix_word_emb,
            fix_bert_emb=not Config.no_fix_bert_emb,
        )

        use_edge_weight = False

        if self.graph_name == "node_emb":
            self.graph_topology = NodeEmbeddingBasedGraphConstruction(
                sim_metric_type=Config.gl_metric_type,
                num_heads=Config.gl_num_heads,
                top_k_neigh=Config.gl_top_k,
                epsilon_neigh=Config.gl_epsilon,
                smoothness_ratio=Config.gl_smoothness_ratio,
                connectivity_ratio=Config.gl_connectivity_ratio,
                sparsity_ratio=Config.gl_sparsity_ratio,
                input_size=Config.num_hidden,
                hidden_size=Config.gl_num_hidden,
            )
            use_edge_weight = True
        elif self.graph_name == "node_emb_refined":
            self.graph_topology = NodeEmbeddingBasedRefinedGraphConstruction(
                Config.init_adj_alpha,
                sim_metric_type=Config.gl_metric_type,
                num_heads=Config.gl_num_heads,
                top_k_neigh=Config.gl_top_k,
                epsilon_neigh=Config.gl_epsilon,
                smoothness_ratio=Config.gl_smoothness_ratio,
                connectivity_ratio=Config.gl_connectivity_ratio,
                sparsity_ratio=Config.gl_sparsity_ratio,
                input_size=Config.num_hidden,
                hidden_size=Config.gl_num_hidden,
            )
            use_edge_weight = True

        if "w2v" in self.graph_initializer.embedding_layer.word_emb_layers:
            self.word_emb = self.graph_initializer.embedding_layer.word_emb_layers[
                "w2v"
            ].word_emb_layer
        else:
            self.word_emb = WordEmbedding(
                self.vocab_model.in_word_vocab.embeddings.shape[0],
                self.vocab_model.in_word_vocab.embeddings.shape[1],
                pretrained_word_emb=self.vocab_model.in_word_vocab.embeddings,
                fix_emb=not Config.no_fix_word_emb,
            ).word_emb_layer

        if Config.gnn == "gat":
            heads = [Config.gat_num_heads] * (Config.gnn_num_layers - 1) + [
                Config.gat_num_out_heads
            ]
            self.gnn = GAT(
                Config.gnn_num_layers,
                Config.num_hidden,
                Config.num_hidden,
                Config.num_hidden,
                heads,
                direction_option=Config.gnn_direction_option,
                feat_drop=Config.gnn_dropout,
                attn_drop=Config.gat_attn_dropout,
                negative_slope=Config.gat_negative_slope,
                residual=Config.gat_residual,
                activation=F.elu,
                allow_zero_in_degree=True,
            )
        elif Config.gnn == "graphsage":
            self.gnn = GraphSAGE(
                Config.gnn_num_layers,
                Config.num_hidden,
                # If `num_layers` is larger than 1, while the `hidden_size` is an int format value, we assume that all the hidden layers have the same size
                Config.num_hidden,
                Config.num_hidden,
                Config.graphsage_aggreagte_type,
                direction_option=Config.gnn_direction_option,
                feat_drop=Config.gnn_dropout,
                bias=True,
                norm=None,
                activation=F.relu,
                use_edge_weight=use_edge_weight,
            )
        elif Config.gnn == "ggnn":
            self.gnn = GGNN(
                Config.gnn_num_layers,
                Config.num_hidden,
                Config.num_hidden,
                Config.num_hidden,
                feat_drop=Config.gnn_dropout,
                direction_option=Config.gnn_direction_option,
                bias=True,
                use_edge_weight=use_edge_weight,
            )
        else:
            raise RuntimeError("Unknown gnn type: {}".format(Config.gnn))

        # create a dictionary that maps the label name to an integer and vice versa
        label2id = {"HC": "0", "AD": "1"}
        id2label = {"0": "HC", "1": "AD"}
        self.audioModel = AutoModelForAudioClassification.from_pretrained(
            Config.audioModelName, num_labels=len(id2label), label2id=label2id, id2label=id2label
        )

        # fix audioModel
        for param in self.audioModel.parameters():
            param.requires_grad = False

        if "wavlm-large" in Config.audioModelName:
            self.audioLinear = nn.Linear(1024, Config.num_hidden, bias=True)
        else: # wavlm-base
            self.audioLinear = nn.Linear(768, Config.num_hidden, bias=True)

        self.graph_pool = AvgPooling()

        self.graphActivation = nn.ReLU()

        if Config.fuse_method == "crossnet":
            self.cross = CrossNet(2 * Config.num_hidden)

        # This is a high-level graph classification prediction module which consists of a graph pooling component and a multilayer perceptron (MLP).
        self.clf = FeedForwardNNLayer(
            # input_size: The dimension of input graph embeddings.
            2 * (2 * Config.num_hidden if Config.gnn_direction_option == "bi_sep" else Config.num_hidden),
            # num_class: The number of classes for classification.
            Config.num_classes,
            # hidden_size: Hidden size per NN layer.
            [Config.num_hidden],
            self.graphActivation,
        )

        # cross entropy loss
        self.loss = GeneralLoss("CrossEntropy")

    def forward(self, graph_list, input_values, attention_mask, tgt=None, require_loss=True):
        audioModelOutput = self.audioModel(input_values=input_values, attention_mask=attention_mask,
                                           output_hidden_states=True, labels=tgt)
        audio_hidden_states = audioModelOutput.hidden_states
        audio_last_hidden_states = audio_hidden_states[-1]
        
        audio_last_hidden_states = torch.mean(audio_last_hidden_states, dim=1)
        
        audio_last_hidden_states = self.audioLinear(audio_last_hidden_states)
        
        batch_gd = self.graph_initializer(graph_list)

        # run dynamic graph construction if turned on
        if hasattr(self, "graph_topology") and hasattr(self.graph_topology, "dynamic_topology"):
            batch_gd = self.graph_topology.dynamic_topology(batch_gd)
        

        if not Config.delete_gnn_layers:
            self.gnn(batch_gd)
        else:
            # if delete_gnn_layers is set to True, we will delete the gnn layers
            batch_gd.node_features['node_emb'] = batch_gd.node_features['node_feat']

        graph_emb = self.graph_pool(batch_gd, "node_emb")

        if Config.fuse_method == "concat":
            # concatanate graph_emb and audio_last_hidden_states
            cat_emb = torch.cat((graph_emb, audio_last_hidden_states), dim=1)
        elif Config.fuse_method == "crossnet":
            cat_emb = torch.cat((graph_emb, audio_last_hidden_states), dim=1)
            # pass cat_emb to crossnet
            cat_emb = self.cross(cat_emb)
        
        # torch.save(cat_emb, "cat_emb.pt")

        # run graph classifier
        logits = self.clf(cat_emb)

        if require_loss:
            loss = self.loss(logits, tgt)
            return logits, loss
        else:
            return logits

    def post_process(self, logits, label_names):
        logits_list = []

        for idx in range(len(logits)):
            logits_list.append(logits[idx].cpu().clone().numpy())

        pred_tags = [label_names[pred.argmax()] for pred in logits_list]
        return pred_tags

    @classmethod
    def load_checkpoint(cls, model_path):
        return torch.load(model_path)


class ModelHandler:
    """A high-level model handler"""
    def __init__(self):
        super(ModelHandler, self).__init__()
        self._init_seed_and_device()
        self.out_dir = os.path.join(Config.out_dir_prefix, Config.out_dir_suffix)
        self.logger = Logger(
            self.out_dir,
            config={k: v for k, v in Config.__dict__.items() if not (k.startswith('__') or k == 'device')},
            overwrite=True,
        )
        self.logger.write(self.out_dir)
        # self._load_dataset()
        # self._build_dataloader()
        # self._build_model()
        # self._build_optimizer()
        # self._build_evaluation()

    def _init_seed_and_device(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(Config.gpu)

        np.random.seed(Config.seed)
        torch.manual_seed(Config.seed)

        if not Config.no_cuda and torch.cuda.is_available():
            # as CUAD_VISIBLE_DEVICES is set before, we need to set the device to be cuda:0
            Config.device = "cuda:0"
            # Config.device = torch.device("cuda" if Config.gpu < 0 else "cuda:%d" % Config.gpu)
            torch.cuda.manual_seed(Config.seed)
            torch.cuda.manual_seed_all(Config.seed)
            torch.backends.cudnn.deterministic = True
            cudnn.benchmark = False
        else:
            Config.device = torch.device("cpu")

    def _load_dataset(self, encoded_pitt_data):
        self.encoded_pitt_data = encoded_pitt_data
        self.graph_name = Config.graph_construction_args["graph_name"]
        topology_subdir = "{}_graph".format(self.graph_name)

        if self.graph_name == "node_emb_refined":
            topology_subdir += "_{}".format(
                Config.graph_construction_args["dynamic_init_graph_name"]
            )

        # the raw data file is stored under the raw directory under the dataset’s root directory. Similarly, the processed data file is stored under the processed sub-directory.
        # MyDataset is defined in multimodal_dataset.py
        dataset = MyDataset(
            # The path of dataset.
            root_dir=Config.graph_construction_args["root_dir"],
            #  The directory name of processed path.
            topology_subdir=topology_subdir,
            # The name of graph construction method.
            # we devide the process of dependency graph building into several steps:
            # 1. Parsing. It will parse the input paragraph into list of sentences. Then for each sentence, we will parse the dependency relations.
            # 2. Sub-graph construction. We will construct subgraph for each sentence.
            # 3. Graph merging. We will merge sub-graphs into one big graph.
            graph_name=self.graph_name,
            dynamic_init_graph_name=Config.graph_construction_args["dynamic_init_graph_name"],
            # The strategy to merge sub-graphs.
            merge_strategy=Config.graph_construction_args["merge_strategy"],
            # 'homogeneous' means we will drop the edge type information and only preserve the connectivity information.
            edge_strategy=Config.graph_construction_args["edge_strategy"],
            min_word_vocab_freq=1,
            word_emb_size=300,
            seed=Config.seed,
            thread_number=Config.graph_construction_args["thread_number"],
            port=Config.graph_construction_args["port"],
            timeout=Config.graph_construction_args["timeout"],
            reused_label_model=None,
            pretrained_word_emb_name=Config.pretrained_word_emb_name,
            audio_data=self.encoded_pitt_data,
        )

        self.dataset = dataset

    def _build_dataloader(self):
        self.train_dataloader = DataLoader(
            self.dataset.train,
            batch_size=Config.batch_size,
            # set to True to have the data reshuffled at every epoch
            shuffle=True,
            # Setting the argument num_workers as a positive integer will turn on multi-process data loading with the specified number of loader worker processes.
            num_workers=Config.num_workers,
            collate_fn=self.dataset.collate_fn,
        )
        if not hasattr(self.dataset, "val"):
            self.dataset.val = self.dataset.train
        self.val_dataloader = DataLoader(
            self.dataset.val,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            collate_fn=self.dataset.collate_fn,
        )
        self.test_dataloader = DataLoader(
            self.dataset.test,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            collate_fn=self.dataset.collate_fn,
        )

        self.vocab_model = self.dataset.vocab_model
        # label mappgings from a label set.
        self.label_model = self.dataset.label_model

        Config.num_classes = self.label_model.num_classes
        self.num_train = len(self.dataset.train)
        self.num_val = len(self.dataset.val)
        self.num_test = len(self.dataset.test)
        print(
            "Train size: {}, Val size: {}, Test size: {}".format(
                self.num_train, self.num_val, self.num_test
            )
        )
        self.logger.write(
            "Train size: {}, Val size: {}, Test size: {}".format(
                self.num_train, self.num_val, self.num_test
            )
        )

    def _build_model(self):
        self.model = MulmodalClassifier(self.vocab_model, self.label_model)
        self.model = self.model.to(Config.device)

    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=Config.lr)
        # used to stop the training process when the model is not improving.
        self.stopper = EarlyStopping(
            # save_model_path
            os.path.join(
                self.out_dir,
                Constants._SAVED_WEIGHTS_FILE,
            ),
            # Number of epochs with no improvement after which training will be stopped.
            patience=Config.stopper_patience,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            # In `min` mode, lr will be reduced when the quantity monitored has stopped decreasing; in `max` mode it will be reduced when the quantity monitored has stopped increasing.
            mode="max",
            # Factor by which the learning rate will be reduced. new_lr = lr * factor.
            factor=Config.lr_reduce_factor,
            # Number of epochs with no improvement after which learning rate will be reduced. For example, if `patience = 2`, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn't improved then.
            patience=Config.scheduler_patience,
            verbose=True,
        )

    def _build_evaluation(self):
        self.metric = Accuracy(["accuracy", "precision", "recall", "F1"])

    def train(self):
        # record how many consecutive epochs the val acc is equal to 1    
        number_of_consecutive_one = 0
        startTime = time.time()
        for epoch in range(Config.epochs):

            # It sets the mode to train
            self.model.train()
            for ind, data in enumerate(self.train_dataloader):
                input_values = data["input_values"].to(Config.device)
                attention_mask = data["attention_mask"].to(Config.device)
                # ground truth
                tgt = to_cuda(data["tgt_tensor"], Config.device)

                # this is a batch of graphs, each graph is a connected component of the large graph
                data["graph_data"] = data["graph_data"].to(Config.device)
                logits, loss = self.model(data["graph_data"], input_values, attention_mask, tgt, require_loss=True)

                # add graph regularization loss if available
                if data["graph_data"].graph_attributes.get("graph_reg", None) is not None:
                    loss = loss + data["graph_data"].graph_attributes["graph_reg"]

                loss = loss / Config.accumulation_steps

                # backward
                loss.backward()

                if (ind + 1) % Config.accumulation_steps == 0 or (ind + 1) == len(self.train_dataloader):
                    # update parameters
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # evaluate on validation set for each epoch
            val_acc = self.evaluate(self.val_dataloader)[0]
            print("val_acc: {}".format(val_acc))
            self.logger.write("val_acc: {}".format(val_acc))

            test_acc = self.evaluate(self.test_dataloader)[0]
            print("test_acc: {}".format(test_acc))
            self.logger.write("test_acc: {}".format(test_acc))


            if val_acc == 1.0:
                number_of_consecutive_one += 1
                if number_of_consecutive_one >= 3:
                    break

            self.scheduler.step(val_acc)

            print("Epoch: [{} / {}] | Val Acc: {:.4f} | Test Acc: {:.4f}".format(epoch + 1, Config.epochs, val_acc, test_acc))
            self.logger.write(
                "Epoch: [{} / {}] | Val Acc: {:.4f} | Test Acc: {:.4f}".format(epoch + 1, Config.epochs, val_acc, test_acc))

            if self.stopper.step(val_acc, self.model):
                break

        print("final_test_acc:{}".format(test_acc))
        self.logger.write("final_test_acc:{}".format(test_acc))

        runtime = time.time() - startTime
        print('train time: {:.2f}s'.format(runtime))
        self.logger.write('train time: {:.2f}s\n'.format(runtime))

        return self.stopper.best_score

    def evaluate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            pred_collect = []
            gt_collect = []
            for data in dataloader:
                tgt = to_cuda(data["tgt_tensor"], Config.device)
                data["graph_data"] = data["graph_data"].to(Config.device)
                input_values = data["input_values"].to(Config.device)
                attention_mask = data["attention_mask"].to(Config.device)
                logits = self.model(data["graph_data"], input_values, attention_mask, require_loss=False)
                pred_collect.append(logits)
                gt_collect.append(tgt)

            pred_collect = torch.max(torch.cat(pred_collect, 0), dim=-1)[1].cpu()
            gt_collect = torch.cat(gt_collect, 0).cpu()
            score = self.metric.calculate_scores(ground_truth=gt_collect, predict=pred_collect)
            print(score)
            return score

    def test(self):
        startTime = time.time()
        # restored best saved model
        self.model = MulmodalClassifier.load_checkpoint(self.stopper.save_model_path)
        acc = self.evaluate(self.test_dataloader)
        dur = time.time() - startTime
        print(
            "Test examples: {} | Time: {:.2f}s |  Test Acc: {:.4f} |  Test Precision: {:.4f} |  Test Recall: {:.4f} |  Test F1: {:.4f}".format(
                self.num_test, dur, acc[0], acc[1][1], acc[2][1], acc[3][1])
        )
        self.logger.write(
            "Test examples: {} | Time: {:.2f}s |  Test Acc: {:.4f} |  Test Precision: {:.4f} |  Test Recall: {:.4f} |  Test F1: {:.4f}".format(
                self.num_test, dur, acc[0], acc[1][1], acc[2][1], acc[3][1])
        )

        return acc



class KFold:

    def __init__(self, k=10, path='pitt.txt', outputPath='raw/'):
        self._k = k

        self._cleanDir()

        self._load_audio_dataset()

    def _load_audio_dataset(self):
        pitt_data = load_dataset(Config.audioDataPath)

        pitt_data["train"] = pitt_data["train"].shuffle()
        # pitt_data = pitt_data["train"].shuffle(seed=42)

        feature_extractor = AutoFeatureExtractor.from_pretrained(Config.audioModelName)
        pitt_data = pitt_data.cast_column("audio", Audio(sampling_rate=Config.sampling_rate))

        def preprocess_function(examples):
            audio_arrays = [x["array"] for x in examples["audio"]]
            inputs = feature_extractor(
                audio_arrays, sampling_rate=feature_extractor.sampling_rate,
                max_length=Config.sampling_rate * Config.truncateAudio, truncation=True, padding='longest'
            )
            return inputs

        self.encoded_pitt_data = pitt_data.map(preprocess_function, remove_columns="audio", batched=True)

        self.encoded_pitt_data.set_format("torch")


    def _cleanDir(self):
        processedDir = os.path.join(Config.graph_construction_args['root_dir'], 'processed/')
        if os.path.exists(processedDir):
            shutil.rmtree(processedDir)


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
        self._cleanDir()
        perLen = len(self.encoded_pitt_data["train"]) // self._k
        startInd = i * perLen
        endInd = (i + 1) * perLen
        self.trainAndValInds = list(range(startInd)) + list(range(endInd, len(self.encoded_pitt_data["train"])))
        
        self.trainSet = self.encoded_pitt_data["train"].select(self.trainAndValInds)

        self.testInds = list(range(startInd, endInd))
        self.testSet = self.encoded_pitt_data["train"].select(self.testInds)

        trainStat = self._statistics(self.trainSet)
        testStat = self._statistics(self.testSet)
        stat = "statistics: train: {}, test: {}".format(trainStat, testStat)
        print(stat)
        self.kFold_encoded_pitt_data = DatasetDict({'train': self.trainSet, 'test': self.testSet})


def train_and_test():
    k = 10
    kFold = KFold(k)
    accList = []
    pList = []
    rList = []
    f1List = []
    for i in range(k):
        kFold.writeIth(i)
        Config.seed = random.randint(1, 20000000)

        runner = ModelHandler()
        runner._load_dataset(kFold.kFold_encoded_pitt_data)

        runner._build_dataloader()
        runner._build_model()
        runner._build_optimizer()
        runner._build_evaluation()


        runner.train()
        acc = runner.test()
        accList.append(acc[0])
        pList.append(acc[1][1])
        rList.append(acc[2][1])
        f1List.append(acc[3][1])
    return {
        "avgAcc": np.mean(accList),
        "stdAcc": np.std(accList),
        "accList": accList,
        "avgP": np.mean(pList),
        "stdP": np.std(pList),
        "pList": pList,
        "avgR": np.mean(rList),
        "stdR": np.std(rList),
        "rList": rList,
        "avgF1": np.mean(f1List),
        "stdF1": np.std(f1List),
        "f1List": f1List
    }

logger = Logger(Config.out_dir_prefix, overwrite=True)

for turn in range(0, 5):
    s = Config.the_name
    Config.out_dir_suffix = s
    result = train_and_test()
    logger.write("{}_{} {}".format(s, turn, str(result)))
        


