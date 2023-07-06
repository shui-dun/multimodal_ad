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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from graph4nlp.pytorch.datasets.trec import TrecDataset
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
from graph4nlp.pytorch.modules.prediction.classification.graph_classification import FeedForwardNN
from graph4nlp.pytorch.modules.utils import constants as Constants
from graph4nlp.pytorch.modules.utils.generic_utils import EarlyStopping, grid, to_cuda
from graph4nlp.pytorch.modules.utils.logger import Logger
from graph4nlp.pytorch.data.data import from_batch
import json
import csv
import random


class Config:
    """Configuration class for model hyperparameters."""
    
    batch_size = 10
    
    accumulation_steps = 2
    
    # Graph embedding initializer strategy
    emb_strategy = 'w2v_bert_bilstm'
    # emb_strategy = 'w2v_bilstm'
    
    epochs = 30
      
    lr = 0.001
    
    # hidden size of Graph embedding initializer, gnn, classifier
    num_hidden = 300
    
    # cuda config
    no_cuda = False
    gpu = 3
    
    graph_construction_args = {
        'graph_name': 'dependency', 
        # 'graph_name': 'node_emb', # dynamic graph
        # 'graph_name': 'node_emb_refined', # fused graph
        # intrinsic graph
        'dynamic_init_graph_name': 'dependency',
        'root_dir': './pitt_data', 
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
    
    # pooler setting
    graph_pooling = 'avg_pool'
    max_pool_linear_proj = False
    
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
    gl_smoothness_ratio = 0.1 # kernel: 0.5, weighted_cosine: 0.4
    # Specify the connectivity ratio (i.e., between 0 and 1) for graph regularization on connectivity
    gl_connectivity_ratio = 0.1 # kernel: 0.1, weighted_cosine: 0.1
    # Specify the sparsity ratio (i.e., between 0 and 1) for graph regularization on sparsity
    gl_sparsity_ratio = 0.3 # kernel: 0.3, weighted_cosine: 0
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

    # data augmentation file name
    augment_method_name = None


class TextClassifier(nn.Module):
    """classification model framework."""
    def __init__(self, vocab, label_model):
        super(TextClassifier, self).__init__()
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

        # This is a high-level graph classification prediction module which consists of a graph pooling component and a multilayer perceptron (MLP).
        self.clf = FeedForwardNN(
            # input_size: The dimension of input graph embeddings.
            2 * Config.num_hidden if Config.gnn_direction_option == "bi_sep" else Config.num_hidden,
            # num_class: The number of classes for classification.
            Config.num_classes,
            # hidden_size: Hidden size per NN layer.
            [Config.num_hidden],
            graph_pool_type=Config.graph_pooling,
            # dim should be specified when use_linear_proj is set to True
            dim=Config.num_hidden,
            # An optional linear projection can be applied to node embeddings before conducting max pooling.
            use_linear_proj=Config.max_pool_linear_proj,
        )
        # cross entropy loss
        self.loss = GeneralLoss("CrossEntropy")

    def forward(self, graph_list, tgt=None, require_loss=True):
        batch_gd = self.graph_initializer(graph_list)
        
        # run dynamic graph construction if turned on
        if hasattr(self, "graph_topology") and hasattr(self.graph_topology, "dynamic_topology"):
            batch_gd = self.graph_topology.dynamic_topology(batch_gd)
        
        if not Config.delete_gnn_layers:
            self.gnn(batch_gd)
        else:
            # if delete_gnn_layers is set to True, we will delete the gnn layers
            batch_gd.node_features['node_emb'] = batch_gd.node_features['node_feat']
        
        # run graph classifier
        self.clf(batch_gd)

        logits = batch_gd.graph_attributes["logits"]
        
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
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_evaluation()

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
        
    def _build_dataloader(self):
        self.graph_name = Config.graph_construction_args["graph_name"]
        topology_subdir = "{}_graph".format(self.graph_name)
        
        if self.graph_name == "node_emb_refined":
            topology_subdir += "_{}".format(
                Config.graph_construction_args["dynamic_init_graph_name"]
            )

        # the raw data file is stored under the raw directory under the dataset’s root directory. Similarly, the processed data file is stored under the processed sub-directory.
        dataset = TrecDataset(
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
        )
        
        self.dataset = dataset
        
        self.train_dataloader = DataLoader(
            dataset.train,
            batch_size=Config.batch_size,
            # set to True to have the data reshuffled at every epoch
            shuffle=True,
            # Setting the argument num_workers as a positive integer will turn on multi-process data loading with the specified number of loader worker processes.
            num_workers=Config.num_workers,
            collate_fn=dataset.collate_fn,
        )
        if not hasattr(dataset, "val"):
            dataset.val = dataset.train
        self.val_dataloader = DataLoader(
            dataset.val,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            collate_fn=dataset.collate_fn,
        )
        self.test_dataloader = DataLoader(
            dataset.test,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            collate_fn=dataset.collate_fn,
        )

        self.vocab_model = dataset.vocab_model
        # label mappgings from a label set.
        self.label_model = dataset.label_model

        Config.num_classes = self.label_model.num_classes
        self.num_train = len(dataset.train)
        self.num_val = len(dataset.val)
        self.num_test = len(dataset.test)
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
        self.model = TextClassifier(self.vocab_model, self.label_model).to(Config.device)

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
            # acc list for each batch
            train_acc = []
            for ind, data in enumerate(self.train_dataloader):
                
                # ground truth
                tgt = to_cuda(data["tgt_tensor"], Config.device)
                # this is a batch of graphs, each graph is a connected component of the large graph
                data["graph_data"] = data["graph_data"].to(Config.device)
                logits, loss = self.model(data["graph_data"], tgt, require_loss=True)

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

                # calculate accuracy
                pred = torch.max(logits, dim=-1)[1].cpu()
                train_acc.append(
                    self.metric.calculate_scores(ground_truth=tgt.cpu(), predict=pred.cpu())[0]
                )

            # evaluate on validation set for each epoch
            val_acc = self.evaluate(self.val_dataloader)[0]
            
            if val_acc == 1.0:
                number_of_consecutive_one += 1
                if number_of_consecutive_one >=3:
                    break
            
            self.scheduler.step(val_acc)
            print("Epoch: [{} / {}] | Train Acc: {:.4f} | Val Acc: {:.4f}".format(epoch + 1, Config.epochs, np.mean(train_acc), val_acc))
            self.logger.write("Epoch: [{} / {}] | Train Acc: {:.4f} | Val Acc: {:.4f}".format(epoch + 1, Config.epochs, np.mean(train_acc), val_acc))

            if self.stopper.step(val_acc, self.model):
                break
        
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
                logits = self.model(data["graph_data"], require_loss=False)
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
        self.model = TextClassifier.load_checkpoint(self.stopper.save_model_path)
        acc = self.evaluate(self.test_dataloader)
        dur = time.time() - startTime
        print(
            "Test examples: {} | Time: {:.2f}s |  Test Acc: {:.4f} |  Test Precision: {:.4f} |  Test Recall: {:.4f} |  Test F1: {:.4f}".format(self.num_test, dur, acc[0], acc[1][1], acc[2][1], acc[3][1])
        )
        self.logger.write(
            "Test examples: {} | Time: {:.2f}s |  Test Acc: {:.4f} |  Test Precision: {:.4f} |  Test Recall: {:.4f} |  Test F1: {:.4f}".format(self.num_test, dur, acc[0], acc[1][1], acc[2][1], acc[3][1])
        )
        
        return acc


def augment(data, augmentDataPath):
    """
    load the augmented data and add it to the original data
    """
    ans = []
    with open(augmentDataPath) as f:
        d = json.load(f)
    for line in data:
        ans.append(line)
        if d.get(line[0]) is not None:
            for newLine in d[line[0]]:
                ans.append([newLine, line[1]])
    return ans


class KFold:
    
    def __init__(self, k=10, path='pitt.txt', outputPath='raw/'):
        path = os.path.join(Config.graph_construction_args['root_dir'], path)
        outputPath = os.path.join(Config.graph_construction_args['root_dir'], outputPath)
        self._k = k
        
        self._outputPath = outputPath
        self._cleanDir()
        
        self._pittData = self._readData(path)
        
        random.shuffle(self._pittData)
    
    
    def _cleanDir(self):
        if os.path.exists(self._outputPath):
            shutil.rmtree(self._outputPath)
            
        processedDir = self._outputPath.replace('/raw', '/processed', 1)
        if os.path.exists(processedDir):
            shutil.rmtree(processedDir)
            
        os.mkdir(self._outputPath)
        
    def _readData(self, path):
        ans = []
        with open(path) as f:
            for line in f:
                line = line[:-1].split('\t')
                ans.append(line)
        return ans

    
    def _writeData(self, path, data):
        with open(path, 'w') as f:
            for line in data:
                f.write('{}\t{}\n'.format(line[0], line[1]))


    def _statistics(self, data):
        ans = dict()
        for d in data:
            if ans.get(d[1]) is None:
                ans[d[1]] = 1
            else:
                ans[d[1]] += 1
        return ans

        
    def writeIth(self, i):
        self._cleanDir()
        perLen = len(self._pittData) // self._k
        startInd = i * perLen
        endInd = (i + 1) * perLen
        # print("[{}, {}]".format(startInd, endInd))
        self.trainSet = self._pittData[:startInd] + self._pittData[endInd:]
        if Config.augment_method_name is not None:
            self.trainSet = augment(self.trainSet, os.path.join(Config.graph_construction_args['root_dir'], Config.augment_method_name))
        self.testSet = self._pittData[startInd:endInd]
        # print("train: {}, test: {}".format(self._statistics(self.trainSet), self._statistics(self.testSet)))
        self._writeData(os.path.join(self._outputPath, 'train.txt'), self.trainSet)
        self._writeData(os.path.join(self._outputPath, 'test.txt'), self.testSet)
        trainStat = self._statistics(self.trainSet)
        testStat = self._statistics(self.testSet)
        stat = "statistics: train: {}, test: {}".format(trainStat, testStat)
        print(stat)


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
    s = "pitt_graphsage"
    Config.out_dir_suffix = s
    result = train_and_test()
    logger.write("{}_{} {}".format(s, turn, str(result)))
        


