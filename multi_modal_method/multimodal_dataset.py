import abc
import json
import os
import warnings
from collections import Counter
from copy import deepcopy
from multiprocessing import Pool
from typing import Union
import numpy as np
import stanfordcorenlp
import torch.utils.data
from nltk.tokenize import word_tokenize

from ..data.data import GraphData, to_batch
from ..modules.graph_construction.base import (
    DynamicGraphConstructionBase,
    StaticGraphConstructionBase,
)
from ..modules.graph_construction.constituency_graph_construction import (
    ConstituencyBasedGraphConstruction,
)
from ..modules.graph_construction.dependency_graph_construction import (
    DependencyBasedGraphConstruction,
)
from ..modules.graph_construction.ie_graph_construction import IEBasedGraphConstruction
from ..modules.graph_construction.node_embedding_based_graph_construction import (
    NodeEmbeddingBasedGraphConstruction,
)
from ..modules.graph_construction.node_embedding_based_refined_graph_construction import (
    NodeEmbeddingBasedRefinedGraphConstruction,
)
from ..modules.utils.generic_utils import LabelModel
from ..modules.utils.padding_utils import pad_2d_vals_no_size
from ..modules.utils.tree_utils import Tree
from ..modules.utils.tree_utils import Vocab as VocabForTree
from ..modules.utils.tree_utils import VocabForAll
from ..modules.utils.vocab_utils import VocabModel

import nltk

from . import DataItem, Dataset


class TextAndAudio2LabelDataItem(DataItem):
    def __init__(self, input_text, output_label=None, tokenizer=None, audio=None, audio_attention_mask=None):
        super(TextAndAudio2LabelDataItem, self).__init__(input_text, tokenizer)
        self.output_label = output_label
        self.audio = audio
        self.audio_attention_mask = audio_attention_mask

    def extract(self):
        """
        Returns
        -------
        Input tokens and output tokens
        """
        g: GraphData = self.graph

        input_tokens = []
        for i in range(g.get_node_num()):
            if self.tokenizer is None:
                tokenized_token = g.node_attributes[i]["token"].strip().split()
            else:
                tokenized_token = self.tokenizer(g.node_attributes[i]["token"])

            input_tokens.extend(tokenized_token)

        return input_tokens


class TextAndAudio2LabelDataset(Dataset):
    """
    The dataset for text-to-label applications.
    Parameters
    ----------
    graph_name: str
        The name of graph construction method. E.g., "dependency".
        Note that if it is in the provided graph names (i.e., "dependency", \
            "constituency", "ie", "node_emb", "node_emb_refine"), the following \
            parameters are set by default and users can't modify them:
            1. ``topology_builder``
            2. ``static_or_dynamic``
        If you need to customize your graph construction method, you should rename the \
            ``graph_name`` and set the parameters above.
    root_dir: str, default=None
        The path of dataset.
    topology_builder: Union[StaticGraphConstructionBase, DynamicGraphConstructionBase], default=None
        The graph construction class.
    topology_subdir: str
        The directory name of processed path.
    static_or_dynamic: str, default='static'
        The graph type. Expected in ('static', 'dynamic')
    dynamic_init_graph_name: str, default=None
        The graph name of the initial graph. Expected in (None, "line", \
            "dependency", "constituency").
        Note that if it is in the provided graph names (i.e., "line", "dependency", \
            "constituency"), the following parameters are set by default and users \
            can't modify them:
            1. ``dynamic_init_topology_builder``
        If you need to customize your graph construction method, you should rename the \
            ``graph_name`` and set the parameters above.
    dynamic_init_topology_builder: StaticGraphConstructionBase
        The graph construction class.
    dynamic_init_topology_aux_args: None,
        TBD.
    """

    def __init__(
            self,
            graph_name: str,
            root_dir: str = None,
            static_or_dynamic: str = None,
            topology_builder: Union[
                StaticGraphConstructionBase, DynamicGraphConstructionBase
            ] = DependencyBasedGraphConstruction,
            topology_subdir: str = None,
            dynamic_init_graph_name: str = None,
            dynamic_init_topology_builder: StaticGraphConstructionBase = None,
            dynamic_init_topology_aux_args=None,
            audio_data=None,
            **kwargs,
    ):
        if kwargs.get("graph_type", None) is not None:
            raise ValueError(
                "The argument ``graph_type`` is disgarded. \
                    Please use ``static_or_dynamic`` instead."
            )
        self.data_item_type = TextAndAudio2LabelDataItem
        if graph_name == "dependency":
            topology_builder = DependencyBasedGraphConstruction
            static_or_dynamic = "static"
        elif graph_name == "constituency":
            topology_builder = ConstituencyBasedGraphConstruction
            static_or_dynamic = "static"
        elif graph_name == "ie":
            topology_builder = IEBasedGraphConstruction
            static_or_dynamic = "static"
        elif graph_name == "node_emb":
            topology_builder = NodeEmbeddingBasedGraphConstruction
            static_or_dynamic = "dynamic"
        elif graph_name == "node_emb_refined":
            topology_builder = NodeEmbeddingBasedRefinedGraphConstruction
            static_or_dynamic = "dynamic"
        else:
            print("Your are customizing the graph construction method.")
            if topology_builder is None:
                raise ValueError("``topology_builder`` can't be None if graph is defined by user.")
            if static_or_dynamic is None:
                raise ValueError("``static_or_dynamic`` can't be None if graph is defined by user.")

        if static_or_dynamic == "dynamic":
            if dynamic_init_graph_name is None or dynamic_init_graph_name == "line":
                dynamic_init_topology_builder = None
            elif dynamic_init_graph_name == "dependency":
                dynamic_init_topology_builder = DependencyBasedGraphConstruction
            elif dynamic_init_graph_name == "constituency":
                dynamic_init_topology_builder = ConstituencyBasedGraphConstruction
            elif dynamic_init_graph_name == "ie":
                topology_builder = IEBasedGraphConstruction
            else:
                if dynamic_init_topology_builder is None:
                    raise ValueError(
                        "``dynamic_init_topology_builder`` can't be None \
                            if ``dynamic_init_graph_name`` is defined by user."
                    )

        self.static_or_dynamic = static_or_dynamic

        self.audio_data = audio_data

        super(TextAndAudio2LabelDataset, self).__init__(
            root=root_dir,
            graph_name=graph_name,
            topology_builder=topology_builder,
            topology_subdir=topology_subdir,
            static_or_dynamic=static_or_dynamic,
            dynamic_init_topology_builder=dynamic_init_topology_builder,
            dynamic_init_topology_aux_args=dynamic_init_topology_aux_args,
            **kwargs,
        )

    def parse_file(self, file_path) -> list:
        """
        Read and parse the file specified by `file_path`. The file format
        is specified by each individual task-specific base class. Returns
        all the indices of data items in this file w.r.t. the whole dataset.

        For TextAndAudio2LabelDataset, the format of the input file should contain
        lines of input, each line representing one record of data. The
        input and output is separated by a tab(\t).

        Examples
        --------
        input: How far is it from Denver to Aspen ?    NUM

        DataItem: input_text="How far is it from Denver to Aspen ?", output_label="NUM"

        Parameters
        ----------
        file_path: str
            The path of the input file.

        Returns
        -------
        list
            The indices of data items in the file w.r.t. the whole dataset.
        """
        basename = os.path.splitext(os.path.basename(file_path))[0]

        data = []
        for item in self.audio_data[basename]:
            data_item = TextAndAudio2LabelDataItem(
                input_text=item["transcription"].strip(), output_label=str(item["label"]), tokenizer=self.tokenizer,
                audio=item["input_values"], audio_attention_mask=item["attention_mask"]
            )
            data.append(data_item)
        # with open(file_path, "r", encoding="utf-8") as f:
        #     for line in f:
        #         input, output = line.split("\t")
        #         data_item = TextAndAudio2LabelDataItem(
        #             input_text=input.strip(), output_label=output.strip(), tokenizer=self.tokenizer
        #         )
        #         data.append(data_item)

        return data

    def build_vocab(self):
        data_for_vocab = self.train
        if self.use_val_for_vocab:
            data_for_vocab = data_for_vocab + self.val

        self.vocab_model = VocabModel.build(
            saved_vocab_file=self.processed_file_paths["vocab"],
            data_set=data_for_vocab,
            tokenizer=self.tokenizer,
            lower_case=self.lower_case,
            max_word_vocab_size=self.max_word_vocab_size,
            min_word_vocab_freq=self.min_word_vocab_freq,
            pretrained_word_emb_name=self.pretrained_word_emb_name,
            pretrained_word_emb_url=self.pretrained_word_emb_url,
            pretrained_word_emb_cache_dir=self.pretrained_word_emb_cache_dir,
            word_emb_size=self.word_emb_size,
            share_vocab=True,
        )

        # label encoding
        all_labels = {item.output_label for item in self.train + self.test}
        if "val" in self.__dict__:
            all_labels = all_labels.union({item.output_label for item in self.val})

        self.label_model = LabelModel.build(
            self.processed_file_paths["label"], all_labels=all_labels
        )

    @classmethod
    def _vectorize_one_dataitem(cls, data_item, vocab_model, label_model=None, use_ie=False):
        item = deepcopy(data_item)
        graph: GraphData = item.graph
        token_matrix = []
        for node_idx in range(graph.get_node_num()):
            node_token = graph.node_attributes[node_idx]["token"]
            node_token_id = vocab_model.in_word_vocab.getIndex(node_token, use_ie)
            graph.node_attributes[node_idx]["token_id"] = node_token_id

            token_matrix.append([node_token_id])
        if use_ie:
            for i in range(len(token_matrix)):
                token_matrix[i] = np.array(token_matrix[i][0])
            token_matrix = pad_2d_vals_no_size(token_matrix)
            token_matrix = torch.tensor(token_matrix, dtype=torch.long)
            graph.node_features["token_id"] = token_matrix
        else:
            token_matrix = torch.tensor(token_matrix, dtype=torch.long)
            graph.node_features["token_id"] = token_matrix

        if use_ie and "token" in graph.edge_attributes[0].keys():
            edge_token_matrix = []
            for edge_idx in range(graph.get_edge_num()):
                edge_token = graph.edge_attributes[edge_idx]["token"]
                edge_token_id = vocab_model.in_word_vocab.getIndex(edge_token, use_ie)
                graph.edge_attributes[edge_idx]["token_id"] = edge_token_id
                edge_token_matrix.append([edge_token_id])
            if use_ie:
                for i in range(len(edge_token_matrix)):
                    edge_token_matrix[i] = np.array(edge_token_matrix[i][0])
                edge_token_matrix = pad_2d_vals_no_size(edge_token_matrix)
                edge_token_matrix = torch.tensor(edge_token_matrix, dtype=torch.long)
                graph.edge_features["token_id"] = edge_token_matrix

        if item.output_label is not None:
            assert label_model is not None, "label_model must be specified."
            item.output = label_model.le.transform([item.output_label])[0]
        return item

    def vectorization(self, data_items):
        if self.topology_builder == IEBasedGraphConstruction:
            use_ie = True
        else:
            use_ie = False
        for idx in range(len(data_items)):
            data_items[idx] = self._vectorize_one_dataitem(
                data_items[idx], self.vocab_model, label_model=self.label_model, use_ie=use_ie
            )

    @staticmethod
    def collate_fn(data_list: [TextAndAudio2LabelDataItem]):

        graph_list = [item.graph for item in data_list]
        graph_data = to_batch(graph_list)

        tgt_tensor = []
        if len(data_list) > 0 and hasattr(data_list[0], "output"):
            tgt = [deepcopy(item.output) for item in data_list]
            tgt_tensor = torch.LongTensor(tgt)

        if len(data_list) > 0 and hasattr(data_list[0], "audio"):
            # input_values = torch.Tensor([deepcopy(item.audio) for item in data_list])
            input_values = torch.stack([item.audio for item in data_list])

        if len(data_list) > 0 and hasattr(data_list[0], "audio_attention_mask"):
            # attention_mask = torch.Tensor([deepcopy(item.audio_attention_mask) for item in data_list])
            attention_mask = torch.stack([item.audio_attention_mask for item in data_list])

        return {"graph_data": graph_data, "tgt_tensor": tgt_tensor, "input_values": input_values,
                "attention_mask": attention_mask}


class MyDataset(TextAndAudio2LabelDataset):
    @property
    def raw_file_names(self):
        """3 reserved keys: 'train', 'val' (optional), 'test'. Represent the split of dataset."""
        return {"train": "train.txt", "test": "test.txt"}

    @property
    def processed_file_names(self):
        """At least 3 reserved keys should be fiiled: 'vocab', 'data' and 'label'."""
        return {"vocab": "vocab.pt", "data": "data.pt", "label": "label.pt"}

    def download(self):
        # raise NotImplementedError(
        #     "This dataset is now under test and cannot be downloaded."
        #     "Please prepare the raw data yourself."
        #     )
        return

    def __init__(
            self,
            root_dir,
            topology_subdir,
            graph_name,
            static_or_dynamic="static",
            topology_builder=None,
            dynamic_init_graph_name=None,
            dynamic_init_topology_builder=None,
            dynamic_init_topology_aux_args=None,
            pretrained_word_emb_name="840B",
            pretrained_word_emb_url=None,
            pretrained_word_emb_cache_dir=None,
            max_word_vocab_size=None,
            min_word_vocab_freq=1,
            tokenizer=nltk.RegexpTokenizer(" ", gaps=True).tokenize,
            word_emb_size=None,
            audio_data=None,
            **kwargs
    ):
        super(MyDataset, self).__init__(
            graph_name,
            root_dir=root_dir,
            static_or_dynamic=static_or_dynamic,
            topology_builder=topology_builder,
            topology_subdir=topology_subdir,
            dynamic_init_graph_name=dynamic_init_graph_name,
            dynamic_init_topology_builder=dynamic_init_topology_builder,
            dynamic_init_topology_aux_args=dynamic_init_topology_aux_args,
            pretrained_word_emb_name=pretrained_word_emb_name,
            pretrained_word_emb_url=pretrained_word_emb_url,
            pretrained_word_emb_cache_dir=pretrained_word_emb_cache_dir,
            max_word_vocab_size=max_word_vocab_size,
            min_word_vocab_freq=min_word_vocab_freq,
            tokenizer=tokenizer,
            word_emb_size=word_emb_size,
            audio_data=audio_data,
            **kwargs
        )
