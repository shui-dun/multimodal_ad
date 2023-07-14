# gnn_based_method

This approach was inspired by https://github.com/graph4ai/graph4nlp_demo.

## How to Run

- download the pitt cookie-theft dataset, and put it in `pitt_data/pitt.txt`, with the following format

  ```
  text1 label1
  text2 label2
  ...
  ```

  where each line is a sample (text-label pair), and the label is either `0` (representing health control) or `1` (representing probable AD).

- install dependencies

  ```
  pip install torch
  pip install torchtext
  pip install graph4nlp-cu110
  pip install numpy<1.20
  ```

- fix bug in graph4nlp: replace line 155 in 

  ```
  /path/to/your/python/environment/lib/python3.8/site-packages/graph4nlp/pytorch/modules/graph_construction/node_embedding_based_refined_graph_construction.py
  ``` 

  from 

  ```
  graph = dynamic_init_topology_builder.topology(
  ```

  with

  ```
  graph = dynamic_init_topology_builder.static_topology(
  ```

- Download StanfordCoreNLP https://stanfordnlp.github.io/CoreNLP/ and run

  ```
  java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9012 -timeout 15000
  ```

- run `python ad_gnn.py`, the final result is saved in `out/pitt_data/metric.log`