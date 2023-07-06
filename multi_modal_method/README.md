# multi_modal_method

- Refer to configuration for AD-GNN in [gnn_based_method/README.md](../gnn_based_method/README.md) to download pitt cookie-theft dataset, install dependencies, fix bugs in graph4nlp and run stanford corenlp server, since we use AD-GNN to process textual data in our multimodal approach.

- Download the pitt cookie-theft speech dataset, and put it in `pitt_audio_and_text_data/`, with the following structure:

  ```
  xxx.mp3
  xxx.mp3
  ...
  metadata.csv
  ```
    
  where `metadata.csv` has the following format:

  ```
  file_name,label,transcription
  ```
  
- `cp multimodal_dataset.py /path/to/your/python/environment/lib/python3.8/site-packages/graph4nlp/pytorch/data`

- Insatll extra dependencies

  ```
  pip install datasets==2.8.0
  pip install evaluate
  pip install soundfile==0.12.1
  pip install torchaudio==2.0.2
  pip install numpy==1.19.5
  ```

- run `python multimodal_classification.py`, the final result is saved in `out/`.