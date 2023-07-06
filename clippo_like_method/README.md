# clippo_like_method

- `conda create -n clippo-like python=3.8`
- `conda activate clippo-like`
- `pip install -r requirements.txt`
- generate `pitt_audio_and_text_data` folder as described in [multi_modal_method/README.md](../multi_modal_method/README.md)
- run `python text_to_speech.py` to generate `pitt_audio_and_text_data_tts` folder. You also need to manually generate `pitt_audio_and_text_data_tts/metadata.csv` file.
- run `python clippo_like_method_parallel.py`.

