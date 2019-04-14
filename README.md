# bert-span-parser

This repo is based on https://github.com/mitchellstern/minimal-span-parser

## Using BERT as Feature Extractor

Download pretrained BERT model: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz

Then using this command `tar xzvf bert-base-multilingual-cased.tar.gz` to extract the content, and then move `pytorch_model.bin` to `models/bert-base-multilingual-cased`

To train the model, simply run: `python -u main.py train > log`
