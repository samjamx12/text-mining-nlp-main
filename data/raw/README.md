# Data Instructions

This repository includes small sample files so the notebooks have a clear expected structure. Larger datasets and pretrained models are intentionally not committed.

Expected data layout:

```text
data/raw/
├── apple_samsung_sample.txt
├── my_tweets.json
├── airlinetweets/
│   ├── negative/
│   ├── neutral/
│   └── positive/
├── CONLL2003/
│   ├── train.txt
│   └── test.txt
└── ner_dataset.csv
```

## Dataset notes

- `apple_samsung_sample.txt`: small text sample used by the NLP toolkit comparison notebook.
- `my_tweets.json`: manually annotated tweet examples for VADER experiments. A tiny sample is included; replace it with a larger annotated set for full evaluation.
- `airlinetweets/`: expected as a folder readable by `sklearn.datasets.load_files`, with one subfolder per class.
- `CONLL2003/`: expected CoNLL-style NER files.
- `ner_dataset.csv`: entity annotated corpus CSV used for the additional NER experiment.
- `models/GoogleNews-vectors-negative300.bin`: optional large pretrained Word2Vec model used by the embedding experiment. Place it in `models/` manually.
