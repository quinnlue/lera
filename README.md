# LeRa

## Project Structure

```
LeRa/
├── lera/
│   ├── __init__.py
│   ├── __version__.py
│   └── model/
│       ├── __init__.py
│       ├── model.py
│       ├── moe.py
│       ├── rope.py
│       ├── standard_model.py
│       └── transformer.py
├── train/
│   ├── __init__.py
│   ├── prototype_train.py
│   ├── standard_train.py
│   └── utils.py
├── tokenizer/
│   ├── __init__.py
│   ├── tokenizer.json
│   ├── tokenizer
│   ├── tokenizer.ipynb
│   └── _deprecated_tokenizer.py
├── data/
│   ├── pretraining/
│   │   ├── cleaned/
│   │   ├── fine-math-4/
│   │   ├── raw/
│   │   └── tokenized/
│   └── scripting/
│       └── pretraining/
├── prototyping/
│   ├── addition-test/
│   ├── conv-test/
│   └── data/
├── logs/
├── pyproject.toml
└── README.md

```

