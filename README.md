# Topic Modelling of ChatGPT Tweets

This repository contains the [Typst](https://typst.app/) markup of a university project report
investigating the public discourse surrounding ChatGPT using topic modeling.

The project applies [BERTopic](https://maartengr.github.io/BERTopic/index.html)
to a dataset of 50,002 tweets tagged with `#chatgpt`,
collected over a three-day period in January 2023
([dataset source](https://www.kaggle.com/datasets/tariqsays/chatgpt-twitter-dataset)).
The goal was to identify and analyse
recurring themes in social media discussions about generative AI.
In total, the analysis identified 74 topics that describe discussions of ChatGPT
across various domains such as education, digital marketing and artificial intelligence.

The rendered report can be found [here](https://github.com/dixslyf/topic-modeling-of-tweets-on-chatgpt/releases/latest/download/report.pdf) (PDF).

Source code for
data preprocessing, hyperparameter tuning, model training and visualisations
can be accessed from the latest release [here](https://github.com/dixslyf/topic-modeling-of-tweets-on-chatgpt/releases/latest/download/chatgpt-topic-modeling.ipynb)
in the form of a Jupyter notebook.
Alternatively, you may also view the notebook on [Kaggle](https://www.kaggle.com/code/dixonseanlowyanfeng/topic-modeling-of-tweets-on-chatgpt).

## Compiling the Report

To compile the report, ensure you have the Typst compiler installed.
Then, run the following in the repository's root directory:

```sh
typst compile report/report.typ
```

This will output a PDF file at `report/report.pdf`.

### Nix

This repository provides a [Nix](https://nixos.org/) flake that you can also use to compile the report.
To do so, run the following:

```sh
# From the repo's root directory:
nix build

# Or, without having to clone the repo:
nix build github:dixslyf/topic-modeling-of-tweets-on-chatgpt
```

You can then view the compiled report (PDF) through the usual `result` symlink.
