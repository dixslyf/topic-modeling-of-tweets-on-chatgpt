#import "@preview/algo:0.3.3": algo, d, i
#import "/constants.typ": *
#import "/lib.typ": *

// SETUP

#set page(
  paper: "a4",
  numbering: "1",
  number-align: right + top,
)

#set heading(numbering: "1.")

#show heading.where(level: 1): it => [
  #set text(size: HEADING_SIZE, weight: "bold")
  #it
]

#show heading.where(level: 2): it => [
  #set text(size: SUBHEADING_SIZE, weight: "regular", style: "italic")
  #it
]

#show heading.where(level: 3): it => [
  #set text(size: SUBSUBHEADING_SIZE, weight: "regular", style: "normal")
  #it
]

#show outline: it => {
  set par(leading: 1em)
  it
}

#set figure(placement: none)

#set block(above: 1.5em)

#set par(justify: true)

#set text(size: TEXT_SIZE, font: TEXT_FONT)

#set list(indent: LIST_INDENT)

#set terms(indent: LIST_INDENT)

#show math.equation: set text(size: 11pt)

#show link: it => {
  set text(fill: blue)
  underline(it)
}

#set list(indent: LIST_INDENT)
#set enum(indent: LIST_INDENT, numbering: "a)")

#show figure: set block(breakable: true)

#show table.cell.where(y: 0): set text(weight: "bold")
#set table(
  stroke: (x, y) => (
    top: if y <= 1 { 0.8pt } else { 0pt },
    bottom: 1pt,
  ),
  inset: 0.5em,
)

// CONTENT

#let date = datetime(
  year: 2024,
  month: 8,
  day: 9,
)

#align(center, [
  #text(size: TITLE_SIZE)[*ICT202 Assignment 2: Topic Modeling*] \
  \
  #text(size: SUBTITLE_SIZE)[
    Dixon Sean Low Yan Feng \
    Murdoch University \
    #date.display("[day] [month repr:long], [year]")
  ]
])

#outline(indent: auto)

= Introduction

ChatGPT is a powerful language model developed by OpenAI
that has generated significant attention worldwide
and considerable discussion on social media platforms.
As its adoption grows,
many views have emerged about how ChatGPT~(and artificial intelligence, in general)
may impact
various sectors, including business, education and creative industries.
Understanding the public discourse surrounding ChatGPT
is a key step for grasping the implications of widely accessible artifical intelligence.

This report applies BERTopic @grootendorst2022bertopic, a topic modeling technique,
to analyze tweets from X~(formerly Twitter) about ChatGPT.
The objective is to identify the main topics discussed in tweets mentioning ChatGPT
in hopes of gaining insights into user perceptions and issues
regarding the increasingly wide accessibility of artificial intelligence.
Such insights can inform stakeholders about the technology's reception
and guide future developments and policies surrounding artficial intelligence.

The following sections detail the data collection and preprocessing steps
and describe the structure and application of BERTopic,
before evaluating and presenting the results
and discussing their implications.

= Data Collection and Preprocessing

The dataset was retrieved from Kaggle and
contains metadata and the textual content of
50,002 tweets posted with the hashtag `#chatgpt`.
The tweets span a time period from 2023-01-22 to 2023-01-24.
The dataset is multilingual, with tweets of varying languages.
The metadata include the tweet's timestamp, user, hyperlinks, retweet count and like count, among others.
// TODO: redirect to appendix for full list of columns
This dataset was chosen due to its high usability score of 10 (calculated by Kaggle)
and relative small but sufficient size,
to reduce topic modeling time
to meet this #text(hyphenate: false)[project]'s time constraints.
Furthermore, the dataset does not contain duplicate samples or missing values,
reducing the effort needed for data cleaning.
// TODO: cite dataset
// https://www.kaggle.com/datasets/tariqsays/chatgpt-twitter-dataset

As multilingual topic modeling is complex,
the task was simplified to consider only tweets in English.
Hence, the dataset was filtered to remove tweets in other languages.
As each sample is already prelabeled with its language,
filtering the dataset only required examining the language label
and removing tweets whose language was not English.
32,076 samples remained after filtering.

All columns except the textual content column
were removed
since they are metadata and unneeded for building the BERTopic model.
The textual content was then preprocessed to remove
emojis,
hash symbols,
user mentions
and hyperlinks
as those introduce noise
(the pretrained model BERTopic uses for generating document embeddings
was not trained on such data)
and could reduce the coherence of the resulting topics.

No further preprocessing (e.g., removal of stop words, lemmatization, etc.) was performed
on the dataset
as BERTopic's use of document embeddings and a transformer-based model
requires keeping the original structure of the text
to understand context.
// TODO: cite https://maartengr.github.io/BERTopic/faq.html#should-i-preprocess-the-data

= Topic Modeling

A BERTopic model was fit to the dataset using the #link("https://maartengr.github.io/BERTopic/index.html")[BERTopic Python library].
// TODO: cite BERTopic
BERTopic @grootendorst2022bertopic processes documents using a pipeline of submodels.
First, it uses a Bidirectional Encoder Representations from Transformers~(BERT) model to generate document embeddings.
These embeddings are then passed to a dimensionality reduction algorithm,
followed by a clustering algorithm to group documents.
By default, BERTopic uses Uniform Manifold Approximation and Projection~(UMAP) for dimensionality reduction
and Hierarchical Density-Based Spatial Clustering of Applications with Noise~(HDBSCAN) for clustering.
Next, BERTopic generates a cluster-level bag-of-words model and
calculates a term frequency–inverse document frequency~(TF-IDF) score.
However, each cluster is treated as a single document when calculating TF-IDF;
hence, the author of BERTopic refers to the score as class-based TF-IDF~(c-TF-IDF).
Each cluster/topic is represented by the words with the highest c-TF-IDF scores.

To build the model,
first, the document embeddings were precomputed
using the pretrained `all-MiniLM-L6-v2` model from the
#link("https://www.sbert.net/")[#text(hyphenate: false)[SentenceTransformers]~(SBERT)] package @reimers-2019-sentence-bert.
`all-MiniLM-L6-v2` was chosen out of the pretrained models
#footnote[A list of pretrained models is available #link("https://sbert.net/docs/sentence_transformer/pretrained_models.htmL")[here].]
as it strikes a good balance between quality and speed.
UMAP and HDBSCAN were used for the dimensionality reduction and clustering respectively
since they are recommended by the BERTopic documentation.
However, GPU-accelerated implementations of the algorithms were used
via the #link("https://docs.rapids.ai/api/cuml/stable/")[cuML library] @raschka2020machine
instead of the default CPU implementations
to speed up the process.
These submodels were then used to initialise the BERTopic model.

Stopwords were removed from the topic representations _after_
determining the topics
using a list provided by the #link("https://www.nltk.org/")[Natural Language Toolkit~(NLTK)] @Bird_Natural_Language_Processing_2009.
Additionally, part-of-speech tagging (provided by spaCy @Honnibal_spaCy_Industrial-strength_Natural_2020) was applied to retain only adjectives and nouns,
followed by maximal marginal relevance~(MMR) to improve the diversity of the words representing each topic.
These steps were specified to the BERTopic model during initialisation.

Several hyperparameters were tuned using a grid search:
(a) number of neighbours for UMAP,
(b) number of resulting components for UMAP,
(c) minimum cluster size for HDBSCAN, and
(d) number of topics to reduce to for BERTopic.
These hyperparameters have the most impact on the resulting topics
in terms of coherence and size.
Each hyperparameter set was evaluated using
a mixture of objective evaluation,
using the $C_v$ topic coherence score from @roder_exploring_2015;
and subjective evaluation,
through human judgement of the topic representations.
Ultimately, the goal of topic modeling is to produce topics
meaningful and interpretable to _humans_.
While a high $C_v$ indicates good statistical coherence,
a mathematically-coherent topic might still be vague to a human,
hence the use of both objective and subjective evaluation.
Results are in @section-evaluation.

= Evaluation <section-evaluation>

= Results

= Discussion and Future Work

#bibliography("references.bib")
