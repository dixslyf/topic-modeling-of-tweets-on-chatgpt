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

#show bibliography: set heading(numbering: "1.")

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

#show table.cell.where(y: 0): set text(style: "italic")
#show table.cell.where(y: 0): set par(justify: false)
#set table(
  align: (x, y) => if y == 0 { center + horizon } else { center + top },
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
  #text(size: TITLE_SIZE)[*ICT202 Assignment 2: Topic Modelling*] \
  \
  #text(size: SUBTITLE_SIZE)[
    Dixon Sean Low Yan Feng \
    Murdoch University \
    #date.display("[day] [month repr:long], [year]")
  ]
])

#outline(indent: auto)

= Introduction

ChatGPT, developed by OpenAI, has garnered significant global attention and discussion on social media.
As its adoption increases,
views have emerged about how it and other artificial intelligence~(AI) tools
may impact sectors like business, education and creative industries.
Understanding the public discourse surrounding ChatGPT
is crucial for assessing the implications of widely accessible artifical intelligence.

This report uses BERTopic @grootendorst2022bertopic
to perform topic modelling on
tweets from X~(formerly Twitter) about ChatGPT.
The objective is to identify the key topics underlying discussions
and gain insights into user perceptions and issues
regarding the increasingly wide accessibility of AI.
These insights can inform stakeholders about the technology's reception
and guide future developments and policies surrounding AI.

The following sections detail the data collection and preprocessing steps
and describe the structure and application of BERTopic
before evaluating and presenting the results
and discussing their implications.

= Data Collection and Preprocessing

The #link("https://www.kaggle.com/datasets/tariqsays/chatgpt-twitter-dataset")[dataset]
was sourced from Kaggle and
contains metadata (e.g., timestamps, user information and like counts)
and textual content from 
50,002 tweets with the `#chatgpt` hashtag.
It spans from 2023-01-22 to 2023-01-24
and is multilingual.
This dataset was chosen for its high usability score of 10 (calculated by Kaggle)
and manageable size.
Furthermore, it is free of duplicate and missing values,
reducing the effort needed for data cleaning.
// TODO: cite dataset
// https://www.kaggle.com/datasets/tariqsays/chatgpt-twitter-dataset

As multilingual topic modelling is complex,
only English tweets were retained.
Each sample was already prelabeled with its language,
so filtering only required examining the language column,
leaving 32,076 samples.
Next, all columns except the textual content
were removed
since they are metadata and unneeded for topic modelling.

The textual content was preprocessed to remove
emojis,
hash symbols,
user mentions
and hyperlinks
to reduce noise
(the pre-trained model BERTopic uses for document embeddings
was not trained on such data).
No further preprocessing (e.g., removal of stop words, lemmatisation) was performed
as BERTopic's use of document embeddings and a transformer-based model
requires keeping the original structure of the text.
// TODO: cite https://maartengr.github.io/BERTopic/faq.html#should-i-preprocess-the-data

= Topic Model

A BERTopic model was fit to the dataset using the #link("https://maartengr.github.io/BERTopic/index.html")[BERTopic library].
// TODO: cite BERTopic
BERTopic @grootendorst2022bertopic processes documents through a pipeline of submodels.
First, it uses a Bidirectional Encoder Representations from Transformers~(BERT) model to generate document embeddings.
These embeddings are then subjected to dimensionality reduction and clustering algorithms.
By default, BERTopic uses Uniform Manifold Approximation and Projection~(UMAP) for dimensionality reduction
and Hierarchical Density-Based Spatial Clustering of Applications with Noise~(HDBSCAN) for clustering.
Following clustering, BERTopic constructs a cluster-level bag-of-words model and
calculates term frequency–inverse document frequency~(TF-IDF) scores.
However, each cluster is treated as a single document when calculating TF-IDF,
leading to what is termed class-based TF-IDF~(c-TF-IDF).
The terms with the highest c-TF-IDF scores represent each topic.

To build the model,
document embeddings were first precomputed
using the `all-MiniLM-L6-v2` model from the
#link("https://www.sbert.net/")[#text(hyphenate: false)[SentenceTransformers]~(SBERT)] package @reimers-2019-sentence-bert.
Although #link("https://www.sbert.net/docs/sentence_transformer/pretrained_models.html")[other models] are available,
`all-MiniLM-L6-v2` strikes a balance between quality and speed.
UMAP and HDBSCAN were used for the dimensionality reduction and clustering, respectively,
as recommended by the BERTopic documentation.
GPU-accelerated implementations of these algorithms were used
via the #link("https://docs.rapids.ai/api/cuml/stable/")[cuML library] @raschka2020machine
to improve processing speed.
Stopwords were removed from the topic representations
using a list provided by the #link("https://www.nltk.org/")[Natural Language Toolkit~(NLTK)] @Bird_Natural_Language_Processing_2009.
Then, part-of-speech tagging (provided by spaCy @Honnibal_spaCy_Industrial-strength_Natural_2020) was applied to retain only adjectives and nouns,
followed by maximal marginal relevance~(MMR) to improve the diversity of the representative words.
These submodels and steps were specified to the initialisation of the BERTopic model.

Hyperparameters were optimised using a grid search:
(a) number of neighbours for UMAP,
(b) number of resulting components for UMAP, and
(c) minimum cluster size for HDBSCAN.
These parameters were selected for their significant impact on topic coherence and size
as determining the number of topics in advance is challenging. 
Each hyperparameter configuration was evaluated through
both objective evaluation (using the $C_v$ topic coherence score from @roder_exploring_2015)
and subjective human judgement of the topic representations.
This dual approach ensures that while high $C_v$ scores indicate statistical coherence,
the final topics are also meaningful and interpretable to humans.
The results of this analysis are in @section-evaluation.

= Evaluation <section-evaluation>

@table-cv shows the $C_v$ scores for each set of hyperparameters tested.
$C_v$ scores range from 0 (low coherence) to 1 (high coherence).
The scores indicate satisfactory topic coherence,
with 0.596 as the lowest and 0.626 as the highest.
These values suggest that the topics are reasonably coherent across different hyperparameter settings.

#figure(
  caption: [$C_V$ scores for tested hyperparameter sets],
  placement: auto,
  {
    set text(size: 11pt)
    table(
      columns: (auto, auto, auto, 15%),
      table.header(
        [No. of neighbours (UMAP)],
        [No. of components (UMAP)],
        [Minimum cluster size (HDBSCAN)],
        [$C_v$],
      ),

      [30],
      [10],
      [30],
      [#calc.round(0.6260656565788427, digits: 3)],

      [15],
      [10],
      [30],
      [#calc.round(0.6251453237522169, digits: 3)],

      [30],
      [5],
      [30],
      [#calc.round(0.6247880561567395, digits: 3)],

      [30],
      [5],
      [20],
      [#calc.round(0.6218670777506773, digits: 3)],

      [30],
      [10],
      [20],
      [#calc.round(0.6178856581242598, digits: 3)],

      [15],
      [10],
      [10],
      [#calc.round(0.6138945314312856, digits: 3)],

      [15],
      [5],
      [30],
      [#calc.round(0.6073894745143839, digits: 3)],

      [30],
      [5],
      [10],
      [#calc.round(0.6059289829719166, digits: 3)],

      [15],
      [5],
      [10],
      [#calc.round(0.6038504287879352, digits: 3)],

      [15],
      [10],
      [20],
      [#calc.round(0.601119136530049, digits: 3)],

      [15],
      [5],
      [20],
      [0.600],
      // [#calc.round(0.6004166027278458, digits: 3)],

      [30],
      [10],
      [10],
      [#calc.round(0.5957065404077899, digits: 3)],
    )
  }
) <table-cv>

@table-topics provides a snippet of the topic representations for the top three hyperparameter sets.
These representations were evaluated through human inspection.
The topics derived from these sets are similar in both coherence and content.
However, it is not obvious what the second topic for each set represents
based on the representative terms alone,
indicating a potential area for further analysis.

Despite the similarity in results,
it is necessary to select a single hyperparameter set for final use.
Given that the differences between sets are minimal,
the set with the highest $C_v$ score will be used for subsequent analysis. 
It is important to note that while the $C_v$ score provides a quantitative measure of coherence,
the final evaluation remains somewhat subjective.

#figure(
  caption: [Topic representations for the four most frequent topics from the BERTopic models for the top three hyperparameter sets],
  table(
    columns: 3,
    table.header[Hyperparameter Set\ $C_v$ Ranking][Topic Count][Representation],

    [1],
    [1688],
    [tools, generative, artificialintelligence, intelligence, artificial, robots, ai, future, humans, art],

    [],
    [848],
    [gpt3, essay, teacher, programming, telugu, referral, autobiography, instructgpt, freud, method],

    [],
    [476],
    [seo, search, engines, engine, searches, query, links, results, keywords, ads],

    [],
    [473],
    [students, education, classroom, schools, ban, cheating, teachers, kids, educators, teacher],

    [2],
    [1858],
    [generative, tools, artificialintelligence, revolution, future, jobs, artists, ethics, intelligence, machinelearning],

    [],
    [1121],
    [gpt3, telugu, teacher, programming, raw, hell, essay, referral, crazy, autobiography],

    [],
    [646],
    [seo, search, engines, searches, keywords, web, threat, ads, results, query],

    [],
    [571],
    [students, teachers, cheating, education, ban, classroom, student, essay, teaching, class],

    [3],
    [1660],
    [generative, tools, artificialintelligence, robots, mind, ai, intelligence, future, jobs, humans],

    [],
    [807],
    [gpt3, telugu, essay, programming, teacher, referral, instructgpt, freud, step, wallet],

    [],
    [513],
    [students, education, classroom, schools, ban, cheating, teacher, educators, teachers, student],

    [],
    [501],
    [seo, search, engines, engine, searches, threat, ads, keywords, query, results],
)
) <table-topics>

= Results

The BERTopic model with the chosen set of hyperparameters identified 74 topics,
which are visualised in a two-dimensional intertopic distance map~(#text(hyphenate: false)[@fig-intertopic-dist]).
Each bubble in the map represents a topic, with its size indicating the topic’s prevalence.
Topics located closer together are more similar to each other.
It is clear that many topics overlap,
a result of the hierarchical clustering performed by HDBSCAN.
This hierarchical clustering is also evident from the high similarity between many topics,
as shown in the topic similarity matrix (@fig-similarity-matrix).
Additionally, @fig-intertopic-dist illustrates approximately ten primary clusters of topics.
Although a reduction to ten topics could be beneficial for further analysis,
this approach is beyond the current report’s scope.

#grid(
  columns: 2,
  align: center + bottom,
  [#figure(
    caption: [Intertopic distance map],
    rect(image("graphics/intertopic-distance.png", width: 85%)),
  ) <fig-intertopic-dist>],
  [#figure(
    caption: [Topic similarity matrix],
    image("graphics/similarity-matrix.png", width: 90%),
  ) <fig-similarity-matrix>],
)

@fig-topic-word-scores shows the terms with the highest c-TF-IDF scores for the ten most frequent topics,
while @fig-wordclouds provides word cloud representations of these terms,
with larger words indicating higher scores.
The topics are ordered by their prevalence (i.e., Topic 0 is the most common).
Topic 0, unsurprisingly, focuses on AI, reflecting the nature of ChatGPT.
However, this topic's insights are somewhat superficial due to its generality.
Topic 1 has somewhat disjointed terms — 
"essay" and "teacher" suggest an educational context,
but "programming" introduces a technological aspect
and "telugu" hints at regional or multilingual discussions.
It is not clear what the topic represents.

#figure(
  caption: [Topic word scores for the ten most frequent topics],
  image("graphics/topic-word-scores.png", width: 90%),
  placement: auto,
) <fig-topic-word-scores>

By contrast, Topic 2 addresses search engine optimisation~(SEO)
and suggests that users are exploring how AI tools like ChatGPT can be leveraged in digital marketing.
This aligns with the increasing interest in using AI to enhance online visibility and content strategies,
as evidenced by related studies @shen_chatgpt_2024 @cutler_chatgpt_2023.
Topic 9 also appears to be related to SEO but is less coherent in its representation.

Topics 3, 7 and 8 clearly revolve around education,
indicating that ChatGPT is actively discussed as a tool for learning and teaching.
This suggests that a growing influence of AI in educational settings,
potentially transforming how students complete assignments and interact with educational content.
However, integrating AI into education also raises concerns about academic integrity,
a topic discussed in @cotton_chatting_2024.
These concerns may have been discussed by users due to the word "ban" in Topic 3.

The less prevalent topics span a range of fields,
including medicine, cybersecurity, mathematics, stocks and employment.
This diversity highlights the extensive influence of ChatGPT across various sectors
and the growing recognition of AI’s versatility and its potential impact on both specialised and general domains.

#figure(
  caption: [Word clouds for the ten most frequent topics],
  image("graphics/wordclouds.png"),
) <fig-wordclouds>

= Conclusion and Future Work

The BERTopic model identified 74 topics from tweets about ChatGPT,
showing a high degree of overlap and hierarchical clustering.
Key topics include artificial intelligence, SEO and education,
highlighting ChatGPT’s impact in digital marketing and educational contexts.
Overall, the topics are diverse and underscore the recognition of
the versatility and expanding relevance of AI.

Future work could focus on refining the topic model to reduce redundancy and capture distinct themes more clearly
--- some topics (e.g., topics 3, 7 and 8) were nearly identical.
Additionally, 
the dataset only contained three days' worth of tweets.
Future research could benefit from analysing a longer time frame and
the change in topics over time,
in addition to
comparing discussions across multiple platforms and AI tools
to gain a deeper understanding of AI's impact and usage.

#bibliography("references.bib")
