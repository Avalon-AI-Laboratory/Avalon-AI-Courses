# 📃 Preface

<div style="width: 50%; margin: 0 auto;">

| Previous Material |        Current         |         Next Material          |
| :---------------: | :--------------------: | :----------------------------: |
| [◁](../../README.md) | "Fundamentals Preface" | [▷](./normalization/README.md) |

</div>

In NLP, like in other branches of machine learning, prior to actually processing the data, what we need to do is turn the modality of natural text into representations which can be analyzed and modelled numerically, AKA., pre-processing.

Simply put, there's a modality gap from the raw lingustic form of, say, `the quick brown fox jumps over the lazy dog` into the numbers which computers and math can model.

In this chapter, we'll learn how to clean textual data up for processing, then to turn them into representations which would be compatible with mathematically-computed representation modelling. Specifically, refer to the following graph for the materials we'll cover:

```mermaid
graph TD
    subgraph Fundamentals
        A[Start] --> B(Normalization);
        B --> C(Tokenization);
    end

    subgraph Tokenization Methods
        C --> D(Rule-Based);
        C --> E(Wordpiece);
        C --> F(Unigram);
        C --> G(Byte-Pair Encoding);
    end
```

Next, go to [**[Normalization]**](./normalization/README.md) for your first material.
