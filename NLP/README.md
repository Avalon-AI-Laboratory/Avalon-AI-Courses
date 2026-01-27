# 🤵‍♀️ Natural Language Processing

Corresponding Author: [Faiz Kautsar](https://github.com/spuuntries)

<div style="width: 50%; margin: 0 auto;">

| Previous Material |   Current   |         Next Material         |
| :---------------: | :---------: | :---------------------------: |
|         ◁         | "NLP Intro" | [▷](./Topics/fundamentals/README.md) |

</div>

Natural Language Processing (NLP) is the branch of computer science which deals with processing of natural language information. This module will teach you from the basics of NLP to some of the more recent developments within this space. By the end of this module, the hope is that you'll be able to have a good grasp of the concepts discussed within NLP and be able to create NLP products with extensive understanding of its components.

This module will be divided into two main chapters, of which each correspond to a sub-branch of NLP: `nlu` and `nlg`. Included prior to those two, is also a `fundamentals` chapter which act as a prerequisite. You're expected to follow this flow in exploring this module:

```mermaid
graph TD
    subgraph Fundamentals
        direction LR
        fundamentals@{ shape: circle, label: "Start" }--(preferably)-->preface[Preface]
        preface-->normalization[Normalization]
        fundamentals-->normalization
        normalization-->tokenization[Tokenization]
        tokenization-->rulebased[Rule-Based]
        tokenization-->wordpiece[Wordpiece]
        tokenization-->unigram[Unigram]
        tokenization-->bpe[Byte-Pair Encoding]
    end
    Fundamentals-->NLU
    subgraph NLU [Natural Language Understanding]
        direction LR
        subgraph architectures
            direction LR
            rnn[RNN]-->attention[Attention]
            attention-->transformers[Transformers]
            transformers-->bert[BERT]
        end
        subgraph embedding_g [embedding]
            direction LR
            embedding[Embedding]-->bow[Bag-of-Words]
            embedding-->tfidf[TF-IDF]
            embedding-->word2vec[Word2Vec]
            embedding-->nnembedding[nn.Embedding]
        end
        embedding_g-->architectures
        architectures-->cls[Classification]
    end
    NLU-->NLG
    subgraph NLG [Natural Language Generation]
        direction LR
        gmodels[Generative Models]-->encoderonly
        gmodels-->encoderdecoder
        gmodels-->decoderonly
        encoderonly[Encoder]-->bertg[BERT]
        encoderdecoder[Encoder-Decoders]-->gpt[GPT]-->T5
        decoderonly[Decoder-Only]-->GPT2-->Llama
        decoderonly-->agentic
        subgraph Agentics
            agentic[Agentic Language Modelling]-->planning[Planning]
            agentic-->tool_use[Tool Use]
            agentic-->memory[Memory]
            planning-->react[ReAct]
            tool_use-->react[ReAct]
        end
    end
```

We'll primarily be using [PyTorch](http://pytorch.org/) as our framework of choice for its simplicity in the execution graph representations (i.e., it's dynamically done, and although TF has sorta moved toward this in recent years w/ 2.x, we believe that torch all in all is still much more beginner-friendly, especially for those who are doing experiments) and object-oriented/pythonic, though the concepts explored will be universally-applicable.

To start, go over to [the fundamentals preamble](./Topics/fundamentals/README.md).
