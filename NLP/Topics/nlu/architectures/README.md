# 👷‍♀️ Architectures

<div style="width: 50%; margin: 0 auto;">

|        Previous Material         |             Current             |        Next Material         |
| :------------------------------: | :-----------------------------: | :--------------------------: |
| [◁](../embedding/nnembedding.md) |    "Architectures Preamble"     |        [▷](./rnn.md)         |

</div>

When we talk about modern language modelling, we have to talk about architectures. What _are_ architectures? Well, it's complicated, but simply put, here's an attempt at _a_ definition tries to encompass all of them: architectures are structural configurations or training paradigms applied to the Neural Network (NN) model which alters its properties from the "vanilla" NN formulation to achieve a certain inductive bias.

NN architectures come in many forms, and especially for NLP, this field is relatively fairly "open" to the use of many kinds of architectures. The main reason for this is that NLP, on its own, is a much more "forgiving" subfield of deep learning. In the sense that most, if not all, of the processing done can be viewed through the lens of sequence modelling of abstract and discrete symbolic data. In contrast, other subfields like CV deal with continuous physical signals that historically required strict spatial biases, which restricts the way you can formularize it.

As a result, NLP can be done in so many different ways so long as it's able to model the relational topology and hierarchies used within the grammatical systems of natural language. In [Word2Vec](../embedding/word2vec.md), I showed you an architecture which did NLP with a CNN, an architecture which historically has mostly only been applied on CV problems for its spatial inductive biases, to contrast its embeddings with Word2Vec. There are many ways to model language in "unorthodox" or "unconventional" ways, and as a result, to talk about all of them would be a long and winded discussion on how to model the aforementioned sequential data.

As such, in this chapter, we'll try to restrict our scope into the several architectures which currently mostly builds modern language modelling: RNNs, LSTMs, Attention-based systems. For starters, let's talk about [RNNs](./rnn.md).
