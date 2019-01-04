![singnetlogo](./assets/singnet-logo.jpg?raw=true 'SingularityNET')

[![CircleCI](https://circleci.com/gh/EskenderTamrat/DSSM-LSTM.svg?style=svg)](https://circleci.com/gh/EskenderTamrat/DSSM-LSTM)

# DSSM with LSTM Network Service
## Service User's Guide

This service provides DSSM neural network modeling for text strings in a continuous semantic space and modeling semantic similarity between two text strings.

It provides the trained model as service for semantic modeling given query and a pair of answers. it returns the cosine similarity between the query and answer pair (higher value indicating a better answer)

### How does it work?

The user must provide a request satisfying the proto descriptions [given](../../service_spec/DSSMService.proto).

* A request with qry: a pharase with words that exist in the sample model vocabluary (../../data/vocab_Q.wl) 
            
* Pair of answers: phrases that exist in the sample model vocabulary (../../data/vocab_A.wl)

### Using the service on the platform

The returned result has the following form: 
```bash
message DSSMResponse {
  float qry_ans_similarity = 1;
  float qry_ans2_similarity = 2;
}
```
An example result obtained after passing the query and answer sets
```bash
DSSMRequest(qry="what contribution did you made to science", ans1 ="book author book_editions_published", ans2="activism address adjoining_relationship")
```

```bash
{
qry_ans_similarity: 0.99989253282547,
qry_ans2_similarity: 0.9998538494110107,
}
```
If the user's input for query and answers include words that doesn't exist in the vocabulary, it returns the message that you should use terms from available sample data.