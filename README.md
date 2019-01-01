# Deep Structured Semantic Modeling with LSTM Networks

This service is an implementation of Deep Structured Semantic Model or Deep Semantic Similarity Model (DSSM) with LSTM Networks. Given a query and a pair of answers from the sample data the network trained, it returns the better answer based on the cosine similarity. It is forked from [CNTK 303: Deep Structured Semantic Modeling with LSTM](https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_303_Deep_Structured_Semantic_Modeling_with_LSTM_Networks.ipynb).  

## Deep Structured Semantic Model (DSSM)

DSSM is a Deep Neural Network (DNN) modeling technique for representing text strings in a continuous semantic space and modeling semantic similarity (a metric defined over a set of documents or terms where the distance between them based on the likeness of their semantic content or meaning) between text strings. Its application include information retrieval, web search ranking, ad selection/relevance, question answering and machine translation.

Given a pair of documents, this model would map the inputs to a pair of feature vectors in a continuous, low dimensional space where one could compare the semantic similarity between the text strings using the cosine similarity between their vectors in that space.

<p align="center"><img src="images/semantic_modeling.png?raw=true" alt="Semantic Modeling"></p>

In the above figure, the given query (**Q**) and set of documents (**D_1, D_2, ..., D_n**) generate semantic features, which can then be used to generate pairwise distance metric.

# Models

The data set is preprocessed into two parts. There are Vocabulary files (one each for question and asnwers in .wl format) and CTF format question and answer files (CTF deserializer is used to read input data). 

With LSTM-RNN (Long-Short Term Memory and Recurrent Neural Network) (1), it sequentially takes each word in a sentence, extracts its information, and embeds it into a semantic vector.

<p align="center"><img src="images/query_vector.png?raw=true" alt="Semantic Feature"></p>

The above figure illustrated how query_vector space projected with the semantic representation of sentences and there would be a similar projection for the answer_vector.

## Parameters

One could refine parameters associated with the network architecture at *Variables.txt*. Default values set to suite the sample model found in *data* directory.

## Setup

A sampling of QA dataset is populated in *data* with vocablary (WL format) and QA (CTF format) files, which would populate data/DSSM 

Set up a virtualenv
	
	  	mkvirtualenv --python=/usr/bin/python3.6 semantic-modeling

Install required packages 

      	pip install -r requirements.txt

Run the following commad to generate gRPC class for Python

		 python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. service_spec/DSSMService.proto

# Using docker
```bash
docker build . -t singnet:semmodeling
```
# How to use the docker image
To run with grpc endpoint
docker run -it --rm -p 8001:8001 singnet:semmodeling python3.6 script.py

# Running the service

The service accepts a query and two answers phrases. The query and answers entry should be from the available sample data for the service to create vector representation and compute the similarity.  

		python3.6 service.py --qry=qry_string --ans1=first_answer --ans2=second_answer

# Authors
- Eskender Tamrat - Maintainer - [SingularityNet.io](https://singularitynet.io)

# Licenses

Microsoft Cognitive Toolkit (CNTK)
Copyright (c) Microsoft Corporation. All rights reserved.
MIT License

## References

(1) https://towardsdatascience.com/recurrent-neural-networks-and-lstm-4b601dd822a5