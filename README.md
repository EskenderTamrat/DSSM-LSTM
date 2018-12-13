# DSSM with LSTM Model

This service is an implementation of Deep Structured Semantic Model or Deep Semantic Similarity Model (DSSM) with LSTM Networks. It is forked from [CNTK 303: Deep Structured Semantic Modeling with LSTM](https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_303_Deep_Structured_Semantic_Modeling_with_LSTM_Networks.ipynb).  

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

One could refine parameters associated with the network architecture at *Variables.txt*. Default values set to suite the sample model found in data/DSSM directory.

## Setup

To download a sampling of QA dataset, run `python download_model.py`, which would populate data/DSSM with vocablary (WL format) and QA (CTF format) files

Set up a virtualenv
	
	  	mkvirtualenv --python=/usr/bin/python3.6 semantic-modeling

Install required packages 

      	pip install -r requirements.txt

Run the following command to generate gRPC classes for Python

        python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. LSTM.proto

## Usage

Make changes to the parameter values at *Variables.txt* for testing with different specifications


		cd service
		#start the server 
		python server.py

Run the client-based code with parameters of the data set (two QA and two Vocabulary files)

		#on a different terminal run 
		python client.py --train_file "PATH_TO_TRAIN_FILE" --validation_file "PATH_TO_VALIDATION_FILE" --query_wf "PATH_TO_QUESTION_VOCABULARY" --answer_wf "PATH_TO_ANSWER_VOCABULARY"


## Example

The results for running the service with the sampling of QA data set:

		python client.py --train_file "../data/DSSM/train.pair.tok.ctf" --validation_file "../data/DSSM/valid.pair.tok.ctf" --query_wf "../data/DSSM/vocab_Q.wl" --answer_wf "../data/DSSM/vocab_A.wl"

Mapping from input stream to network inputs

		Learning rate per 1 samples: 0.0015625
		Momentum per 50 samples: 0.0
		Finished Epoch[1 of 5]: [Training] loss = 0.419417 * 1522, metric = 0.00% * 1522 1.168s (1303.1 samples/s);
		Finished Epoch[2 of 5]: [Training] loss = 0.096080 * 1530, metric = 0.00% * 1530 1.066s (1435.3 samples/s);
		Finished Epoch[3 of 5]: [Training] loss = 0.062401 * 1525, metric = 0.00% * 1525 1.398s (1090.8 samples/s);
		Finished Epoch[4 of 5]: [Training] loss = 0.044918 * 1534, metric = 0.00% * 1534 1.521s (1008.5 samples/s);
		Finished Epoch[5 of 5]: [Training] loss = 0.032787 * 1510, metric = 0.00% * 1510 0.968s (1559.9 samples/s);


		Finished Evaluation [1]: Minibatch[1-35]: metric = 0.13% * 410;

It will compute the cosine similarity between the query and answer pair with the higher value indicating a better answeer for the query.

		Query to Answer similarity:0.9997578263282776
		Query to poor-answer similarity:0.9997633099555969

## References

(1) https://towardsdatascience.com/recurrent-neural-networks-and-lstm-4b601dd822a5