# DSSM with LSTM Model

This service is an implementation of Deep Structured Semantic Model or Deep Semantic Similarity Model (DSSM) with LSTM Networks. It is forked from [CNTK 303: Deep Structured Semantic Modeling with LSTM](https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_303_Deep_Structured_Semantic_Modeling_with_LSTM_Networks.ipynb). Given a pair of documents, the model would map the inputs to a pair of feature vectors in a continuous, low dimensional space where one could compare the semantic similarity between the text strings using the cosine similarity between their vectors in that space. 

<p align="center"><img src="images/semantic_modeling.png?raw=true" alt="Semantic Modeling"></p>

In the above figure, the given query (**Q**) and set of documents (**D_1, D_2, ..., D_n**) generate semantic features, which can then be used to generate pairwise distance metric.

More details on the implementation could be found here: https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_303_Deep_Structured_Semantic_Modeling_with_LSTM_Networks.ipynb

## Setup

To download a sampling of QA dataset, run `python download.py`, which would populate data/DSSM with vocablary (WL format) and QA (CTF format) files

## Usage

		cd service
		#start the server 
		python server.py

		#on a different terminal run 
		python client.py --train_file "PATH_TO_TRAIN_FILE" --validation_file "PATH_TO_VALIDATION_FILE" --query_wf "PATH_TO_QUESTION_VOCABULARY" --answer_wf "PATH_TO_ANSWER_VOCABULARY"


## Example

The results for running the service with the sampling of QA data set looks the following

		python client.py --train_file "../data/DSSM/train.pair.tok.ctf" --validation_file "../data/DSSM/valid.pair.tok.ctf" --query_wf "../data/DSSM/vocab_Q.wl" --answer_wf "../data/DSSM/vocab_A.wl"

		Finished Evaluation [1]: Minibatch[1-35]: metric = 0.13% * 410;
		Query to Answer similarity:0.9997578263282776
		Query to poor-answer similarity:0.9997633099555969