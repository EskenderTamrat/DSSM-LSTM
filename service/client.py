import grpc

import LSTM_pb2
import LSTM_pb2_grpc
import argparse
import os
import sys

parser = argparse.ArgumentParser()

# open a gRPC channel
channel = grpc.insecure_channel('localhost:50051')
stub = LSTM_pb2_grpc.ReplyStub(channel)

if __name__ == "__main__":
	parser.add_argument("--train_file", type=str, help='path to the training file (for available sampling data use: data/DSSM/train.pair.tok.ctf)')
	parser.add_argument("--validation_file", type=str, help='path to the validation file (for available sampling data use: data/DSSM/valid.pair.tok.ctf)')
	parser.add_argument("--query_wf", type=str, help='path to the vocabulary file for questions (for available sampling data use: data/DSSM/vocab.Q.wl)')
	parser.add_argument("--answer_wf", type=str, help='path to the vocabulary file for answers (for available sampling data use: data/DSSM/vocab.A.wl)')
	args = parser.parse_args()
	if  len(sys.argv) == 1:
		parser.print_help()
		sys.exit()
	response = stub.semantic_modeling(LSTM_pb2.Query(train_file = args.train_file, validation_file = args.validation_file, query_wf = args.query_wf, answer_wf = args.answer_wf))	
	print("Query to Answer similarity: ", response.qry_ans_similarity)
	print("Query to poor-answer similarity: ", response.qry_poor_ans_similarity)