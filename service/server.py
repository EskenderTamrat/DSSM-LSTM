import grpc
from concurrent import futures
import time

import LSTM_pb2
import LSTM_pb2_grpc

import LSTM

class ReplyServicer(LSTM_pb2_grpc.ReplyServicer):
	def semantic_modeling(self, request, context):
		response = LSTM_pb2.Response()
		response.qry_ans_similarity, response.qry_poor_ans_similarity = LSTM.lstm(request.train_file, request.validation_file, request.query_wf, request.answer_wf)
		return response

# create a gRPC server
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

LSTM_pb2_grpc.add_ReplyServicer_to_server(
				ReplyServicer(), server)

# listen on port 50051
print('Starting server. Listening on port 8001.')
server.add_insecure_port('[::]:8001')
server.start()

# since server.start() will not block,
# a sleep-loop is added to keep alive
try:
    while True:
        time.sleep(86400)
except KeyboardInterrupt:
    server.stop(0)