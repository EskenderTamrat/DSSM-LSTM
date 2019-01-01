import base64
import grpc
import subprocess
import sys
import argparse
import time
from script import create_server
from service import searchWord
import unittest

from service_spec.DSSMService_pb2_grpc import DSSMStub
from service_spec.DSSMService_pb2 import DSSMRequest, DSSMResponse

class TestSuiteGrpc(unittest.TestCase):
	def setUp(self):
		self.port = "8001"
		self.server = create_server(self.port)
		self.server.start()
		self.result = ('qry_ans_similarity: 0.9998714923858643\n'
			'qry_ans2_similarity: 0.9998154044151306\n')

	def test_grpc_call(self):
		with grpc.insecure_channel('localhost:' + self.port) as channel:
			stub = DSSMStub(channel)
			request = DSSMRequest(qry="what contribution did you made to science", ans1 ="book author book_editions_published", ans2="activism address adjoining_relationship")
			feature = stub.semantic_modeling(request)
			#print("feature:\n ", feature)
			#print("self:\n ", self.result)
			self.assertMultiLineEqual(self.result, str(feature), "gRPC is working accordingly")

if __name__ == '__main__':
	suite = unittest.TestSuite()
	suite.addTest(TestSuiteGrpc("test_grpc_call"))	
	unittest.main()