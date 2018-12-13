## gRPC service for [Semantic Modeling with LSTM Network](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/02/LSTM_DSSM_IEEE_TASLP.pdf) implementation

## Setup

  - Make sure you read the readme in the parent folder and installed all requirements
  - Run the following command to generate gRPC classes for Python

        python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. LSTM.proto

## Running Demo

  - You can use the sampling data available in data/DSSM

        python client.py --train_file "PATH_TO_TRAIN_FILE" --validation_file "PATH_TO_VALIDATION_FILE" --query_wf "PATH_TO_QUESTION_VOCABULARY" --answer_wf "PATH_TO_ANSWER_VOCABULARY"

    - Optional arguments:
     
      -h, --help show the ff help message and exit
      
      --train_file TRAIN_FILE
          path to the training file (for available sampling data use: data/DSSM/train.pair.tok.ctf)
      --validation_file VALIDATION_FILE
          path to the validation file (for available sampling data use: data/DSSM/valid.pair.tok.ctf)
      --query_wf QUERY_WF   
          path to the vocabulary file for questions (for available sampling data use: data/DSSM/vocab.Q.wl)
      --answer_wf ANSWER_WF
          path to the vocabulary file for answers (for available sampling data use: data/DSSM/vocab.A.wl)
