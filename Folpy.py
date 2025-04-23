import numpy as np
import pandas as pd

class cop:

    def nodcnn(word_index, embeddings_index, nclasses, MAX_SEQUENCE_LENGTH=500, EMBEDDING_DIM=100):
        kernel_size = 2
        filters = 256
        pool_size = 2
        gru_node = 256
        embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                if len(embedding_matrix[i]) !=len(embedding_vector):
                    print("could not broadcast input array from shape",str(len(embedding_matrix[i])),
                                     "into shape",str(len(embedding_vector))," Please make sure your"
                                     " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                    exit(1)

                embedding_matrix[i] = embedding_vector
        model = Sequential()
        model.add(Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=True))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters, kernel_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Conv1D(filters, kernel_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Conv1D(filters, kernel_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(Conv1D(filters, kernel_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2))
        model.add(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2))
        model.add(LSTM(gru_node, return_sequences=True, recurrent_dropout=0.2))
        model.add(LSTM(gru_node, recurrent_dropout=0.2))
        model.add(Dense(1024,activation='relu'))
        model.add(Dense(nclasses))
        model.add(Activation('softmax'))
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model
    
    
    
    def correletion():
        bt = {'school','address','sex','famsize','Pstatus','Mjob','Fjob','reason','guardian','aff_no'}
        return bt