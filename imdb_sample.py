'''This example demonstrates the use of Convolution1D for text classification.
Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
'''
from __future__ import print_function

from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Conv1D, GlobalMaxPooling1D, BatchNormalization, np, MaxPooling1D
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import sequence

from data_loader import load_split_data


def arg_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./drive/my_data')
    parser.add_argument('--mode', default='train')
    args = parser.parse_args(args=[])
    return args


def main(args):
    # set parameters:
    max_features = 5000
    maxlen = 400
    batch_size = 32
    embedding_dims = 128
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    epochs = 10  # we start off with an efficient embedding layer which maps

    print('Loading data...')
    X_train, y_train, X_test, y_test, tokenizer_train, tokenizer_test = load_split_data(args)

    y_train = np.asarray(y_train).astype('float32')
    y_test = np.asarray(y_test).astype('float32')
    vocab_size_train = len(tokenizer_train.word_index) + 1
    vocab_size_test = len(tokenizer_test.word_index) + 1
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, padding='post')
    X_test = sequence.pad_sequences(X_test, maxlen=len(X_train[0]), padding='post')
    print('x_train shape:', X_train.shape)
    print('x_test shape:', X_test.shape)
    print('vocab_size_train', vocab_size_train)

    seqX_len = len(X_train[0])
    print('seqX_len', seqX_len)
    seqY_len = len(y_train[0])
    print('Build model...')

    model = Sequential()

    # our vocab indices into embedding_dims dimensions
    model.add(
        Embedding(input_dim=vocab_size_train, output_dim=embedding_dims, input_length=seqX_len))
    model.add(Dropout(0.2))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='same',
                     strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # we use max pooling:
    model.add(MaxPooling1D(strides=1))

    model.add(Conv1D(filters,
                     kernel_size,
                     padding='same',
                     strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(units=5))
    model.add(Activation('sigmoid'))

    optimizer = Adam(lr=0.000001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    checkpointer = ModelCheckpoint(
        filepath='./drive/text_cnn' + '.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1,
        save_best_only=True, monitor='val_acc', mode='max')
    csv_logger = CSVLogger('./drive/text_cnn.log')

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test),
              callbacks=[checkpointer, csv_logger])

    with(open('./drive/text_cnn_imdb_model.json', 'w')) as f:
        f.write(model.to_json())


if __name__ == '__main__':
    main(arg_parser())
