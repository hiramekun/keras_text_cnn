from __future__ import print_function

import numpy as np
from keras import Input, Model
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Reshape, Conv2D, MaxPooling2D, merge, Flatten
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
    batch_size = 32
    embedding_dims = 128
    filters = 128
    epochs = 20

    print('Loading data...')
    x_train, y_train, x_test, y_test, tokenizer_train, tokenizer_test = load_split_data(args)
    y_train = np.asarray(y_train).astype('float32')
    y_test = np.asarray(y_test).astype('float32')
    vocab_size_train = len(tokenizer_train.word_index) + 1
    vocab_size_test = len(tokenizer_test.word_index) + 1
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, padding='post')
    x_test = sequence.pad_sequences(x_test, maxlen=len(x_train[0]), padding='post')
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    seq_x_len = len(x_train[0])
    seq_y_len = len(y_train[0])
    print('Build model...')

    inputs = Input(shape=(seq_x_len,))
    x = Embedding(input_dim=vocab_size_train, output_dim=embedding_dims, input_length=seq_x_len)(
        inputs)
    print(x.shape)
    reshape = Reshape((seq_x_len, embedding_dims, 1))(x)
    print(reshape.shape)
    conv1 = Conv2D(filters=filters, kernel_size=(seq_x_len, 3), padding='same', activation='relu')(
        reshape)
    conv2 = Conv2D(filters=filters, kernel_size=(seq_x_len, 4), padding='same', activation='relu')(
        reshape)
    conv3 = Conv2D(filters=filters, kernel_size=(seq_x_len, 5), padding='same', activation='relu')(
        reshape)
    print('conv...')
    print(conv1.shape)
    print(conv2.shape)
    print(conv3.shape)
    pool1 = MaxPooling2D(pool_size=(seq_x_len, embedding_dims), strides=(1, 1))(conv1)
    pool2 = MaxPooling2D(pool_size=(seq_x_len, embedding_dims), strides=(1, 1))(conv2)
    pool3 = MaxPooling2D(pool_size=(seq_x_len, embedding_dims), strides=(1, 1))(conv3)
    print('pooling...')
    print(pool1.shape)
    print(pool2.shape)
    print(pool3.shape)
    x = merge([pool1, pool2, pool3], mode='concat', concat_axis=1)
    x = Flatten()(x)
    output = Dense(units=5, activation='softmax')(x)
    model = Model(inputs, output)

    optimizer = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    checkpointer = ModelCheckpoint(
        filepath='./drive/text_cnn' + '.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1,
        save_best_only=True, monitor='val_acc', mode='max')
    csv_logger = CSVLogger('./text_cnn.log')

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              callbacks=[checkpointer, csv_logger])

    with(open('./drive/model.json', 'w')) as f:
        f.write(model.to_json())


if __name__ == '__main__':
    main(arg_parser())
