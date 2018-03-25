import os

import numpy as np

context_label = (
    'background',
    'conclusions',
    'methods',
    'objective',
    'results'
)


def load_split_data(args):
    from keras.preprocessing.text import Tokenizer
    from sklearn.model_selection import train_test_split
    x_train, y_train = load_all_data(args, 'train')
    x_test, y_test = load_all_data(args, 'test')
    print(x_train.shape)
    print(y_train.shape)
    x_all = np.append(x_train, x_test)
    y_all = np.append(y_train, y_test, axis=0)
    print(x_all.shape)
    print(y_all.shape)
    x_train, x_test, y_train, y_test = train_test_split(x_all.reshape((len(x_all), 1)),
                                                        y_all.reshape(
                                                            (len(y_all[:]), len(y_all[0][:]), 1)),
                                                        test_size=0.33,
                                                        random_state=0)

    tokenizer_train = Tokenizer(filters="")
    tokenizer_test = Tokenizer(filters="")
    x_train = x_train.reshape((len(x_train[:]),))
    x_test = x_test.reshape((len(x_test[:]),))
    y_train = y_train.reshape((len(y_train[:]), len(y_train[0][:])))
    y_test = y_test.reshape((len(y_test[:]), len(y_test[0][:])))
    tokenizer_train.fit_on_texts(x_train)
    tokenizer_test.fit_on_texts(x_test)

    print(x_train.shape)
    print(x_test.shape)
    x_train = tokenizer_train.texts_to_sequences(x_train)
    x_test = tokenizer_test.texts_to_sequences(x_test)

    return x_train, y_train, x_test, y_test, tokenizer_train, tokenizer_test


def load_all_data(args, mode):
    def load_data(data_dir, fname):
        texts = []
        labels = []
        with open(os.path.join('{}/{}'.format(data_dir, fname)), encoding='utf-8') as f:
            for line in f:
                texts.append("<s> " + line.strip() + " </s>")
                labels.append(context_label.index(os.path.splitext(fname)[0]))
        return texts, labels

    whole_texts = []
    whole_labels = []
    data_dir = '{}/{}'.format(args.data_path, mode)

    for fname in os.listdir(data_dir):
        x, y = load_data(data_dir, fname)
        whole_texts.extend(x)
        whole_labels.extend(y)

    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(whole_labels, num_classes=5)

    return np.asarray(whole_texts), categorical_labels
