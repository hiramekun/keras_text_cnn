import os

from keras.preprocessing.text import Tokenizer

context_label = (
    'background',
    'conclusions',
    'methods',
    'objective',
    'results'
)


def load_all_data(args, mode):
    def load_data(data_dir, fname):
        texts = []
        labels = []
        with open(os.path.join('{}/{}'.format(data_dir, fname)), encoding='utf-8') as f:
            for line in f:
                texts.append("<s> " + line.strip() + " </s>")
                labels.append(context_label.index(os.path.splitext(fname)[0]))
        return texts, labels

    tokenizer = Tokenizer(filters="")
    whole_texts = []
    whole_labels = []
    data_dir = '{}/{}'.format(args.data_path, mode)

    for fname in os.listdir(data_dir):
        x, y = load_data(data_dir, fname)
        whole_texts.extend(x)
        whole_labels.extend(y)

    from keras.utils.np_utils import to_categorical
    categorical_labels = to_categorical(whole_labels, num_classes=5)
    tokenizer.fit_on_texts(whole_texts)

    return tokenizer.texts_to_sequences(whole_texts), categorical_labels, tokenizer
