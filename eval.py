def arg_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=32)
    parser.add_argument('--data_path', default='./drive/my_data')
    parser.add_argument('--param_path', default='./drive/text_cnn.01-1.31.hdf5')
    parser.add_argument('--model_path', default='./drive/model.json')
    args = parser.parse_args(args=[])
    return args


def main(args):
    from keras.optimizers import Adam
    from keras.models import model_from_json
    from keras.preprocessing import sequence
    from data_loader import load_all_data

    x_train, _, _ = load_all_data(args, mode='train')
    x_test, y_test, _ = load_all_data(args, mode='test')
    x_train = sequence.pad_sequences(x_train, padding='post')
    x_test = sequence.pad_sequences(x_test, maxlen=len(x_train[0]), padding='post')

    with(open(args.model_path)) as f:
        model = model_from_json(f.read())

    optimizer = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.load_weights(args.param_path)
    loss, acc = model.evaluate(x_test, y_test, batch_size=args.batch)
    print(f'loss: {loss}')
    print(f'acc: {acc}')


if __name__ == '__main__':
    main(arg_parser())
