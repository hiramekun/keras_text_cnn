# keras_text_cnn
kerasを用いてPubMed RCTデータセットの文章分類タスクを実装しました。
## imdb_sample.py
kerasでCNNを用いて実装されていたimdbに対する文章分類タスクを、自分の用いるデータセット用に変更したものです。
元実装はこちらにあります。https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py
## text_cnn.py
Convolutional Neural Networks for Sentence Classificationの論文を参考にして、CNNで文章分類タスクを実装しました。
論文はこちらにあります。http://www.aclweb.org/anthology/D14-1181
## データセット
[PubMed RCT 200k](https://github.com/Franck-Dernoncourt/pubmed-rct)を用いており、こちらからダウンロードする。
