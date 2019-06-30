# setup directories
mkdir data
mkdir logging
mkdir models
mkdir results

# get ResNet101 COCO image features and GloVe embedding
cd data

wget -O train.pickle https://www.dropbox.com/s/r56cyszpi7dpokn/train.pickle?dl=0
wget -O dev.pickle   https://www.dropbox.com/s/rubxestevay06y7/val.pickle?dl=0

wget -O glove.zip http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
unzip glove.zip
rm glove.zip

cd ..

# get annotations
cd annotations
unzip annotations.zip
cd ..
