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

wget -O train.zip https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip train.zip
rm train.zip
mv v2_mscoco_train2014_annotations.json train_answers.json

wget -O dev.zip https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip dev.zip
rm dev.zip
mv v2_mscoco_val2014_annotations.json dev_answers.json

cd ..
