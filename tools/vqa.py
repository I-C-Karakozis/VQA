import json
import torch
import pickle
import numpy as np
import torch.utils.data as data

class VQA(data.Dataset):
    def __init__(self, glove_word2idx, top_n_answers, split, id2answer=None):
        # load annotations and data
        if split == 0:
            print("LOADING TRAIN SET...")
            self.img_features = pickle.load(open("data/train.pickle", "rb"))
            self.questions = json.load(open("annotations/train_questions.json"))['questions']
            answers = json.load(open("annotations/train_answers.json"))['annotations']
        elif split == 1:
            print("LOADING TEST SET...")
            self.img_features = pickle.load(open("data/dev.pickle", "rb"))
            self.questions = json.load(open("annotations/dev_questions.json"))['questions']
            answers = json.load(open("annotations/dev_answers.json"))['annotations']
        else:
            exit("Bad split id input.")

        # map ids to answers in decreasing order of frequency
        if id2answer is None:
            self.answer_freq = {}
            for i in range(len(answers)):
                answer = answers[i]['multiple_choice_answer']
                if answer in self.answer_freq:
                    self.answer_freq[answer] += 1
                else:
                    self.answer_freq[answer] = 1
            self.id2answer = np.array(sorted(self.answer_freq.items(), key=lambda x:x[1], reverse=True))[:, 0].tolist()
            self.id2answer = self.id2answer[:top_n_answers]
        else:
            self.id2answer = id2answer

        # combine answer and question annotations
        for i in range(len(self.questions)):
            assert(self.questions[i]['question_id'] == answers[i]['question_id'])
            answer = answers[i]['multiple_choice_answer']
            self.questions[i]['answer'] = answer
            try:
                self.questions[i]['answer_id'] = self.id2answer.index(answer)
            except ValueError:
                self.questions[i]['answer_id'] = len(self.id2answer)

        # remove training questions with answers that are not included in the top_n_answers
        if split == 0:
            filtered_questions = []
            for question in self.questions:
                if question['answer_id'] < len(self.id2answer):
                    filtered_questions.append(question)
            self.questions = filtered_questions

        # replace words with GloVe indices
        unk_count = 0
        for i in range(len(self.questions)):
            q_str_split = self.questions[i]['question'][:-1].lower().split()
            q_id_split = []
            for w in q_str_split:
                if w in glove_word2idx:
                    q_id_split.append(glove_word2idx[w])
                else:
                    q_id_split.append(len(glove_word2idx)-1)
                    unk_count += 1
            self.questions[i]['glove_question'] = q_id_split
        print("{} unknown words seen.".format(unk_count))

        # dataset statistics
        self.img_feat_dim = len(self.img_features[self.questions[0]['image_id']])
        self.n_imgs = len(self.img_features)
        self.n_questions = len(self.questions)
        print("{} Images | {} Questions".format(self.n_imgs, self.n_questions))
        print("{} Possible Answers".format(len(self.id2answer)))

    def __getitem__(self, idx):
        q_id   = self.questions[idx]['question_id']
        q_str  = self.questions[idx]['question']
        img_id = self.questions[idx]['image_id']
        img_feat = torch.FloatTensor(self.img_features[img_id])
        q_id_split = torch.LongTensor(self.questions[idx]['glove_question'])
        answer = torch.LongTensor([self.questions[idx]['answer_id']])

        return img_id, q_id, q_str, img_feat, q_id_split, answer

    def __len__(self):
        return self.n_questions

def test_get_item(dataset):
    img_feat, q_str, answer = dataset.__getitem__(0)
    print(img_feat.size(), q_str, answer, dataset.id2answer[answer.item()])

if __name__ == "__main__":
    trainset = VQA(split=0)
    devset   = VQA(split=1)

    test_get_item(trainset)
    test_get_item(devset)
