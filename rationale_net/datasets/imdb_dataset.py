import os
import gzip
import re
import tqdm
import torch
from rationale_net.utils.embedding import get_indices_tensor
from rationale_net.datasets.factory import RegisterDataset
from rationale_net.datasets.abstract_dataset import AbstractDataset
from sklearn.datasets import fetch_20newsgroups
import random
random.seed(0)

def preprocess_data(set_type):
    '''
    input: specifies test or train directory
    output: tuple of (features, labels) for data where labels are 0 for negative 1 for
    positive and features is a list of each review
    '''
    label_dictionary = {'neg': 0, 'pos': 1}
    label_name_dictionary = ['Bad', 'Good']

    filename = "raw_data/imdb/%s/" % (set_type)
    preprocessed_data = []
    
    for sentiment in ('neg', 'pos'):
        for file in os.listdir(filename + sentiment):
            label = label_dictionary[sentiment]

            text = open(filename + sentiment + '/' + file, 'r').read()
            text = re.sub('\W+', ' ', text).lower().strip()

            label_name = label_name_dictionary[label]

            preprocessed_data.append((text, label, label_name))
    return preprocessed_data

@RegisterDataset('imdb')
class IMDBDataset(AbstractDataset):

    def __init__(self, args, word_to_indx, char_to_indx, name):
        self.args = args
        self.args.num_class = 2
        self.name = name
        self.dataset = []
        self.word_to_indx  = word_to_indx
        self.char_to_indx = char_to_indx
        self.max_word_length = args.max_word_length
        self.max_char_length = args.max_char_length
        self.class_balance = {}

        if name in ['train', 'dev']:
            data = preprocess_data('train')
            random.seed(0)
            random.shuffle(data)
            num_train = int(len(data)*.8)
            if name == 'train':
                data = data[:num_train]
            else:
                data = data[num_train:]
        else:
            data = preprocess_data('test')

        for indx, _sample in tqdm.tqdm(enumerate(data)):
            sample = self.processLine(_sample)

            if not sample['y'] in self.class_balance:
                self.class_balance[ sample['y'] ] = 0
            self.class_balance[ sample['y'] ] += 1
            self.dataset.append(sample)

        print ("Class balance", self.class_balance)

        if args.class_balance:
            raise NotImplementedError("IMDB dataset doesn't support balanced sampling")
        if args.objective == 'mse':
            raise NotImplementedError("IMDB dataset does not support Regression objective")

    def processLine(self, row):
        text, label, label_name = row

        chars= [c for c in text[::-1]]


        x1 = get_indices_tensor(chars, self.char_to_indx, self.max_char_length)
        x2 =  get_indices_tensor(text.split(), self.word_to_indx, self.max_word_length)
        sample = {'text':text,'x_char':x1, 'x_word':x2, 'y':label, 'y_name': label_name}
        return sample
