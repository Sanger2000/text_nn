import argparse
import copy
import rationale_net.datasets.factory as dataset_factory
import rationale_net.utils.embedding as embedding
import rationale_net.models.factory as model_factory
import rationale_net.utils.generic as generic
import rationale_net.learn.train as train
import os
import torch
import datetime
import pickle
import pdb
import csv

models = ('composite, cnn, transformer, rnn')
input_types = dict(zip(models, (['both'], ['word', 'char'], ['word'], ['word', 'char']))
fully_connected_layers = [16, 32, 62, 128]
hidden_dims = [[4], [16], [32], [64], [16, 16], [32, 32]]
pretrain_embeddings = {'char': [False], 'word': [False, True]. 'both': [False]}
N = dict(zip(models, ([0], [0], [1, 2, 3, 4], [0])))
d_ff = dict(zip(models, ([0], [0], [256, 512, 1024, 2048], [256, 512, 1024, 2048])))
heads = dict(zip(models, ([0], [0]. [4, 8, 16]. [0])))
filter_num = dict(zip(models, ([[32], [64], [128], [256]], [[32], [64], [128], [256]], [None], [None])))
input_types = {'composite': ['both']. 'cnn': ['word', 'char'], 'transformer': ['word'], 'rnn': ['word', 'char']}
iterate_through(args):
    args = generic.parse_args()
    word_embeddings, word_to_indx = embedding.get_embedding_tensor(args)
        
    args.embedding = 'char'
    char_embeddings, char_to_indx = embedding.get_embedding_tensor(args)

    model_type = args.model_form
    f = open(model_type + '_performance.csv', 'w')
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Rep Type', 'dataset', 'train_acc', 'dev_acc', 'test_acc', 'n_epochs', 'fully_connected', 'hidden_dims', 'pretrain_e', 'N', 'd_ff', 'heads', 'filter_num'])

    for dataset in ('imdb', 'news-group'):
        args.dataset = dataset
        for rep_type in input_types[model_type]:
            args.representation_type = rep_type

            if args.representation_type == 'word':
                if args.embedding_size is None:
                    args.embedding_size = word_embeddings.shape[1]
                args.vocab_size = len(word_to_indx)

            elif args.representation_type == 'x_char':
                if args.embedding_size is None:
                    args.embedding_size = char_embeddings.shape[1]
                args.vocab_size = len(char_to_indx)

            elif args.representation_type == 'both':
                args.vocab_size = len(word_to_indx)

            train_data, dev_data, test_data = dataset_factory.get_dataset(args, word_to_indx, char_to_indx)
            for epochs in range(20, 30, 40, 50):
                args.epochs = epochs
                for fully_connected_layer in fully_connected_layers:
                    args.fully_connected_layer = fully_connected_layer
                    for h_d in hidden_dims:
                        args.hidden_dim = h_d
                        for p_e in pretrain_embeddings[rep_type]:
                            args.pretrain_embeddings=p_e
                            for n in N[model_type]:
                                args.N = n
                                for d in d_ff[model_type]:
                                    args.d_ff = d
                                    for h in heads[model_type]:
                                        args.heads = h
                                        for f_n in filter_num[model_type]:
                                            args.filter_num = f_n
                                        

def train(args, gs, char_idx):
    model = model_factory.get_model(args, char_embeddings, word_embeddings)

    # train
    if args.train :
        epoch_stats, model = train.train_model(train_data, dev_data, model, args)
        args.epoch_stats = epoch_stats
        save_path = args.results_path
        print("Save train/dev results to", save_path)
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path,'wb') )


    # test
    if args.test :
        test_stats = train.test_model(test_data, model, args)
        args.test_stats = test_stats
        args.train_data = train_data
        args.test_data = test_data

        save_path = args.results_path
        print("Save test results to", save_path)
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path,'wb') )

