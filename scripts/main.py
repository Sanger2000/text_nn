from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))
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


if __name__ == '__main__':
    # update args and print
    args = generic.parse_args()
    word_embeddings, word_to_indx = embedding.get_embedding_tensor(args)
        
    args.embedding = 'char'
    char_embeddings, char_to_indx = embedding.get_embedding_tensor(args) 
    if args.dataset == "news_group":
        args.num_class = 20

    if args.representation_type == 'word':
        if args.embedding_size is None:
            args.embedding_size = word_embeddings.shape[1]
        args.vocab_size = len(word_to_indx)

    elif args.representation_type == 'char':
        if args.embedding_size is None:
            args.embedding_size = char_embeddings.shape[1]
        args.vocab_size = len(char_to_indx)

    elif args.representation_type == 'both':
        args.vocab_size = len(word_to_indx)

    train_data, dev_data, test_data = dataset_factory.get_dataset(args, word_to_indx, char_to_indx)

    results_path_stem = args.results_path.split('/')[-1].split('.')[0]
    args.model_path = '{}.pt'.format(os.path.join(args.save_dir, results_path_stem))

    # model
    model = model_factory.get_model(args, char_embeddings, word_embeddings)

    # train
    if args.train :
        print('test')
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
