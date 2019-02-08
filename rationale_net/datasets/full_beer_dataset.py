import gzip
import tqdm
from rationale_net.utils.embedding import get_indices_tensor
from rationale_net.datasets.factory import RegisterDataset
from rationale_net.datasets.abstract_dataset import AbstractDataset


SMALL_TRAIN_SIZE = 800

@RegisterDataset('full_beer')
class FullBeerDataset(AbstractDataset):

    def __init__(self, args, word_to_indx, char_to_indx, mode, stem='raw_data/beer_review/reviews.aspect'):
        aspect = args.aspect
        self.args= args
        self.name = mode
        self.objective = args.objective
        self.dataset = []
        self.char_to_indx  = char_to_indx
        self.word_to_indx = word_to_indx 
        self.max_word_length = args.max_word_length
        self.max_char_length = args.max_char_length
        self.aspects_to_num = {'appearance':0, 'aroma':1, 'palate':2,'taste':3, 'overall':4}
        self.class_map = {0: 0, 1:0, 2:0, 3:0, 4:1, 5:1, 6:1, 7:1, 8:2, 9:2, 10:2}
        self.name_to_key = {'train':'train', 'dev':'heldout', 'test':'heldout'}
        self.class_balance = {}
        with gzip.open(stem+str(self.aspects_to_num[aspect])+'.'+self.name_to_key[self.name]+'.txt.gz') as gfile:
            lines = gfile.readlines()
            lines = list(zip( range(len(lines)), lines) )


            if args.debug_mode:
                lines = lines[:SMALL_TRAIN_SIZE]
            elif self.name == 'dev':
                lines = lines[:5000]
            elif self.name == 'test':
                lines = lines[5000:10000]
            elif self.name == 'train':
                lines = lines[0:20000]



            for indx, line in tqdm.tqdm(enumerate(lines)):
                uid, line_content = line
                sample = self.processLine(line_content, self.aspects_to_num[aspect], indx, args.model_form=="char_cnn")

                if not sample['y'] in self.class_balance:
                    self.class_balance[ sample['y'] ] = 0
                self.class_balance[ sample['y'] ] += 1
                sample['uid'] = uid
                self.dataset.append(sample)
            gfile.close()
        print ("Class balance", self.class_balance)

        if args.class_balance:
            raise NotImplementedError("Beer review dataset doesn't support balanced sampling!")

    ## Convert one line from beer dataset to {Text, Tensor, Labels}
    def processLine(self, line, aspect_num, i, is_char_cnn=False):
        if isinstance(line, bytes):
            line = line.decode()
        labels = [ float(v) for v in line.split()[:5] ]
        if self.objective == 'mse':
            label = float(labels[aspect_num])
            self.args.num_class = 1
        else:
            label = int(self.class_map[ int(labels[aspect_num] *10) ])
            self.args.num_class = 3
        text_list = line.split('\t')[-1]

        char_list = [char for char in text_list[::-1]]
        text_list = text_list.split()

        char_list = char_list[:max_char_length]
        text_list = text_list[:max_word_length]

        word_text = " ".join(text_list)
        text = "".join(text_list)[::-1]
        x1 = get_indices_tensor(char_list, self.char_to_indx, self.max_char_length)
        x2 = get_indices_tensor(text_list, self.word_to_indx, self.max_word_length)
        #make x tuple of vectors
        sample = {'text':text,'x_char': x1, 'x_word':x2, 'y':label, 'i':i}
        return sample
