import codecs
from random import shuffle

from datasets import Dataset, DatasetDict

from hyperparam import Hyperparams as hp

def load_data() -> DatasetDict:
    """
    Load tsv data into huggingface DatasetDict.
    """
    input_data_list = []
    for line in codecs.open('data/bible_ko.tsv', 'r', 'utf-8'):
        hangul_sent, hanja_sent = line.strip().split("\t")
        if len(hangul_sent) != len(hanja_sent):
            continue
        if len(hangul_sent) > hp.maxlen:
            continue
        input_data_list.append({'input_tokens': hangul_sent, 'output_tokens': hanja_sent})

    test_length = int(len(input_data_list) * hp.test_ratio)

    shuffle(input_data_list)
    train_dataset = Dataset.from_list(input_data_list[:-test_length])
    test_dataset = Dataset.from_list(input_data_list[-test_length:])

    input_dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset,
    })

    return input_dataset
