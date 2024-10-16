import codecs
from typing import List, Tuple, Dict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

UNK_LABEL = 0
UNK_ID = 3
IGNORE_LABEL = -100

def labelize_input(sentences: Tuple[str, str], tokenizer: AutoTokenizer, num_labels: int, labels_to_hanja_words: Dict[int, str], hanja_words_to_labels: Dict[str, int]) -> Tuple[Dict[str, torch.Tensor], int]:
    input_chars = sentences[0]
    output_chars = sentences[1]
    tokenized_input = tokenizer.encode_plus(input_chars, return_offsets_mapping=True, return_tensors='pt')

    # TODO: Batching and parallelizing is skipped here.
    # TODO: The sentence is not padded to `max_length` and PAD token is not processed here. Also attention masks are not padded with zeros.
    # TODO: Add a limit that restricts tokens length to `max_length`. If exceeded, pass and return None for first argument.
    input_offset = tokenized_input['offset_mapping'][0]
    labels = []

    for interval in input_offset:
        input_token = input_chars[interval[0]:interval[1]]
        output_token = output_chars[interval[0]:interval[1]]
        if not input_token:
            labels.append(IGNORE_LABEL)
        elif input_token == output_token:
            labels.append(UNK_LABEL)
        else:
            if output_token in hanja_words_to_labels:
                labels.append(hanja_words_to_labels[output_token])
            else:
                labels_to_hanja_words[num_labels] = output_token
                hanja_words_to_labels[output_token] = num_labels
                labels.append(num_labels)
                num_labels += 1

    labeled_input = {
        'input_ids': tokenized_input['input_ids'],
        'attention_mask': tokenized_input['attention_mask'],
        'labels': torch.tensor(labels, dtype=torch.int64).unsqueeze(0),
    }

    return labeled_input, num_labels


def labelize_inputs(input_data_list: List[Tuple[str, str]], tokenizer: AutoTokenizer) -> Tuple[List[Dict[str, torch.Tensor]], Dict[int, str]]:
    # Convert input data list to labelized ones.
    labels_to_hanja_words = {UNK_LABEL: tokenizer.decode(UNK_ID)}
    hanja_words_to_labels = {tokenizer.decode(UNK_ID): UNK_LABEL}
    num_labels = len(labels_to_hanja_words)

    labeled_inputs = []
    for item in tqdm(input_data_list):
        labeled_input, num_labels = labelize_input(item, tokenizer, num_labels, labels_to_hanja_words, hanja_words_to_labels)
        labeled_inputs.append(labeled_input)

    return labeled_inputs, labels_to_hanja_words


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("KoichiYasuoka/roberta-large-korean-hanja")
    model = AutoModelForMaskedLM.from_pretrained("KoichiYasuoka/roberta-large-korean-hanja")

    input_data_list = []
    for line in codecs.open('data/bible_ko.tsv', 'r', 'utf-8'):
        hangul_sent, hanja_sent = line.strip().split("\t")
        input_data_list.append((hangul_sent, hanja_sent))

    labeled_inputs, labels_to_hanja_words = labelize_inputs(input_data_list, tokenizer)
