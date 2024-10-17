
from typing import Dict, List, Optional, Tuple

import torch
from datasets import DatasetDict
from transformers import AutoTokenizer

from hyperparam import Hyperparams as hp


IGNORE_LABEL = -100
UNK_LABEL = 0
PAD_ID = 1
UNK_ID = 3


def labelize_input(sentences: Dict[str, List[str]], tokenizer: AutoTokenizer, labels_to_hanja_words: Dict[int, str], hanja_words_to_labels: Dict[str, int]) -> Dict[str, torch.Tensor]:
    """
    Labelize a single batch of input data.
    Arguments:
        sentences: a slice of `datasets.Dataset`, but here to be more specially, `Dict[str, List[str]]` which has keys 'input_tokens' and 'output_tokens'.
        tokenizer: a `transformers.AutoTokenizer` instance.
        labels_to_hanja_words: self-explanatory name in `Dict[int, str]` type.
        hanja_words_to_labels: same above.
    Returns:
        A slice of labelized input data.
    """
    input_sentences = sentences['input_tokens']
    output_sentences = sentences['output_tokens']
    tokenized_inputs = tokenizer(input_sentences, return_offsets_mapping=True, padding=True, return_tensors='pt')
    input_ids = tokenized_inputs['input_ids']
    input_offsets = tokenized_inputs['offset_mapping']

    batched_labels = []
    for i, input_offset in enumerate(input_offsets):
        labels = []
        for j, interval in enumerate(input_offset):
            input_token = input_sentences[i][interval[0]:interval[1]]
            output_token = output_sentences[i][interval[0]:interval[1]]
            if not input_token:
                if input_ids[i][j] == PAD_ID:
                    labels.extend([IGNORE_LABEL] * (input_ids.shape[1] - j))
                    break
                labels.append(IGNORE_LABEL)
            elif input_token == output_token:
                labels.append(UNK_LABEL)
            else:
                if output_token in hanja_words_to_labels:
                    labels.append(hanja_words_to_labels[output_token])
                else:
                    # TODO: Parallelization ignored here. This is not a thread-safe approach to this question.
                    num_labels = len(labels_to_hanja_words)
                    labels_to_hanja_words[num_labels] = output_token
                    hanja_words_to_labels[output_token] = num_labels
                    labels.append(num_labels)

        batched_labels.append(labels)

    tokenized_inputs['labels'] = torch.tensor(batched_labels, dtype=torch.int64)

    return tokenized_inputs

def currier(labelize_input: callable, tokenizer: AutoTokenizer, labels_to_hanja_words: Dict[int, str], hanja_words_to_labels: Dict[str, int]) -> callable:
    def curried(sentences: Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        return labelize_input(sentences, tokenizer, labels_to_hanja_words, hanja_words_to_labels)

    return curried


def labelize_inputs(input_dataset: DatasetDict, tokenizer: AutoTokenizer, labels_to_hanja_words: Optional[Dict[int, str]] = None, hanja_words_to_labels: Optional[Dict[str, int]] = None) -> Tuple[List[Dict[str, torch.Tensor]], Dict[int, str], Dict[str, int]]:
    """
    Convert input dataset to labelized ones.
    """
    # TODO: Needs a mechanism to save these 2 dicts to local files, in case there is more data in future to be processed.
    labels_to_hanja_words = labels_to_hanja_words if labels_to_hanja_words else {UNK_LABEL: tokenizer.decode(UNK_ID)}
    hanja_words_to_labels = hanja_words_to_labels if hanja_words_to_labels else {tokenizer.decode(UNK_ID): UNK_LABEL}

    curried = currier(labelize_input, tokenizer, labels_to_hanja_words, hanja_words_to_labels)
    labeled_inputs = input_dataset.map(curried, batched=True, batch_size=hp.batch_size)

    return labeled_inputs, labels_to_hanja_words, hanja_words_to_labels
