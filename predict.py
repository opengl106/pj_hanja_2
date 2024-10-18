from typing import Any, Dict, List, Union

import torch
from transformers import pipeline

from hyperparam import Hyperparams as hp


UNK_TOKEN = '[UNK]'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transcriber = pipeline("token-classification", model=f"{hp.user_repo}/{hp.model_output_dir}", device=device)


def process_single(result_dict: List[Dict[str, Any]], text: str) -> str:
    """
    Process one single result into string.
    """
    result = ""
    last_end = 0

    for item in result_dict:
        result += text[last_end:item['start']]

        if item['entity'] == UNK_TOKEN:
            result += text[item['start']:item['end']]
        else:
            result += item['entity']

        last_end = item['end']

    return result


def predict(text: Union[List[str], str]) -> Union[List[str], str]:
    """
    Transcript hangul sentences to hanjas.
    """
    result_dict = transcriber(text)
    if isinstance(text, list):
        result = []
        for rd, t in zip(result_dict, text):
            result.append(process_single(rd, t))
        return result
    else:
        return process_single(result_dict, text)
