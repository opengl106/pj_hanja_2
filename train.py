
from transformers import AutoTokenizer

from dataloader import load_data
from tokenizer import labelize_inputs

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

    input_data_list = load_data()

    labeled_inputs, labels_to_hanja_words, hanja_words_to_labels = labelize_inputs(input_data_list, tokenizer)
