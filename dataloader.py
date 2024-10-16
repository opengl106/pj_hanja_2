import codecs


def load_data():
    """
    Load tsv data in the form of tuples of two strings.
    """
    input_data_list = []
    for line in codecs.open('data/bible_ko.tsv', 'r', 'utf-8'):
        hangul_sent, hanja_sent = line.strip().split("\t")
        input_data_list.append((hangul_sent, hanja_sent))

    return input_data_list
