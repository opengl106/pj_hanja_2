from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("KoichiYasuoka/roberta-large-korean-hanja")
model = AutoModelForMaskedLM.from_pretrained("KoichiYasuoka/roberta-large-korean-hanja")
