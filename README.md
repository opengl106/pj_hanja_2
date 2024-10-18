# Hangul to Chinse Character Conversion
A fine-tuned LLM transcripting Sino-Korean Hangul words into respective Hanja orthograph.

Also see the [manually-built transformer version](https://github.com/opengl106/pj_hanja) and the [RNN version](https://github.com/opengl106/t2_h2h_converter/tree/RNN_version_archive) under my space.

## Requirements
See `requirements.txt`. Simply install them with `python3 -m pip install -r requirements.txt`.

## Data
The KRV Bible separated on each line by a '\t'. This, along with the original project inspiration, is credited to [Kyubyong](https://github.com/Kyubyong/h2h_converter).

## Model Architecture
This model is a fine-tuned version based on [klue/roberta-small](https://huggingface.co/klue/roberta-small), which is a 2021 project. The original authors and researchers used a RoBERTa architecture and a fill-mask task to pre-train the base model and obtained excellent results on Korean language understanding. Thus, I decided to use this model to perform this kind of task that requires understanding of the context. See [the KLUE paper](https://arxiv.org/abs/2105.09680) for more information about the base model.

The task is designed to be a "token classification" one, i. e. each Hangul word is either assigned a class labeled as its Hanja orthograph, or the class "[UNK]" if it is a Korean-etymology word. For example, the classification of "학교" would be "學校", but that of "오늘" would be "[UNK]" as it is not a Chinese borrow word[1]. In this methodology I fine-tuned this model with 4925 Hanja word labels from the Bible (see [labels_to_hanja_words.csv](https://github.com/opengl106/pj_hanja_2/blob/main/roberta-small-hangul-2-hanja/labels_to_hanja_words.csv) for a list of them), which contains some of our everyday speech but generally is different from modern colloquial Korean corpus. The resulting fine-tuned model is pushed to [opengl106/roberta-small-hangul-2-hanja](https://huggingface.co/opengl106/roberta-small-hangul-2-hanja).

[1] 이두 (吏讀) and other forms of *kun'yomi* are not discussed here since they serve no good for understanding modern Korean.

Our next steps would be:
* 1 Fine-tune this model on more Hanja corpus like 승정원일기 or 조선왕조실록, which is also an appropriated idea from Prof. Kyunghyun et. al.'s lab's another great work, [the HUE paper](https://aclanthology.org/2022.findings-naacl.140/).
* 2 Fine-tune this model on some NAVER-provided corpus, albeit I don't know whether NAVER has ever provided these kinda shits in open access.
* 3 Try to fine-tune two larger KLUE RoBERTa models, either by more skillful fine-tuning technologies, or by cloud computation.
* 4 Establish a web server for this service.
* 5 Write a frontend-available web app for the above-mentioned service, if necessary.
* 6 (Main Objective) Write a Chrome extension connected to the above-mentioned service, which instantly transcripts your Hangul webpages to Hanja-annotated ones when you browse the Internet.

## Training
* Adjust hyperparameters in the `hyperparams.py` if necessary.
* Run `python train.py`.

## Predicting
```
>>> from predict import predict
>>> predict('디두모라 하는 도마가 다른 제자들에게 말하되 우리도 주와 함께 죽으러 가자 하니라')
'디두모라 하는 도마가 다른 弟子들에게 말하되 우리도 主와 함께 죽으러 가자 하니라'
>>> predict(['디두모라 하는 도마가 다른 제자들에게 말하되 우리도 주와 함께 죽으 러 가자 하니라', '돌을 옮겨 놓으니 예수께서 눈을 들어 우러러 보시고 가라사대 아 버 지여 내 말을 들으신 것을 감사하나이다'])
['디두모라 하는 도마가 다른 弟子들에게 말하되 우리도 主와 함께 죽으러 가자 하니라', '돌을 옮겨 놓으니 예수께서 눈을 들어 우러러 보시고 가라사대 아버 지여 내 말을 들으신 것을 感謝하나이다']
>>> import time
>>> import codecs
>>> strings = []
>>> i = 0
>>> for line in codecs.open('data/bible_ko.tsv', 'r', 'utf-8'):
...     if len(line) <= 500:
...         strings.append(line.strip().split("\t")[0])
...     i += 1
...     if i >= 1000:
...         break
...
>>> strings[962]
'여종과 그 자식들은 앞에 두고 레아와 그 자식들은 다음에 두고 라헬과 요셉은 뒤에 두고'
>>>
>>> t = time.time()
>>> c = predict(strings)
>>> dt = time.time() - t
>>> dt
4.27865743637085
>>> c[962]
'女종과 그 子息들은 앞에 두고 레아와 그 子息들은 다음에 두고 라헬과 요셉은 뒤에 두고'
>>> s = "승정원일기는 행정과 사무, 왕명, 출납 등을 맡은 승정원의 사무를 기록한 일기이다. 단일 사료로서는 가장 방대한 양으로서 사료적 가치가 높게 평가된다. 모두 3,245책, 글자 수 2억 4,250만자다. 1960년부터 1977년까지 국사편찬위원회에서 초서체였던 승정원일기를 해서체로 고쳐쓰는 작업을 하였다. 2000년부터 2010년까지는 승정원일기 정보화사업을 진행하여 영인본 1책~111책, 127책~129책에 대한 전산화가 진행되었다. 원본 1부밖에 없는 귀중한 자료로 국보 제303호(1999.4.9)로 지정되어 있다. 이는 세계 최대 및 1차 사료로서의 가치를 인정받아 2001년 9월 유네스코세계기록유산으로 등재되었다."
>>> strings = s.split(".")
'勝內殿日記는 會計과 事務, 王命, 出納 등을 맡은 勝內侍의 事務를 記錄한 日記이다. 萬을 殞命로서는 가장 廣闊한 樣으로서 資料的 價値가 높게 判斷된다. 모두 四,九冊, 글字 數 三百 四,七萬子다. 泄精年부터 論年까지 國事便饌圍에서 初書體였던 勝內殿日記를 해서체로 고쳐쓰는 작업을 하였다. 天年부터 每年年까지는 勝色日記 庫和事業을 進行하여 營人本 一冊~太初冊, 十二冊~十二冊에 對한 機械譁가 進行되었다. 願 一夫밖에 없는 貴重한 資料로 寶物 第四三호(이番.4.9)로 指定되어 있다. 이는 世界 第一 및 一次 殞命로서의 價値를 인정받아 古代年 9月 大聲世界記錄寶物으로 天下되었다.'
```
## Efficiency
Great. What else can I say? 1,000 lines in 4.28 secs, and mostly appreciated, the model knows the Korean language (of **freaking** course) and it even maps 전산화 into 機械譁 as "電算" apparently did not exist in the Biblical ages. However, I mean, it **really** needs some corpus update.
