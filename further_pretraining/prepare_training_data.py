#-*-coding:utf-8-*- 
# author lyl
import torch
from transformers import BertTokenizer
import argparse
import random
import os
from tqdm import tqdm
import numpy as np
import json
from ark_nlp.factory.utils.conlleval import get_entity_bio


def get_args():
    parser = argparse.ArgumentParser(description="pytorch for bert pretraining data process")
    parser.add_argument("--data_path", default="data/unlabeled_train_data.txt", type=str, help="data path")
    parser.add_argument("--output_dir", default="data", type=str, help="output data path")
    parser.add_argument("--model_type", type=str, default="pretrained_model/nezha-cn-base")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--dupe_factor", type=int, default=1, help="document dupe factor")
    parser.add_argument("--seed", type=int, default=40, help="seed")
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--max_mask_len", type=int, default=100)

    return parser.parse_args()


class PreTrainingDataProcess:

    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.model_type)
        self.vocab = self.tokenizer.get_vocab()
        self.words = list(self.vocab.keys())
        self.documents = self.data_reader()

    def data_reader(self):
        data_path = self.args.data_path
        if not os.path.isfile(data_path):
            raise FileExistsError("data file path {} not exists".format(data_path))
        documents = []
        with open(data_path, "rt", encoding="utf-8") as reader:
            document = []
            for line in tqdm(reader.readlines()[:100000]):
                line = self.tokenizer.tokenize(line.strip())
                if not line:
                    if len(document) > 0:
                        documents.append(document)
                        document = []
                    continue
                if len(line) >= 10:
                    document.append(line)
            if len(document) >= 0:
                documents.append(document)
        return documents

    def generate_sentence(self, sentence: list):
        if len(sentence) > self.args.max_seq_len-2:
            sentence = sentence[: self.args.max_seq_len-2]
        gen_sentence = []
        train_mask = []
        truth_word = []
        for token in sentence:
            prob = random.random()
            if prob > self.args.mask_prob or len(truth_word) >= self.args.max_mask_len:
                gen_sentence.append(token)
                train_mask.append(0)
                continue
            prob /= self.args.mask_prob
            train_mask.append(1)
            truth_word.append(token)
            if prob < 0.8:
                gen_sentence.append(self.tokenizer.mask_token)
            elif prob < 0.9:
                gen_sentence.append(token)
            else:
                gen_sentence.append(random.choice(self.words))

        # 处理mask数量为0
        if len(truth_word) == 0:
            index = [i for i in range(len(sentence))]
            random.shuffle(index)
            chose_index = index[:int(self.args.mask_prob*len(sentence))][:self.args.max_mask_len]
            for i in chose_index:
                truth_word.append(gen_sentence[i])
                gen_sentence[i] = self.tokenizer.mask_token
                train_mask[i] = 1
        assert len(truth_word) <= self.args.max_mask_len
        return gen_sentence, train_mask, truth_word

    def generate_data(self):
        sents = []
        list(map(sents.extend, self.documents))
        max_len = args.max_seq_len
        result = []
        for document in tqdm(self.documents):
            for _ in range(self.args.dupe_factor):
                for idx in tqdm(range(len(document))):
                    sent = document[idx]
                    gen_sentence, train_mask, truth_word = self.generate_sentence(sent)

                    sentence_token = [101] + [self.vocab.get(token, self.tokenizer.unk_token_id) for token in gen_sentence] + [102]
          
                    input_ids = sentence_token
                    input_mask = len(input_ids) * [1]
                    
                    segment_ids = len(sentence_token) * [0] + (max_len - len(input_ids))*[0]
                    input_ids += (max_len - len(input_ids)) * [self.tokenizer.pad_token_id]
                    input_mask += (max_len - len(input_mask)) * [0]

                    masked_lm_positions = (np.where(np.array(train_mask) == 1)[0] + 1).tolist()

                    masked_lm_positions += (self.args.max_mask_len - len(masked_lm_positions)) * [0]

                    masked_lm_ids = [self.vocab.get(t, self.tokenizer.unk_token_id) for t in truth_word] 
                               
                    masked_lm_ids += (self.args.max_mask_len  - len(masked_lm_ids)) * [0]

                    masked_lm_weights = [1.0] * len(masked_lm_ids) + (self.args.max_mask_len  - len(masked_lm_ids)) * [0.0]
                    
            
                    # print(len(input_ids), len(segment_ids), len(masked_lm_ids), len(masked_lm_positions), len(masked_lm_weights))
                    assert len(input_ids) == max_len and \
                           len(input_mask) == max_len and \
                           len(segment_ids) == max_len and \
                           len(masked_lm_ids) == self.args.max_mask_len  and \
                           len(masked_lm_positions) == self.args.max_mask_len and \
                           len(masked_lm_weights) == self.args.max_mask_len

                    result.append({
                        "input_ids": input_ids,
                        "input_mask": input_mask,
                        "segment_ids": segment_ids,
                        "masked_lm_positions": masked_lm_positions,
                        "masked_lm_ids": masked_lm_ids,
                        "masked_lm_weights": masked_lm_weights,
                    })

        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        with open(os.path.join(self.args.output_dir, "pretrained_unlabel_data_entity_20000.json"), "wt", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False, indent=2))


class PreTrainingDataProcessEnt:
    
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.model_type)
        self.vocab = self.tokenizer.get_vocab()
        self.words = list(self.vocab.keys())
        self.documents = self.data_reader()
        '''
            self.documents:
                [{"text": "", "entitys": [["3", 0, 2],...]}, {}, ...]
        '''

    def data_reader(self):
        data_path = self.args.data_path
        if not os.path.isfile(data_path):
            raise FileExistsError("data file path {} not exists".format(data_path))
        with open(data_path, "rt", encoding="utf-8") as reader:
            data = json.load(reader)
        documents = []
        for d in data:     
            item = {
                "text": d[0],
                "entitys": get_entity_bio(d[1], id2label=None)
            }
            documents.append(item)
        return documents
    def generate_sentence(self, item):
        sentence = item["text"]
        tag_id = len(sentence) * [0]
        for tag in item["entitys"]:
            # tag_id[tag[1]: tag[2]+1] = 1
            for i in range(tag[1], tag[2]+1):
                tag_id[i] = 1
        if len(sentence) > self.args.max_seq_len-2:
            sentence = sentence[: self.args.max_seq_len-2]
            tag_id = tag_id[: self.args.max_seq_len-2]
        gen_sentence = []
        train_mask = []
        truth_word = []
        for token, tad_id in zip(sentence, tag_id):
            if tad_id == 1:
                gen_sentence.append(self.tokenizer.mask_token)
                train_mask.append(1)
                truth_word.append(token)
            else:
                gen_sentence.append(token)
                train_mask.append(0)

        # 处理mask数量为0
        if len(truth_word) == 0:
            index = [i for i in range(len(sentence))]
            random.shuffle(index)
            chose_index = index[:int(self.args.mask_prob*len(sentence))]
            for i in chose_index:
                truth_word.append(gen_sentence[i])
                gen_sentence[i] = self.tokenizer.mask_token
                train_mask[i] = 1
        # self.args.max_mask_len = max(self.args.max_mask_len, len(truth_word))
        return gen_sentence, train_mask, truth_word

    def generate_data(self):
        max_len = args.max_seq_len
        result = []
        for document in tqdm(self.documents):
            for _ in range(self.args.dupe_factor):
                gen_sentence, train_mask, truth_word = self.generate_sentence(document)
                assert sum(train_mask) == len(truth_word)

                sentence_token = [101] + [self.vocab.get(token, self.tokenizer.unk_token_id) for token in gen_sentence] + [102]
        
                input_ids = sentence_token
                input_mask = len(input_ids) * [1]
                
                segment_ids = len(sentence_token) * [0] + (max_len - len(input_ids))*[0]
                input_ids += (max_len - len(input_ids)) * [self.tokenizer.pad_token_id]
                input_mask += (max_len - len(input_mask)) * [0]

                masked_lm_positions = (np.where(np.array(train_mask) == 1)[0] + 1).tolist()

                masked_lm_positions += (self.args.max_mask_len - len(masked_lm_positions)) * [0]

                masked_lm_ids = [self.vocab.get(t, self.tokenizer.unk_token_id) for t in truth_word] 
                            
                masked_lm_ids += (self.args.max_mask_len  - len(masked_lm_ids)) * [0]

                masked_lm_weights = [1.0] * len(truth_word) + (self.args.max_mask_len  - len(truth_word)) * [0.0]
                
        
                # print(len(input_ids), len(segment_ids), len(masked_lm_ids), len(masked_lm_positions), len(masked_lm_weights))
                assert len(input_ids) == max_len and \
                        len(input_mask) == max_len and \
                        len(segment_ids) == max_len and \
                        len(masked_lm_ids) == self.args.max_mask_len  and \
                        len(masked_lm_positions) == self.args.max_mask_len and \
                        len(masked_lm_weights) == self.args.max_mask_len

                result.append({
                    "input_ids": input_ids,
                    "input_mask": input_mask,
                    "segment_ids": segment_ids,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_ids": masked_lm_ids,
                    "masked_lm_weights": masked_lm_weights,
                })

        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        with open(os.path.join(self.args.output_dir, "pretrained_unlabel_data_entity_100000.json"), "wt", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False))


class PreTrainingDataProcessNgram:
    
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.model_type)
        self.vocab = self.tokenizer.get_vocab()
        self.words = list(self.vocab.keys())
        self.additional_special_split_token='[unused1]'
        self.documents = self.data_reader()

    def data_reader(self):
        data_path = self.args.data_path
        if not os.path.isfile(data_path):
            raise FileExistsError("data file path {} not exists".format(data_path))
        documents = []
        with open(data_path, "rt", encoding="utf-8") as reader:
            for line in tqdm(reader.readlines()[:20000]):
                line = self.tokenize(line.strip())
                if len(line) >= 10:
                    documents.append(line)
        return documents
    
    def tokenize(self, text, split_token=" "):
        tokens = []
        for span_ in text.split(split_token):
            tokens += self.tokenizer.tokenize(span_)
            tokens += [self.additional_special_split_token]
        return tokens[:-1]


    def generate_sentence(self, sentence: list):
        if len(sentence) > self.args.max_seq_len-2:
            sentence = sentence[: self.args.max_seq_len-2]
        gen_sentence = []
        train_mask = []
        truth_word = []
        rands = np.random.random(len(sentence))
        idx = 0
        while idx < len(rands):
            if rands[idx] < 0.15:
                ngram = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
                if ngram == 3 and len(rands) < 7:
                    ngram = 2
                if ngram == 2 and len(rands) < 4:
                    ngram = 1
                L = idx
                R = idx + ngram
                while L < R and L < len(rands):
                    rands[L] = np.random.random()*0.15
                    L += 1
                idx = R
                if idx < len(rands):
                    rands[idx] = 1
            idx += 1

        for r, token in zip(rands, sentence):
            if r >= 0.15:
                gen_sentence.append(token)
                train_mask.append(0)
                continue
            train_mask.append(1) 
            truth_word.append(token)
            if r < 0.15 * 0.8:
                gen_sentence.append(self.tokenizer.mask_token)    
            elif r < 0.15 * 0.9:
                gen_sentence.append(token)
            else:
                gen_sentence.append(random.choice(self.words))

        return gen_sentence, train_mask, truth_word


    def generate_data(self):
        max_len = args.max_seq_len
        result = []
        for document in tqdm(self.documents):
            for _ in range(self.args.dupe_factor):
                gen_sentence, train_mask, truth_word = self.generate_sentence(document)

                sentence_token = [101] + [self.vocab.get(token, self.tokenizer.unk_token_id) for token in gen_sentence] + [102]
        
                input_ids = sentence_token
                input_mask = len(input_ids) * [1]
                
                segment_ids = len(sentence_token) * [0] + (max_len - len(input_ids))*[0]
                input_ids += (max_len - len(input_ids)) * [self.tokenizer.pad_token_id]
                input_mask += (max_len - len(input_mask)) * [0]

                masked_lm_positions = (np.where(np.array(train_mask) == 1)[0] + 1).tolist()

                masked_lm_positions += (self.args.max_mask_len - len(masked_lm_positions)) * [0]

                masked_lm_ids = [self.vocab.get(t, self.tokenizer.unk_token_id) for t in truth_word] 
                            
                masked_lm_ids += (self.args.max_mask_len  - len(masked_lm_ids)) * [0]

                masked_lm_weights = [1.0] * len(truth_word) + (self.args.max_mask_len  - len(truth_word)) * [0.0]
                
                assert len(input_ids) == max_len and \
                        len(input_mask) == max_len and \
                        len(segment_ids) == max_len and \
                        len(masked_lm_ids) == self.args.max_mask_len  and \
                        len(masked_lm_positions) == self.args.max_mask_len and \
                        len(masked_lm_weights) == self.args.max_mask_len

                result.append({
                    "input_ids": input_ids,
                    "input_mask": input_mask,
                    "segment_ids": segment_ids,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_ids": masked_lm_ids,
                    "masked_lm_weights": masked_lm_weights,
                })

        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        with open(os.path.join(self.args.output_dir, "eval_unlabeled_data_2000.json"), "wt", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    args = get_args()
    processor = PreTrainingDataProcessNgram(args)
    processor.generate_data()

