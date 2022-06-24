import warnings

warnings.filterwarnings("ignore")

import torch

import os
import argparse
import numpy as np
import json
from tqdm import tqdm

from ark_nlp.factory.utils.seed import set_seed
from ark_nlp.model.ner.global_pointer_bert import Tokenizer
from models.global_pointer_nezha import GlobalPointerNeZha
from models.configuration_nezha import NeZhaConfig

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="ner")
parser.add_argument("--test_data_path", default=os.path.join(BASE_DIR, "data/contest_data/preliminary_test_a"),
                    type=str, help="test data file path")
parser.add_argument("--test_file_name", default="sample_per_line_preliminary_A",
                    type=str, help="predict file name")
parser.add_argument("--test_data_file_path", default="", type=str)
parser.add_argument("--best_model_name", default=os.path.join(BASE_DIR, "data/best_model"), type=str)
parser.add_argument("--result_save_path", default=os.path.join(BASE_DIR, "submission"),
                    type=str, help="result file path")
parser.add_argument("--result_name", default="result.txt", type=str)

parser.add_argument("--cuda_device", default=4, type=int, help="the number of cuda to use")
parser.add_argument("--seed", default=42, type=int)

args = parser.parse_args()

set_seed(args.seed)

class GlobalPointerNERPredictor(object):
    """
    GlobalPointer命名实体识别的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (:obj:`dict`): 标签映射
    """  # noqa: ignore flake8"

    def __init__(
            self,
            module,
            tokernizer,
            cat2id
    ):
        self.module = module
        self.module.task = 'TokenLevel'

        self.cat2id = cat2id
        self.tokenizer = tokernizer
        self.device = list(self.module.parameters())[0].device

        self.id2cat = {}
        for cat_, idx_ in self.cat2id.items():
            self.id2cat[idx_] = cat_

    def _convert_to_transfomer_ids(
            self,
            text
    ):

        tokens = self.tokenizer.tokenize(text)
        token_mapping = self.tokenizer.get_token_mapping(text, tokens)

        input_ids = self.tokenizer.sequence_to_ids(tokens)
        input_ids, input_mask, segment_ids = input_ids

        zero = [0 for i in range(self.tokenizer.max_seq_len)]
        span_mask = [input_mask for i in range(sum(input_mask))]
        span_mask.extend([zero for i in range(sum(input_mask), self.tokenizer.max_seq_len)])
        span_mask = np.array(span_mask)

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'span_mask': span_mask
        }

        return features, token_mapping

    def _get_input_ids(
            self,
            text
    ):
        if self.tokenizer.tokenizer_type == 'vanilla':
            return self._convert_to_vanilla_ids(text)
        elif self.tokenizer.tokenizer_type == 'transfomer':
            return self._convert_to_transfomer_ids(text)
        elif self.tokenizer.tokenizer_type == 'customized':
            return self._convert_to_customized_ids(text)
        else:
            raise ValueError("The tokenizer type does not exist")

    def _get_module_one_sample_inputs(
            self,
            features
    ):
        return {col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device) for col in features}

    def predict_one_sample(
            self,
            text='',
            threshold=0
    ):
        """
        单样本预测

        Args:
            text (:obj:`string`): 输入文本
            threshold (:obj:`float`, optional, defaults to 0): 预测的阈值
        """  # noqa: ignore flake8"

        features, token_mapping = self._get_input_ids(text)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            scores = self.module(**inputs)[0].cpu()

        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf

        entities = []

        for category, start, end in zip(*np.where(scores > threshold)):
            if end - 1 > token_mapping[-1][-1]:
                break
            if token_mapping[start - 1][0] <= token_mapping[end - 1][-1]:
                entitie_ = {
                    "start_idx": token_mapping[start - 1][0],
                    "end_idx": token_mapping[end - 1][-1],
                    "entity": text[token_mapping[start - 1][0]: token_mapping[end - 1][-1] + 1],
                    "type": self.id2cat[category]
                }

                if entitie_['entity'] == '':
                    continue

                entities.append(entitie_)

        return entities


def predict(model, tokenizer, cat2id):
    ner_predictor_instance = GlobalPointerNERPredictor(model, tokenizer, cat2id)

    predict_results = []

    if args.test_data_file_path and os.path.isfile(args.test_data_file_path):
        test_data_file_path = args.test_data_file_path
    else:
        test_data_file_path = os.path.join(args.test_data_path, args.test_file_name)

    with open(test_data_file_path, 'rt', encoding='utf-8') as f:
        lines = f.readlines()
        for _line in tqdm(lines):
            label = len(_line) * ['O']
            for _preditc in ner_predictor_instance.predict_one_sample(_line[:-1]):
                if 'I' in label[_preditc['start_idx']]:
                    continue
                if 'B' in label[_preditc['start_idx']] and 'O' not in label[_preditc['end_idx']]:
                    continue
                if 'O' in label[_preditc['start_idx']] and 'B' in label[_preditc['end_idx']]:
                    continue

                label[_preditc['start_idx']] = 'B-' + _preditc['type']
                label[_preditc['start_idx'] + 1: _preditc['end_idx'] + 1] = (_preditc['end_idx'] - _preditc[
                    'start_idx']) * [('I-' + _preditc['type'])]

            predict_results.append([_line, label])
    # with open("pre_train_unlabeled_data_100000.txt", "wt", encoding="utf-8") as f:
    #     f.write(json.dumps(predict_results, ensure_ascii=False))
    return predict_results


def save_result(predict_results, path_dir="./"):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    with open(os.path.join(path_dir, args.result_name), 'wt', encoding='utf-8') as f:
        for _result in predict_results:
            for word, tag in zip(_result[0], _result[1]):
                if word == '\n':
                    continue
                f.write(f'{word} {tag}\n')
            f.write('\n')


def get_label_dict():
    cat2id_list = sorted([str(i) for i in range(1, 55) if i != 27 and i != 45] + ["O"])
    return {key:idx for idx, key in enumerate(cat2id_list)}


if __name__ == "__main__":

    tokenizer = Tokenizer(vocab=args.best_model_name, max_seq_len=128)
    cat2id = get_label_dict()
    config = NeZhaConfig.from_pretrained(args.best_model_name, num_labels=len(cat2id))
    model = GlobalPointerNeZha.from_pretrained(args.best_model_name,
                                               config=config)
    device = torch.device("cuda:{}".format(args.cuda_device) if torch.cuda.is_available() else "cpu")
    model.to(device)
    predict_results = predict(model, tokenizer, cat2id)
    save_result(predict_results, args.result_save_path)



