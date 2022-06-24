import warnings
warnings.filterwarnings("ignore")

import torch
from torchcontrib.optim import SWA
import torch.nn.functional as F
import pandas as pd
import os
import argparse
import numpy as np
import time
import json
from tqdm import tqdm

from ark_nlp.factory.utils.seed import set_seed
from ark_nlp.model.ner.global_pointer_bert import Dataset
from ark_nlp.model.ner.global_pointer_bert import Task
from ark_nlp.model.ner.global_pointer_bert import get_default_model_optimizer
from ark_nlp.model.ner.global_pointer_bert import Tokenizer
from ark_nlp.factory.utils.conlleval import get_entity_bio
from models.global_pointer_nezha import GlobalPointerNeZha
from models.configuration_nezha import NeZhaConfig
from transformers import get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, CONFIG_NAME


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="ner")
parser.add_argument("--train_data_path", default=os.path.join(BASE_DIR, "data/contest_data/train_data"),
                    type=str, help="train data file path")
parser.add_argument("--train_data_file_path", default="",
                    type=str, help="train data file path")
parser.add_argument("--model_name", default="data/pretrain_model/nezha-cn-base", type=str)
parser.add_argument("--model_save_dir", default="output_model", type=str)

parser.add_argument("--learning_rate", default=2e-5, type=float)
parser.add_argument("--num_epoches", default=6, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.", )
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--weight_decay", default=0.01, type=float,
                    help="Weight decay if we apply some.")

parser.add_argument("--cuda_device", default=4, type=int, help="the number of cuda to use")
parser.add_argument("--seed", default=42, type=int)

# adversarial training
parser.add_argument("--do_adv", action="store_true",
                    help="Whether to adversarial training.")
parser.add_argument('--adv_epsilon', default=0.5, type=float,
                    help="Epsilon for adversarial.")
parser.add_argument('--adv_name', default='word_embeddings', type=str,
                    help="name for adversarial layer.")

parser.add_argument("--type", default="train", type=str, help="train or predict")
parser.add_argument("--predict_model_path", type=str, default="output_model/model_nezha_fgm_epoch_6.pth")

# 伪标签训练
parser.add_argument("--do_fake_label", action="store_true", help="whether to use fake label training.")
parser.add_argument("--fake_train_data_path", type=str, default=os.path.join(BASE_DIR, "data/orther"))
parser.add_argument("--fake_train_data_name", type=str, default="fake_train_data_20000.txt")

# 使用swa
parser.add_argument("--swa", action="store_true")
parser.add_argument("--swa_start", default=10, type=int)
parser.add_argument("--swa_freq", default=5, type=int)
parser.add_argument("--swa_lr", default=0.01, type=float)

# 使用rdrop
parser.add_argument("--rdrop", action="store_true")
parser.add_argument("--rdrop_rate", default=0.5, type=float)

args = parser.parse_args()

set_seed(args.seed)

def load_train_data(train_data_path):
    datalist = []
    label_set = set()
    with open(train_data_path, 'rt', encoding='utf-8') as f:
        lines = f.readlines()
        lines.append('\n')
        
        text = []
        labels = []
        
        for line in tqdm(lines): 
            if line == '\n':                
                text = ''.join(text)
                entity_labels = []
                for _type, _start_idx, _end_idx in get_entity_bio(labels, id2label=None):
                    entity_labels.append({
                        'start_idx': _start_idx,
                        'end_idx': _end_idx,
                        'type': _type,
                        'entity': text[_start_idx: _end_idx+1]
                    })
                    
                if text == '':
                    continue
                
                datalist.append({
                    'text': text,
                    'label': entity_labels
                })
                
                text = []
                labels = []
                
            elif line == '  O\n':
                text.append(' ')
                labels.append('O')
            else:
                line = line.strip('\n').split()
                if len(line) == 1:
                    term = ' '
                    label = line[0]
                else:
                    term, label = line
                text.append(term)
                label_set.add(label.split('-')[-1])
                labels.append(label)
    return datalist, label_set

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
            if end-1 > token_mapping[-1][-1]:
                break
            if token_mapping[start-1][0] <= token_mapping[end-1][-1]:
                entitie_ = {
                    "start_idx": token_mapping[start-1][0],
                    "end_idx": token_mapping[end-1][-1],
                    "entity": text[token_mapping[start-1][0]: token_mapping[end-1][-1]+1],
                    "type": self.id2cat[category]
                }

                if entitie_['entity'] == '':
                    continue

                entities.append(entitie_)

        return entities

class FGM(object):
     
    def __init__(self, model, emb_name, epsilon=1.0):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}
 
    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)
 
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

def train():
    if args.train_data_file_path and os.path.isfile(args.train_data_file_path):
        train_data_file_path = args.train_data_file_path
    else:
        train_data_file_path = os.path.join(args.train_data_path, "train.txt")
    datalist, label_set = load_train_data(train_data_file_path)

    fake_datalist = []
    if args.do_fake_label:
        assert os.path.isfile(os.path.join(args.fake_train_data_path, args.fake_train_data_name)), "you must input the fake label data path"
        fake_datalist, _ = load_train_data(os.path.join(args.fake_train_data_path, args.fake_train_data_name))
    
    datalist.extend(fake_datalist)
    train_data_df = pd.DataFrame(datalist)
    train_data_df["label"] = train_data_df["label"].apply(lambda x: str(x))

    if len(fake_datalist) > 0:
        dev_data_df = pd.DataFrame(datalist[-(400+len(fake_datalist)):-len(fake_datalist)])
    else:
        dev_data_df = pd.DataFrame(datalist[-400:])
    dev_data_df['label'] = dev_data_df['label'].apply(lambda x: str(x))

    label_list = sorted(list(label_set))

    ner_train_dataset = Dataset(train_data_df, categories=label_list)
    ner_dev_dataset = Dataset(dev_data_df, categories=ner_train_dataset.categories)

    tokenizer = Tokenizer(vocab=args.model_name, max_seq_len=128)

    ner_train_dataset.convert_to_ids(tokenizer)
    ner_dev_dataset.convert_to_ids(tokenizer)

    config = NeZhaConfig.from_pretrained(args.model_name, 
                                                    num_labels=len(ner_train_dataset.cat2id))

    torch.cuda.empty_cache()

    dl_module = GlobalPointerNeZha.from_pretrained(args.model_name, 
                                                config=config)

    optimizer = get_default_model_optimizer(dl_module, weight_decay=args.weight_decay)
    if args.swa:
        optimizer = SWA(optimizer, swa_start=args.swa_start, swa_freq=args.swa_freq, swa_lr=args.swa_lr)

    t_total = (len(ner_train_dataset) / args.batch_size) // args.gradient_accumulation_steps * args.num_epoches
    warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                    num_warmup_steps=warmup_steps,
                                    num_training_steps=t_total)

    model = Task(module=dl_module, 
                 optimizer=optimizer, 
                 loss_function='gpce',
                 scheduler = scheduler, 
                 cuda_device=args.cuda_device)



    fgm = FGM(model.module, args.adv_name, args.adv_epsilon)
    model.fit(ner_train_dataset, 
            ner_dev_dataset,
            lr=args.learning_rate,
            epochs=args.num_epoches, 
            batch_size=args.batch_size,
            grad_clip=args.max_grad_norm,
            fgm=fgm if args.do_adv else None, 
            compute_kl_loss=compute_kl_loss if args.rdrop else None, 
            rdrop_rate=args.rdrop_rate)

    return model, tokenizer, ner_train_dataset.cat2id

def save_model(model, tokenizer):
    model_state_dict_dir = os.path.join(args.model_save_dir,
                                        time.strftime("%Y-%m-%d_%H.%M.%S", time.gmtime()))
    if not os.path.exists(model_state_dict_dir):
        os.makedirs(model_state_dict_dir)

    train_args = {
        "model_name": args.model_name,
        "num_epoches": args.num_epoches,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "warmup_proportion": args.warmup_proportion,
        "weight_decay": args.weight_decay,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    if args.do_adv:
        train_args.update({
            "do_adv": True,
            "adv_epsilon": args.adv_epsilon,
            "adv_name": args.adv_name
        })
    if args.swa:
        train_args.update({
            "swa": True,
            "swa_start": args.swa_start,
            "swa_freq": args.swa_freq,
            "swa_lr": args.swa_lr
        })
    if args.do_fake_label:
        train_args.update({
            "fake_train_data_path": args.fake_train_data_path
        })
    if args.rdrop:
        train_args.update({
            "rdrop_rate": args.rdrop_rate
        })
    with open(os.path.join(model_state_dict_dir, "train_config.json"), "wt", encoding="utf-8") as f:
        f.write(json.dumps(train_args, indent=4, ensure_ascii=False))
    model_to_save = model.module if hasattr(model, 'module') else model
    # 保存模型权重pytorch_model.bin
    torch.save(model_to_save.state_dict(), os.path.join(model_state_dict_dir, WEIGHTS_NAME))
    # 保存模型配置文件config.json
    model_to_save.config.to_json_file(os.path.join(model_state_dict_dir, CONFIG_NAME))
    # 保存vocab.txt
    tokenizer.vocab.save_vocabulary(model_state_dict_dir)
    return model_state_dict_dir


if __name__ == "__main__":

    model, tokenizer, cat2id = train()
    model_state_dict_dir = save_model(model, tokenizer)



    