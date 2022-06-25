import torch
from torch.utils.data import DataLoader
from callback.optimizer import AdamW
import transformers
from transformers import BertConfig, get_linear_schedule_with_warmup, AdamW, BertTokenizer
from transformers import BertForPreTraining
from models.modeling_nezha import NeZhaForMaskedLM
from loguru import logger


import os
import logging
from tqdm import tqdm, trange


# logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset, val_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = NeZhaForMaskedLM.from_pretrained(args.model_name_or_path)
        self.device = torch.device("cuda:{}".format(args.cuda_device) if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, len(args.n_gpu))
        self.model.to(self.device)

        self.log_file = "log_loss.txt"

    def gather_indexes(self, positions, masked_lm_weights):
        flat_offsets = torch.reshape(
            torch.arange(0, positions.shape[0], dtype=torch.int32) * 128, [-1, 1]).to(self.device)
        flat_positions = torch.reshape((positions + flat_offsets) * masked_lm_weights, [-1])
        flat_positions = flat_positions[torch.nonzero(flat_positions).squeeze(-1)]
        return flat_positions.to(self.device)

    def evaluate(self):
        assert self.val_dataset is not None
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.args.train_batch_size, shuffle=False)
        self.model.eval()
        val_epoch_iterator = tqdm(val_dataloader, desc="valid_Iteration")
        total_loss = 0.0
        with torch.no_grad():
            for step, batch in enumerate(val_epoch_iterator):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]}
                masked_positions = batch[3]
                labels = batch[4]
                outputs = self.model(**inputs)
                hidden_states= outputs[0]

                hidden_states = torch.reshape(hidden_states, [-1, hidden_states.shape[-1]])
                masked_lm_weights = batch[5].long()
                index = self.gather_indexes(masked_positions, masked_lm_weights)
                prediction_scores = hidden_states[index]

                loss_fct = torch.nn.CrossEntropyLoss()
                labels = labels.reshape([-1])
                labels = labels[torch.nonzero(labels).squeeze(-1)]
                masked_lm_loss = loss_fct(prediction_scores.view(-1, prediction_scores.shape[-1]), labels.view(-1))
                total_loss += masked_lm_loss
        return total_loss / len(val_epoch_iterator)


    def train(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True)
        
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        bert_param_optimizer = list(self.model.bert.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay, 'lr': self.args.learning_rate},
            {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            'lr': self.args.learning_rate},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                num_training_steps=t_total)
        if self.args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.args.fp16_opt_level)
        # multi-gpu training (should be after apex fp16 initialization)
        if len(self.args.n_gpu) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.n_gpu)
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = {}".format(len(self.train_dataset)))
        logger.info("  Num Epochs = {}".format(self.args.num_train_epochs))
        logger.info("  Total train batch size = {}".format(self.args.train_batch_size))
        logger.info("  Gradient Accumulation steps = {}".format(self.args.gradient_accumulation_steps))
        logger.info("  Total optimization steps = {}".format(t_total))
        logger.info("  Logging steps = {}".format(self.args.logging_steps))
        logger.info("  Save steps = {}".format(self.args.save_steps))

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        best_loss = 100.0
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(self.device) for t in batch)
                self.model.train()
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2]}
                masked_positions = batch[3]
                labels = batch[4]
                # next_sentence_label = batch[5]

                outputs = self.model(**inputs)
                hidden_states= outputs[0]
                # print(prediction_scores.shape, seq_relationship_score.shape)

                total_loss = None
                # if labels is not None and next_sentence_label is not None:
                if labels is not None:
                    hidden_states = torch.reshape(hidden_states, [-1, hidden_states.shape[-1]])
                    masked_lm_weights = batch[5].long()
                    index = self.gather_indexes(masked_positions, masked_lm_weights)
                    prediction_scores = hidden_states[index]

                    loss_fct = torch.nn.CrossEntropyLoss()
                    labels = labels.reshape([-1])
                    labels = labels[torch.nonzero(labels).squeeze(-1)]
                    masked_lm_loss = loss_fct(prediction_scores.view(-1, prediction_scores.shape[-1]), labels.view(-1))
                    # next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
                    # total_loss = masked_lm_loss + next_sentence_loss
                    total_loss = masked_lm_loss


                    if self.args.gradient_accumulation_steps > 1:
                        total_loss = total_loss / self.args.gradient_accumulation_steps

                    total_loss.backward()

                    tr_loss += total_loss.item()
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        # print(total_loss.item())
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        self.model.zero_grad()
                        global_step += 1

                        if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                            self.save_model(scheduler, str(global_step))
                        if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                            if self.val_dataset is not None:
                                total_loss = self.evaluate()
                                if total_loss < best_loss and total_loss < 0.6:
                                    self.save_model(scheduler, "best_model", global_step=global_step)
                                    best_loss = total_loss
                            with open(self.log_file, "at", encoding="utf-8") as f:
                                f.write(" global_steps: {}, total_loss: {}\n".format(global_step, total_loss))
                            
                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break
        self.save_model(scheduler, "final")
        return global_step, tr_loss / global_step

    def save_model(self, scheduler, sub_dir, **kwargs):
        global_step = kwargs.pop("global_step", None)
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        save_dir = os.path.join(self.args.model_dir, sub_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(save_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(save_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", save_dir)
 
        # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", save_dir)

        if global_step is not None:
            with open(os.path.join(save_dir, "best_global_step.txt"), "at", encoding="utf-8") as f:
                f.write(str(global_step) + "\n")