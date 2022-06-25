import argparse

from trainer import Trainer
from data_loader import DataReader
from torch.utils.data import Dataset


def main(args):

    train_dataset = DataReader(args.train_data_dir)
    if args.do_eval:
        val_dataset = DataReader(args.eval_data_dir)
    else:
        val_dataset = None
    trainer = Trainer(args, train_dataset, val_dataset)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data_dir", default="data/pretrained_unlabel_data.json", type=str, help="The input data dir")
    parser.add_argument("--eval_data_dir", default="data/pretrained_unlabel_data.json", type=str, help="The input data dir")
    parser.add_argument("--model_dir", default="output_model", type=str, help="Path to save, load model")
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int)
    parser.add_argument("--num_train_epochs", default=3, type=int)
    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--logging_steps', type=int, default=1000, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--model_name_or_path", default="", type=str, help="")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument('--n_gpu', type=list, nargs='+', default=[5])
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html", )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--do_eval", action="store_true")

    args = parser.parse_args()
    main(args)