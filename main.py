import argparse
import logging
import math
import os
from datetime import datetime

import torch
import transformers

from common.logging import get_logger
from dataset import get_data_module
from model import get_model
from trainer import finetune_ZO, finetune_FO


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for Model Fine-tuning')

    # common
    parser.add_argument('--model_name', type=str, default='distilbert-base-cased',
                        help='Name of the pre-trained model')
    parser.add_argument('--task', type=str, default='mnli', help='Task for model training')
    parser.add_argument('--algorithm', type=str, default='ZO', help='Algorithm to use',
                        choices=["FOSGD", "FOAdam", "ZO", "ZOAdam", "ZOSVRG", "LOZO", "HiZOO", "LOREN"])
    parser.add_argument('--device', type=int, default=0, help='GPU Number')
    parser.add_argument('--results', type=str, default='results_demo',
                        help='Name of folder to store results')
    # training
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size for training')
    parser.add_argument('--batchsize_limit', type=int, default=64,
                        help='Max batch size to be used to avoid memory error')
    parser.add_argument('--full_parameter', action='store_true',
                        help='True for full parameter fine-tuning')
    # dataset
    parser.add_argument('--samplesize', type=int, default=512, help='Training data sample size')
    parser.add_argument('--samplesize_validation', type=int, default=256,
                        help='Validation data sample size')
    parser.add_argument('--max_seq_length', type=int, default=256,
                        help='Max sequence length for inputs')
    # optimization
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight_decay')
    parser.add_argument('--perturbation_scale', type=float, default=1e-3,
                        help='Perturbation scale for SPSA estimators')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience; '
                        'if the validation accuracy does not improve for this number of evaluations, '
                        'stop training')
    parser.add_argument('--num_queries', type=int, default=10,
                        help='Number of queries to evaluate in large batch update')
    # experiment
    parser.add_argument('--soft_prompt', action='store_true', help='True for using soft prompt')
    parser.add_argument('--save_ckpt', action='store_true', help='Save checkpoint')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every n epochs')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Evaluate on validation set every n epochs')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 precision')
    parser.add_argument('--bf16', action='store_true', help='Use BF16 precision')
    parser.add_argument('--logging', type=str, default="csv", choices=['tensorboard', 'csv'],
                        help="Choose logging method; either wandb or tensorboard or none")
    parser.add_argument('--trial', type=int, default=0, help='Trial number')
    parser.add_argument('--init_seed', type=int, default=None, help='Random seed for model initialization')
    parser.add_argument('--clipping', action='store_true', help='set to True to apply gradient clipping')
    # SVRG
    parser.add_argument('--q', type=int, default=2, help='q parameter used only for ZO-SVRG')
    parser.add_argument('--lr_mezosvrg_mb', type=float, default=1e-6,
                        help='Mini-batch learning rate for MeZO-SVRG')
    parser.add_argument('--anneal', type=float, default=1.5, help='Annealing parameter')
    # Adam
    parser.add_argument('--eps', type=float, default=1e-8, help='the level of adaptivity (eps for adam)')
    parser.add_argument('--low_bit_adam', type=int, default=0,
                        help='Use Adam with quantized states; options: 4 or 8')
    # LOZO
    parser.add_argument('--rank', type=int, default=2, help='rank of U and V')
    parser.add_argument('--step_interval', type=int, default=50, help='frequency of updating matrix V')
    # HiZOO
    parser.add_argument('--hessian_smooth_type', type=str, default="constant1e-8",
                        help="hessian smooth type")
    # LOREN
    parser.add_argument('--damping', type=float, default=1.0, help='damping')
    parser.add_argument('--lr_cov', type=float, default=1e-3, help='Learning rate for covariance')

    args = parser.parse_args()
    return args


def setup_logger():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    logfile = os.path.join(args.output_dir, f'log_{timestamp}.txt')
    logger = get_logger(logfile, level="DEBUG")
    get_logger(level='INFO')

    """
    The default of training_args.log_level is passive, so we set log level at info here
    to have that default.
    """
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    return logger


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    args = parse_arguments()
    args.hf_token = os.getenv('HF_TOKEN')

    trimmed_model_name = args.model_name.split('/')[-1]
    num_steps = math.ceil(args.samplesize / args.batchsize) * args.epochs
    args.run_name = f'{args.algorithm}_lr{args.lr:.0e}_nq-{args.num_queries}'

    if args.algorithm == 'ZOSVRG':
        args.run_name += f'_lr_mb_{args.lr_mezosvrg_mb:.0e}_q_{args.q:.0e}_anneal_{args.anneal:.0e}'
    elif args.algorithm == 'ZOAdam':
        args.run_name += f'_eps_{args.eps:.0e}'
        if args.low_bit_adam == 4 or args.low_bit_adam == 8:
            args.run_name += f'_{args.low_bit_adam}bit'
    elif args.algorithm == 'LOZO':
        args.run_name += f'_rank_{args.rank}_step-interval_{args.step_interval}'
    elif args.algorithm == 'HiZOO':
        args.run_name += f'_hessian_smooth-{args.hessian_smooth_type}'
    elif args.algorithm == 'LOREN':
        args.run_name += f'_damp-{args.damping:.0e}_covlr{args.lr_cov:.0e}'

    args.run_name += f'_bsz{args.batchsize}_steps{num_steps}'
    args.output_dir = os.path.join(args.results, trimmed_model_name, args.task, args.run_name)

    if args.logging == "tensorboard":
        from lightning.pytorch.loggers import TensorBoardLogger

        writer = TensorBoardLogger(args.output_dir)
    elif args.logging == 'csv':
        from lightning.pytorch.loggers import CSVLogger

        writer = CSVLogger(args.output_dir,
                           flush_logs_every_n_steps=10)
    os.makedirs(args.output_dir, exist_ok=True)

    # logging setup
    logger = setup_logger()
    logger.debug(f"Training Arguments {args}")

    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    dm = get_data_module(args)
    transformer = get_model(args, dm)

    if 'FO' in args.algorithm:
        finetune_FO(args, device, dm, transformer, writer)
    else:
        finetune_ZO(args, device, dm, transformer, writer)
