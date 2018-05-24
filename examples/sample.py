import os
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

import sys
sys.path.insert(0, '/home/jiaruizou/research/Arabic/pytorch-seq2seq')

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path', default='/home/jiaruizou/research/Arabic/pytorch-seq2seq/data/train.tsv', help='Path to train.txt data')
parser.add_argument('--dev_path', action='store', dest='dev_path', default='/home/jiaruizou/research/Arabic/pytorch-seq2seq/data/eval.tsv', help='Path to dev data')

parser.add_argument('--epoch', type=int, default=30, help='number of epochs')

parser.add_argument('--rnn_cell', default='gru', help='gru or lstm')
# parser.add_argument('--rnn_cell', default='lstm', help='gru or lstm')
parser.add_argument('--bi', default=True, help='true or false')
parser.add_argument('--dropout', default=0.2, help='')
parser.add_argument('--hidden_size', type=int, default=256, help='')
parser.add_argument('--batch_size', type=int, default=32, help='')
parser.add_argument('--teacher_forcing_ratio', default=0.8, help='')



parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='/home/jiaruizou/research/Arabic/pytorch-seq2seq/checkpoint', help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', default='', help='name of checkpoint to load')

parser.add_argument('--resume', action='store_true', dest='resume', default=False, help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level', default='info', help='Logging level.')

parser.add_argument('--save_checkpoint', default=True, help='Whether to store the model.')
parser.add_argument('--save_checkpoint_path', default='/home/jiaruizou/research/Arabic/pytorch-seq2seq/checkpoint/', help='Path to store the model.')

opt = parser.parse_args()


LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

src = SourceField()
tgt = TargetField()
max_len = 50


def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len


train = torchtext.data.TabularDataset(
    path=opt.train_path, format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)
dev = torchtext.data.TabularDataset(
    path=opt.dev_path, format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)
src.build_vocab(train, max_size=50000)
tgt.build_vocab(train, max_size=50000)
input_vocab = src.vocab
output_vocab = tgt.vocab

# NOTE: If the source field name and the target field name
# are different from 'src' and 'tgt' respectively, they have
# to be set explicitly before any training or inference
# seq2seq.src_field_name = 'src'
# seq2seq.tgt_field_name = 'tgt'

# Prepare loss
weight = torch.ones(len(tgt.vocab))
pad = tgt.vocab.stoi[tgt.pad_token]
loss = Perplexity(weight, pad)
if torch.cuda.is_available():
    loss.cuda()

seq2seq = None
optimizer = None

# Initialize model
hidden_size=opt.hidden_size
bidirectional = opt.bi
encoder = EncoderRNN(len(src.vocab), max_len, hidden_size,
                     bidirectional=bidirectional, variable_lengths=True)
decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                     dropout_p=0.2, use_attention=True, bidirectional=bidirectional, rnn_cell=opt.rnn_cell,
                     eos_id=tgt.eos_id, sos_id=tgt.sos_id)
seq2seq = Seq2seq(encoder, decoder)

if torch.cuda.is_available():
    seq2seq.cuda()

if len(opt.load_checkpoint) > 0:
    path = os.path.join(opt.expt_dir, opt.load_checkpoint)
    logging.info("loading checkpoint from {}".format(path))
    seq2seq.load_state_dict(torch.load(path))

else:
    # Prepare dataset

    if not opt.resume:


        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        # optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        # scheduler = StepLR(optimizer.optimizer, 1)
        # optimizer.set_scheduler(scheduler)

# train.txt
t = SupervisedTrainer(loss=loss, batch_size=opt.batch_size,
                      print_every=10, expt_dir=opt.expt_dir)

seq2seq = t.train(seq2seq, train,
                  num_epochs=opt.epoch, dev_data=dev,
                  optimizer=optimizer,
                  teacher_forcing_ratio=opt.teacher_forcing_ratio,
                  resume=opt.resume)

if opt.save_checkpoint:
    if len(opt.load_checkpoint) > 0:
        torch.save(seq2seq.state_dict(), opt.save_checkpoint_path + str(opt.epoch + int(opt.load_checkpoint)) + opt.rnn_cell + str(opt.hidden_size)+ str(opt.teacher_forcing_ratio) + str(opt.batch_size) + str(opt.dropout))
    else:
        torch.save(seq2seq.state_dict(), opt.save_checkpoint_path + str(opt.epoch) + opt.rnn_cell + str(opt.hidden_size) + str(opt.teacher_forcing_ratio) + str(opt.batch_size) + str(opt.dropout))

# predictor = Predictor(seq2seq, input_vocab, output_vocab)

# while True:
#     seq_str = raw_input("6 32 20 22")
#     seq = seq_str.strip().split()
#     print(predictor.predict(seq))
