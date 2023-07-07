import argparse
import torch
import pickle
import random
from mmpg.train.train_iter import train_iters
from mmpg.models.model import Encoder, Decoder, Key_encoder

use_cuda = torch.cuda.is_available()


def padding(sequence, length):
    sequence = sequence[:length]
    while len(sequence) < length:
        sequence.append(0)
    return sequence


def make_batches(data, batch_size):
    batch_num = len(data) // batch_size if len(data) % batch_size == 0 else len(data) // batch_size + 1
    batches = []
    for batch in range(batch_num):
        mini_batch = data[batch * batch_size:(batch + 1) * batch_size]
        en_max_len = max([len(p[0]) for p in mini_batch])
        de_max_len = max([len(p[1]) for p in mini_batch])
        key_max_len = max([len(p[2]) for p in mini_batch])
        en_mini_batch = [padding(p[0], en_max_len) for p in mini_batch]
        de_mini_batch = [padding(p[1], de_max_len) for p in mini_batch]
        key_mini_batch = [padding(p[2], key_max_len) for p in mini_batch]
        batches.append((en_mini_batch, de_mini_batch, key_mini_batch))

    return batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_size', default=128, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--teacher_forcing_ratio', default=0.8, type=float)
    parser.add_argument('--bidirectional', default=True, action='store_false')
    parser.add_argument( '--epoch_num', default=10, type=int)
    parser.add_argument('--attention_cover', default=True, action='store_true')
    parser.add_argument('--key_encoder', default=True, action='store_true')
    parser.add_argument('--mu', default=0.5, type=float)
    parser.add_argument('--is_mu', default=True, action='is_mu')
    parser.add_argument('--lamda', default=0.3, type=float)
    parser.add_argument('--is_lamda', default=True, action='is_lamda')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if use_cuda: torch.cuda.set_device(args.gpu)
    data, test_data_text, med_lang, cure_lang, key_lang= pickle.load(
        open('./data.pkl', 'rb'))
    encoder = Encoder(voc_size=len(cure_lang.word2id), emb_size=args.emb_size, hidden_size=args.hidden_size,
                      bidirectional=args.bidirectional)
    decoder = Decoder(args=args, voc_size=len(med_lang.word2id), hidden_size=args.hidden_size, emb_size=args.emb_size,
                      bidirectional=args.bidirectional)
    key_encoder = Key_encoder(voc_size=len(key_lang.word2id), emb_size=args.emb_size, hidden_size=args.hidden_size,
                      bidirectional=args.bidirectional)
    encoder.load_emb(cure_lang.emb)
    decoder.load_emb(med_lang.emb)
    key_encoder.load_emb(key_lang.emb)
    if use_cuda:
        encoder.cuda()
        decoder.cuda()
        key_encoder.cuda()
    random.seed(255)
    random.shuffle(data)
    train_data = data[:int(len(data) * 0.9)]
    dev_data = data[int(len(data) * 0.9):int(len(data) * 0.95)]
    test_data = data[int(len(data) * 0.95):]
    batches = make_batches(train_data, args.batch_size)
    train_iters(args, encoder, decoder, key_encoder, batches, dev_data, test_data, test_data_text, args.epoch_num, tgt_lang=med_lang,
                src_lang=cure_lang, key_lang=key_lang)
