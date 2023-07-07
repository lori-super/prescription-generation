from mmpg.models.model import Encoder, Decoder, Key_encoder
import pickle
import torch
import argparse
from torch.autograd import Variable

device = "cuda" if torch.cuda.is_available() else 'cpu'
use_cuda = torch.cuda.is_available()
batch_size = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_size', default=128, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--bidirectional', default=True, action='store_false')
    return parser.parse_args()


def remove_space(item):
    return item.split(' ')


if __name__ == '__main__':
    args = parse_args()
    data, test_data_text, med_lang, cure_lang, key_lang = pickle.load(
        open('./data.pkl', 'rb'))
    encoder = Encoder(voc_size=len(cure_lang.word2id), emb_size=args.emb_size, hidden_size=args.hidden_size,
                          bidirectional=args.bidirectional)
    decoder = Decoder(args=args, voc_size=len(med_lang.word2id), hidden_size=args.hidden_size, emb_size=args.emb_size,
                          bidirectional=args.bidirectional)
    key_encoder = Key_encoder(voc_size=len(key_lang.word2id), emb_size=args.emb_size, hidden_size=args.hidden_size,
                              bidirectional=args.bidirectional)

    encoder.load_state_dict(torch.load('encoder.pt'), strict=False)
    decoder.load_state_dict(torch.load('decoder.pt'), strict=False)
    key_encoder.load_state_dict(torch.load('key_encoder.pt'), strict=False)

    if use_cuda:
        encoder.cuda()
        decoder.cuda()
        key_encoder.cuda()
    symptom = ''
    prescription = ''
    key = ''
    tokens = [tok for tok in symptom]

    tokens_idx = [cure_lang.word2id[char] for char in tokens] + [cure_lang.EOS_token]
    tokens_idx = torch.tensor(tokens_idx)

    key_idx = [key_lang.word2id[key]] + [key_lang.EOS_token]
    key_idx = torch.tensor(key_idx)

    if use_cuda:
        tokens_idx = tokens_idx.unsqueeze(0).cuda()
        key_idx = key_idx.unsqueeze(0).cuda()

    voc_size = len(med_lang.word2id)
    input_length = len(tokens_idx)
    encoder_output, encoder_state = encoder(tokens_idx)
    encoder_key_output, encoder_key_hidden = key_encoder(key_idx)
    decoder_hidden = torch.squeeze(encoder_state, 0) if encoder_state.dim() == 3 else encoder_state

    decoder_input = Variable(torch.LongTensor([med_lang.SOS_token] * batch_size))
    attention_outputs = torch.zeros((batch_size, input_length))
    if use_cuda:
        decoder_input = decoder_input.cuda()
        attention_outputs = attention_outputs.cuda()

    result = []
    prescription = remove_space(prescription)
    for t in range(len(prescription)):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_output, Variable(attention_outputs), encoder_key_hidden)
        attention_outputs = attention_outputs + decoder_attention.data
        _, topi = decoder_output.data.topk(1)
        word = med_lang.id2word[topi]
        result.append(word)

        decoder_input = Variable(torch.squeeze(topi, dim=0))
        decoder_input = decoder_input.cuda()
    print(' '.join(result))