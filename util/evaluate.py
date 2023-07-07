import torch
from tqdm import tqdm
from torch.autograd import Variable
import os
import codecs
from nltk.translate.bleu_score import sentence_bleu


use_cuda = torch.cuda.is_available()
MAX_LENGTH = 20


def eval_belu(log_path):
    ground_true = []
    generation = []
    all_belu = 0
    counter = 0

    gf_file = log_path + '/ground_true.txt'
    gen_file = log_path + '/generation.txt'

    with open(gf_file, 'r', encoding='utf-8') as gf, open(gen_file, 'r', encoding='utf-8') as gen:
      ground = gf.readlines()
      genera = gen.readlines()
      for gf, gen in zip(ground, genera):
          gf = gf.strip()
          gf = gf[0: -4]
          gen = gen.strip()
          gen = gen[0: -4]

          ground_true.append(gf.split(' '))
          generation = (gen.split())

          belu_1 = sentence_bleu(ground_true, generation, weights=(1, 0, 0, 0))
          all_belu += belu_1
          counter += 1
    return all_belu / counter


# 评价指标F
def eval_F(ground_true, generation, log_path):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    gf_file = log_path + '/ground_true.txt'
    gen_file = log_path + '/generation.txt'
    with codecs.open(gf_file, 'w', 'utf-8') as f:
        for s in ground_true:
            f.write(" ".join(s) + '\n')
    with codecs.open(gen_file, 'w', 'utf-8') as f:
        for s in generation:
            f.write(" ".join(s).strip() + '\n')

    total_right = 0.
    total_gf = 0.
    total_gen = 0.
    for r, c in zip(ground_true, generation):
        r_set = set(r) - {'EOS'}
        c_set = set(c) - {'EOS'}
        right = set.intersection(r_set, c_set)
        total_right += len(right)
        total_gf += len(r_set)
        total_gen += len(c_set)
    total_gen = total_gen if total_gen != 0 else 1
    precision = total_right / float(total_gen)
    recall = total_right / float(total_gf)
    if precision == 0 or recall == 0:
        F = 0.
    else:
        F = precision * recall * 2. / (precision + recall)
    return precision, recall, F


def variable_from_sentence(lang, sentence, from_text=False):

    if from_text:
        result = []
        for c in sentence:
            if c in lang.word2id:
                result.append(lang.word2id[c])
            else:
                result.append(lang.word2id['OOV'])
        sentence = result
        sentence.append(lang.EOS_token)
    result = Variable(torch.LongTensor(sentence))
    if use_cuda:
        return result.cuda()
    return result


def evaluate(args, encoder, decoder, key_encoder, sentence, key, src_lang, tgt_lang, key_lang, from_text=False, max_length=MAX_LENGTH):
    input_variable = variable_from_sentence(src_lang, sentence, from_text)
    input_variable = torch.unsqueeze(input_variable, 0)
    input_length = input_variable.size()[1]

    key_variable = variable_from_sentence(key_lang, key, from_text)
    key_variable = torch.unsqueeze(key_variable, 0)

    encoder_output, encoder_state = encoder(input_variable)
    encoder_key_output, encoder_key_hidden = key_encoder(key_variable)

    decoder_input = Variable(torch.LongTensor([tgt_lang.SOS_token]))
    decoder_input = decoder_input.cuda()
    decoder_hidden = torch.squeeze(encoder_state, 0) if encoder_state.dim() == 3 else encoder_state

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, input_length)

    attention_outputs = torch.zeros((1, input_length))
    if use_cuda:
        attention_outputs = attention_outputs.cuda()
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output,
                                                                    Variable(attention_outputs), encoder_key_hidden)
        attention_outputs += decoder_attention.data
        decoder_attentions[di] = decoder_attention.data
        _, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == tgt_lang.EOS_token:
            decoded_words.append('EOS')
            break
        else:
            if ni != tgt_lang.word2id['OOV']:
                decoded_words.append(tgt_lang.id2word[ni])
        decoder_input = Variable(torch.LongTensor([ni]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    return decoded_words, decoder_attentions[:di + 1]


def evaluate_all(args, data, encoder, decoder, key_encoder, src_lang, tgt_lang, key_lang, from_text):
    generation = []
    ground_true = []
    for s, t, k in tqdm(data, disable=not args.verbose):
        result, att = evaluate(args, encoder, decoder, key_encoder, s, k, src_lang=src_lang, tgt_lang=tgt_lang, key_lang = key_lang,
                               from_text=from_text)
        generation.append(result)
        ground_true.append([tgt_lang.id2word[word] for word in t])
    if not from_text:
        precision, recall, F = eval_F(ground_true, generation, log_path='./big_seq2seq' )
    else:
        precision, recall, F = eval_F(ground_true, generation, log_path='./small_seq2seq')
    return precision, recall, F
