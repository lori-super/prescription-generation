import torch
import random
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
use_cuda = torch.cuda.is_available()
logsoftmax = torch.nn.LogSoftmax()


def multi_hot(ids, nclass):
    result = Variable(torch.zeros((ids.size()[0], nclass)))
    padding_index = Variable(torch.zeros((ids.size()[0], 1)).long())
    if use_cuda:
        result = result.cuda()
        padding_index = padding_index.cuda()
    result.scatter_(1, ids, 1.)
    result.scatter_(1, padding_index, 0.)
    assert result.data[0][0] == 0.
    return result


def cross_entropy(prob, targets, weight):
    H = -logsoftmax(prob) * targets
    return torch.sum(H * weight)


def js_div(p_output, q_output, get_softmax=True):

    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output, dim=1)
        q_output = F.softmax(q_output, dim=1)
    log_mean_output = ((p_output + q_output)/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2


def train(args, inputs, targets, key, encoder, decoder, key_encoder, optimizer, tgt_lang):
    optimizer.zero_grad()

    input_length = inputs.size()[1]
    target_length = targets.size()[1]
    batch_size = inputs.size()[0]
    assert batch_size == targets.size()[0]
    loss = 0.

    encoder_output, encoder_state = encoder(inputs)
    encoder_key_output, encoder_key_hidden = key_encoder(key)

    decoder_input = Variable(torch.LongTensor([tgt_lang.SOS_token] * batch_size))
    if use_cuda: decoder_input = decoder_input.cuda()

    decoder_hidden = torch.squeeze(encoder_state, 0) if encoder_state.dim() == 3 else encoder_state
    use_teacher_forcing = True if random.random() < args.teacher_forcing_ratio else False

    attention_outputs = torch.zeros((batch_size, input_length))
    out_weight = Variable(torch.ones(len(tgt_lang.word2id)))
    out_weight[0] = 0.

    out_weight = torch.unsqueeze(out_weight, 0)
    if use_cuda:
        attention_outputs = attention_outputs.cuda()
        out_weight = out_weight.cuda()
    all_med = F.softmax(multi_hot(targets, len(tgt_lang.word2id)))
    if use_teacher_forcing:
        for time in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output,  Variable(attention_outputs), encoder_key_hidden)

            if args.is_lamda:
                cov_loss = js_div(attention_outputs, decoder_attention, get_softmax=True)
                loss = loss + cross_entropy(decoder_output, (
                        multi_hot(torch.unsqueeze(targets[:, time], 1), len(tgt_lang.word2id)) + all_med) / 2.,
                                      weight=out_weight)
                loss = loss + args.lamda * cov_loss
            else:
                loss = loss + cross_entropy(decoder_output, (
                        multi_hot(torch.unsqueeze(targets[:, time], 1), len(tgt_lang.word2id)) + all_med) / 2.,
                                            weight=out_weight)
            decoder_input = targets[:, time]
            if args.is_mu:
                attention_outputs = args.mu * attention_outputs + (1 - args.mu) * torch.exp(-decoder_attention.data)
            else:
                attention_outputs = attention_outputs + decoder_attention.data
    else:
        for time in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output,  Variable(attention_outputs), encoder_key_hidden)
            if args.is_mu:
                attention_outputs = args.mu * attention_outputs + (1 - args.mu) * torch.exp(-decoder_attention.data)
            else:
                attention_outputs = attention_outputs + decoder_attention.data
            _, topi = decoder_output.data.topk(1)
            if args.is_lamda:

                cov_loss = js_div(attention_outputs, decoder_attention, get_softmax=True)
                loss = loss + cross_entropy(decoder_output, (
                        multi_hot(torch.unsqueeze(targets[:, time], 1), len(tgt_lang.word2id)) + all_med) / 2.,
                                      weight=out_weight)
                loss = loss + args.lamda * cov_loss
            else:
                loss = loss + cross_entropy(decoder_output, (
                        multi_hot(torch.unsqueeze(targets[:, time], 1), len(tgt_lang.word2id)) + all_med) / 2.,
                                            weight=out_weight)

            decoder_input = Variable(torch.squeeze(topi))
            decoder_input = decoder_input.cuda()
    loss.backward()
    optimizer.step()

    return loss.data / target_length
