import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional as F
import numpy as np


use_cuda = torch.cuda.is_available()


class Encoder(nn.Module):
    def __init__(self, voc_size, emb_size, hidden_size, n_layers=1, bidirectional=False):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(voc_size, emb_size, padding_idx=0)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers=self.n_layers, batch_first=False,
                          bidirectional=self.bidirectional)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
    def load_emb(self, emb):
        emb = torch.from_numpy(np.array(emb, dtype=np.float32))
        self.embedding.weight.data.copy_(emb)

    def forward(self, input):
        batch_size = input.size()[0]
        init_state = self.initHidden(batch_size)
        output, state = self.encode(input, init_state)
        return output, state

    def initHidden(self, batch_size):
        bid = 2 if self.bidirectional else 1
        result = Variable(torch.zeros(self.n_layers * bid, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        return result

    def encode(self, input, hidden):
        mask = torch.gt(input.data, 0)
        input_length = torch.sum((mask.long()), dim=1)
        lengths, indices = torch.sort(input_length, dim=0,
                                      descending=True)
        _, ind = torch.sort(indices, dim=0)
        input_length = torch.unbind(lengths, dim=0)

        embedded = self.embedding(torch.index_select(input, dim=0, index=Variable(indices)))
        output, hidden = self.gru(pack(embedded, input_length, batch_first=True), hidden)

        output = torch.index_select(unpack(output, batch_first=True)[0], dim=0, index=Variable(ind)) * Variable(
            torch.unsqueeze(mask.float(), -1))
        hidden = torch.index_select(self.fc(torch.cat((hidden[-1, :, :], hidden[-2, :, :]), dim=1)), dim=0, index=Variable(ind))
        direction = 2 if self.bidirectional else 1
        assert hidden.size() == (input.size()[0], self.hidden_size) and output.size() == (
            input.size()[0], input.size()[1], self.hidden_size * direction)
        return output, hidden


class Key_encoder(nn.Module):
    def __init__(self, voc_size, emb_size, hidden_size, n_layers=1, bidirectional=False):
        super(Key_encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(voc_size, emb_size, padding_idx=0)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers=self.n_layers, batch_first=False,
                          bidirectional=self.bidirectional)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def load_emb(self, emb):
        emb = torch.from_numpy(np.array(emb, dtype=np.float32))
        self.embedding.weight.data.copy_(emb)

    def initHidden(self, batch_size):
        bid = 2 if self.bidirectional else 1
        result = Variable(torch.zeros(self.n_layers * bid, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        return result

    def forward(self, input):
        batch_size = input.size()[0]
        init_state = self.initHidden(batch_size)
        output, state = self.encode(input, init_state)
        return output, state

    def encode(self, input, hidden):
        mask = torch.gt(input.data, 0)
        input_length = torch.sum((mask.long()), dim=1)
        lengths, indices = torch.sort(input_length, dim=0,
                                      descending=True)
        _, ind = torch.sort(indices, dim=0)
        input_length = torch.unbind(lengths, dim=0)

        embedded = self.embedding(torch.index_select(input, dim=0, index=Variable(indices)))
        output, hidden = self.gru(pack(embedded, input_length, batch_first=True), hidden)

        output = torch.index_select(unpack(output, batch_first=True)[0], dim=0, index=Variable(ind)) * Variable(
            torch.unsqueeze(mask.float(), -1))
        hidden = torch.index_select(self.fc(torch.cat((hidden[-1, :, :], hidden[1, :, :]), dim=1)), dim=0,
                                    index=Variable(ind))
        direction = 2 if self.bidirectional else 1
        assert hidden.size() == (input.size()[0], self.hidden_size) and output.size() == (
            input.size()[0], input.size()[1], self.hidden_size * direction)
        return output, hidden


class Bah_Attn(nn.Module):
    def __init__(self, query_size, hidden_size, direction):
        super(Bah_Attn, self).__init__()
        self.lin1 = nn.Linear(query_size, hidden_size)  # query
        self.lin2 = nn.Linear(hidden_size * direction, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, query, key):
        query = self.lin1(query)
        Linear_key = self.lin2(key)
        score = self.v(F.tanh(torch.unsqueeze(query, 1) + Linear_key))
        score = F.softmax(torch.squeeze(score, -1))
        context = torch.sum(torch.unsqueeze(score, -1) * key, 1)
        return context, score


class Cover(nn.Module):
    def __init__(self, input_size, hidden_size, direction):
        super(Cover, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size * direction, hidden_size)
        self.lin3 = nn.Linear(1, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, query, key, cover):
        h_x = self.lin1(query)
        h_m = self.lin2(key)
        h_c = self.lin3(torch.unsqueeze(cover, -1))
        score = self.v(F.tanh(torch.unsqueeze(h_x, 1) + h_m + h_c))
        score = F.softmax(torch.squeeze(score, -1))
        context = torch.sum(torch.unsqueeze(score, -1) * key, 1)
        return context, score


class Decoder(nn.Module):
    def __init__(self, args, voc_size, emb_size, hidden_size, bidirectional=False):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.voc_size = voc_size
        self.args = args
        self.bidirectional = bidirectional
        self.direction = 2 if self.bidirectional else 1

        self.embedding = nn.Embedding(self.voc_size, self.emb_size, padding_idx=0)


        if args.key_encoder:
            self.gru_cell = nn.GRUCell(self.hidden_size * (self.direction + 1) + self.emb_size, self.hidden_size)
        else:
            self.gru_cell = nn.GRUCell(self.hidden_size * self.direction + self.emb_size, self.hidden_size)

        self.attn = Bah_Attn(hidden_size, hidden_size, self.direction)
        self.attn_cover = Cover(hidden_size, hidden_size, self.direction)
        self.out = nn.Linear(self.hidden_size, self.voc_size)

    def forward(self, word, hidden, encoder_outputs, encoder_attention, encoder_key_hidden):
        word_vec = self.embedding(word)

        if self.args.attention_cover:
            context, score = self.attn_cover(hidden, encoder_outputs,
                                             encoder_attention)
        else:
            context, score = self.attn(hidden, encoder_outputs)

        if self.args.key_encoder:
            hidden = self.gru_cell(torch.cat([context, encoder_key_hidden, word_vec], dim=1), hidden)
        else:
            hidden = self.gru_cell(torch.cat([context, word_vec], 1), hidden)

        pred_word = self.out(hidden)
        return pred_word, hidden, score

    def load_emb(self, emb):
        emb = torch.from_numpy(np.array(emb, dtype=np.float32))
        self.embedding.weight.data.copy_(emb)

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        return result

