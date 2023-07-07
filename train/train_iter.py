import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.autograd import Variable
from train import train
from mmpg.util.evaluate import evaluate_all, eval_belu
import sys
use_cuda = torch.cuda.is_available()


def train_iters(args, encoder, decoder, key_encoder, train_data, dev_data, test_data, test_data_text, n_iters, src_lang, tgt_lang, key_lang):
    parameters = [p for p in encoder.parameters()] + [p for p in decoder.parameters()] + [p for p in key_encoder.parameters()]
    optimizer = optim.Adam(parameters)

    best_F = -1.
    for iter in range(1, n_iters + 1):
        total_loss = 0.
        if args.debug:
            train_data = train_data[:2]
        pbar = tqdm(train_data)
        pbar.set_description("[Train Epoch {}]".format(iter))
        for batch in pbar:
            src, tgt, key = batch
            src = np.array([np.array(s, dtype=np.long) for s in src], dtype=np.long)
            tgt = np.array([np.array(s, dtype=np.long) for s in tgt], dtype='int64')
            key = np.array([np.array(s, dtype=np.long) for s in key], dtype=np.long)
            input_variable = Variable(torch.from_numpy(src))
            target_variable = Variable(torch.from_numpy(tgt))
            key_variable = Variable(torch.from_numpy(key))
            if use_cuda:
                input_variable = input_variable.cuda()
                target_variable = target_variable.cuda()
                key_variable = key_variable.cuda()
            loss = train(args, input_variable, target_variable, key_variable, encoder, decoder, key_encoder, optimizer, tgt_lang)
            total_loss += loss
            pbar.set_postfix(loss=loss.item())
        print('epoch %d, total loss %.2f' % (iter, total_loss.item()))
        precision, recall, F = evaluate_all(args, dev_data, encoder, decoder, key_encoder, src_lang=src_lang, tgt_lang=tgt_lang, key_lang=key_lang,
                                            from_text=False)
        print('dev_data Precision %.3f, recall %.3f, F %.3f' % (precision * 100, recall * 100, F * 100))
        sys.stdout.flush()

        if best_F < F:
            best_F = F
            torch.save(encoder.state_dict(), 'models/encoder.pt', pickle_protocol=3)
            torch.save(decoder.state_dict(), 'models/decoder.pt', pickle_protocol=3)
            torch.save(key_encoder.state_dict(), 'models/key_encoder.pt', pickle_protocol=3)
    encoder.load_state_dict(torch.load('models/encoder.pt'))
    decoder.load_state_dict(torch.load('models/decoder.pt'))
    key_encoder.load_state_dict(torch.load('models/key_encoder.pt'))

    precision, recall, F = evaluate_all(args, dev_data, encoder, decoder, key_encoder, src_lang=src_lang, tgt_lang=tgt_lang, key_lang=key_lang,
                                        from_text=False)
    print('Precision %.3f, recall %.3f, F %.3f' % (precision * 100, recall * 100, F * 100))
    test_precision, test_recall, test_F = evaluate_all(args, test_data, encoder, decoder, key_encoder, src_lang=src_lang,
                                                       tgt_lang=tgt_lang, key_lang=key_lang, from_text=False)
    text_precision, text_recall, text_F = evaluate_all(args, test_data_text, encoder, decoder, key_encoder, src_lang=src_lang,
                                                       tgt_lang=tgt_lang, key_lang=key_lang, from_text=True)
    print('test precision %.3f, recall %.3f, F value %.3f' % (test_precision * 100, test_recall * 100, test_F * 100))
    print('text precision %.3f, recall %.3f, F value %.3f' % (text_precision * 100, text_recall * 100, text_F * 100))
    test_belu_1 = eval_belu(log_path='./log/big_')
    text_belu_1 = eval_belu(log_path='./log/small_')
    print('test belu_1 %.3f, text belu_1 %.3f' % (test_belu_1 * 100, text_belu_1 * 100))

    sys.stdout.flush()
