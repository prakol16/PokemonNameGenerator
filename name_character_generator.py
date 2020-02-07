from typing import List

import torch
from torch import nn, optim
from operator import attrgetter
import random
from load_train_data import get_etymologies_char_arrays, char2ind, \
    get_etymology_char_array_len, get_name_char_arrays, load_from_file

# Source for much of this file: CS 224N neural machine translation code

BATCH_SIZE = 32

class NameGenerator(nn.Module):
    def __init__(self, hidden_size, p_dropout=0.2):
        super().__init__()

        self.num_chars = char2ind['<end>']
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(input_size=self.num_chars, hidden_size=hidden_size, bidirectional=True)
        self.decoder = nn.LSTMCell(input_size=self.num_chars+hidden_size, hidden_size=hidden_size)

        # Create one-hot embeddings
        self.char_embeddings = nn.Embedding.from_pretrained(
            torch.cat([torch.zeros(self.num_chars)[None,:], torch.eye(self.num_chars)]), padding_idx=0
        )

        self.decoder_hidden_init = nn.Linear(in_features=2*hidden_size, out_features=hidden_size)
        self.decoder_cell_init = nn.Linear(in_features=2*hidden_size, out_features=hidden_size)

        # Given ith hidden state of encoder and current decoder hidden state, compute attention score
        # by projecting encoder state size (2*hidden size) to hidden_size
        self.att_projection = nn.Linear(in_features=2*hidden_size, out_features=hidden_size, bias=False)
        # Given attention score (2*hidden_size) and decoder state (hidden_size), output some output vector
        self.combined_output_projection = nn.Linear(in_features=3*hidden_size, out_features=hidden_size, bias=False)
        # Transform tanh(combined_output) to character probabilities
        self.character_prediction = nn.Linear(in_features=hidden_size, out_features=self.num_chars, bias=False)
        # Dropout for regularization
        self.dropout = nn.Dropout(p=p_dropout)

    @staticmethod
    def create_mask(lens, max_len):
        mask = torch.zeros((len(lens), max_len))
        for i, l in enumerate(lens):
            mask[i,l:] = 1
        return mask.bool()

    def forward(self, etymologies, true_names, etymology_lens):
        # etymologies: (max_et_len, batch_size)
        # true_names: (max_name_len, batch_size)
        enc_hiddens, dec_init_state = self.encode(etymologies, etymology_lens)
        mask = NameGenerator.create_mask(etymology_lens, etymologies.size(0))
        combined_outputs = self.decode(enc_hiddens, mask, dec_init_state, true_names)
        # (max_name_len, batch_size, num_chars) -> (max_name_len * batch_size, num_chars)
        predictions = self.character_prediction(combined_outputs).view(-1, self.num_chars)
        return predictions

    def encode(self, etymologies, etymology_lens):
        # (max_et_len, batch_size, num_chars)
        embeddings = self.char_embeddings(etymologies)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, etymology_lens)
        # enc_hiddens: (batch_size, max_et_len, 2*hidden_size)
        # hidden_enc_layers, cell_enc_layers: (2, batch_size, hidden_size)
        output, (last_hidden_enc, last_cell_enc) = self.encoder(packed_embeddings)

        enc_hiddens, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        init_decoder_hidden = self.decoder_hidden_init(last_hidden_enc.permute(1, 0, 2).reshape(-1, 2 * self.hidden_size))
        init_decoder_cell = self.decoder_cell_init(last_cell_enc.permute(1, 0, 2).reshape(-1, 2 * self.hidden_size))

        dec_init_state = (init_decoder_hidden, init_decoder_cell)

        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens, etymology_mask, dec_init_state, true_names):
        # (max_name_len, batch_size)

        dec_state = dec_init_state

        batch_size = enc_hiddens.size(0)
        true_names_adjusted = torch.cat((torch.zeros(batch_size).long()[None,:], true_names[:-1, :]), dim=0)
        output_prev = torch.zeros(batch_size, self.hidden_size)

        combined_outputs = []

        # (batch_size, max_et_len, hidden_size)
        # this is what will be dotted with the decoder hidden state
        # to compute the attention scores
        enc_hiddens_proj = self.att_projection(enc_hiddens)
        # (max_name_len, batch_size, num_chars)
        true_names_chars = self.char_embeddings(true_names_adjusted)

        for true_names_char in true_names_chars.split(1):
            # (batch_size, num_chars)
            n_char_sq = true_names_char.squeeze(dim=0)
            # (batch_size, num_chars+hidden_size)
            decoder_input = torch.cat((n_char_sq, output_prev), dim=1)
            output_prev, dec_state = self.step(decoder_input, dec_state, enc_hiddens_proj, enc_hiddens, etymology_mask)

            combined_outputs.append(output_prev)

        # (max_name_len, batch_size, num_chars)
        return torch.stack(combined_outputs)

    def step(self, decoder_input, dec_state, enc_hiddens_proj, enc_hiddens, etymology_mask):
        dec_state = self.decoder(decoder_input, dec_state)
        hidden_dec, cell_dec = dec_state
        # (batch_size, max_et_len)
        attention_logits = enc_hiddens_proj.bmm(hidden_dec[:, :, None]).squeeze(dim=2)

        if etymology_mask is not None:
            attention_logits.masked_fill_(etymology_mask, float('-inf'))

        attention = nn.functional.softmax(attention_logits, dim=1)
        # (batch_size, 2*hidden_layer)
        attention_weighted_sum = attention[:, None, :].bmm(enc_hiddens).squeeze(dim=1)
        # (batch_size, 3*hidden_layer)
        final_layer_input = torch.cat((attention_weighted_sum, hidden_dec), dim=1)
        # (batch_size, hidden_layer)
        final_layer_output = self.dropout(torch.tanh(self.combined_output_projection(final_layer_input)))
        return final_layer_output, dec_state

    def do_search(self, etymology_arr, top_k=10, max_len=50):
        ets, et_lens = get_etymologies_char_arrays([etymology_arr])
        enc_hiddens_, dec_init_state = self.encode(ets, et_lens)
        # (1, et_len, 2*hidden_size) -> (top_k, et_len, 2*hidden_size)
        enc_hiddens = enc_hiddens_.expand(top_k, -1, -1)
        enc_hiddens_proj_ = self.att_projection(enc_hiddens_)
        enc_hiddens_proj = enc_hiddens_proj_.expand(top_k, -1, -1)
        # output_prev = torch.zeros((top_k, self.hidden_size))
        # n_char_seq = torch.zeros((top_k, self.num_chars))
        # dec_state = dec_init_state

        # stopped = torch.zeros(top_k).bool()
        # cum_log_prob = torch.zeros(top_k)
        # name_lens = torch.zeros(top_k)

        results = torch.zeros(top_k, max_len).long()

        stop_token = char2ind['<end>'] - 1

        # Run a single step to initialize
        output_prev_start, dec_state_start = self.step(torch.zeros(1, self.num_chars + self.hidden_size),
                                                       dec_init_state,
                                                       enc_hiddens_proj_, enc_hiddens_, None)
        predictions_start = nn.functional.log_softmax(self.character_prediction(output_prev_start), dim=1)
        topk_start = predictions_start.squeeze(dim=0).topk(top_k)
        n_char_seq = self.char_embeddings(topk_start.indices + 1)

        cum_log_prob = topk_start.values[:]
        name_lens = torch.ones(top_k).long()
        output_prev = output_prev_start.repeat(top_k, 1)
        dec_state = dec_state_start[0].repeat(top_k, 1), dec_state_start[1].repeat(top_k, 1)

        stopped = topk_start.indices == stop_token
        results[:, 0] = topk_start.indices

        for i in range(1, max_len):
            #             print(results[:, :i])
            decoder_input = torch.cat((n_char_seq, output_prev), dim=1)
            output_prev, dec_state = self.step(decoder_input, dec_state, enc_hiddens_proj, enc_hiddens, None)
            # (top_k, num_chars)
            predictions = nn.functional.log_softmax(self.character_prediction(output_prev), dim=1)
            #             print("Predictions", predictions)
            predictions[stopped, :stop_token] = float('-inf')
            predictions[stopped, stop_token] = 0
            pot_log_prob = cum_log_prob[:, None] + predictions
            name_lens[~stopped] += 1
            av_pot_log_prob = pot_log_prob / name_lens[:, None]

            topk_log_prob = av_pot_log_prob.flatten().topk(top_k)
            xs = topk_log_prob.indices % self.num_chars
            ys = topk_log_prob.indices // self.num_chars
            n_char_seq = self.char_embeddings(xs + 1)

            cum_log_prob[:] = pot_log_prob[ys, xs]
            output_prev[:, :] = output_prev[ys, :]
            name_lens[:] = name_lens[ys]
            dec_state = (dec_state[0][ys], dec_state[1][ys])

            stopped = xs == stop_token
            results[:, :] = results[ys, :]
            results[:, i] = xs
            if stopped.all():
                break
        max_name_len = name_lens.max().item()
        return results[:, :max_name_len]


def get_batched_name_gen_data(pokemon):
    for i in range(len(pokemon) // BATCH_SIZE):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(pokemon))
        pokés = pokemon[start:end]
        pokés.sort(key=get_etymology_char_array_len, reverse=True)
        batch = (
            get_etymologies_char_arrays(list(map(attrgetter('etymology'), pokés))),
            get_name_char_arrays(list(map(attrgetter('name'), pokés)))
        )
        yield batch


def train():
    pokemon = list(load_from_file("train.json"))
    random.seed(0)
    random.shuffle(pokemon)

    model = NameGenerator(20)
    model.train()
    opt = optim.Adam(model.parameters(), lr=0.001)

    load_pre = True

    if load_pre:
        model.load_state_dict(torch.load("name_gen.pt"))
        opt.load_state_dict(torch.load("name_gen_opt.pt"))

    scheduler = optim.lr_scheduler.StepLR(opt, step_size=60)
    batches = list(get_batched_name_gen_data(pokemon))

    NUM_EPOCHS = 100
    for epoch in range(NUM_EPOCHS):
        loss = do_epoch(model, opt, batches)
        print("Loss", loss)
        if epoch % 10 == 9:
            print("Saving...")
            torch.save(model.state_dict(),  "name_gen.pt")
            torch.save(opt.state_dict(), "name_gen_opt.pt")
            torch.save(scheduler.state_dict(), "name_gen_scheduler.pt")
        scheduler.step()


def do_epoch(model, opt: torch.optim.Adam, batches):
    cum_loss = 0

    for ets, names in batches:
        opt.zero_grad()

        ets_arr, ets_lens = ets
        names_arr, true_name_lens = names
        predictions = model(ets_arr, names_arr, ets_lens)
        loss = nn.functional.cross_entropy(predictions, names_arr.flatten()-1, ignore_index=-1)
        loss.backward()
        opt.step()

        cum_loss += loss.item()

    return cum_loss / len(batches)


if __name__ == "__main__":
    train()

