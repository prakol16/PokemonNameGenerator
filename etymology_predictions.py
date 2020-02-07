import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import  StepLR

from load_train_data import all_types, load_from_file, get_species_array

BATCH_SIZE = 32


class ReverseEmbeddings(nn.Module):
    def __init__(self, word_vectors):
        super().__init__()
        # shape: (n_words, embed_size)
        self.wv = nn.Parameter(torch.Tensor(word_vectors.vectors), requires_grad=False)
        self.wv /= torch.norm(self.wv, dim=1)[:,None]

    def forward(self, wvs):
        # wvs shape: (batch_size, embed_size)
        return wvs.mm(self.wv.T)


class EtymologyNet(nn.Module):
    def __init__(self, word_vecs, hidden_size):
        super().__init__()

        self.embed_size = word_vecs.vectors.shape[1]
        self.reverse_embeddings = ReverseEmbeddings(word_vecs)
        self.num_types = len(all_types)
        self.model = nn.Sequential(
            nn.Linear(in_features=2*self.embed_size+self.num_types, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=self.embed_size),
            self.reverse_embeddings
        )

    def forward(self, species):
        return self.model(species)


def train():
    from load_train_data import load_from_file, load_pokemon, \
        save_to_file, get_etymology_indices, get_species_array, get_type_array
    import os
    import gensim.downloader as api
    from operator import attrgetter

    load_pre = True
    model_save_file = "etymology_gen.pt"
    optim_save_file = "etymology_gen_optim.pt"

    print("Loading word vectors...")
    word_vectors = api.load("glove-wiki-gigaword-50")
    print("Loading training data...")

    if not os.path.exists("train.json"):
        pokemon = list(load_pokemon())
        save_to_file("train.json", pokemon)
    else:
        pokemon = load_from_file("train.json")

    indices, words = get_etymology_indices(list(map(attrgetter("etymology"), pokemon)), word_vectors)
    species = get_species_array(list(map(attrgetter("species"), pokemon)), word_vectors)
    types = get_type_array(list(map(attrgetter("types"), pokemon)))

    NUM_EPOCHS = 2001
    model = EtymologyNet(word_vectors, 100)
    opt = optim.SGD(model.parameters(), lr=0.01)

    if load_pre:
        print("Loading saved model...")
        model.load_state_dict(torch.load(model_save_file))
        # opt.load_state_dict(torch.load(optim_save_file))

    scheduler = StepLR(opt, step_size=500, gamma=0.3)

    print("Training...")
    for epoch in range(250, NUM_EPOCHS):
        average_loss = do_epoch(model, opt, indices, words, species, types)
        print("Epoch", epoch, "Loss", average_loss)
        if epoch % 10 == 9:
            print("Saving model...")
            torch.save(model.state_dict(), model_save_file)
            print("Saving optimizer...")
            torch.save(opt.state_dict(), optim_save_file)
        if epoch % 500 == 499:
            print("Saving model to out folder...")
            torch.save(model.state_dict(), "etymology_out/etymology_gen_{}_epochs.pt".format(epoch + 3101))
            torch.save(opt.state_dict(), "etymology_out/etymology_gen_{}_epochs_optim.pt".format(epoch + 3101))
        scheduler.step()


def do_epoch(model, opt, indices, words, species, types):
    cum_loss = 0
    num_batches = 1 + len(indices) // BATCH_SIZE
    for i in range(num_batches):
        opt.zero_grad()

        start = i * BATCH_SIZE
        end = min(len(indices), start + BATCH_SIZE)
        index = indices[start:end]
        true_ets = words[start:end]
        inputs = torch.cat((species[index], types[index]), dim=1)
        outputs = model(inputs)

        loss = nn.functional.cross_entropy(outputs, true_ets, reduction='mean')
        loss.backward()
        opt.step()

        cum_loss += loss.item() * (end - start)

    return cum_loss / len(indices)


if __name__ == "__main__":
    train()



