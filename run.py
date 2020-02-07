from load_train_data import ind2char, load_from_file, load_pokemon, \
    get_etymologies_char_arrays
import gensim.downloader as api
from name_character_generator import NameGenerator
from etymology_predictions import EtymologyNet
from gensim.models import KeyedVectors
import torch

if __name__ == "__main__":
    name_gen_model = NameGenerator(20)
    name_gen_model.load_state_dict(torch.load("name_gen.pt"))

    print(ind2char[name_gen_model.do_search(["cat", "rat", "bat"])])