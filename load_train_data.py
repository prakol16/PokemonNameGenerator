from collections import namedtuple
from operator import attrgetter
from lxml import html
import requests
import re
import json
import random

import torch
import numpy as np
from gensim.models import KeyedVectors

PokemonName = namedtuple('Pokemon', ['name', 'number', 'species', 'etymology', 'types'])
etymology_page_url = "https://pokemondb.net/etymology"
species_page_url = "https://bulbapedia.bulbagarden.net/wiki/Pok%C3%A9mon_category"
types_page_url = "https://pokemondb.net/pokedex/all"

# Clean up some typos/prefixes
word_replacements = {
    "-saur": "dinosaur",
    "char": "charcoal",
    "aero-": "air",
    "mudskipper": "mud skipper",
    "cu": "copper",
    "thieve": "thief",
    "anacondo": "anaconda",
    "rat-a-tat-tat": "ratatat",
    "palindrome": "",  # the page was explaining that the name is a palindrome
    "armadilo": "armadillo",
    "x-ray": "x+ray",
    "mime artist": "mime",
    "lop-eared": "lop+eared",
    "rock-n-roll": "rock+n+roll"
}

char2ind = {" ": 27, "'": 28, ".": 29, "-": 30, ":": 31, "2": 32, "<end>": 33}
for i in range(26):
    char2ind[chr(ord('a') + i)] = i+1
char2ind["é"] = char2ind["e"]

ind2char_ = [None] * char2ind['<end>']
for c, i in char2ind.items():
    if c != "é":
        ind2char_[i - 1] = c
ind2char = np.array(ind2char_)

all_types = ["normal", "fire", "water", "grass", "electric", "ice", "fighting", "poison", "ground", "flying", "psychic", "bug", "rock", "ghost", "dark", "dragon", "steel", "fairy"]
type2ind = {tp: i for i, tp in enumerate(all_types)}


def load_etymology():
    page = requests.get(etymology_page_url)
    tree = html.fromstring(page.content)
    rows = tree.xpath("//table/tbody/tr")

    pk_name = None
    pk_num = -1
    pk_et_so_far = []
    for row in rows:
        if row.find_class("cell-name"):
            if pk_name is not None:
                yield pk_name.lower(), int(pk_num), pk_et_so_far
            pk_num = row.find_class("cell-num")[0].text_content().replace("\n", "").strip()
            pk_name = row.find_class("cell-name")[0].text_content().replace("\n", "").strip()
            pk_et_so_far = [row.find_class("cell-etym-word")[0].text_content().lower()]
        else:
            pk_et_so_far.append(row.find_class("cell-etym-word")[0].text_content())
    yield pk_name.lower(), int(pk_num), pk_et_so_far


def load_species():
    page = requests.get(species_page_url)
    tree = html.fromstring(page.content)
    rows = tree.find_class("sortable")[0].findall("tr")[1:]

    prev_num = 0
    for row in rows:
        tds = row.findall("td")
        num = int(tds[0].text_content().strip())
        if num == prev_num:
            continue
        prev_num = num
        pk_name = tds[2].find("a").text_content().strip()
        species = tds[3].text_content().strip().lower().split(" ")[-3:-1]  # remove the word "pokemon"
        yield pk_name.lower(), num, species


def load_types():
    page = requests.get(types_page_url)
    tree = html.fromstring(page.content)
    table = tree.get_element_by_id("pokedex")
    rows = table.find("tbody").findall("tr")

    prev_num = 0
    for row in rows:
        tds = row.findall("td")
        n = int(tds[0].text_content().strip())
        if n == prev_num:
            continue
        prev_num = n
        name = tds[1].find("a").text_content().lower()
        a = tds[2].findall("a")
        yield name, n, [l.text_content().lower() for l in a]


def load_pokemon():
    for et, sp, tp in zip(load_etymology(), load_species(), load_types()):
        name, num, etymology = et
        name2, num2, species = sp
        name3, num3, tps = tp
        assert name == name2 == name3, "{}, {}, {} should be equal".format(name, name2, name3)
        assert num == num2, "{}, {}, {} should be equal".format(num, num2, num3)
        yield PokemonName(name.replace("♀", "").replace("♂", ""), num, species, clean_etymologies(etymology), tps)


def clean_etymologies(etymology):
    def clean_etymology(et):
        et = et.lower()
        if et in word_replacements:
            et = word_replacements[et]
        et = re.sub(r'[^a-z\s\-\+]', '', et)
        return list(map(lambda s: re.sub(r'\+', '-', s), re.split(r'[\s\-]+', et)))

    def remove_duplicates(et):
        seen = set()
        for e in et:
            if e not in seen:
                seen.add(e)
                yield e

    return list(remove_duplicates(e for et in etymology for e in clean_etymology(et) if e))



def save_to_file(filename, pokemon):
    data = list(map(lambda pk: dict(pk._asdict()), pokemon))
    with open(filename, 'w') as f:
        f.write(json.dumps(data))


def load_from_file(filename):
    with open(filename, 'r') as f:
        data = json.loads(f.read())
    return list(map(lambda pk: PokemonName(**pk), data))


def pad_names(names):
    true_lens = list(map(len, names))
    max_len_name = max(true_lens)
    ret = torch.zeros((len(names), max_len_name+1)).long()
    for i, name in enumerate(names):
        for j, c in enumerate(name):
            ret[i][j] = char2ind[c]
        ret[i][len(name)] = char2ind['<end>']
    return ret.T, 1+torch.LongTensor(true_lens)


def get_species_array(species, word_vectors):
    embed_size = word_vectors.vectors.shape[1]
    ret = torch.zeros((len(species), 2 * embed_size))
    for i, sp in enumerate(species):
        ret[i][embed_size:] = torch.tensor(word_vectors.word_vec(sp[-1]) if sp[-1] in word_vectors.vocab else 0)
        if len(sp) > 1:
            ret[i][:embed_size] = torch.tensor(word_vectors.word_vec(sp[-2]) if sp[-2] in word_vectors.vocab else 0)
    return ret


def get_etymology_indices(etymologies, word_vecs):
    ret = []
    for i, ets in enumerate(etymologies):
        for et in ets:
            if et in word_vecs.vocab:
                ret.append((i, word_vecs.vocab[et].index))
    random.shuffle(ret)
    indices, words = zip(*ret)
    return torch.LongTensor(indices), torch.LongTensor(words)


def get_etymologies_char_arrays(etymologies):
    etymologies_chars = [" ".join(et) for et in etymologies]
    return pad_names(etymologies_chars)


def get_etymology_char_array_len(pokemon):
    join_char_len = 1
    return sum(map(len, pokemon.etymology)) + len(pokemon.etymology) * join_char_len - 1


def get_name_char_arrays(names):
    return pad_names(names)


def get_type_array(types):
    ret = torch.zeros(len(types), len(all_types))
    for i, tp in enumerate(types):
        for t in tp:
            ret[i][type2ind[t]] = 1
    return ret


def create_train_data(pokemon, word_vectors):
    pokemon_names, name_lens = pad_names(list(map(attrgetter('name'), pokemon)))
    species = get_species_array(list(map(attrgetter('species'), pokemon)), word_vectors)
    etymologies_char = get_etymologies_char_arrays(list(map(attrgetter('etymology'), pokemon)))
    type_arr = get_type_array(list(map(attrgetter('types'), pokemon)))
    print(type_arr[-1], pokemon[-1])
    return pokemon_names, name_lens, species, etymologies_char


if __name__ == "__main__":
    # word_vectors = api.load("glove-wiki-gigaword-50")
    word_vectors = KeyedVectors(50)
    pokemon = load_from_file('train.json')
    names, _, __, et_char = create_train_data(pokemon, word_vectors)
    print(et_char)
