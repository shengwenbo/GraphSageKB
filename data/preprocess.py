
import os
from collections import Counter
import pickle as pkl

def stat(dir_path):
    train_path = os.path.join(dir_path, "train.txt")
    valid_path = os.path.join(dir_path, "valid.txt")
    test_path = os.path.join(dir_path, "test.txt")
    text_path = os.path.join(dir_path, "text_cvsc.txt")

    entities_path = os.path.join(dir_path, "entities.txt")
    relations_path = os.path.join(dir_path, "relations.txt")
    dictionary_path = os.path.join(dir_path, "dictionary.txt")
    dependence_pat = os.path.join(dir_path, "dependence.txt")

    edges = []
    [edges.extend(es) for es in [load_kb_edges(p) for p in [train_path, valid_path, test_path]]]

    entities = []
    relations = []

    for edge in edges:
        e1, r, e2 = edge
        entities.append(e1)
        entities.append(e2)
        relations.append(r)
    entities = Counter(entities).items()
    entities = sorted(entities, key=lambda x: x[1], reverse=True)
    relations = Counter(relations).items()
    relations = sorted(relations, key=lambda x: x[1], reverse=True)

    sentences = []
    dependence = []
    dictionary = []

    with open(text_path, "r", encoding="utf-8") as fin:
        for line in fin.readlines():
            e1, sent, e2, freq = line.strip().split("\t")
            sent = sent.split(":")
            if len(sent) < 3:
                continue
            sentences.append(sent)
            for token in sent:
                if token.startswith("<"):
                    # dep
                    dependence.append(token.replace("-", ""))
                elif token.startswith("["):
                    # [XXX] [YYY]
                    pass
                else:
                    # word
                    dictionary.append(token)
    dependence = Counter(dependence).items()
    dependence = sorted(dependence, key=lambda x: x[1], reverse=True)
    dictionary = Counter(dictionary).items()
    dictionary = sorted(dictionary, key=lambda x: x[1], reverse=True)

    dump_word_list(entities, entities_path)
    dump_word_list(relations, relations_path)
    dump_word_list(dependence, dependence_pat)
    dump_word_list(dictionary, dictionary_path)


def load_kb_edges(path):
    edges = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin.readlines():
            e1, r, e2 = line.strip().split("\t")
            edges.append((e1, r, e2))
    return edges


def dump_word_list(word_list, path):
    with open(path, "wb", encoding="utf-8") as fout:
        pkl.dump(word_list, fout)


if __name__ == "__main__":
    stat("./FB15K-237.2")

