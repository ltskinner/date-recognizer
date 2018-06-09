


"""
Scalable Requirements

1) Take data from dbase instead of csv (so can work mobile)
2) word2vec on RAM elements read from db
3) REMOVE LONGS BEFORE POOLING TT 

1) Really make sure labeling alg is on point
2) More data, 1 GB by end of day? If anyone can.....


def get_rand_batch():

    # select random strings pre or post query?
    data = getStrings() #either from db or local dat, modular

    inputs = []
    labels = []
    seqlens = []
    for text in data:
        vector = []
        label = np.zeros(wide)
        doc = nlp(text)
        for c, word in enumerate(doc):
            vector.append(word.vector)
            if findDateToken(word.text): # mod alg to find stand alones, need stateful solution to see if +- 1 index is also dateful
                label[c] = 1

        while len(vector) < wide:
            vector.append(np.zeros(vec_len))

        inputs.append(vector)
        labels.append(label)
        seqlens.append(len(doc))

    # data_X of shape (batch, wide, vec_len)
    # data_Y of shape (batch, wide)
    # seqlen_X of shape (batch)
    return data_X, data_Y, seqlen_X, orig_strings







"""