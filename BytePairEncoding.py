import json

class BPE():

    def __init__(self,vocab_size=None,corpus=None):
        self.vocab_size = vocab_size
        self.corpus = corpus
        self.merges = {}
        self.vocab = {}

    def get_stats(self,ids):
        count = {}

        for pair in zip(ids, ids[1:]):
            count[pair] = count.get(pair, 0)+1

        return count
    
    def merge(self,ids, pair, idx):
        i = 0
        newids = []

        while(i < len(ids)):
            if(i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]):
                newids.append(idx)
                i+=2
            else:
                newids.append(ids[i])
                i+=1
                
        return newids
    

    def tokensEncode(self,text):
        with open('English_Merges.json', 'r') as f:
            data = json.load(f)

        self.merges = {eval(key):val for key,val in data.items()}
        tokens = list(text.encode('utf-8'))

        while(len(tokens) > 2):
            stats = self.get_stats(tokens)
            pair = max(stats, key=stats.get)

            if pair not in self.merges:
                break

            tokens = self.merge(tokens, pair, self.merges[pair])

        return tokens


    def tokensDecode(self, tokens):

        with open('index_to_vocab.json','r') as f:
            data = json.load(f)

        self.vocab = {eval(key):eval(val) for key,val in data.items()}

        tokens = b"".join(self.vocab[idx] for idx in tokens)

        return tokens.decode('utf-8',errors='replace')


    def train(self):
        num_merges = self.vocab_size-256
        ids = list(self.corpus.encode('utf-8'))

        for i in range(num_merges):

            stats = self.get_stats(ids)

            pair = max(stats, key=stats.get)

            idx = 256+i
            ids = self.merge(ids,pair,idx)

            self.merges[pair] = idx

        print("Merging Completed successfully!!!")

        self.vocab = {idx:bytes([idx]) for idx in range(256)}

        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

        print(f"Vocab and merges Completed with length :- {len(self.vocab)} {len(self.merges)} and returned !!!")
        return self.vocab,self.merges

