import pickle as pkl

class dict_use:
    '''
    Manages (sentence --> indices) and (indices --> sentence) conversion
    '''
    def __init__(self, dict_path):
        self.Dict1, self.Dict2 = pkl.load(open(dict_path, 'rb'))
        # Increase length to add UNK, BOS, EOS
        self.vocab_size = len(self.Dict1) + 3  
        self.UNK = self.vocab_size - 3  # Unknown
        self.BOS = self.vocab_size - 1  # Beginning of sentence
        self.EOS = self.vocab_size - 2  # End of sentence

    def sen2id(self, s):
        if s == []:
            return []

        Dict = self.Dict1
        # dict_size = len(Dict)

        s_new = []
        if not isinstance(s[0], list):
            for item in s:
                if item in Dict:
                    s_new.append(Dict[item])
                else:
                    s_new.append(self.UNK)  # Replaced dict_size in original code
            return s_new
        else:
            # Recursively call sen2id if the element type is 'list'
            return [self.sen2id(x) for x in s]

    def id2sen(self, s):
        if s == []:
            return []

        Dict = self.Dict2
        # dict_size = len(Dict)
        s_new = []
        if not isinstance(s[0], list):
            for item in s:
                if item in Dict:
                    s_new.append(Dict[item])
                elif item == self.UNK:  # Replaced dict_size in original code
                    s_new.append('UNK')
                else:
                    pass
            return s_new
        else:
            return [self.id2sen(x) for x in s]