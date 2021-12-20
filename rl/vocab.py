class Vocabulary:
    def __init__(self, token_list):
        self.token_list = [self.pad_token()] + token_list
        self.vocab = {}
        for idx, token in enumerate(self.token_list):
            self.vocab[token] = idx
    
    @property
    def size(self):
        return len(self.token_list)
    
    @property
    def pad_code(self):
        return self.vocab[self.pad_token()]
    
    @staticmethod
    def pad_token():
        return '#PAD'
    
    def get_code(self, in_token):
        if isinstance(in_token, list):
            return [self.vocab[token] for token in in_token]
        else:
            return self.vocab[in_token]
    
    def get_token(self, in_code):
        if isinstance(in_code, list):
            return [self.token_list[code] for code in in_code]
        else:
            return self.token_list[in_code]
