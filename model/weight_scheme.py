

def get_weight_scheme(weight_scheme: str='usif', **kwargs):
    if weight_scheme.lower() == 'sif':
        return SIF()
    elif weight_scheme.lower() == 'usif':
        return USIF()
    else:
        return BOW()


class SIF:
    def __init__(self, count_fn='enwiki_vocab_min200.txt', a=1e-3):
        self.weight = {}
        with open(count_fn) as f:
            lines = f.readlines()
        N = 0
        for line in lines:
            line = line.strip()
            if len(line) > 0:
                if len(line.split()) == 2:
                    k, v = line.split()
                    k = k.lower()
                    v = float(v)
                    self.weight[k] = v
                    N += v
                else:
                    print(line)
        for key, value in self.weight.items():
            self.weight[key] = a / (a + value / N)

    def __getitem__(self, item):
        return self.weight.get(item.lower(), 1)


class USIF:
    """Map words to their probabilities."""
    def __init__(self, count_fn='enwiki_vocab_min200.txt', n=11):
        """Initialize a word2prob object.

        Args:
            count_fn: word count file name (one word per line)
        """
        self.prob = {}
        total = 0.0

        for line in open(count_fn, encoding='utf8'):
            k, v = line.split()
            v = int(v)
            k = k.lower()

            self.prob[k] = v
            total += v

        self.prob = {k: (self.prob[k] / total) for k in self.prob}
        self.min_prob = min(self.prob.values())
        self.count = total

        vocab_size = float(len(self.prob))
        threshold = 1 - (1 - 1 / vocab_size) ** n
        alpha = len([w for w in self.prob if self.prob[w] > threshold]) / vocab_size
        Z = 0.5 * vocab_size
        self.a = (1 - alpha) / (alpha * Z)

    def __getitem__(self, w):
        return self.a / (0.5 * self.a + self.prob.get(w, self.min_prob))


class BOW:
    def __init__(self):
        pass

    def __getitem__(self, item):
        return 1
