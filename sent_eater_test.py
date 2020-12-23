import numpy as np
import spacy
from sentence_parser import Word2Vector, get_uSIFw
from sentence_parser import SentenceParser


if __name__ == '__main__':
	nlp = spacy.load('en_core_web_sm')
	weight = get_uSIFw(count_fn="enwiki_vocab_min200.txt")
	vector = Word2Vector(vector_fn="vectors/psl.txt")
	SEater = SentenceParser(vector, weight, nlp)

	text = ['a man with a hard hat is dancing',
			'a man is spreading shredded cheese on an uncooked pizza',
			'two men are playing chess',
			'the man spanked the other man with a stick']
	SEater.add(text)
	SEater.emb()
