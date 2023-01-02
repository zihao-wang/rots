# ROTS, recursive optimal transport similarity

Code for paper [Unsupervised Sentence Textual Similarity with Compositional Phrase Semantics](https://arxiv.org/abs/2210.02284)

## requirement
- python OT
- spacy
- tqdm
- pandas
- numpy
- sklearn
- scipy
- scikits.bootstrap

## Setup

1. prepare data yourself
  1. prepare the vectors in `vectors` folder, see 'model/word_vector.py' for path specifications and see appendix for downloadable links.
  2. prepare the datasets in `dataset` folder, see 'model/dataset.py' for path specifications and see appendix for ways to obtain and preprocess.
2. run pipline.py file for evaluation, note that you may need to set the config files

For sample configs, please see the `config` folder.


## Bibliography
```
@inproceedings{wang2022Unsupervised,
  title={Unsupervised Sentence Textual Similarity with Compositional Phrase Semantics},
  author={Zihao Wang and Jiaheng Dou and Yong Zhang},
  booktitle={Proceedings of the 29th International Conference on Computational Linguistics},
  year={2022}
}
```
