# uniblock

Tired of scripting rules to remove **special punctuation marks, full-width characters, emojis, mathematical symbols, Latin characters, currency symbols, scientific notations, ...** for your NLP model?

This repository contains code for the paper [uniblock: Scoring and Filtering Corpus with Unicode Block Information](https://arxiv.org/abs/1908.09716) in [EMNLP-IJCNLP 2019](https://www.emnlp-ijcnlp2019.org/).

Using [Unicode](https://home.unicode.org/) [Block](https://en.wikipedia.org/wiki/Unicode_block) information (and more) as features, a [Bayesian Gaussian Mixture Model (BGM model)](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html) is trained to score and filter corpus. This method provides an alternative to the traditional rule-based approaches for "illegal" character filtering. It is statistical, extendable, simple and effective.

# Getting Started

## installation

To install the package, please use the following command (uniblock uses [f-strings](https://www.python.org/dev/peps/pep-0498/) and requires python 3.6+):
```
pip install uniblock
```
After correct installation, the "uniblock" command should be added to the PATH:
```
$ uniblock -h
usage: uniblock.py <command> [<args>]

uniblock, scoring and filtering corpus with Unicode block information (and
more).

positional arguments:
  {setup,train,score,filter}
                        command to execute

optional arguments:
  -h, --help            show this help message and exit

```

## getting help

uniblock uses a git-like "command \<subcommand\> [\<args\>]" interface. To get help about available subcommands and details about each subcommand, please use the following commands:
```
uniblock --help
uniblock setup --help
uniblock train --help
uniblock score --help
uniblock filter --help
```

## an example run

The best way to get started with uniblock is to do an example run to score and filter some text. Following the steps below takes about 20 minutes and you should be comfortable using uniblock for your own purposes afterwards.

### 1. get data

For this run, we will use some Machine Translation (MT) data prepared from [WMT19](http://www.statmt.org/wmt19/). The language pair is Chinese-English, and training data is subsampled down to 2M sentence pairs for a quick and realistic run of uniblock. To download the data:

[Google Drive](https://drive.google.com/drive/folders/1TK28o9ShMsub4PM7e1lL58C4BlLsezI5?usp=sharing) or [OneDrive](https://1drv.ms/u/s!AqfzE52eeEFjg0ExRoJl6N4mrk0T?e=dhYApM)

After download and decompression, you should see 4 data files: "train.2M.en", "train.2M.zh", "valid.en" and "valid.zh". Take a look at the data and get a feeling of the quality:
```
less train.2M.en
less train.2M.zh
less valid.en
less valid.zh
```
We will use "valid.xx" to train our BGM model and score and filter "train.2M.xx".

### 2. create folders

Create a folder called "uniblock-an-example-run" and subfolders called "zh" and "en":
```
mkdir uniblock-an-example-run
mkdir uniblock-an-example-run/zh
mkdir uniblock-an-example-run/en
```

### 3. setup features
We will filter sentences by their lengths and character Unicode distributions. It is also possible to discriminate the ASCII digits and ASCII punctuation and symbols from other ASCII characters, which is achieved by adding "pseudo-blocks".
```
uniblock setup --help
uniblock setup --exp-path uniblock-an-example-run/zh --char-count --word-count --pseudo-blocks '0030..0039; ASCII digits' '0020..002F 003A..0040 005B..0060 007B..007E; ASCII punctuation and symbols'
uniblock setup --exp-path uniblock-an-example-run/en --char-count --word-count --pseudo-blocks '0030..0039; ASCII digits' '0020..002F 003A..0040 005B..0060 007B..007E; ASCII punctuation and symbols'
```
"uniblock-an-example/xx/uniblock" folders are created to store uniblock-related files and "uniblock-an-example/xx/uniblock/feature.dump" are dumped for fast conversion from sentences to feature vectors.

### 4. train zh and en BGM models
In this step we train BGM models, one for "zh" and one for "en". With the "--verbose" flag, we can see that with the default hyperparameters the EM training converged.
```
uniblock train --help
uniblock train uniblock-data/valid.zh --exp-path uniblock-an-example-run/zh/ --verbose
uniblock train uniblock-data/valid.en --exp-path uniblock-an-example-run/en/ --verbose
```
It is also possible to examine the score files (replace "xx" with "zh" or "en"):
```
sort -g uniblock-an-example-run/xx/uniblock/valid.xx.score | less
```
and score statistics (replace "xx" with "zh" or "en"):
```
less uniblock-an-example-run/xx/uniblock/valid.xx.stat
```

### 5. score corpus
NLP corpus can often go up to over 100M lines. Therefore in this step we split our 2M line corpus files into subfiles each containing 1M lines to simulate a more realistic run.
First split the corpus files:
```
split uniblock-data/train.2M.zh -l 1000000 uniblock-data/train.2M.zh.split.
split uniblock-data/train.2M.en -l 1000000 uniblock-data/train.2M.en.split.
```
Then score the subfiles with uniblock:
```
uniblock score --help
uniblock score uniblock-data/train.2M.zh.split.aa --exp-path uniblock-an-example-run/zh/
uniblock score uniblock-data/train.2M.zh.split.ab --exp-path uniblock-an-example-run/zh/
uniblock score uniblock-data/train.2M.en.split.aa --exp-path uniblock-an-example-run/en/
uniblock score uniblock-data/train.2M.en.split.ab --exp-path uniblock-an-example-run/en/
```
When the original corpus is large, there may be many subfiles. In this case, you can use your favorite job submitting software for your cluster and easilly parallelize this step.
Finally, merge the scores:
```
cat uniblock-an-example-run/zh/uniblock/train.2M.zh.split.*.score > uniblock-an-example-run/zh/uniblock/train.2M.zh.score
cat uniblock-an-example-run/en/uniblock/train.2M.en.split.*.score > uniblock-an-example-run/en/uniblock/train.2M.en.score
```
Optionally, check the results qualitatively:
```
sort -g uniblock-an-example-run/zh/uniblock/train.2M.zh.score | less
sort -g uniblock-an-example-run/en/uniblock/train.2M.en.score | less
```
At this point, you should see sentences with low qualities (really short ones, really long ones and ones with illegal characters) are assigned with low scores. Scrolling down to the end of the sorted score files ("shift + G"), you should also see high quality sentences with high scores.

### 6. filter corpus
uniblock supports several ways to filter the corpus with scores. For example, an absolute threshold (inclusive) or a relative threshold ([0, 1]) could be used. Alternatively, one could use the lowest score seen during the training of the BGM model and use it as an absolute threshold. uniblock also supports parallel corpus filtering. In this case, a reduction (the --combine flag) method needs to be specified to combine the scores across the parallel score files (different languages in the case of MT) and then apply the threshold. To see the usage of these options:
```
uniblock filter --help
```
To filter with an absolute threshold of 10.0:
```
$ uniblock filter uniblock-an-example-run/zh/uniblock/train.2M.zh.score uniblock-an-example-run/en/uniblock/train.2M.en.score --thres-abs 10.0
@ 59830 / 2000000 ≈ 2.99% lines filtered out
```
To filter with a relative threshold of 20%:
```
$ uniblock filter uniblock-an-example-run/zh/uniblock/train.2M.zh.score uniblock-an-example-run/en/uniblock/train.2M.en.score --thres-rel 0.2
@ 400000 / 2000000 ≈ 20.00% lines filtered out
```
To filter with min scores seen during the training of the BGM model:
```
$ uniblock filter uniblock-an-example-run/zh/uniblock/train.2M.zh.score uniblock-an-example-run/en/uniblock/train.2M.en.score --combine none --stat-paths uniblock-an-example-run/zh/uniblock/valid.zh.stat uniblock-an-example-run/en/uniblock/valid.en.stat 
@ 106350 / 2000000 ≈ 5.32% lines filtered out
```
To filter by linearly combining the scores for different languages:
```
$ uniblock filter uniblock-an-example-run/zh/uniblock/train.2M.zh.score uniblock-an-example-run/en/uniblock/train.2M.en.score --combine weighted_sum --lambdas 0.9 0.1 --thres-rel 0.3
@ 600000 / 2000000 ≈ 30.00% lines filtered out
```
After filtering, it is important to check the ".filter" files and get a feeling for the cleaned corpus:
```
less uniblock-an-example-run/zh/uniblock/train.2M.zh.score.filter
less uniblock-an-example-run/en/uniblock/train.2M.en.score.filter
```
These two files are the cleaned version of the original "uniblock-data/train.2M.zh" "uniblock-data/train.2M.en" files.

### 7. next steps
Following the previous steps, you successfully scored and filtered a training corpus for MT. For other tasks that work with monolingual data, it is easy to adapt the steps and obtain "clean" training data. As in the paper, we have experimented with Sentiment Analysis, Language Modeling and Machine Translation to show that uniblocks is robust and works quite well across different tasks and languages. As next steps, you should train your NLP model with the uniblock-cleaned data, and see if there is expected improvement.

# Cite This Work
To cite this work, please use the following .bib:
```
@InProceedings{gao19:uniblock,
	author={Gao, Yingbo and Wang, Weiyue and Ney, Hermann},  	
	title={uniblock: Scoring and Filtering Corpus with Unicode Block Information},  
	booktitle={Conference on Empirical Methods in Natural Language Processing},
	year=2019,  
	address={Hong Kong, China},  
	month=nov,  
	booktitlelink={https://www.emnlp-ijcnlp2019.org/},
}
```

> Written with [StackEdit](https://stackedit.io/).
