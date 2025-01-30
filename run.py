
from functools import cache
from collections import Counter

from itertools import tee, islice

import re

import polars as pl
import collections
from ccstuff import repetition_signals, fasttext

def ngrams(xs, n):
    iterables = tee(xs, n)
    for i, sub in enumerate(iterables):
        list(islice(sub, i)) # drop the first i values from the ith iterable (list forces the slice)
    return zip(*iterables)


def ngram_duplicate_chars(xss):
    seen = set()
    lb = 0 # left bound of duplicates already accounted for (in the ngram)
    dup_chars = 0
    for xs in xss:
        # we observe one more word, so decrement the left bound.
        lb = max(0, lb - 1)
        if xs in seen:
            # add word lens for words not already accounted for.
            # and set the left bound to the rightmost position
            dup_chars += sum(len(x) for x in xs[lb:])
            #dup_chars += sum(1 for x in xs[lb:])
            lb = len(xs)
        else:
            seen.add(xs)
    return dup_chars

def top_ngram(txt, n):
    words = re.split(r'\s+', txt)
    tot = sum(len(w) for w in words)
    top = 0

    for k, v in Counter(ngrams(words, n)).most_common():
        l = sum(len(w) for w in k)
        top = max(top, l*v)

    return top / tot

def ngram_duplicates(txt, n):
    words = re.split(r'\s+', txt)
    tot = sum(len(w) for w in words)
    return ngram_duplicate_chars(ngrams(words, n)) / tot

def all_signals(txt):
    ret = {}
    for i in range(1, 5):
        ret[f'top_{i}_gram_char_ratio'] = top_ngram(txt, i)
    for i in range(5, 11):
        ret[f'dup_{i}_gram_char_ratio'] = ngram_duplicates(txt, i)
    return ret



#r, p = pl.scan_parquet('/home/amaru/Downloads/000_00000.parquet').select('text').head(10000).with_columns(repetition=repetition_signals("text")).profile()
#print(r)
#print(p)

HEAD = 10_000
import time
duration = -time.perf_counter()
lf = pl.scan_parquet('/home/amaru/Downloads/000_00000.parquet')
#rep = lf.with_columns(repetition=repetition_signals("text"))
#lid = lf.with_columns(langid=langid("text", path="model.bin", labels=["__label__swe_Latn", "__label__eng_Latn"]))
lf = lf.with_columns(langid=fasttext("text", path="model.bin", labels=["__label__swe_Latn", "__label__eng_Latn"]))
lf = lf.with_columns(repetition=repetition_signals("text"))
r, p = lf.head(HEAD).profile() #.unnest('langid').unnest('repetition').filter(pl.col('dup_5_gram_char_ratio') == 0.0).sink_parquet('dump.parquet')
duration += time.perf_counter()
print('rust:  ', duration)
print(r)
print(p)
#print(pl.read_parquet('dump.parquet'))
#print(both.head(HEAD).filter(pl.col('langid').struct.field('total_prob') > .90).head(10).collect())
#for txt in txts['text']:
#    s = ngram_all(re.split(r'\s+', txt))
#print('python:', duration)
#print(s)

#for i in range(100):
#    print(signals['langid'][i])
#    print('rust', list(signals['repetition'][i].values()))
#    txt = signals['text'][i]
#    print('pyth', ngram_all(re.split(r'\s+', txt)))
#    print('pyth', list(all_signals(txt).values()))
#    print()
#
