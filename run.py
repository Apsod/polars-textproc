
from functools import cache
from collections import Counter
from itertools import tee, islice

import re

import polars as pl
import collections
from ccstuff import repetition_signals

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

txts = pl.scan_parquet('/home/amaru/Downloads/000_00000.parquet').select('text').head(100_000).collect()

import time

duration = -time.perf_counter()
for txt in txts['text']:
    s = ngram_duplicates(txt,10)
duration += time.perf_counter()
print(duration)

duration = -time.perf_counter()
signals = txts.with_columns(repetition=repetition_signals("text"))#.write_parquet('dump.parquet')
duration += time.perf_counter()
print(duration)


#result = lf.with_columns(repetition=repetition_signals("text")).collect()
#print(result)
#for i in range(100):
#    print('rust', result['repetition'][i])
#    txt = result['text'][i]
#    print('pyth', all_signals(txt))
#    print()
