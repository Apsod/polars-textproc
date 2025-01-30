
from functools import cache
from collections import Counter

from itertools import tee, islice

import re

import polars as pl
import collections
from ccstuff import repetition_signals, fasttext

HEAD = 10_000
import time
duration = -time.perf_counter()

df = pl.DataFrame({
    "text": ["Hello, this is a text in English, I hope the LID-model gets it right.", "Detta är en svensk text", "This is both an engelsk text men också lite svenska, det är en mix of both Swedish and English"],
    })
print(df)

lf = df.lazy()


lf = lf.with_columns(repetition=repetition_signals("text"))
lf = lf.with_columns(langid=fasttext("text", path="model.bin", labels=["__label__swe_Latn", "__label__eng_Latn"]))
print(lf.explain(streaming=True))
res = lf.collect()
#r, p = lf.head(HEAD).profile() #.unnest('langid').unnest('repetition').filter(pl.col('dup_5_gram_char_ratio') == 0.0).sink_parquet('dump.parquet')
duration += time.perf_counter()
print('rust:  ', duration)
print(res.select('text', 'langid').unnest('langid'))
print(res.select('text', pl.col('repetition').struct.field('top_1_gram_char_ratio')))
#print(r)
#print(p)
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
