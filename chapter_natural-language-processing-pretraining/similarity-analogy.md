# Szóhasonlóság és szóanalógia
:label:`sec_synonyms`

A :numref:`sec_word2vec_pretraining` fejezetben
egy word2vec modellt tanítottunk egy kis adathalmazon,
majd alkalmaztuk
szemantikailag hasonló szavak keresésére
egy bemeneti szóhoz.
A gyakorlatban
nagy korpuszokon előre tanított szóvektorok
alkalmazhatók downstream
természetes nyelvfeldolgozási feladatokra,
amelyekkel később a :numref:`chap_nlp_app` fejezetben foglalkozunk.
Az előre tanított szóvektorok
nagy korpuszokon tárolt szemantikájának
szemléletes bemutatásához
alkalmazzuk őket
szóhasonlósági és szóanalógia-feladatokban.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
```

## Előre tanított szóvektorok betöltése

Az alábbiakban felsorolt, 50, 100 és 300 dimenziós előre tanított GloVe beágyazások
letölthetők a [GloVe weboldaláról](https://nlp.stanford.edu/projects/glove/).
Az előre tanított fastText beágyazások több nyelven is elérhetők.
Itt az egyik angol változatot vizsgáljuk (300 dimenziós „wiki.en"), amely letölthető a
[fastText weboldaláról](https://fasttext.cc/).

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

#@save
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

#@save
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

#@save
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')
```

Az előre tanított GloVe és fastText beágyazások betöltéséhez definiáljuk az alábbi `TokenEmbedding` osztályt.

```{.python .input}
#@tab all
#@save
class TokenEmbedding:
    """Token beágyazás."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Fejléc-információk kihagyása, pl. a fastText első sora
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

Az alábbiakban betöltjük az
50 dimenziós GloVe beágyazásokat
(amelyeket a Wikipédia egy részhalmazán tanítottak elő).
A `TokenEmbedding` példány létrehozásakor
a megadott beágyazási fájl letöltésre kerül,
ha még nem volt elérhető.

```{.python .input}
#@tab all
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

Írassuk ki a szótár méretét. A szótár 400 000 szót (tokent) és egy speciális ismeretlen tokent tartalmaz.

```{.python .input}
#@tab all
len(glove_6b50d)
```

Lekérhetjük egy szó szótárbeli indexét, és fordítva.

```{.python .input}
#@tab all
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## Az előre tanított szóvektorok alkalmazása

A betöltött GloVe-vektorok segítségével
bemutatjuk azok szemantikáját
az alábbi szóhasonlósági és szóanalógia-feladatokban.


### Szóhasonlóság

A :numref:`subsec_apply-word-embed` fejezethez hasonlóan,
egy bemeneti szóhoz szemantikailag hasonló szavak megkereséséhez
a szóvektorok közötti koszinuszos hasonlóság alapján
az alábbi `knn`
($k$-legközelebbi szomszéd) függvényt valósítjuk meg.

```{.python .input}
#@tab mxnet
def knn(W, x, k):
    # Numerikus stabilitás érdekében 1e-9 hozzáadása
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

```{.python .input}
#@tab pytorch
def knn(W, x, k):
    # Numerikus stabilitás érdekében 1e-9 hozzáadása
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]
```

Ezután hasonló szavakat keresünk
az `embed` `TokenEmbedding` példányból
betöltött előre tanított szóvektorok segítségével.

```{.python .input}
#@tab all
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # A bemeneti szó kizárása
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

A `glove_6b50d` előre tanított szóvektorkészletének szótára
400 000 szót és egy speciális ismeretlen tokent tartalmaz.
A bemeneti szót és az ismeretlen tokent kizárva
keressük meg e szótárban
a „chip" szóhoz szemantikailag legközelebb eső
három szót.

```{.python .input}
#@tab all
get_similar_tokens('chip', 3, glove_6b50d)
```

Az alábbiakban a „baby" és a „beautiful" szavakhoz hasonló szavakat kapjuk meg.

```{.python .input}
#@tab all
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
#@tab all
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### Szóanalógia

A hasonló szavak megkeresése mellett
szóvektorokat szóanalógia-feladatokhoz is felhasználhatunk.
Például
a „man":"woman"::"son":"daughter"
egy szóanalógia alakja:
„man" úgy viszonyul „woman"-hoz, ahogy „son" viszonyul „daughter"-hoz.
Pontosabban,
a szóanalógia-kiegészítési feladat
a következőképpen definiálható:
egy $a : b :: c : d$ alakú szóanalógiában,
adott az első három szó $a$, $b$ és $c$, és meg kell találni $d$-t.
Jelöljük a $w$ szó vektorát $\textrm{vec}(w)$-vel.
Az analógia kiegészítéséhez
azt a szót keressük,
amelynek vektora leginkább hasonlít
$\textrm{vec}(c)+\textrm{vec}(b)-\textrm{vec}(a)$ eredményéhez.

```{.python .input}
#@tab all
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Ismeretlen szavak eltávolítása
```

Ellenőrizzük a „férfi–nő" analógiát a betöltött szóvektorok segítségével.

```{.python .input}
#@tab all
get_analogy('man', 'woman', 'son', glove_6b50d)
```

Az alábbi egy „főváros–ország" analógiát old meg:
„beijing":"china"::"tokyo":"japan".
Ez az előre tanított szóvektorok szemantikáját szemlélteti.

```{.python .input}
#@tab all
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

A
„melléknév–felsőfokú melléknév" analógiára –
például „bad":"worst"::"big":"biggest" –
látható, hogy az előre tanított szóvektorok
szintaktikai információt is megragadhatnak.

```{.python .input}
#@tab all
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

Az előre tanított szóvektorokban
rögzített múlt idő fogalmának bemutatásához
tesztelhetjük a szintaxist
a „jelen idő–múlt idő" analógiával: „do":"did"::"go":"went".

```{.python .input}
#@tab all
get_analogy('do', 'did', 'go', glove_6b50d)
```

## Összefoglalás

* A gyakorlatban a nagy korpuszokon előre tanított szóvektorok alkalmazhatók downstream természetes nyelvfeldolgozási feladatokra.
* Az előre tanított szóvektorok felhasználhatók szóhasonlósági és szóanalógia-feladatokhoz.


## Feladatok

1. Teszteld a fastText eredményeit a `TokenEmbedding('wiki.en')` segítségével!
1. Ha a szótár rendkívül nagy, hogyan lehet gyorsabban megtalálni a hasonló szavakat, vagy kiegészíteni egy szóanalógiát?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/387)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1336)
:end_tab:
