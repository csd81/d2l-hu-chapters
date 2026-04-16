# Nyers szöveg átalakítása sorozatadattá
:label:`sec_text-sequence`

Ebben a könyvben
gyakran fogunk szöveges adatokkal dolgozni,
amelyek szavak, karakterek vagy szórészletek sorozataként vannak ábrázolva.
A kezdéshez szükségünk lesz néhány alapvető eszközre,
amelyek a nyers szöveget a megfelelő formátumú sorozatokká alakítják.
A tipikus előfeldolgozási folyamatok
a következő lépéseket hajtják végre:

1. A szöveg betöltése karakterláncként a memóriába.
1. A karakterláncok tokenekre bontása (pl. szavakra vagy karakterekre).
1. Szókincs-szótár felépítése, amely minden szókincsbeli elemet numerikus indexhez rendel.
1. A szöveg átalakítása numerikus indexek sorozatává.

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

```{.python .input  n=2}
%%tab mxnet
import collections
import re
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
import collections
import re
from d2l import torch as d2l
import torch
import random
```

```{.python .input  n=4}
%%tab tensorflow
import collections
import re
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
%%tab jax
import collections
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import random
import re
```

## Az adathalmaz beolvasása

Itt H. G. Wells
[Az időgép](http://www.gutenberg.org/ebooks/35) c. könyvével dolgozunk,
amely valamivel több mint 30 000 szót tartalmaz.
Bár a valódi alkalmazások általában
lényegesen nagyobb adathalmazokat tartalmaznak,
ez elegendő az előfeldolgozási folyamat bemutatásához.
A következő `_download` metódus
(**a nyers szöveget karakterláncként olvassa be**).

```{.python .input  n=5}
%%tab all
class TimeMachine(d2l.DataModule): #@save
    """Az Időgép adathalmaz."""
    def _download(self):
        fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', self.root,
                             '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

data = TimeMachine()
raw_text = data._download()
raw_text[:60]
```

Az egyszerűség kedvéért a nyers szöveg előfeldolgozásakor figyelmen kívül hagyjuk az írásjeleket és a nagybetűket.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _preprocess(self, text):
    return re.sub('[^A-Za-z]+', ' ', text).lower()

text = data._preprocess(raw_text)
text[:60]
```

## Tokenizálás

A *tokenek* a szöveg atomi (oszthatatlan) egységei.
Minden időlépés 1 tokennek felel meg,
de hogy pontosan mi alkotja a tokent, az tervezési döntés.
Például a "Baby needs a new pair of shoes" mondatot
ábrázolhatjuk 7 szóból álló sorozatként,
ahol az összes szó egy nagy szókincset alkot (általában tíz-
vagy százezer szót).
Vagy ugyanezt a mondatot ábrázolhatjuk
30 karakterből álló, jóval hosszabb sorozatként,
sokkal kisebb szókincset használva
(összesen csak 256 különböző ASCII karakter van).
Az alábbiakban előfeldolgozott szövegünket
karakterek sorozatává tokenizáljuk.

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _tokenize(self, text):
    return list(text)

tokens = data._tokenize(text)
','.join(tokens[:30])
```

## Szókincs

Ezek a tokenek még mindig karakterláncok.
Modelljeink bemenetei azonban
végső soron numerikus bemenetekből kell álljanak.
**Következőleg bemutatunk egy osztályt
a *szókincsek* felépítéséhez,
azaz olyan objektumokhoz, amelyek
minden különálló tokenértékhez
egyedi indexet rendelnek.**
Először meghatározzuk a tanítási *korpuszunkban* lévő egyedi tokenek halmazát.
Ezután numerikus indexet rendelünk minden egyedi tokenhez.
A ritka szókincsbeli elemeket kényelmi okokból gyakran eldobjuk.
Amikor tanítás vagy tesztelés során olyan tokennel találkozunk,
amelyet korábban nem láttunk, vagy kizártunk a szókincsből,
egy speciális "&lt;unk&gt;" tokennel ábrázoljuk,
jelezve, hogy ez egy *ismeretlen* érték.

```{.python .input  n=8}
%%tab all
class Vocab:  #@save
    """Szókincs szöveghez."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # 2D lista lapítása, ha szükséges
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Token-gyakoriságok megszámlálása
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # Az egyedi tokenek listája
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Az ismeretlen token indexe
        return self.token_to_idx['<unk>']
```

Most **szókincset építünk** az adathalmazunkhoz,
a karakterláncok sorozatát
numerikus indexek listájává alakítva.
Megjegyezzük, hogy nem veszítettünk el semmilyen információt,
és könnyen visszaalakíthatjuk adathalmazunkat
az eredeti (karakterlánc) formájára.

```{.python .input  n=9}
%%tab all
vocab = Vocab(tokens)
indices = vocab[tokens[:10]]
print('indices:', indices)
print('words:', vocab.to_tokens(indices))
```

## Mindent összefogva

A fenti osztályok és metódusok segítségével
**mindent becsomagolunk a `TimeMachine` osztály következő
`build` metódusába**,
amely visszaadja a `corpus`-t, token indexek listáját, és a `vocab`-ot,
*Az időgép* korpuszának szókincsét.
Az általunk elvégzett módosítások:
(i) a szöveget karakterekre tokenizáljuk, nem szavakra,
hogy egyszerűsítsük a tanítást a későbbi szakaszokban;
(ii) a `corpus` egyetlen lista, nem tokenlista-lista,
mivel *Az időgép* adathalmaz minden szövegsorja
nem feltétlenül mondat vagy bekezdés.

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def build(self, raw_text, vocab=None):
    tokens = self._tokenize(self._preprocess(raw_text))
    if vocab is None: vocab = Vocab(tokens)
    corpus = [vocab[token] for token in tokens]
    return corpus, vocab

corpus, vocab = data.build(raw_text)
len(corpus), len(vocab)
```

## A nyelv feltáró statisztikái
:label:`subsec_natural-lang-stat`

A valódi korpusz és a szavak felett definiált `Vocab` osztály segítségével
megvizsgálhatjuk a korpuszban lévő szóhasználat alapvető statisztikáit.
Az alábbiakban *Az időgépben* használt szavakból szókincset készítünk
és kiírjuk a tíz leggyakrabban előforduló szót.

```{.python .input  n=11}
%%tab all
words = text.split()
vocab = Vocab(words)
vocab.token_freqs[:10]
```

Vegyük észre, hogy (**a tíz leggyakoribb szó**)
nem különösebben leíró jellegű.
Talán el is képzelhetjük,
hogy egy nagyon hasonló listát kapnánk,
ha véletlenszerűen választottunk volna bármelyik könyvet.
Az olyan névelők, mint a "the" és "a",
az olyan névmások, mint az "i" és "my",
és az olyan elöljárók, mint az "of", "to" és "in"
gyakran előfordulnak, mert közönséges szintaktikai szerepeket töltenek be.
Az ilyen szavakat, amelyek gyakoriak, de nem különösebben leíróak,
gyakran (***stop szavaknak***) nevezik, és
az úgynevezett bag-of-words reprezentációkon alapuló szövegosztályozók
korábbi generációiban
leggyakrabban kiszűrték őket.
Azonban jelentést hordoznak, és
nem szükséges kiszűrni őket,
ha modern, RNN- és
Transformer-alapú neurális modellekkel dolgozunk.
Ha lejjebb nézünk a listában,
észrevesszük,
hogy a szógyakoriság gyorsan csökken.
A $10.$ leggyakoribb szó
kevesebb mint $1/5$-szer olyan közönséges, mint a legnépszerűbb.
A szógyakoriság hajlamos hatványeloszlást követni
(pontosabban a Zipf-eloszlást), ahogy lejjebb haladunk a rangsorban.
Jobb képet alkotni, **rajzoljuk meg a szógyakoriság ábrát**.

```{.python .input  n=12}
%%tab all
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

Miután az első néhány szót kivételként kezeltük,
az összes többi szó körülbelül egyenes vonalat követ egy log-log ábrán.
Ezt a jelenséget a *Zipf-törvény* ragadja meg,
amely kimondja, hogy a $i^\textrm{th}$ leggyakoribb szó $n_i$ gyakorisága:

$$n_i \propto \frac{1}{i^\alpha},$$
:eqlabel:`eq_zipf_law`

ami ekvivalens azzal, hogy

$$\log n_i = -\alpha \log i + c,$$

ahol $\alpha$ az eloszlást jellemző kitevő, és $c$ egy konstans.
Ennek már el kell gondolkoztatnia bennünket, ha szavakat szeretnénk modellezni
számolási statisztikák alapján.
Végül is, jelentősen túlbecsüljük a farok, azaz a ritka szavak gyakoriságát. De **mi a helyzet más szókombinációkkal, például két egymást követő szóval (bigram), három egymást követő szóval (trigram)** és továbbiakkal?
Nézzük meg, hogy a bigram-gyakoriság ugyanolyan módon viselkedik-e, mint az egyszavas (unigram) gyakorisága.

```{.python .input  n=13}
%%tab all
bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
bigram_vocab = Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

Egy dolog figyelemre méltó itt. A tíz leggyakoribb szópár közül kilenc stop szavakból áll, és csak egy releváns a tényleges könyvhöz: "the time". Ezenkívül nézzük meg, hogy a trigram-gyakoriság ugyanolyan módon viselkedik-e.

```{.python .input  n=14}
%%tab all
trigram_tokens = ['--'.join(triple) for triple in zip(
    words[:-2], words[1:-1], words[2:])]
trigram_vocab = Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

Most **vizualizáljuk a token-gyakoriságot** a három modell között: unigram, bigram és trigram.

```{.python .input  n=15}
%%tab all
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

Ez az ábra egészen izgalmas.
Először is, az unigrammális szavakon túl, a szósorozatok
szintén látszanak követni a Zipf-törvényt,
bár kisebb $\alpha$ kitevővel
a :eqref:`eq_zipf_law`-ban,
a sorozathossztól függően.
Másodszor, a különböző $n$-gramok száma nem olyan nagy.
Ez reményt ad számunkra, hogy a nyelvben számos struktúra van.
Harmadszor, sok $n$-gram nagyon ritkán fordul elő.
Ez egyes módszereket alkalmatlanná tesz a nyelvmodellezésre,
és motiválja a mélytanulás modellek használatát.
Erről a következő szakaszban tárgyalunk.


## Összefoglalás

A szöveg a mélytanulásban leggyakrabban előforduló sorozatadatok egyike.
A tokent alkotó elemek közönséges választásai a karakterek, szavak és szórészletek.
A szöveg előfeldolgozásához általában (i) szövegeket tokenekre bontunk; (ii) szókincset készítünk a token-karakterláncok numerikus indexekre való leképezéséhez; és (iii) szöveges adatokat token indexekké alakítunk, amelyeket a modellek manipulálni tudnak.
A gyakorlatban a szavak gyakorisága hajlamos a Zipf-törvényt követni. Ez nemcsak az egyes szavakra (unigramok) igaz, hanem az $n$-gramokra is.


## Feladatok

1. Ebben a fejezetben lévő kísérletben tokenizáld a szöveget szavakra, és változtasd a `Vocab` példány `min_freq` argumentumának értékét! Kvalitatívan jellemezd, hogyan befolyásolja a `min_freq` változása a kapott szókincs méretét!
1. Becsüld meg az unigramok, bigramok és trigramok Zipf-eloszlásának kitevőjét ebben a korpuszban!
1. Keress más adatforrásokat (tölts le egy standard gépi tanulási adathalmazt, válassz egy másik közkincsben lévő könyvet, scrape-elj egy weboldalt stb.)! Minden esetben tokenizáld az adatokat mind szó, mind karakter szinten. Hogyan hasonlíthatók a szókincsek méretei *Az időgép* korpuszával az egyenértékű `min_freq` értékeknél? Becsüld meg a Zipf-eloszlás kitevőjét az unigram és bigram eloszlásokhoz ezeken a korpuszokon! Hogyan hasonlíthatók az *Az időgép* korpuszon megfigyelt értékekhez?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1049)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18011)
:end_tab:
