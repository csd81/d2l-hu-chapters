```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Gépi Fordítás és az Adathalmaz
:label:`sec_machine_translation`

A modern RNN-ek iránti széles körű érdeklődést kiváltó nagy áttörések egyike
a statisztikai *gépi fordítás* alkalmazott területén bekövetkezett jelentős előrelépés volt.
Ebben a feladatban a modell egy mondatot kap az egyik nyelven,
és a megfelelő mondatot kell megjósolnia egy másik nyelven.
Vegyük figyelembe, hogy a mondatok különböző hosszúságúak lehetnek,
és a két mondatban lévő megfelelő szavak
nem feltétlenül fordulnak elő ugyanabban a sorrendben,
a két nyelv grammatikai szerkezete közötti különbségek miatt.


Sok problémának van ilyen jellegű leképezése
két ilyen "nem igazított" sorozat között.
Példák közé tartozik a párbeszéd utasításaiból válaszokra való leképezés
vagy kérdésekből válaszokra való leképezés.
Általánosan ezeket a problémákat
*sorozatból sorozatba* (seq2seq) problémáknak nevezzük,
és ezek lesznek a fókuszunk
ebben a fejezet hátralévő részében
és a :numref:`chap_attention-and-transformers` nagy részében is.

Ebben a részben bemutatjuk a gépi fordítás problémáját
és egy példaadathalmazt, amelyet a következő példákban fogunk használni.
Évtizedekig a nyelvek közötti fordítás statisztikai megfogalmazásai
népszerűek voltak :cite:`Brown.Cocke.Della-Pietra.ea.1988,Brown.Cocke.Della-Pietra.ea.1990`,
még mielőtt a kutatók a neurális hálózatos megközelítéseket sikeresen alkalmazni kezdték
(ezeket a módszereket sokszor összefoglalóan *neurális gépi fordítás* névvel illetik).


Először szükségünk lesz néhány új kódra az adatok feldolgozásához.
Ellentétben a :numref:`sec_language-model` fejezetben látott nyelvi modellezéssel,
ahol minden példa egyetlen szövegsorozatból állt,
itt minden példa két külön szövegsorozatból áll:
egy a forrásnyelvből és egy (a fordítás) a célnyelvből.
A következő kódrészletek megmutatják, hogyan
tölthetők be az előfeldolgozott adatok minibatchekbe tanítás céljából.

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
import os
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import os
```

```{.python .input  n=4}
%%tab jax
from d2l import jax as d2l
from jax import numpy as jnp
import os
```

## **Az Adathalmaz Letöltése és Előfeldolgozása**

Kezdésként letöltünk egy angol–francia adathalmazt,
amely [kétnyelvű mondatpárokat tartalmaz a Tatoeba projektből](http://www.manythings.org/anki/).
Az adathalmaz minden sora tabulátorral elválasztott pár,
amely egy angol szövegsorozatból (a *forrás*) áll
és a lefordított francia szövegsorozatból (a *cél*).
Vegyük figyelembe, hogy minden szövegsorozat
lehet egyetlen mondat
vagy több mondatból álló bekezdés.

```{.python .input  n=5}
%%tab all
class MTFraEng(d2l.DataModule):  #@save
    """Az angol–francia adathalmaz."""
    def _download(self):
        d2l.extract(d2l.download(
            d2l.DATA_URL+'fra-eng.zip', self.root, 
            '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()
```

```{.python .input}
%%tab all
data = MTFraEng() 
raw_text = data._download()
print(raw_text[:75])
```

Az adathalmaz letöltése után
**több előfeldolgozási lépést hajtunk végre**
a nyers szöveges adatokon.
Például lecseréljük a nem törő szóközt szóközre,
a nagybetűket kisbetűkre alakítjuk,
és szóközöket szúrunk be a szavak és a írásjelek közé.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _preprocess(self, text):
    # Nem törő szóköz cseréje szóközre
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Szóköz beszúrása a szavak és az írásjelek közé
    no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text.lower())]
    return ''.join(out)
```

```{.python .input}
%%tab all
text = data._preprocess(raw_text)
print(text[:80])
```

## **Tokenizálás**

Ellentétben a :numref:`sec_language-model` fejezetben alkalmazott
karakterszintű tokenizálással,
gépi fordításhoz szószintű tokenizálást alkalmazunk
(a mai legkorszerűbb modellek
összetettebb tokenizálási technikákat használnak).
Az alábbi `_tokenize` metódus
tokenizálja az első `max_examples` szövegsorozat-párt,
ahol minden token egy szó vagy egy írásjel.
Minden sorozat végéhez hozzáfűzzük a speciális "&lt;eos&gt;" tokent,
jelezve a sorozat végét.
Amikor egy modell token után tokent generálva jósol,
a "&lt;eos&gt;" token generálása
jelezheti, hogy a kimeneti sorozat teljes.
Végül az alábbi metódus
két token-lista listát ad vissza: `src` és `tgt`.
Konkrétan, `src[i]` az $i$-edik szövegsorozat tokenjeinek listája a forrásnyelven (itt angolul),
és `tgt[i]` ugyanez a célnyelven (itt franciául).

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _tokenize(self, text, max_examples=None):
    src, tgt = [], []
    for i, line in enumerate(text.split('\n')):
        if max_examples and i > max_examples: break
        parts = line.split('\t')
        if len(parts) == 2:
            # Üres tokenek kihagyása
            src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
            tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
    return src, tgt
```

```{.python .input}
%%tab all
src, tgt = data._tokenize(text)
src[:6], tgt[:6]
```

**Ábrázoljuk a tokenek számának hisztogramját szövegsorozatonként.**
Ebben az egyszerű angol–francia adathalmazban
a legtöbb szövegsorozat kevesebb mint 20 tokent tartalmaz.

```{.python .input  n=8}
%%tab all
#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """A listahossz-párok hisztogramjának ábrázolása."""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)
```

```{.python .input}
%%tab all
show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', src, tgt);
```

## Rögzített Hosszúságú Sorozatok Betöltése
:label:`subsec_loading-seq-fixed-len`

Felidézzük, hogy a nyelvi modellezésben
**minden példasorozat**,
akár egy mondat egy szegmense,
akár több mondatot átívelő szakasz,
**rögzített hosszúságú volt.**
Ezt a `num_steps`
(időlépések vagy tokenek száma) argumentum határozta meg a :numref:`sec_language-model` fejezetből.
A gépi fordításban minden példa
egy forrás- és célszöveg-sorozat pár,
ahol a két szövegsorozat különböző hosszúságú lehet.

A számítási hatékonyság érdekében
egy minibatch szövegsorozatot egyszerre is feldolgozhatunk
*csonkítás* és *kitöltés* segítségével.
Tegyük fel, hogy ugyanabban a minibatchben minden sorozatnak
azonos `num_steps` hosszúságúnak kell lennie.
Ha egy szövegsorozatnak kevesebb mint `num_steps` tokenje van,
a végéhez folyamatosan hozzáfűzzük a speciális "&lt;pad&gt;" tokent,
amíg el nem éri a `num_steps` hosszt.
Ellenkező esetben a szövegsorozatot csonkítjuk,
csak az első `num_steps` tokenjét tartva meg és a maradékot elvetve.
Így minden szövegsorozatnak
azonos hossza lesz,
és azonos alakú minibatchekbe tölthető.
Ezen felül rögzítjük a forrássorozat hosszát is a kitöltési tokenek nélkül.
Erre az információra néhány modellnek szüksége lesz, amelyeket később tárgyalunk.


Mivel a gépi fordítás adathalmazai
nyelvpárokból állnak,
mindkét nyelvhez külön-külön szókincset építhetünk:
mind a forrásnyelv, mind a célnyelv számára.
Szószintű tokenizálásnál
a szókincs mérete lényegesen nagyobb lesz,
mint karakterszintű tokenizálásnál.
Ennek enyhítésére
az egynél kevesebbszer előforduló ritka tokeneket
ugyanolyan ismeretlen ("&lt;unk&gt;") tokenként kezeljük.
Ahogy később elmagyarázunk (:numref:`fig_seq2seq`),
amikor célsorozatokkal tanítunk,
a dekódoló kimenet (címke tokenek)
ugyanaz lehet, mint a dekódoló bemenet (cél tokenek),
egy tokennel eltolva;
a speciális sorozat-kezdő "&lt;bos&gt;" token
lesz az első bemeneti token
a célsorozat előrejelzésekor (:numref:`fig_seq2seq_predict`).

```{.python .input  n=9}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
    super(MTFraEng, self).__init__()
    self.save_hyperparameters()
    self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
        self._download())
```

```{.python .input}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
    def _build_array(sentences, vocab, is_tgt=False):
        pad_or_trim = lambda seq, t: (
            seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
        sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
        if is_tgt:
            sentences = [['<bos>'] + s for s in sentences]
        if vocab is None:
            vocab = d2l.Vocab(sentences, min_freq=2)
        array = d2l.tensor([vocab[s] for s in sentences])
        valid_len = d2l.reduce_sum(
            d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
        return array, vocab, valid_len
    src, tgt = self._tokenize(self._preprocess(raw_text), 
                              self.num_train + self.num_val)
    src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
    tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
    return ((src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]),
            src_vocab, tgt_vocab)
```

## **Az Adathalmaz Olvasása**

Végül definiáljuk a `get_dataloader` metódust,
amely visszaadja az adatiterátort.

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader(self.arrays, train, idx)
```

**Olvassuk be az angol–francia adathalmaz első minibatchét.**

```{.python .input  n=11}
%%tab all
data = MTFraEng(batch_size=3)
src, tgt, src_valid_len, label = next(iter(data.train_dataloader()))
print('source:', d2l.astype(src, d2l.int32))
print('decoder input:', d2l.astype(tgt, d2l.int32))
print('source len excluding pad:', d2l.astype(src_valid_len, d2l.int32))
print('label:', d2l.astype(label, d2l.int32))
```

Megmutatunk egy forrás- és célsorozatpárt,
amelyet a fenti `_build_arrays` metódus dolgozott fel
(karakterlánc formátumban).

```{.python .input  n=12}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def build(self, src_sentences, tgt_sentences):
    raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
        src_sentences, tgt_sentences)])
    arrays, _, _ = self._build_arrays(
        raw_text, self.src_vocab, self.tgt_vocab)
    return arrays
```

```{.python .input  n=13}
%%tab all
src, tgt, _,  _ = data.build(['hi .'], ['salut .'])
print('source:', data.src_vocab.to_tokens(d2l.astype(src[0], d2l.int32)))
print('target:', data.tgt_vocab.to_tokens(d2l.astype(tgt[0], d2l.int32)))
```

## Összefoglalás

A természetes nyelvi feldolgozásban a *gépi fordítás* az a feladat, amelynek során automatikusan leképezünk egy sorozatot, amely egy *forrás*nyelven lévő szövegláncot reprezentál, egy másik sorozatra, amely egy ésszerű fordítást reprezentál a *cél* nyelven. Szószintű tokenizálásnál a szókincs mérete lényegesen nagyobb lesz, mint karakterszintű tokenizálásnál, de a sorozathosszak lényegesen rövidebbek lesznek. A nagy szókincsméret enyhítésére a ritka tokeneket "ismeretlen" tokenként kezelhetjük. Csonkítással és kitöltéssel a szövegsorozatokat azonos hosszúságúvá tehetjük, hogy minibatchekbe tölthetők legyenek. A modern implementációk sokszor hasonló hosszúságú sorozatokat csoportosítanak, hogy elkerüljék a kitöltéssel járó felesleges számítást.


## Feladatok

1. Próbálj ki különböző értékeket a `_tokenize` metódus `max_examples` argumentumára. Hogyan befolyásolja ez a forrásnyelv és a célnyelv szókincsméretét?
1. Egyes nyelvek, például a kínai és a japán szövegekben nincsenek szóhatár-jelölők (pl. szóközök). Ezekben az esetekben is jó ötlet a szószintű tokenizálás? Miért, vagy miért nem?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/344)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1060)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3863)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18020)
:end_tab:
