# Az Adathalmaz Szóbeágyazások Előtanításához
:label:`sec_word2vec_data`

Most, hogy megismertük a word2vec modellek és a közelítő tanítási módszerek technikai részleteit,
nézzük meg az implementációjukat.
Konkrétan
a :numref:`sec_word2vec` fejezetbeli skip-gram modellt
és a :numref:`sec_approx_train` fejezetbeli negatív mintavételezést
vesszük példaként.
Ebben a szakaszban
az adathalmazzal kezdjük,
amelyet a szóbeágyazási modell előtanításához használunk:
az adatok eredeti formátumát
mini-batch-ekké alakítjuk,
amelyeken tanítás közben iterálni lehet.

```{.python .input}
#@tab mxnet
import collections
from d2l import mxnet as d2l
import math
from mxnet import gluon, np
import os
import random
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
import os
import random
```

## Az Adathalmaz Beolvasása

Az itt használt adathalmaz a [Penn Tree Bank (PTB)]( https://catalog.ldc.upenn.edu/LDC99T42).
Ez a korpusz Wall Street Journal cikkekből van mintavételezve,
tanítási, validációs és teszthalmazra bontva.
Az eredeti formátumban
a szövegfájl minden sora
egy szóközökkel elválasztott szavakból álló mondatot képvisel.
Itt minden szót tokenként kezelünk.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

#@save
def read_ptb():
    """A PTB adathalmazt szövegsorok listájába tölti be."""
    data_dir = d2l.download_extract('ptb')
    # A tanítókészlet beolvasása
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
f'# sentences: {len(sentences)}'
```

A tanítóhalmaz beolvasása után felépítünk egy szókincset a korpuszhoz,
ahol minden szó, amely kevesebb mint 10-szer szerepel, a „&lt;unk&gt;" tokennel helyettesítjük.
Megjegyezzük, hogy az eredeti adathalmaz
szintén tartalmaz „&lt;unk&gt;" tokeneket, amelyek ritka (ismeretlen) szavakat jelölnek.

```{.python .input}
#@tab all
vocab = d2l.Vocab(sentences, min_freq=10)
f'vocab size: {len(vocab)}'
```

## Alulmintavételezés

A szöveges adatok
általában magas frekvenciájú szavakat tartalmaznak,
mint például a „the", az „a" és az „in":
ezek akár milliárdszor is előfordulhatnak
nagyon nagy korpuszokban.
Azonban
ezek a szavak gyakran
sok különböző szóval fordulnak elő együtt
a kontextusablakokban, kevés hasznos jelet adva.
Például
gondoljunk a „chip" szóra egy kontextusablakban:
intuitívan
az alacsony frekvenciájú „intel" szóval való együttes előfordulása
hasznosabb a tanításban,
mint a magas frekvenciájú „a" szóval való együttes előfordulás.
Továbbá a hatalmas mennyiségű (magas frekvenciájú) szóval való tanítás
lassú.
Ezért, szóbeágyazási modellek tanításakor,
a magas frekvenciájú szavak *alulmintavételezhetők* :cite:`Mikolov.Sutskever.Chen.ea.2013`.
Konkrétan,
az adathalmazban az $i$ indexű $w_i$ szót
a következő valószínűséggel vetjük el:


$$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$

ahol $f(w_i)$ a $w_i$ szavak száma
az adathalmazban lévő összes szóhoz viszonyítva,
a $t$ konstans pedig egy hiperparaméter
(a kísérletben $10^{-4}$).
Láthatjuk, hogy csak akkor vethető el a (magas frekvenciájú) $w_i$ szó,
ha a relatív frekvencia
$f(w_i) > t$,
és minél magasabb a szó relatív frekvenciája,
annál nagyobb az elvetés valószínűsége.

```{.python .input}
#@tab all
#@save
def subsample(sentences, vocab):
    """Magas frekvenciájú szavak alulmintavételezése."""
    # Az ismeretlen tokenek ('<unk>') kizárása
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = collections.Counter([
        token for line in sentences for token in line])
    num_tokens = sum(counter.values())

    # Igazat ad vissza, ha a `token` megmarad az alulmintavételezés során
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

subsampled, counter = subsample(sentences, vocab)
```

A következő kódrészlet
az alulmintavételezés előtti és utáni mondatonkénti tokenszám hisztogramját ábrázolja.
Az elvárásoknak megfelelően
az alulmintavételezés jelentősen lerövidíti a mondatokat
a magas frekvenciájú szavak elhagyásával,
ami gyorsabb tanításhoz vezet.

```{.python .input}
#@tab all
d2l.show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence',
                            'count', sentences, subsampled);
```

Az egyes tokenek esetén a magas frekvenciájú „the" szó mintavételi aránya kisebb, mint 1/20.

```{.python .input}
#@tab all
def compare_counts(token):
    return (f'# of "{token}": '
            f'before={sum([l.count(token) for l in sentences])}, '
            f'after={sum([l.count(token) for l in subsampled])}')

compare_counts('the')
```

Ezzel szemben
az alacsony frekvenciájú „join" szavakat teljesen megőrizzük.

```{.python .input}
#@tab all
compare_counts('join')
```

Az alulmintavételezés után a tokeneket indexekre képezzük le a korpuszban.

```{.python .input}
#@tab all
corpus = [vocab[line] for line in subsampled]
corpus[:3]
```

## Középső Szavak és Kontextusszavak Kinyerése


A következő `get_centers_and_contexts`
függvény kinyeri az összes
középső szót és azok kontextusszavait
a `corpus`-ból.
Véletlenszerűen, egyenletes eloszlással mintavételez egy 1 és `max_window_size` közötti egész számot
a kontextusablak méretéként.
Bármely középső szóhoz
azok a szavak,
amelyek tőle való távolsága
nem haladja meg a mintavételezett
kontextusablak méretét,
kontextusszavak.

```{.python .input}
#@tab all
#@save
def get_centers_and_contexts(corpus, max_window_size):
    """A skip-gram középső szavait és kontextusszavait adja vissza."""
    centers, contexts = [], []
    for line in corpus:
        # A "középső szó--kontextusszó" pár alkotásához minden mondatnak
        # legalább 2 szóból kell állnia
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Az `i`-re középpontosított kontextusablak
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # A középső szó kizárása a kontextusszavakból
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
```

Ezután létrehozunk egy mesterséges adathalmazt, amely 7 illetve 3 szóból álló két mondatot tartalmaz.
Legyen a maximális kontextusablak mérete 2,
és nyomtassuk ki az összes középső szót és azok kontextusszavait.

```{.python .input}
#@tab all
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)
```

A PTB adathalmazon való tanításkor
a maximális kontextusablak méretet 5-re állítjuk.
A következő kód kinyeri az adathalmazban lévő összes középső szót és azok kontextusszavait.

```{.python .input}
#@tab all
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# center-context pairs: {sum([len(contexts) for contexts in all_contexts])}'
```

## Negatív Mintavételezés

Negatív mintavételezést alkalmazunk a közelítő tanításhoz.
A zajszavaknak egy előre meghatározott eloszlás szerinti mintavételezéséhez
definiáljuk a következő `RandomGenerator` osztályt,
ahol a (esetleg nem normalizált) mintavételezési eloszlást
a `sampling_weights` argumentumon keresztül adjuk meg.

```{.python .input}
#@tab all
#@save
class RandomGenerator:
    """Véletlenszerűen húz az {1, ..., n} halmazból n mintavételezési súly szerint."""
    def __init__(self, sampling_weights):
        # Kizárás
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # `k` véletlen mintavételezési eredmény gyorsítótárazása
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]
```

Például
10 $X$ véletlen változót húzhatunk az 1, 2 és 3 indexek közül
$P(X=1)=2/9, P(X=2)=3/9$ és $P(X=3)=4/9$ mintavételezési valószínűségekkel az alábbiak szerint.

```{.python .input}
#@tab mxnet
generator = RandomGenerator([2, 3, 4])
[generator.draw() for _ in range(10)]
```

Egy középső szó és kontextusszó párhoz
véletlenszerűen mintavételezünk `K` (a kísérletben 5) zajszót. A word2vec cikk javaslata szerint
a $w$ zajszó $P(w)$ mintavételezési valószínűsége
a szótárban lévő relatív frekvenciájának
0,75-ös hatványára van beállítva :cite:`Mikolov.Sutskever.Chen.ea.2013`.

```{.python .input}
#@tab all
#@save
def get_negatives(all_contexts, vocab, counter, K):
    """A negatív mintavételezés zajszavait adja vissza."""
    # Az 1, 2, ... indexű szavak mintavételezési súlyai (a 0. index a
    # kizárt ismeretlen token) a szókincsben
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # A zajszavak nem lehetnek kontextusszavak
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, vocab, counter, 5)
```

## Tanítási Példák Betöltése mini-batch-ekben
:label:`subsec_word2vec-mini-batch-loading`

Miután
az összes középső szót
a kontextusszavaikkal és a mintavételezett zajszavaikkal együtt kinyertük,
ezeket mini-batch példákká alakítjuk,
amelyek tanítás közben iteratívan tölthetők be.



Egy mini-batch-ben
az $i$-ik példa egy középső szót,
annak $n_i$ kontextusszavát és $m_i$ zajszavát tartalmazza.
A változó kontextusablak-méretek miatt
$n_i+m_i$ eltérő lehet különböző $i$-knél.
Ezért
minden példánál összefűzzük a kontextusszavait és zajszavait
a `contexts_negatives` változóban,
és nullákkal tömítjük ki, amíg az összefűzés hossza
el nem éri a $\max_i n_i+m_i$ értéket (`max_len`).
A tömítések kizárásához
a veszteség számításából,
definiálunk egy `masks` maszk változót.
Egy-egy megfeleltetés áll fenn
a `masks` elemei és a `contexts_negatives` elemei között,
ahol a `masks`-ban lévő nullák (máskülönben egyek) a `contexts_negatives`-ban lévő tömítéseknek felelnek meg.


A pozitív és negatív példák megkülönböztetéséhez
a `contexts_negatives`-ban elkülönítjük a kontextusszavakat a zajszavaktól egy `labels` változón keresztül.
A `masks`-hoz hasonlóan
szintén egy-egy megfeleltetés áll fenn
a `labels` elemei és a `contexts_negatives` elemei között,
ahol a `labels`-ban lévő egyek (máskülönben nullák) a `contexts_negatives`-ban lévő kontextusszavaknak (pozitív példáknak) felelnek meg.


A fenti ötletet a következő `batchify` függvény valósítja meg.
Bemenete a `data`, egy lista, amelynek hossza
egyenlő a batch mérettel,
ahol minden elem egy példa,
amely a `center` középső szóból, annak `context` kontextusszavaiból és `negative` zajszavaiból áll.
Ez a függvény visszaad
egy mini-batch-et, amely betölthető tanítás közbeni számításokhoz,
beleértve a maszk változót.

```{.python .input}
#@tab all
#@save
def batchify(data):
    """Egy minibatch példát ad vissza skip-gram negatív mintavételezéssel."""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (d2l.reshape(d2l.tensor(centers), (-1, 1)), d2l.tensor(
        contexts_negatives), d2l.tensor(masks), d2l.tensor(labels))
```

Teszteljük ezt a függvényt két példából álló mini-batch-csel.

```{.python .input}
#@tab all
x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)
```

## Mindent Összerakva

Végül definiáljuk a `load_data_ptb` függvényt, amely beolvassa a PTB adathalmazt, és visszaadja az adatiteratort és a szókincset.

```{.python .input}
#@tab mxnet
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Letölti a PTB adathalmazt, majd betölti a memóriába."""
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)
    dataset = gluon.data.ArrayDataset(
        all_centers, all_contexts, all_negatives)
    data_iter = gluon.data.DataLoader(
        dataset, batch_size, shuffle=True,batchify_fn=batchify,
        num_workers=d2l.get_dataloader_workers())
    return data_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Letölti a PTB adathalmazt, majd betölti a memóriába."""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                      collate_fn=batchify,
                                      num_workers=num_workers)
    return data_iter, vocab
```

Nyomtassuk ki az adatiterator első mini-batch-jét.

```{.python .input}
#@tab all
data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

## Összefoglalás

* A magas frekvenciájú szavak nem feltétlenül hasznosak a tanításban. Alulmintavételezhetjük őket a tanítás gyorsítása érdekében.
* A számítási hatékonyság érdekében a példákat mini-batch-ekben töltjük be. Más változókat definiálhatunk a tömítések és nem tömítések, valamint a pozitív és negatív példák megkülönböztetéséhez.



## Gyakorló feladatok

1. Hogyan változik a szakasz kódjának futási ideje, ha nem alkalmazunk alulmintavételezést?
1. A `RandomGenerator` osztály `k` véletlen mintavételezési eredményt tárol. Állítsd `k`-t más értékre, és figyeld meg, hogyan befolyásolja az adatbetöltési sebességet.
1. Milyen más hiperparaméterek befolyásolhatják a szakasz kódjában az adatbetöltési sebességet?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/383)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1330)
:end_tab:
