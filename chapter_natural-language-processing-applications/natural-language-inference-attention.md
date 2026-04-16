# Természetes nyelvi inferencia: Figyelem alkalmazásával
:label:`sec_natural-language-inference-attention`

A természetes nyelvi inferencia feladatát és az SNLI adathalmazt a :numref:`sec_natural-language-inference-and-dataset` szakaszban mutattuk be. Számos összetett és mély architektúrán alapuló modell ismeretében :citet:`Parikh.Tackstrom.Das.ea.2016` figyelemmechanizmusok segítségével javasolta a természetes nyelvi inferencia megoldását, és ezt a megközelítést „lebontható figyelemmodellnek" (decomposable attention model) nevezte.
Ez egy olyan modellt eredményez, amely rekurrens vagy konvolúciós rétegek nélkül éri el a legjobb akkori eredményt az SNLI adathalmazon, jóval kevesebb paraméterrel.
Ebben a szakaszban ismertetjük és implementáljuk ezt a figyelemalapú módszert (MLP-kkel) a természetes nyelvi inferencia számára, ahogyan az :numref:`fig_nlp-map-nli-attention` ábrán látható.

![Ez a szakasz előre tanított GloVe-ot táplál egy figyelem- és MLP-alapú architektúrába a természetes nyelvi inferencia céljából.](../img/nlp-map-nli-attention.svg)
:label:`fig_nlp-map-nli-attention`


## A modell

A tokenek sorrendjének megőrzésénél egyszerűbb megközelítéssel
illeszthetjük az egyik szövegszekvencia tokenjeit a másik szekvencia minden egyes tokenjéhez, és fordítva,
majd összehasonlíthatjuk és összesíthetjük ezeket az információkat a premisszák és hipotézisek közötti logikai összefüggések előrejelzéséhez.
Hasonlóan a forrás- és célmondatok közötti tokenillesztéshez a gépi fordításban,
a premisszák és hipotézisek közötti tokenillesztés
figyelemmechanizmusokkal elegánsan megvalósítható.

![Természetes nyelvi inferencia figyelemmechanizmusok segítségével.](../img/nli-attention.svg)
:label:`fig_nli_attention`

A :numref:`fig_nli_attention` ábra a figyelemmechanizmusokat alkalmazó természetes nyelvi inferencia módszert szemlélteti.
Magas szinten három együttesen tanított lépésből áll: odafigyelés, összehasonlítás és összesítés.
A következőkben ezeket lépésről lépésre ismertetjük.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

### Odafigyelés

Az első lépés az egyik szövegszekvencia tokenjeit a másik szekvencia minden egyes tokenjéhez igazítani.
Tegyük fel, hogy a premissza „i do need sleep", a hipotézis pedig „i am tired".
Szemantikai hasonlóság alapján
érdemes lehet a hipotézisben lévő „i"-t a premissza „i"-jével illeszteni,
és a hipotézisbeli „tired"-et a premissza „sleep" szavával illeszteni.
Hasonlóképpen, a premissza „i"-jét a hipotézis „i"-jével érdemes illeszteni,
a premissza „need" és „sleep" szavait pedig a hipotézis „tired" szavával.
Megjegyezzük, hogy ez az illesztés *puha* súlyozott átlag segítségével,
ahol ideális esetben nagy súlyok tartoznak az illesztendő tokenekhez.
A szemléltetés kedvéért a :numref:`fig_nli_attention` ábra *kemény* módon mutatja be ezt az illesztést.

Most részletesebben leírjuk a puha illesztést figyelemmechanizmusok segítségével.
Jelölje $\mathbf{A} = (\mathbf{a}_1, \ldots, \mathbf{a}_m)$
és $\mathbf{B} = (\mathbf{b}_1, \ldots, \mathbf{b}_n)$ a premisszát és a hipotézist,
amelyek tokeneinek száma rendre $m$ és $n$,
ahol $\mathbf{a}_i, \mathbf{b}_j \in \mathbb{R}^{d}$ ($i = 1, \ldots, m, j = 1, \ldots, n$) egy $d$-dimenziós szóvektor.
A puha illesztéshez az $e_{ij} \in \mathbb{R}$ figyelmi súlyokat a következőképpen számítjuk:

$$e_{ij} = f(\mathbf{a}_i)^\top f(\mathbf{b}_j),$$
:eqlabel:`eq_nli_e`

ahol az $f$ függvény egy MLP, amelyet az alábbi `mlp` függvénnyel definiálunk.
Az $f$ kimeneti dimenziója az `mlp` `num_hiddens` argumentumával adható meg.

```{.python .input}
#@tab mxnet
def mlp(num_hiddens, flatten):
    net = nn.Sequential()
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    return net
```

```{.python .input}
#@tab pytorch
def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)
```

Fontos kiemelni, hogy a :eqref:`eq_nli_e` képletben
az $f$ az $\mathbf{a}_i$ és $\mathbf{b}_j$ bemeneteket külön-külön dolgozza fel, nem párban együtt.
Ez a *lebontási* trükk az $f$ $m + n$ alkalmazásához (lineáris komplexitás) vezet az $mn$ alkalmazás
(négyzetes komplexitás) helyett.


A :eqref:`eq_nli_e` figyelmi súlyok normalizálásával
kiszámítjuk a hipotézis összes tokenvektorának súlyozott átlagát,
hogy megkapjuk a hipotézis azon reprezentációját, amely puhán illeszkedik a premissza $i$-edik tokenjéhez:

$$
\boldsymbol{\beta}_i = \sum_{j=1}^{n}\frac{\exp(e_{ij})}{ \sum_{k=1}^{n} \exp(e_{ik})} \mathbf{b}_j.
$$

Hasonlóképpen, kiszámítjuk a premissza tokeneinek puha illesztését a hipotézis $j$-edik tokenjéhez:

$$
\boldsymbol{\alpha}_j = \sum_{i=1}^{m}\frac{\exp(e_{ij})}{ \sum_{k=1}^{m} \exp(e_{kj})} \mathbf{a}_i.
$$

Az alábbiakban definiáljuk az `Attend` osztályt, amely kiszámítja a hipotézisek puha illesztését (`beta`) a bemeneti premisszákhoz `A`, valamint a premisszák puha illesztését (`alpha`) a bemeneti hipotézisekhez `B`.

```{.python .input}
#@tab mxnet
class Attend(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B):
        # Az `A`/`B` alakja: (`batch_size`, az A/B szekvencia tokeneinek
        # száma, `embed_size`)
        # Az `f_A`/`f_B` alakja: (`batch_size`, az A/B szekvencia
        # tokeneinek száma, `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Az `e` alakja: (`batch_size`, az A szekvencia tokeneinek száma,
        # a B szekvencia tokeneinek száma)
        e = npx.batch_dot(f_A, f_B, transpose_b=True)
        # A `beta` alakja: (`batch_size`, az A szekvencia tokeneinek száma,
        # `embed_size`), ahol a B szekvencia puhán illeszkedik az A
        # szekvencia minden tokenjéhez (`beta` 1. tengelye)
        beta = npx.batch_dot(npx.softmax(e), B)
        # Az `alpha` alakja: (`batch_size`, a B szekvencia tokeneinek
        # száma, `embed_size`), ahol az A szekvencia puhán illeszkedik a B
        # szekvencia minden tokenjéhez (`alpha` 1. tengelye)
        alpha = npx.batch_dot(npx.softmax(e.transpose(0, 2, 1)), A)
        return beta, alpha
```

```{.python .input}
#@tab pytorch
class Attend(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B):
        # Az `A`/`B` alakja: (`batch_size`, az A/B szekvencia tokeneinek
        # száma, `embed_size`)
        # Az `f_A`/`f_B` alakja: (`batch_size`, az A/B szekvencia
        # tokeneinek száma, `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Az `e` alakja: (`batch_size`, az A szekvencia tokeneinek száma,
        # a B szekvencia tokeneinek száma)
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        # A `beta` alakja: (`batch_size`, az A szekvencia tokeneinek száma,
        # `embed_size`), ahol a B szekvencia puhán illeszkedik az A
        # szekvencia minden tokenjéhez (`beta` 1. tengelye)
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # Az `alpha` alakja: (`batch_size`, a B szekvencia tokeneinek
        # száma, `embed_size`), ahol az A szekvencia puhán illeszkedik a B
        # szekvencia minden tokenjéhez (`alpha` 1. tengelye)
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha
```

### Összehasonlítás

A következő lépésben összehasonlítjuk az egyik szekvencia egy tokenjét a másik szekvenciával, amely puhán illeszkedik az adott tokenhez.
Megjegyezzük, hogy a puha illesztés során az egyik szekvencia összes tokenje – habár valószínűleg különböző figyelmi súlyokkal – összehasonlításra kerül a másik szekvencia egy tokenjével.
A szemléltetés érdekében a :numref:`fig_nli_attention` ábra *kemény* módon párosítja a tokeneket az illesztett tokenekkel.
Például tegyük fel, hogy az odafigyelési lépés meghatározza, hogy a premissza „need" és „sleep" szavai mindkettő a hipotézis „tired" szavával illeszkednek; ekkor a „tired--need sleep" pár kerül összehasonlításra.

Az összehasonlítási lépésben az egyik szekvencia tokenjeit és a másik szekvencia illesztett tokenjeit összekapcsoljuk (a $[\cdot, \cdot]$ operátorral), majd egy $g$ függvénybe (egy MLP-be) táplálva:

$$\mathbf{v}_{A,i} = g([\mathbf{a}_i, \boldsymbol{\beta}_i]), i = 1, \ldots, m\\ \mathbf{v}_{B,j} = g([\mathbf{b}_j, \boldsymbol{\alpha}_j]), j = 1, \ldots, n.$$

:eqlabel:`eq_nli_v_ab`


A :eqref:`eq_nli_v_ab` képletben a $\mathbf{v}_{A,i}$ a premissza $i$-edik tokenjének összehasonlítása az összes olyan hipotézistokennel, amelyek puhán illeszkednek az $i$-edik tokenhez;
míg $\mathbf{v}_{B,j}$ a hipotézis $j$-edik tokenjének összehasonlítása az összes olyan premisszatokennel, amelyek puhán illeszkednek a $j$-edik tokenhez.
Az alábbi `Compare` osztály definiálja ezt az összehasonlítási lépést.

```{.python .input}
#@tab mxnet
class Compare(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(np.concatenate([A, beta], axis=2))
        V_B = self.g(np.concatenate([B, alpha], axis=2))
        return V_A, V_B
```

```{.python .input}
#@tab pytorch
class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B
```

### Összesítés

Miután rendelkezésünkre áll a két összehasonlító vektorkészlet, $\mathbf{v}_{A,i}$ ($i = 1, \ldots, m$) és $\mathbf{v}_{B,j}$ ($j = 1, \ldots, n$),
az utolsó lépésben összesítjük ezeket az információkat a logikai összefüggés következtetéséhez.
Kezdjük mindkét készlet összegzésével:

$$
\mathbf{v}_A = \sum_{i=1}^{m} \mathbf{v}_{A,i}, \quad \mathbf{v}_B = \sum_{j=1}^{n}\mathbf{v}_{B,j}.
$$

Ezután mindkét összegzési eredmény konkatenációját egy $h$ függvénybe (MLP-be) táplálva megkapjuk a logikai összefüggés osztályozási eredményét:

$$
\hat{\mathbf{y}} = h([\mathbf{v}_A, \mathbf{v}_B]).
$$

Az összesítési lépést az alábbi `Aggregate` osztály definiálja.

```{.python .input}
#@tab mxnet
class Aggregate(nn.Block):
    def __init__(self, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_hiddens=num_hiddens, flatten=True)
        self.h.add(nn.Dense(num_outputs))

    def forward(self, V_A, V_B):
        # Mindkét összehasonlítóvektor-készlet összegzése
        V_A = V_A.sum(axis=1)
        V_B = V_B.sum(axis=1)
        # Mindkét összegzési eredmény konkatenációjának betáplálása egy MLP-be
        Y_hat = self.h(np.concatenate([V_A, V_B], axis=1))
        return Y_hat
```

```{.python .input}
#@tab pytorch
class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):
        # Mindkét összehasonlítóvektor-készlet összegzése
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        # Mindkét összegzési eredmény konkatenációjának betáplálása egy MLP-be
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat
```

### Összerakás

Az odafigyelési, összehasonlítási és összesítési lépések összekapcsolásával
definiáljuk a lebontható figyelemmodellt, amely együttesen tanítja e három lépést.

```{.python .input}
#@tab mxnet
class DecomposableAttention(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_hiddens)
        self.compare = Compare(num_hiddens)
        # 3 lehetséges kimenet van: következmény, ellentmondás és semleges
        self.aggregate = Aggregate(num_hiddens, 3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

```{.python .input}
#@tab pytorch
class DecomposableAttention(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100,
                 num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        # 3 lehetséges kimenet van: következmény, ellentmondás és semleges
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

## A modell tanítása és kiértékelése

Most tanítjuk és kiértékeljük a definiált lebontható figyelemmodellt az SNLI adathalmazon.
Kezdjük az adathalmaz beolvasásával.


### Az adathalmaz beolvasása

A :numref:`sec_natural-language-inference-and-dataset` szakaszban definiált függvénnyel töltjük le és olvassuk be az SNLI adathalmazt. A batch mérete és a szekvencia hossza rendre $256$ és $50$.

```{.python .input}
#@tab all
batch_size, num_steps = 256, 50
train_iter, test_iter, vocab = d2l.load_data_snli(batch_size, num_steps)
```

### A modell létrehozása

A bemeneti tokenek reprezentálásához az előre tanított 100-dimenziós GloVe beágyazást használjuk.
Ezért előre meghatározzuk a $\mathbf{a}_i$ és $\mathbf{b}_j$ vektorok dimenzióját a :eqref:`eq_nli_e` képletben 100-ként.
A :eqref:`eq_nli_e` $f$ és a :eqref:`eq_nli_v_ab` $g$ függvények kimeneti dimenziója 200-ra van beállítva.
Ezután létrehozunk egy modellpéldányt, inicializáljuk paramétereit,
és betöltjük a GloVe beágyazást a bemeneti tokenek vektorainak inicializálásához.

```{.python .input}
#@tab mxnet
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
net.initialize(init.Xavier(), ctx=devices)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
```

```{.python .input}
#@tab pytorch
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds);
```

### A modell tanítása és kiértékelése

A :numref:`sec_multi_gpu` `split_batch` függvényével ellentétben, amely egyetlen bemenetet vesz fel, például szövegszekvenciákat (vagy képeket),
definiálunk egy `split_batch_multi_inputs` függvényt, amely több bemenetet kezel, például premisszákat és hipotéziseket mini-batchekben.

```{.python .input}
#@tab mxnet
#@save
def split_batch_multi_inputs(X, y, devices):
    """A többbemenetes `X` és `y` felosztása több eszközre."""
    X = list(zip(*[gluon.utils.split_and_load(
        feature, devices, even_split=False) for feature in X]))
    return (X, gluon.utils.split_and_load(y, devices, even_split=False))
```

Most taníthatjuk és kiértékelhetjük a modellt az SNLI adathalmazon.

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.001, 4
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               split_batch_multi_inputs)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.001, 4
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

### A modell használata

Végül definiáljuk az előrejelző függvényt, amely egy premissza-hipotézis pár logikai összefüggését adja meg.

```{.python .input}
#@tab mxnet
#@save
def predict_snli(net, vocab, premise, hypothesis):
    """A premissza és a hipotézis közötti logikai összefüggés előrejelzése."""
    premise = np.array(vocab[premise], ctx=d2l.try_gpu())
    hypothesis = np.array(vocab[hypothesis], ctx=d2l.try_gpu())
    label = np.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), axis=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

```{.python .input}
#@tab pytorch
#@save
def predict_snli(net, vocab, premise, hypothesis):
    """A premissza és a hipotézis közötti logikai összefüggés előrejelzése."""
    net.eval()
    premise = torch.tensor(vocab[premise], device=d2l.try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=d2l.try_gpu())
    label = torch.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

A tanított modell segítségével megkaphatjuk egy mintamondatpár természetes nyelvi inferencia eredményét.

```{.python .input}
#@tab all
predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.'])
```

## Összefoglalás

* A lebontható figyelemmodell három lépésből áll a premisszák és hipotézisek közötti logikai összefüggések előrejelzéséhez: odafigyelés, összehasonlítás és összesítés.
* Figyelemmechanizmusok segítségével az egyik szövegszekvencia tokenjeit illeszthetjük a másik szekvencia minden egyes tokenjéhez, és fordítva. Ez az illesztés puha, súlyozott átlag révén, ahol ideális esetben nagy súlyok tartoznak az illesztendő tokenekhez.
* A lebontási trükk a figyelmi súlyok számításakor kívánatosabb lineáris komplexitáshoz vezet a négyzetes komplexitás helyett.
* Előre tanított szóvektorokat használhatunk bemeneti reprezentációként az olyan downstream természetes nyelvfeldolgozási feladatokhoz, mint a természetes nyelvi inferencia.


## Feladatok

1. Tanítsd a modellt más hiperparaméter-kombinációkkal. Elérsz-e jobb pontosságot a teszthalmazon?
1. Melyek a lebontható figyelemmodell fő hiányosságai a természetes nyelvi inferencia szempontjából?
1. Tegyük fel, hogy bármely mondatpár szemantikai hasonlóságának szintjét szeretnénk meghatározni (pl. 0 és 1 közötti folytonos értékként). Hogyan gyűjtsük és címkézzük az adathalmazt? Tudsz-e tervezni egy figyelemmechanizmusokat alkalmazó modellt?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/395)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1530)
:end_tab:
