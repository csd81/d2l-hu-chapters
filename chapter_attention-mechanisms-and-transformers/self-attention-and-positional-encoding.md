```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Önfigyelem és pozíciókódolás
:label:`sec_self-attention-and-positional-encoding`

A mélytanulásban gyakran CNN-eket vagy RNN-eket használunk szekvenciák kódolásához.
Most a figyelemmechanizmusokat szem előtt tartva,
képzeljük el, hogy egy tokenekből álló szekvenciát táplálunk
egy figyelemmechanizmusba
úgy, hogy minden lépésben
minden tokennek megvan a saját lekérdezése, kulcsai és értékei.
Itt, amikor kiszámítjuk egy token reprezentációjának értékét a következő rétegben,
a token (a lekérdezésvektorán keresztül) figyelhet bármely másik token felé
(a kulcsvektor alapján megvalósuló egyezés szerint).
A lekérdezés-kulcs kompatibilitási pontszámok teljes halmazát felhasználva
minden tokenre kiszámíthatunk egy reprezentációt
a többi token megfelelő súlyozott összegének felépítésével.
Mivel minden token figyel minden más tokenre
(ellentétben azzal az esettel, amikor a dekódoló lépések a kódoló lépéseire figyelnek),
az ilyen architektúrákat általában *önfigyelmi* modelleknek nevezik :cite:`Lin.Feng.Santos.ea.2017,Vaswani.Shazeer.Parmar.ea.2017`,
máshol *intrafigyelmi* modelleknek :cite:`Cheng.Dong.Lapata.2016,Parikh.Tackstrom.Das.ea.2016,Paulus.Xiong.Socher.2017`.
Ebben a szakaszban az önfigyelmet használó szekvenciakódolást tárgyaljuk,
beleértve a szekvencia sorrendjére vonatkozó kiegészítő információ felhasználását.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
import math
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
```

## **Önfigyelem**

Adott egy $\mathbf{x}_1, \ldots, \mathbf{x}_n$ bemeneti token szekvenciája, ahol bármely $\mathbf{x}_i \in \mathbb{R}^d$ ($1 \leq i \leq n$),
az önfigyelem kimenetként egy ugyanolyan hosszúságú
$\mathbf{y}_1, \ldots, \mathbf{y}_n$ szekvenciát produkál,
ahol

$$\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d$$

a :eqref:`eq_attention_pooling`-ban lévő figyelempooling definíciója szerint.
Többfejű figyelmet alkalmazva,
a következő kódrészlet
kiszámítja egy tenzor önfigyelmét
(batch méret, időlépések száma vagy szekvenciahossz tokenekben, $d$) alakban.
A kimeneti tenzor azonos alakú.

```{.python .input}
%%tab pytorch
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab mxnet
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()
```

```{.python .input}
%%tab jax
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
```

```{.python .input}
%%tab tensorflow
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
```

```{.python .input}
%%tab mxnet
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab tensorflow
batch_size, num_queries, valid_lens = 2, 4, tf.constant([3, 2])
X = tf.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens, training=False),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab jax
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention.init_with_output(d2l.get_key(), X, X, X, valid_lens,
                                           training=False)[0][0],
                (batch_size, num_queries, num_hiddens))
```

## CNN-ek, RNN-ek és Önfigyelem Összehasonlítása
:label:`subsec_cnn-rnn-self-attention`

Hasonlítsuk össze az architektúrákat $n$ token szekvenciájának
ugyanolyan hosszúságú másikba való leképezéséhez,
ahol minden bemeneti vagy kimeneti tokent
egy $d$-dimenziós vektor reprezentál.
Konkrétan
CNN-eket, RNN-eket és önfigyelmet fogunk vizsgálni.
Összehasonlítjuk a
számítási bonyolultságukat,
a szekvenciális műveletek számát,
és a maximális úthosszakat.
Vegyük észre, hogy a szekvenciális műveletek akadályozzák a párhuzamos számítást,
míg a rövidebb útvonal
a szekvenciapozíciók bármely kombinációja között
megkönnyíti a hosszú hatótávolságú függőségek megtanulását
a szekvencián belül :cite:`Hochreiter.Bengio.Frasconi.ea.2001`.


![CNN (a kitöltő tokenek kihagyva), RNN és önfigyelmi architektúrák összehasonlítása.](../img/cnn-rnn-self-attention.svg)
:label:`fig_cnn-rnn-self-attention`



Tekintsünk minden szöveges szekvenciát „egydimenziós képként". Hasonlóképpen, egydimenziós CNN-ek feldolgozhatnak lokális jellemzőket, mint például az $n$-gramokat a szövegben.
Adott egy $n$ hosszúságú szekvencia esetén
tekintsünk egy olyan konvolúciós réteget, amelynek kernel-mérete $k$,
és amelynek bemeneti és kimeneti csatornáinak száma egyaránt $d$.
A konvolúciós réteg számítási bonyolultsága $\mathcal{O}(knd^2)$.
Ahogy a :numref:`fig_cnn-rnn-self-attention` mutatja,
a CNN-ek hierarchikusak,
tehát $\mathcal{O}(1)$ szekvenciális műveletek vannak,
és a maximális úthossz $\mathcal{O}(n/k)$.
Például $\mathbf{x}_1$ és $\mathbf{x}_5$
egy kétréteges CNN befogadómezőjén belül vannak
3-as kernel-mérettel a :numref:`fig_cnn-rnn-self-attention`-ban.

Az RNN-ek rejtett állapotának frissítésekor
a $d \times d$ súlymátrix
és a $d$-dimenziós rejtett állapot szorzásának
számítási bonyolultsága $\mathcal{O}(d^2)$.
Mivel a szekvenciahossz $n$,
a rekurrens réteg számítási bonyolultsága
$\mathcal{O}(nd^2)$.
A :numref:`fig_cnn-rnn-self-attention` szerint
$\mathcal{O}(n)$ szekvenciális művelet van,
amely nem párhuzamosítható,
és a maximális úthossz is $\mathcal{O}(n)$.

Az önfigyelemben
a lekérdezések, kulcsok és értékek
mind $n \times d$ mátrixok.
Tekintsük a skálázott skaláris szorzat figyelmet a
:eqref:`eq_softmax_QK_V`-ben,
ahol egy $n \times d$ mátrixot megszorzunk
egy $d \times n$ mátrixszal,
majd a kimeneti $n \times n$ mátrixot megszorozzák
egy $n \times d$ mátrixszal.
Ennek eredményeként
az önfigyelemnek $\mathcal{O}(n^2d)$ a számítási bonyolultsága.
Ahogy a :numref:`fig_cnn-rnn-self-attention`-ból látható,
minden token közvetlenül kapcsolódik
bármely más tokenhez az önfigyelmen keresztül.
Ezért
a számítás párhuzamos lehet $\mathcal{O}(1)$ szekvenciális művelettel
és a maximális úthossz is $\mathcal{O}(1)$.

Összességében
mind a CNN-ek, mind az önfigyelem párhuzamos számítást élveznek,
és az önfigyelemnek van a legrövidebb maximális úthossza.
Azonban a szekvenciahosszhoz képest négyzetes számítási bonyolultság
az önfigyelmet tiltóan lassúvá teszi nagyon hosszú szekvenciák esetén.


## **pozíciókódolás**
:label:`subsec_positional-encoding`


Az RNN-ektől eltérően, amelyek rekurzívan dolgozzák fel
a szekvencia tokenjeit egyenként,
az önfigyelem elveti
a szekvenciális műveleteket a
párhuzamos számítás javára.
Vegyük észre, hogy az önfigyelem önmagában
nem őrzi meg a szekvencia sorrendjét.
Mit tegyünk, ha valóban fontos,
hogy a modell tudja, milyen sorrendben
érkezett a bemeneti szekvencia?

A tokenek sorrendjével kapcsolatos
információk megőrzésének domináns megközelítése
az, hogy ezt a modell számára
minden tokenhez társított
kiegészítő bemenetként reprezentáljuk.
Ezeket a bemeneteket *pozíciókódolásoknak* nevezik,
és megtanulhatók vagy *a priori* rögzíthetők.
Most egy egyszerű sémát írunk le a rögzített pozíciókódolásokhoz
szinusz és koszinusz függvényeken alapulva :cite:`Vaswani.Shazeer.Parmar.ea.2017`.

Tételezzük fel, hogy a
$\mathbf{X} \in \mathbb{R}^{n \times d}$ bemeneti reprezentáció
egy szekvencia $n$ tokenjének $d$-dimenziós beágyazásait tartalmazza.
A pozíciókódolás kimenete
$\mathbf{X} + \mathbf{P}$
egy azonos alakú pozícióbeágyazási mátrixot
$\mathbf{P} \in \mathbb{R}^{n \times d}$ felhasználva,
amelynek eleme az $i$-edik sorban
és a $(2j)$-edik
vagy a $(2j + 1)$-edik oszlopban:

$$\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right),\\p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right).\end{aligned}$$
:eqlabel:`eq_positional-encoding-def`

Első pillantásra
ez a trigonometrikus függvény-tervezés furcsának tűnik.
Mielőtt magyarázatot adnánk erre a tervezésre,
implementáljuk először a következő `PositionalEncoding` osztályban.

```{.python .input}
%%tab mxnet
class PositionalEncoding(nn.Block):  #@save
    """Pozícióenkódolás."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Hozzunk létre egy elég hosszú P-t
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].as_in_ctx(X.ctx)
        return self.dropout(X)
```

```{.python .input}
%%tab pytorch
class PositionalEncoding(nn.Module):  #@save
    """Pozícióenkódolás."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Hozzunk létre egy elég hosszú P-t
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
```

```{.python .input}
%%tab tensorflow
class PositionalEncoding(tf.keras.layers.Layer):  #@save
    """Pozícióenkódolás."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        # Hozzunk létre egy elég hosszú P-t
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len, dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, num_hiddens, 2, dtype=np.float32) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)
        
    def call(self, X, **kwargs):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X, **kwargs)
```

```{.python .input}
%%tab jax
class PositionalEncoding(nn.Module):  #@save
    """Pozícióenkódolás."""
    num_hiddens: int
    dropout: float
    max_len: int = 1000

    def setup(self):
        # Hozzunk létre egy elég hosszú P-t
        self.P = d2l.zeros((1, self.max_len, self.num_hiddens))
        X = d2l.arange(self.max_len, dtype=jnp.float32).reshape(
            -1, 1) / jnp.power(10000, jnp.arange(
            0, self.num_hiddens, 2, dtype=jnp.float32) / self.num_hiddens)
        self.P = self.P.at[:, :, 0::2].set(jnp.sin(X))
        self.P = self.P.at[:, :, 1::2].set(jnp.cos(X))

    @nn.compact
    def __call__(self, X, training=False):
        # A Flax sow API köztes változók rögzítésére szolgál
        self.sow('intermediates', 'P', self.P)
        X = X + self.P[:, :X.shape[1], :]
        return nn.Dropout(self.dropout)(X, deterministic=not training)
```

A pozícióbeágyazási mátrixban $\mathbf{P}$,
**a sorok a szekvencián belüli pozícióknak felelnek meg,
az oszlopok különböző pozíciókódolási dimenziókat reprezentálnak**.
Az alábbi példában láthatjuk,
hogy a pozícióbeágyazási mátrix
$6.$ és $7.$ oszlopának
magasabb frekvenciája van, mint a
$8.$ és $9.$ oszlopnak.
A $6.$ és $7.$ oszlop közötti eltolás (ugyanez érvényes a $8.$ és $9.$ oszlopra)
a szinusz és koszinusz függvények váltakozásának köszönhető.

```{.python .input}
%%tab mxnet
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.initialize()
X = pos_encoding(np.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
%%tab pytorch
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(d2l.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
%%tab tensorflow
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(tf.zeros((1, num_steps, encoding_dim)), training=False)
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(np.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in np.arange(6, 10)])
```

```{.python .input}
%%tab jax
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
params = pos_encoding.init(d2l.get_key(), d2l.zeros((1, num_steps, encoding_dim)))
X, inter_vars = pos_encoding.apply(params, d2l.zeros((1, num_steps, encoding_dim)),
                                   mutable='intermediates')
P = inter_vars['intermediates']['P'][0]  # retrieve intermediate value P
P = P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

### Abszolút pozicionális információ

Annak megértéséhez, hogy a kódolási dimenzió mentén monoton csökkenő frekvencia hogyan kapcsolódik az abszolút pozicionális információhoz,
nyomtassuk ki $0, 1, \ldots, 7$ **bináris reprezentációit**.
Amint láthatjuk, a legalacsonyabb bit, a második legalacsonyabb bit,
és a harmadik legalacsonyabb bit felváltva fordulnak elő minden számnál,
minden két számnál, illetve minden négy számnál.

```{.python .input}
%%tab all
for i in range(8):
    print(f'{i} in binary is {i:>03b}')
```

A bináris reprezentációkban a magasabb bit
alacsonyabb frekvenciájú, mint egy alacsonyabb bit.
Hasonlóképpen, ahogyan az alábbi hőtérképen látható,
**a pozíciókódolás csökkenti
a frekvenciákat a kódolási dimenzió mentén**
trigonometrikus függvények segítségével.
Mivel a kimenetek lebegőpontos számok,
az ilyen folytonos reprezentációk
térkihasználás szempontjából hatékonyabbak
a bináris reprezentációknál.

```{.python .input}
%%tab mxnet
P = np.expand_dims(np.expand_dims(P[0, :, :], 0), 0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
%%tab pytorch
P = P[0, :, :].unsqueeze(0).unsqueeze(0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
%%tab tensorflow
P = tf.expand_dims(tf.expand_dims(P[0, :, :], axis=0), axis=0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
%%tab jax
P = jnp.expand_dims(jnp.expand_dims(P[0, :, :], axis=0), axis=0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

### Relatív pozicionális információ

Az abszolút pozicionális információ megragadásán túl,
a fenti pozíciókódolás lehetővé teszi
a modell számára is, hogy könnyen megtanulja a relatív pozíciók szerint figyelni.
Ennek oka az, hogy
bármely rögzített $\delta$ pozícióeltolásra,
az $i + \delta$ pozíción lévő pozíciókódolás
lineáris vetítéssel reprezentálható
az $i$ pozíción lévőből.


Ezt a vetítést matematikailag is megmagyarázhatjuk.
Az $\omega_j = 1/10000^{2j/d}$ jelöléssel,
a :eqref:`eq_positional-encoding-def`-ben lévő bármely $(p_{i, 2j}, p_{i, 2j+1})$ pár
lineárisan vetíthető $(p_{i+\delta, 2j}, p_{i+\delta, 2j+1})$-ba
bármely rögzített $\delta$ eltolásra:

$$\begin{aligned}
\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\  -\sin(\delta \omega_j) & \cos(\delta \omega_j) \\ \end{bmatrix}
\begin{bmatrix} p_{i, 2j} \\  p_{i, 2j+1} \\ \end{bmatrix}
=&\begin{bmatrix} \cos(\delta \omega_j) \sin(i \omega_j) + \sin(\delta \omega_j) \cos(i \omega_j) \\  -\sin(\delta \omega_j) \sin(i \omega_j) + \cos(\delta \omega_j) \cos(i \omega_j) \\ \end{bmatrix}\\
=&\begin{bmatrix} \sin\left((i+\delta) \omega_j\right) \\  \cos\left((i+\delta) \omega_j\right) \\ \end{bmatrix}\\
=& 
\begin{bmatrix} p_{i+\delta, 2j} \\  p_{i+\delta, 2j+1} \\ \end{bmatrix},
\end{aligned}$$

ahol a $2\times 2$ vetítési mátrix nem függ semmilyen $i$ pozícióindextől.

## Összefoglalás

Az önfigyelemben a lekérdezések, kulcsok és értékek mind ugyanonnan jönnek.
Mind a CNN-ek, mind az önfigyelem párhuzamos számítást élveznek,
és az önfigyelemnek van a legrövidebb maximális úthossza.
Azonban a szekvenciahosszhoz képest négyzetes számítási bonyolultság
az önfigyelmet tiltóan lassúvá teszi
nagyon hosszú szekvenciák esetén.
A szekvencia sorrendinformációjának felhasználásához
abszolút vagy relatív pozicionális információt injektálhatunk
pozíciókódolás hozzáadásával a bemeneti reprezentációkhoz.

## Feladatok

1. Tételezzük fel, hogy mély architektúrát tervezünk szekvencia reprezentálásához önfigyelmi rétegek egymásra rakásával pozíciókódolással. Melyek lennének a lehetséges problémák?
1. Tudsz-e tanítható pozíciókódolási módszert tervezni?
1. Rendelhetünk-e különböző tanult beágyazásokat a lekérdezések és kulcsok között az önfigyelemben összehasonlított különböző eltolásokhoz? Tipp: hivatkozhatsz a relatív pozícióbeágyazásokra :cite:`shaw2018self,huang2018music`.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1651)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1652)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3870)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18030)
:end_tab:
