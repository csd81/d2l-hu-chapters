```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Többfejű figyelem
:label:`sec_multihead-attention`


A gyakorlatban ugyanazon lekérdezések, kulcsok és értékek mellett érdemes lehet, hogy a modell kombinálja a tudást
az ugyanazon figyelemmechanizmus különböző viselkedéseiből,
mint például különböző hatótávolságú függőségek megragadása
(pl. rövidebb vs. hosszabb hatótávolságú) egy szekvencián belül.
Így előnyös lehet, ha a figyelemmechanizmus közösen használhat különböző reprezentációs altereket a lekérdezések, kulcsok és értékek számára.


E cél érdekében egyetlen figyelempooling helyett
a lekérdezések, kulcsok és értékek
$h$ függetlenül tanult lineáris vetítéssel transzformálhatók.
Ezután ezt a $h$ vetített lekérdezést, kulcsot és értéket
párhuzamosan táplálják a figyelempoolingba.
Végül
$h$ figyelempooling-kimenet össze van fűzve és
egy másik tanult lineáris vetítéssel transzformálva
a végső kimenet előállítása érdekében.
Ezt a tervezést
*többfejű figyelemnek* nevezik,
ahol a $h$ figyelempooling kimenet mindegyike
egy *fej* :cite:`Vaswani.Shazeer.Parmar.ea.2017`.
Teljes összeköttetésű rétegeket használva
tanítható lineáris transzformációk végrehajtásához,
a :numref:`fig_multi-head-attention`
írja le a többfejű figyelmet.

![Többfejű figyelem, ahol több fejet összefűznek, majd lineárisan transzformálnak.](../img/multi-head-attention.svg)
:label:`fig_multi-head-attention`

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
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
```

## Modell

A többfejű figyelem implementációjának megadása előtt
formalizáljuk matematikailag ezt a modellt.
Adott $\mathbf{q} \in \mathbb{R}^{d_q}$ lekérdezés,
$\mathbf{k} \in \mathbb{R}^{d_k}$ kulcs,
és $\mathbf{v} \in \mathbb{R}^{d_v}$ érték esetén,
minden $\mathbf{h}_i$ figyelemfej ($i = 1, \ldots, h$)
kiszámítása a következőképpen történik:

$$\mathbf{h}_i = f(\mathbf W_i^{(q)}\mathbf q, \mathbf W_i^{(k)}\mathbf k,\mathbf W_i^{(v)}\mathbf v) \in \mathbb R^{p_v},$$

ahol
$\mathbf W_i^{(q)}\in\mathbb R^{p_q\times d_q}$,
$\mathbf W_i^{(k)}\in\mathbb R^{p_k\times d_k}$,
és $\mathbf W_i^{(v)}\in\mathbb R^{p_v\times d_v}$
tanítható paraméterek,
és $f$ figyelempooling,
mint például
az additív figyelem és a skálázott dot product figyelem
a :numref:`sec_attention-scoring-functions`-ben.
A többfejű figyelem kimenete
egy másik lineáris transzformáció a
tanítható $\mathbf W_o\in\mathbb R^{p_o\times h p_v}$
paramétereken keresztül
a $h$ fej összefűzéséből:

$$\mathbf W_o \begin{bmatrix}\mathbf h_1\\\vdots\\\mathbf h_h\end{bmatrix} \in \mathbb{R}^{p_o}.$$

E tervezés alapján minden fej a bemenet különböző részeire figyelhet.
Az egyszerű súlyozott átlagnál bonyolultabb függvények is kifejezhetők.

## Implementáció

Az implementációban
[**a skálázott dot product figyelmet választjuk
minden fejhez**] a többfejű figyelemben.
A számítási és parametrizációs költség jelentős növekedésének elkerülése érdekében,
beállítjuk, hogy $p_q = p_k = p_v = p_o / h$.
Vegyük észre, hogy $h$ fej párhuzamosan számítható,
ha a lekérdezés, kulcs és érték
lineáris transzformációk kimeneteinek számát
$p_q h = p_k h = p_v h = p_o$-ra állítjuk.
A következő implementációban
$p_o$ a `num_hiddens` argumentumon keresztül kerül megadásra.

```{.python .input}
%%tab mxnet
class MultiHeadAttention(d2l.Module):  #@save
    """Multi-head attention."""
    def __init__(self, num_hiddens, num_heads, dropout, use_bias=False,
                 **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_k = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_v = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_o = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)

    def forward(self, queries, keys, values, valid_lens):
        # A queries, keys vagy values alakja:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # A valid_lens alakja: (batch_size,) vagy (batch_size, no. of queries)
        # Transzponálás után a queries, keys vagy values alakja:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # A 0. tengely mentén az első elemet (skalárt vagy vektort)
            # num_heads alkalommal másoljuk, majd a következőt és így tovább
            valid_lens = valid_lens.repeat(self.num_heads, axis=0)

        # Az output alakja: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        
        # Az output_concat alakja: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
```

```{.python .input}
%%tab pytorch
class MultiHeadAttention(d2l.Module):  #@save
    """Multi-head attention."""
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # A queries, keys vagy values alakja:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # A valid_lens alakja: (batch_size,) vagy (batch_size, no. of queries)
        # Transzponálás után a queries, keys vagy values alakja:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # A 0. tengely mentén az első elemet (skalárt vagy vektort)
            # num_heads alkalommal másoljuk, majd a következőt és így tovább
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Az output alakja: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # Az output_concat alakja: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
```

```{.python .input}
%%tab tensorflow
class MultiHeadAttention(d2l.Module):  #@save
    """Multi-head attention."""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_v = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_o = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
    
    def call(self, queries, keys, values, valid_lens, **kwargs):
        # A queries, keys vagy values alakja:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # A valid_lens alakja: (batch_size,) vagy (batch_size, no. of queries)
        # Transzponálás után a queries, keys vagy values alakja:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        
        if valid_lens is not None:
            # A 0. tengely mentén az első elemet (skalárt vagy vektort)
            # num_heads alkalommal másoljuk, majd a következőt és így tovább
            valid_lens = tf.repeat(valid_lens, repeats=self.num_heads, axis=0)
            
        # Az output alakja: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens, **kwargs)
        
        # Az output_concat alakja: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
```

```{.python .input}
%%tab jax
class MultiHeadAttention(nn.Module):  #@save
    num_hiddens: int
    num_heads: int
    dropout: float
    bias: bool = False

    def setup(self):
        self.attention = d2l.DotProductAttention(self.dropout)
        self.W_q = nn.Dense(self.num_hiddens, use_bias=self.bias)
        self.W_k = nn.Dense(self.num_hiddens, use_bias=self.bias)
        self.W_v = nn.Dense(self.num_hiddens, use_bias=self.bias)
        self.W_o = nn.Dense(self.num_hiddens, use_bias=self.bias)

    @nn.compact
    def __call__(self, queries, keys, values, valid_lens, training=False):
        # A queries, keys vagy values alakja:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # A valid_lens alakja: (batch_size,) vagy (batch_size, no. of queries)
        # Transzponálás után a queries, keys vagy values alakja:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # A 0. tengely mentén az első elemet (skalárt vagy vektort)
            # num_heads alkalommal másoljuk, majd a következőt és így tovább
            valid_lens = jnp.repeat(valid_lens, self.num_heads, axis=0)

        # Az output alakja: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output, attention_weights = self.attention(
            queries, keys, values, valid_lens, training=training)
        # Az output_concat alakja: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat), attention_weights
```

A [**több fej párhuzamos számításának lehetővé tételéhez**],
a fenti `MultiHeadAttention` osztály két, alább definiált transzpozíciós módszert használ.
Konkrétan
a `transpose_output` módszer megfordítja a
`transpose_qkv` módszer műveletét.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_qkv(self, X):
    """Transzponálás több figyelemfej párhuzamos számításához."""
    # A bemeneti X alakja: (batch_size, no. of queries or key-value pairs,
    # num_hiddens). A kimeneti X alakja: (batch_size, no. of queries or
    # key-value pairs, num_heads, num_hiddens / num_heads)
    X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
    # A kimeneti X alakja: (batch_size, num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    X = X.transpose(0, 2, 1, 3)
    # Az output alakja: (batch_size * num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_output(self, X):
    """A transpose_qkv műveletének visszafordítása."""
    X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
    X = X.transpose(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_qkv(self, X):
    """Transzponálás több figyelemfej párhuzamos számításához."""
    # A bemeneti X alakja: (batch_size, no. of queries or key-value pairs,
    # num_hiddens). A kimeneti X alakja: (batch_size, no. of queries or
    # key-value pairs, num_heads, num_hiddens / num_heads)
    X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
    # A kimeneti X alakja: (batch_size, num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    X = X.permute(0, 2, 1, 3)
    # Az output alakja: (batch_size * num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_output(self, X):
    """A transpose_qkv műveletének visszafordítása."""
    X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_qkv(self, X):
    """Transzponálás több figyelemfej párhuzamos számításához."""
    # A bemeneti X alakja: (batch_size, no. of queries or key-value pairs,
    # num_hiddens). A kimeneti X alakja: (batch_size, no. of queries or
    # key-value pairs, num_heads, num_hiddens / num_heads)
    X = tf.reshape(X, shape=(X.shape[0], X.shape[1], self.num_heads, -1))
    # A kimeneti X alakja: (batch_size, num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    # Az output alakja: (batch_size * num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    return tf.reshape(X, shape=(-1, X.shape[2], X.shape[3]))

@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_output(self, X):
    """A transpose_qkv műveletének visszafordítása."""
    X = tf.reshape(X, shape=(-1, self.num_heads, X.shape[1], X.shape[2]))
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    return tf.reshape(X, shape=(X.shape[0], X.shape[1], -1))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_qkv(self, X):
    """Transzponálás több figyelemfej párhuzamos számításához."""
    # A bemeneti X alakja: (batch_size, no. of queries or key-value pairs,
    # num_hiddens). A kimeneti X alakja: (batch_size, no. of queries or
    # key-value pairs, num_heads, num_hiddens / num_heads)
    X = X.reshape((X.shape[0], X.shape[1], self.num_heads, -1))
    # A kimeneti X alakja: (batch_size, num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    X = jnp.transpose(X, (0, 2, 1, 3))
    # Az output alakja: (batch_size * num_heads, no. of queries or key-value
    # pairs, num_hiddens / num_heads)
    return X.reshape((-1, X.shape[2], X.shape[3]))

@d2l.add_to_class(MultiHeadAttention)  #@save
def transpose_output(self, X):
    """A transpose_qkv műveletének visszafordítása."""
    X = X.reshape((-1, self.num_heads, X.shape[1], X.shape[2]))
    X = jnp.transpose(X, (0, 2, 1, 3))
    return X.reshape((X.shape[0], X.shape[1], -1))
```

[**Teszteljük az implementált**] `MultiHeadAttention` osztályt
egy olyan játék-példával, ahol a kulcsok és értékek azonosak.
Ennek eredményeként
a többfejű figyelem kimenetének alakja
(`batch_size`, `num_queries`, `num_hiddens`).

```{.python .input}
%%tab pytorch
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
Y = d2l.ones((batch_size, num_kvpairs, num_hiddens))
d2l.check_shape(attention(X, Y, Y, valid_lens),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab mxnet
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()
```

```{.python .input}
%%tab jax
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
```

```{.python .input}
%%tab tensorflow
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
```

```{.python .input}
%%tab mxnet
batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
Y = d2l.ones((batch_size, num_kvpairs, num_hiddens))
d2l.check_shape(attention(X, Y, Y, valid_lens),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab tensorflow
batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = d2l.tensor([3, 2])
X = tf.ones((batch_size, num_queries, num_hiddens))
Y = tf.ones((batch_size, num_kvpairs, num_hiddens))
d2l.check_shape(attention(X, Y, Y, valid_lens, training=False),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab jax
batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
Y = d2l.ones((batch_size, num_kvpairs, num_hiddens))
d2l.check_shape(attention.init_with_output(d2l.get_key(), X, Y, Y, valid_lens,
                                           training=False)[0][0],
                (batch_size, num_queries, num_hiddens))
```

## Összefoglalás

A többfejű figyelem kombinálja az ugyanazon figyelempooling tudását
a lekérdezések, kulcsok és értékek különböző reprezentációs alterei révén.
Több fejből álló többfejű figyelem párhuzamos kiszámításához
megfelelő tenzormanipulációra van szükség.


## Feladatok

1. Vizualizáld több fej figyelemsúlyait ebben a kísérletben.
1. Tételezzük fel, hogy van egy többfejű figyelemen alapuló betanított modellünk, és a kevésbé fontos figyelemfejeket le akarjuk metszeni az előrejelzési sebesség növelése érdekében. Hogyan tudnánk kísérleteket tervezni egy figyelemfej fontosságának mérésére?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1634)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1635)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3869)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18029)
:end_tab:
