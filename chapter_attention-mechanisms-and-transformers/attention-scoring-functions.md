```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Figyelempontozási függvények
:label:`sec_attention-scoring-functions`


A :numref:`sec_attention-pooling`-ban számos különböző távolságalapú kernelt használtunk, köztük egy Gauss-kernelt a lekérdezések és kulcsok közötti interakciók modellezéséhez. Kiderül, hogy a távolságfüggvények kiszámítása kissé drágább, mint a skaláris szorzatoké. Ezért, a nemnegatív figyelemsúlyokat biztosító softmax-művelettel együtt, sok munka ment a :eqref:`eq_softmax_attention`-ban és a :numref:`fig_attention_output`-ban szereplő, egyszerűbben kiszámítható *figyelempuntozási függvényekbe*.

![A figyelempooling kimenetének kiszámítása az értékek súlyozott átlagaként, ahol a súlyokat az $\mathit{a}$ figyelempuntozási függvény és a softmax-művelet számítja ki.](../img/attention-output.svg)
:label:`fig_attention_output`

```{.python .input}
%%tab mxnet
import math
from d2l import mxnet as d2l
from mxnet import np, npx
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
import math
```

## **Skaláris szorzat alapú figyelem**


Tekintsük át egy pillanatra a Gauss-kernelből (hatványozás nélkül) kapott figyelemfüggvényt:

$$
a(\mathbf{q}, \mathbf{k}_i) = -\frac{1}{2} \|\mathbf{q} - \mathbf{k}_i\|^2  = \mathbf{q}^\top \mathbf{k}_i -\frac{1}{2} \|\mathbf{k}_i\|^2  -\frac{1}{2} \|\mathbf{q}\|^2.
$$

Először is vegyük észre, hogy az utolsó tag csak $\mathbf{q}$-tól függ. Mint ilyen, azonos az összes $(\mathbf{q}, \mathbf{k}_i)$ párra. A figyelemsúlyok $1$-re normalizálása, ahogyan az :eqref:`eq_softmax_attention`-ban történik, biztosítja, hogy ez a tag teljesen eltűnjön. Másodszor vegyük észre, hogy mind a batch, mind a rétegnormalizáció (amelyet később tárgyalunk) olyan aktivációkhoz vezet, amelyeknek jól korlátozott, és gyakran konstans normái vannak, $\|\mathbf{k}_i\|$. Ez a helyzet például akkor, ha a kulcsokat $\mathbf{k}_i$ rétegnormalizáló réteg generálta. Így ezt is elhagyhatjuk $a$ definíciójából az eredmény lényeges megváltoztatása nélkül.

Végül az exponenciális függvény argumentumainak nagyságrendját kell kézben tartanunk. Tételezzük fel, hogy a $\mathbf{q} \in \mathbb{R}^d$ lekérdezés és a $\mathbf{k}_i \in \mathbb{R}^d$ kulcs összes eleme egymástól független, azonos eloszlású, nulla átlagú és egységnyi szórású véletlen változó. A két vektor skaláris szorzata nulla átlagú és $d$ szórású. Annak biztosítására, hogy a skaláris szorzat szórása $1$ maradjon a vektorhossztól függetlenül, a *skálázott dot product figyelempuntozási függvényt* használjuk. Azaz a skaláris szorzatot $1/\sqrt{d}$-vel skálázzuk. Így jutunk el az első általánosan használt figyelemfüggvényhez, amelyet például a Transformerekben is alkalmaznak :cite:`Vaswani.Shazeer.Parmar.ea.2017`:

$$ a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i / \sqrt{d}.$$
:eqlabel:`eq_dot_product_attention`

Vegyük észre, hogy az $\alpha$ figyelemsúlyokat még normalizálni kell. Ezt tovább egyszerűsíthetjük a :eqref:`eq_softmax_attention`-n keresztül a softmax-művelet segítségével:

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \mathrm{softmax}(a(\mathbf{q}, \mathbf{k}_i)) = \frac{\exp(\mathbf{q}^\top \mathbf{k}_i / \sqrt{d})}{\sum_{j=1} \exp(\mathbf{q}^\top \mathbf{k}_j / \sqrt{d})}.$$
:eqlabel:`eq_attn-scoring-alpha`

Kiderül, hogy az összes népszerű figyelemmechanizmus a softmax-ot használja, ezért a fejezet további részében erre korlátozzuk magunkat.

## Segédfüggvények

Néhány függvényre van szükségünk ahhoz, hogy a figyelemmechanizmust hatékonyan tudjuk alkalmazni. Ez magában foglalja a változó hosszúságú karakterláncok kezelésére szolgáló eszközöket (amelyek természetes nyelvfeldolgozásban gyakoriak) és a mini-batch-eken való hatékony kiértékelés eszközeit (batch mátrixszorzás).


### **Maszkolású Softmax-Művelet**

A figyelemmechanizmus egyik legelterjedtebb alkalmazása a szekvenciamodellek. Ezért képesnek kell lennünk különböző hosszúságú szekvenciák kezelésére. Egyes esetekben ilyen szekvenciák ugyanabba a mini-batch-be kerülhetnek, ami szükségessé teszi a rövidebb szekvenciák kitöltését üres tokenekkel (lásd például :numref:`sec_machine_translation`). Ezek a speciális tokenek nem hordoznak jelentést. Tételezzük fel például, hogy a következő három mondatunk van:

```
Dive  into  Deep    Learning 
Learn to    code    <blank>
Hello world <blank> <blank>
```


Mivel nem akarunk üres helyeket a figyelemmodellünkben, egyszerűen korlátozni kell a $\sum_{i=1}^n \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i$-t $\sum_{i=1}^l \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i$-re, amennyire, $l \leq n$, a tényleges mondat terjed. Mivel ez egy olyan általános probléma, névvel rendelkezik: a *maszkolású softmax-művelet*.

Implementáljuk. Valójában az implementáció egy kicsit „csal", azzal, hogy a $\mathbf{v}_i$ értékeit $i > l$ esetén nullára állítja. Ráadásul a figyelemsúlyokat egy nagy negatív számra állítja, például $-10^{6}$-ra, hogy a gradiensekre és értékekre való hozzájárulásuk a gyakorlatban eltűnjön. Ezt azért tesszük, mert a lineáris algebrai kernelek és operátorok erősen optimalizáltak a GPU-khoz, és gyorsabb kissé pazarló a számításban, mint feltételes (if then else) utasításokat tartalmazó kódot futtatni.

```{.python .input}
%%tab mxnet
def masked_softmax(X, valid_lens):  #@save
    """Softmax művelet végrehajtása az utolsó tengely elemeinek maszkolásával."""
    # X: 3D tenzor, valid_lens: 1D vagy 2D tenzor
    if valid_lens is None:
        return npx.softmax(X)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = valid_lens.repeat(shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # Az utolsó tengely mentén a maszkolt elemeket nagyon nagy negatív
        # értékre cseréljük, amelynek exponenciálisa 0 lesz
        X = npx.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, True,
                              value=-1e6, axis=1)
        return npx.softmax(X).reshape(shape)
```

```{.python .input}
%%tab pytorch
def masked_softmax(X, valid_lens):  #@save
    """Softmax művelet végrehajtása az utolsó tengely elemeinek maszkolásával."""
    # X: 3D tenzor, valid_lens: 1D vagy 2D tenzor 
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X
    
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # Az utolsó tengely mentén a maszkolt elemeket nagyon nagy negatív
        # értékre cseréljük, amelynek exponenciálisa 0 lesz
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
```

```{.python .input}
%%tab tensorflow
def masked_softmax(X, valid_lens):  #@save
    """Softmax művelet végrehajtása az utolsó tengely elemeinek maszkolásával."""
    # X: 3D tenzor, valid_lens: 1D vagy 2D tenzor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[
            None, :] < tf.cast(valid_len[:, None], dtype=tf.float32)

        if len(X.shape) == 3:
            return tf.where(tf.expand_dims(mask, axis=-1), X, value)
        else:
            return tf.where(mask, X, value)
    
    if valid_lens is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if len(valid_lens.shape) == 1:
            valid_lens = tf.repeat(valid_lens, repeats=shape[1])
            
        else:
            valid_lens = tf.reshape(valid_lens, shape=-1)
        # Az utolsó tengely mentén a maszkolt elemeket nagyon nagy negatív
        # értékre cseréljük, amelynek exponenciálisa 0 lesz    
        X = _sequence_mask(tf.reshape(X, shape=(-1, shape[-1])), valid_lens,
                           value=-1e6)    
        return tf.nn.softmax(tf.reshape(X, shape=shape), axis=-1)
```

```{.python .input}
%%tab jax
def masked_softmax(X, valid_lens):  #@save
    """Softmax művelet végrehajtása az utolsó tengely elemeinek maszkolásával."""
    # X: 3D tenzor, valid_lens: 1D vagy 2D tenzor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = jnp.arange((maxlen),
                          dtype=jnp.float32)[None, :] < valid_len[:, None]
        return jnp.where(mask, X, value)

    if valid_lens is None:
        return nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = jnp.repeat(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # Az utolsó tengely mentén a maszkolt elemeket nagyon nagy negatív
        # értékre cseréljük, amelynek exponenciálisa 0 lesz
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.softmax(X.reshape(shape), axis=-1)
```

A **függvény működésének szemléltetéséhez**
tekintsünk egy $2 \times 4$ méretű két példányból álló mini-batch-et,
ahol az érvényes hosszaik rendre $2$ és $3$.
A maszkolású softmax-művelet eredményeként
az egyes vektorpárokra vonatkozó érvényes hosszakon túli értékek mind nulla maszkot kapnak.

```{.python .input}
%%tab mxnet
masked_softmax(np.random.uniform(size=(2, 2, 4)), d2l.tensor([2, 3]))
```

```{.python .input}
%%tab pytorch
masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
```

```{.python .input}
%%tab tensorflow
masked_softmax(tf.random.uniform(shape=(2, 2, 4)), tf.constant([2, 3]))
```

```{.python .input}
%%tab jax
masked_softmax(jax.random.uniform(d2l.get_key(), (2, 2, 4)), jnp.array([2, 3]))
```

Ha finomabb vezérlésre van szükségünk az egyes példányok mindkét vektorára vonatkozó érvényes hosszak megadásához, egyszerűen egy kétdimenziós érvényes hosszak tenzort használunk. Ez a következőt adja:

```{.python .input}
%%tab mxnet
masked_softmax(np.random.uniform(size=(2, 2, 4)),
               d2l.tensor([[1, 3], [2, 4]]))
```

```{.python .input}
%%tab pytorch
masked_softmax(torch.rand(2, 2, 4), d2l.tensor([[1, 3], [2, 4]]))
```

```{.python .input}
%%tab tensorflow
masked_softmax(tf.random.uniform((2, 2, 4)), tf.constant([[1, 3], [2, 4]]))
```

```{.python .input}
%%tab jax
masked_softmax(jax.random.uniform(d2l.get_key(), (2, 2, 4)),
               jnp.array([[1, 3], [2, 4]]))
```

### Batch mátrixszorzás
:label:`subsec_batch_dot`

Egy másik általánosan használt művelet a mátrixok batch-einek egymással való szorzása. Ez akkor hasznos, ha lekérdezések, kulcsok és értékek mini-batch-jei vannak. Pontosabban, tételezzük fel, hogy

$$\mathbf{Q} = [\mathbf{Q}_1, \mathbf{Q}_2, \ldots, \mathbf{Q}_n]  \in \mathbb{R}^{n \times a \times b}, \\
    \mathbf{K} = [\mathbf{K}_1, \mathbf{K}_2, \ldots, \mathbf{K}_n]  \in \mathbb{R}^{n \times b \times c}.
$$

Ekkor a batch mátrixszorzás (BMM) az elemenként vett szorzatot számítja ki:

$$\textrm{BMM}(\mathbf{Q}, \mathbf{K}) = [\mathbf{Q}_1 \mathbf{K}_1, \mathbf{Q}_2 \mathbf{K}_2, \ldots, \mathbf{Q}_n \mathbf{K}_n] \in \mathbb{R}^{n \times a \times c}.$$
:eqlabel:`eq_batch-matrix-mul`

Lássuk ezt a gyakorlatban egy mélytanulási keretrendszerben.

```{.python .input}
%%tab mxnet
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(npx.batch_dot(Q, K), (2, 3, 6))
```

```{.python .input}
%%tab pytorch
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(torch.bmm(Q, K), (2, 3, 6))
```

```{.python .input}
%%tab tensorflow
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(tf.matmul(Q, K).numpy(), (2, 3, 6))
```

```{.python .input}
%%tab jax
Q = d2l.ones((2, 3, 4))
K = d2l.ones((2, 4, 6))
d2l.check_shape(jax.lax.batch_matmul(Q, K), (2, 3, 6))
```

## **Skálázott Dot Product Figyelem**

Térjünk vissza a :eqref:`eq_dot_product_attention`-ban bevezetett dot product figyelemhez.
Általánosságban megköveteli, hogy a lekérdezés és a kulcs azonos vektorhosszúsággal rendelkezzen, mondjuk $d$, bár ezt könnyen kezelhetjük azzal, hogy $\mathbf{q}^\top \mathbf{k}$-t helyettesítjük $\mathbf{q}^\top \mathbf{M} \mathbf{k}$-val, ahol $\mathbf{M}$ egy megfelelően választott mátrix a két tér közötti fordításhoz. Egyelőre tételezzük fel, hogy a dimenziók egyeznek.

A gyakorlatban hatékonysági szempontból gyakran mini-batch-ekre gondolunk,
például az $n$ lekérdezés és $m$ kulcs-érték pár figyelmének kiszámítása,
ahol a lekérdezések és kulcsok hossza $d$,
és az értékek hossza $v$. A $\mathbf Q\in\mathbb R^{n\times d}$ lekérdezések,
$\mathbf K\in\mathbb R^{m\times d}$ kulcsok
és $\mathbf V\in\mathbb R^{m\times v}$ értékek skálázott dot product figyelme
így felírható:

$$ \mathrm{softmax}\left(\frac{\mathbf Q \mathbf K^\top }{\sqrt{d}}\right) \mathbf V \in \mathbb{R}^{n\times v}.$$
:eqlabel:`eq_softmax_QK_V`

Vegyük észre, hogy amikor ezt mini-batch-re alkalmazzuk, szükségünk van a :eqref:`eq_batch-matrix-mul`-ban bevezetett batch mátrixszorzásra. A skálázott dot product figyelem következő implementációjában dropout-ot használunk a modell regularizációjához.

```{.python .input}
%%tab mxnet
class DotProductAttention(nn.Block):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # A queries alakja: (batch_size, no. of queries, d)
    # A keys alakja: (batch_size, no. of key-value pairs, d)
    # A values alakja: (batch_size, no. of key-value pairs, value dimension)
    # A valid_lens alakja: (batch_size,) vagy (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # A transpose_b=True felcseréli a keys utolsó két dimenzióját
        scores = npx.batch_dot(queries, keys, transpose_b=True) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab pytorch
class DotProductAttention(nn.Module):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # A queries alakja: (batch_size, no. of queries, d)
    # A keys alakja: (batch_size, no. of key-value pairs, d)
    # A values alakja: (batch_size, no. of key-value pairs, value dimension)
    # A valid_lens alakja: (batch_size,) vagy (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # A keys utolsó két dimenzióját a keys.transpose(1, 2) cseréli fel
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab tensorflow
class DotProductAttention(tf.keras.layers.Layer):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    # A queries alakja: (batch_size, no. of queries, d)
    # A keys alakja: (batch_size, no. of key-value pairs, d)
    # A values alakja: (batch_size, no. of key-value pairs, value dimension)
    # A valid_lens alakja: (batch_size,) vagy (batch_size, no. of queries)
    def call(self, queries, keys, values, valid_lens=None, **kwargs):
        d = queries.shape[-1]
        scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(
            tf.cast(d, dtype=tf.float32))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(self.attention_weights, **kwargs), values)
```

```{.python .input}
%%tab jax
class DotProductAttention(nn.Module):  #@save
    """Scaled dot product attention."""
    dropout: float

    # A queries alakja: (batch_size, no. of queries, d)
    # A keys alakja: (batch_size, no. of key-value pairs, d)
    # A values alakja: (batch_size, no. of key-value pairs, value dimension)
    # A valid_lens alakja: (batch_size,) vagy (batch_size, no. of queries)
    @nn.compact
    def __call__(self, queries, keys, values, valid_lens=None,
                 training=False):
        d = queries.shape[-1]
        # A keys utolsó két dimenzióját a keys.swapaxes(1, 2) cseréli fel
        scores = queries@(keys.swapaxes(1, 2)) / math.sqrt(d)
        attention_weights = masked_softmax(scores, valid_lens)
        dropout_layer = nn.Dropout(self.dropout, deterministic=not training)
        return dropout_layer(attention_weights)@values, attention_weights
```

A `DotProductAttention` osztály működésének **szemléltetéséhez**
ugyanazokat a kulcsokat, értékeket és érvényes hosszakat használjuk az additív figyelemre vonatkozó korábbi játék-példából. Példánk céljaira feltételezzük, hogy a mini-batch-méretünk $2$, összesen $10$ kulcsunk és értékünk van, és az értékek dimenziója $4$. Végül feltételezzük, hogy az érvényes hossz megfigyelésenként rendre $2$ és $6$. Ennek alapján a kimenet várhatóan egy $2 \times 1 \times 4$ alakú tenzor, azaz a mini-batch minden példányára egy sor.

```{.python .input}
%%tab mxnet
queries = d2l.normal(0, 1, (2, 1, 2))
keys = d2l.normal(0, 1, (2, 10, 2))
values = d2l.normal(0, 1, (2, 10, 4))
valid_lens = d2l.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
attention.initialize()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab pytorch
queries = d2l.normal(0, 1, (2, 1, 2))
keys = d2l.normal(0, 1, (2, 10, 2))
values = d2l.normal(0, 1, (2, 10, 4))
valid_lens = d2l.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
attention.eval()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab tensorflow
queries = tf.random.normal(shape=(2, 1, 2))
keys = tf.random.normal(shape=(2, 10, 2))
values = tf.random.normal(shape=(2, 10, 4))
valid_lens = tf.constant([2, 6])

attention = DotProductAttention(dropout=0.5)
d2l.check_shape(attention(queries, keys, values, valid_lens, training=False),
                (2, 1, 4))
```

```{.python .input}
%%tab jax
queries = jax.random.normal(d2l.get_key(), (2, 1, 2))
keys = jax.random.normal(d2l.get_key(), (2, 10, 2))
values = jax.random.normal(d2l.get_key(), (2, 10, 4))
valid_lens = d2l.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
(output, attention_weights), params = attention.init_with_output(
    d2l.get_key(), queries, keys, values, valid_lens)
print(output)
```

Ellenőrizzük, hogy a figyelemsúlyok valóban eltűnnek-e a második, illetve hatodik oszlopon túl (az érvényes hossz $2$-re és $6$-ra állítása miatt).

```{.python .input}
%%tab pytorch, mxnet, tensorflow
d2l.show_heatmaps(d2l.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

```{.python .input}
%%tab jax
d2l.show_heatmaps(d2l.reshape(attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## **Additív Figyelem**
:label:`subsec_additive-attention`

Amikor a $\mathbf{q}$ lekérdezések és $\mathbf{k}$ kulcsok különböző dimenziójú vektorok,
vagy egy mátrixot használhatunk az eltérés kezelésére $\mathbf{q}^\top \mathbf{M} \mathbf{k}$ formájában, vagy additív figyelmet alkalmazhatunk pontozófüggvényként. Egy további előny az, hogy ahogyan nevéből is következik, a figyelem additív. Ez kisebb számítási megtakarításhoz vezethet.
Adott $\mathbf{q} \in \mathbb{R}^q$ lekérdezés
és $\mathbf{k} \in \mathbb{R}^k$ kulcs esetén
az *additív figyelem* pontozófüggvénye :cite:`Bahdanau.Cho.Bengio.2014`:

$$a(\mathbf q, \mathbf k) = \mathbf w_v^\top \textrm{tanh}(\mathbf W_q\mathbf q + \mathbf W_k \mathbf k) \in \mathbb{R},$$
:eqlabel:`eq_additive-attn`

ahol $\mathbf W_q\in\mathbb R^{h\times q}$, $\mathbf W_k\in\mathbb R^{h\times k}$,
és $\mathbf w_v\in\mathbb R^{h}$ a tanítható paraméterek. Ezt a tagot aztán softmax-ba táplálják a nemnegatívás és a normalizáció biztosítása érdekében.
A :eqref:`eq_additive-attn` egyenértelmű értelmezése az, hogy a lekérdezést és a kulcsot összefűzik,
majd egy egyetlen rejtett réteget tartalmazó MLP-be táplálják.
Az aktiválási függvényként $\tanh$-t használva és az eltolási tagokat kikapcsolva
az additív figyelmet a következőképpen implementáljuk:

```{.python .input}
%%tab mxnet
class AdditiveAttention(nn.Block):  #@save
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # Use flatten=False to only transform the last axis so that the
        # shapes for the other axes are kept the same
        self.W_k = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.W_q = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.w_v = nn.Dense(1, use_bias=False, flatten=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1,
        # no. of key-value pairs, num_hiddens). Sum them up with
        # broadcasting
        features = np.expand_dims(queries, axis=2) + np.expand_dims(
            keys, axis=1)
        features = np.tanh(features)
        # A self.w_v kimenete egydimenziós, ezért eltávolítjuk az alak utolsó
        # egydimenziós elemét. A scores alakja:
        # (batch_size, no. of queries, no. of key-value pairs)
        scores = np.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # A values alakja: (batch_size, no. of key-value pairs, value
        # dimension)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab pytorch
class AdditiveAttention(nn.Module):  #@save
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # A self.w_v kimenete egydimenziós, ezért eltávolítjuk az alak utolsó
        # egydimenziós elemét. A scores alakja: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # A values alakja: (batch_size, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

```{.python .input}
%%tab tensorflow
class AdditiveAttention(tf.keras.layers.Layer):  #@save
    """Additive attention."""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.w_v = tf.keras.layers.Dense(1, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, queries, keys, values, valid_lens, **kwargs):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = tf.expand_dims(queries, axis=2) + tf.expand_dims(
            keys, axis=1)
        features = tf.nn.tanh(features)
        # A self.w_v kimenete egydimenziós, ezért eltávolítjuk az alak utolsó
        # egydimenziós elemét. A scores alakja: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = tf.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # A values alakja: (batch_size, no. of key-value pairs, value
        # dimension)
        return tf.matmul(self.dropout(
            self.attention_weights, **kwargs), values)
```

```{.python .input}
%%tab jax
class AdditiveAttention(nn.Module):  #@save
    num_hiddens: int
    dropout: float

    def setup(self):
        self.W_k = nn.Dense(self.num_hiddens, use_bias=False)
        self.W_q = nn.Dense(self.num_hiddens, use_bias=False)
        self.w_v = nn.Dense(1, use_bias=False)

    @nn.compact
    def __call__(self, queries, keys, values, valid_lens, training=False):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = jnp.expand_dims(queries, axis=2) + jnp.expand_dims(keys, axis=1)
        features = nn.tanh(features)
        # A self.w_v kimenete egydimenziós, ezért eltávolítjuk az alak utolsó
        # egydimenziós elemét. A scores alakja: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        attention_weights = masked_softmax(scores, valid_lens)
        dropout_layer = nn.Dropout(self.dropout, deterministic=not training)
        # A values alakja: (batch_size, no. of key-value pairs, value
        # dimension)
        return dropout_layer(attention_weights)@values, attention_weights
```

Nézzük meg, **hogyan működik az `AdditiveAttention`**. Játék-példánkban $(2, 1, 20)$, $(2, 10, 2)$ és $(2, 10, 4)$ méretű lekérdezéseket, kulcsokat és értékeket választunk. Ez megegyezik a `DotProductAttention`-höz választottakkal, kivéve, hogy a lekérdezések most $20$-dimenziósak. Hasonlóképpen, $(2, 6)$-ot választunk a mini-batch szekvenciáinak érvényes hosszaként.

```{.python .input}
%%tab mxnet
queries = d2l.normal(0, 1, (2, 1, 20))

attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
attention.initialize()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab pytorch
queries = d2l.normal(0, 1, (2, 1, 20))

attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
attention.eval()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))
```

```{.python .input}
%%tab tensorflow
queries = tf.random.normal(shape=(2, 1, 20))

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1)
d2l.check_shape(attention(queries, keys, values, valid_lens, training=False),
                (2, 1, 4))
```

```{.python .input}
%%tab jax
queries = jax.random.normal(d2l.get_key(), (2, 1, 20))
attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
(output, attention_weights), params = attention.init_with_output(
    d2l.get_key(), queries, keys, values, valid_lens)
print(output)
```

A figyelemfüggvény áttekintésekor minőségileg meglehetősen hasonló viselkedést látunk a `DotProductAttention`-éhoz. Azaz csak a kiválasztott érvényes hosszon ($2, 6$) belüli tagok nullától különbözők.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
d2l.show_heatmaps(d2l.reshape(attention.attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

```{.python .input}
%%tab jax
d2l.show_heatmaps(d2l.reshape(attention_weights, (1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')
```

## Összefoglalás

Ebben a szakaszban bevezettük a két legfontosabb figyelempuntozási függvényt: a dot product és az additív figyelmet. Ezek hatékony eszközök a változó hosszúságú szekvenciákon való összesítéshez. Különösen a dot product figyelem a modern Transformer architektúrák alapköve. Amikor a lekérdezések és kulcsok különböző hosszúságú vektorok, az additív figyelempuntozási függvényt alkalmazhatjuk helyette. Ezeknek a rétegeknek az optimalizálása az elmúlt évek egyik kulcsterülete. Például az [NVIDIA's Transformer Library](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html) és a Megatron :cite:`shoeybi2019megatron` döntően a figyelemmechanizmus hatékony változataira támaszkodik. Ezt a Transformerek áttekintésekor egy kicsit részletesebben fogjuk tárgyalni a következő szakaszokban.

## Feladatok

1. Implementálj távolságalapú figyelmet a `DotProductAttention` kód módosításával. Vegyük észre, hogy a hatékony implementációhoz csak a kulcsok négyzetes normáira $\|\mathbf{k}_i\|^2$ van szükség.
1. Módosítsd a dot product figyelmet, hogy különböző dimenziójú lekérdezések és kulcsok esetén is működjön egy mátrix alkalmazásával a dimenziók igazításához.
1. Hogyan skálázódik a számítási költség a kulcsok, lekérdezések, értékek dimenziójával és számukkal? Mi a helyzet a memória-sávszélesség követelményeivel?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/346)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1064)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3867)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18027)
:end_tab:
