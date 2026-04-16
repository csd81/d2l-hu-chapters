```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# A Transformer architektúra
:label:`sec_transformer`


A :numref:`subsec_cnn-rnn-self-attention`-ban összehasonlítottuk a CNN-eket, RNN-eket és az önfigyelmet.
Figyelemre méltó, hogy az önfigyelem
egyszerre élvezi a párhuzamos számítást és
a legrövidebb maximális úthosszt.
Ezért
vonzó mély architektúrákat tervezni
az önfigyelem felhasználásával.
A korábbi önfigyelmi modellektől eltérően,
amelyek még mindig RNN-ekre támaszkodnak a bemeneti reprezentációkhoz :cite:`Cheng.Dong.Lapata.2016,Lin.Feng.Santos.ea.2017,Paulus.Xiong.Socher.2017`,
a Transformer modell
kizárólag figyelemmechanizmusokon alapul
konvolúciós vagy rekurrens réteg nélkül :cite:`Vaswani.Shazeer.Parmar.ea.2017`.
Bár eredetileg szöveges adatokon végzett szekvencia-szekvencia tanuláshoz javasolták,
a Transformerek elterjedtek
a modern mélytanulás alkalmazásainak széles körében,
mint például a nyelvvel, látással, beszéddel és megerősítéses tanulással kapcsolatos területeken.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
import math
from mxnet import autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import math
import pandas as pd
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import pandas as pd
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
import math
import pandas as pd
```

## Modell

A kódoló–dekódoló architektúra példányaként,
a Transformer teljes architektúrája
a :numref:`fig_transformer`-ben látható.
Amint látható,
a Transformer egy kódolóból és egy dekódolóból áll.
A :numref:`fig_s2s_attention_details`-ben látható
Bahdanau-figyelemmel ellentétben
a szekvencia-szekvencia tanuláshoz,
a bemeneti (forrás) és kimeneti (cél)
szekvencia-beágyazásokhoz
pozícióenkódolás kerül hozzáadásra
mielőtt betáplálják őket
az önfigyelmen alapuló modulokat egymásra rakó
kódolóba és dekódolóba.

![A Transformer architektúra.](../img/transformer.svg)
:width:`320px`
:label:`fig_transformer`


Most áttekintjük a
Transformer architektúrát a :numref:`fig_transformer`-ben.
Magas szinten,
a Transformer kódoló több azonos réteg egymásra rakásából áll,
ahol minden rétegnek
két alrétege van (mindkettőt $\textrm{sublayer}$-ként jelöljük).
Az első
egy többfejű önfigyelmi pooling,
a második pedig egy pozíciószerinti előre irányított hálózat.
Konkrétan,
a kódoló önfigyelemben,
a lekérdezések, kulcsok és értékek mind az
előző kódolóréteg kimenetéből származnak.
A :numref:`sec_resnet`-beli ResNet tervezéstől ihletve,
maradék összeköttetést alkalmaznak
mindkét alréteg körül.
A Transformerben,
a szekvencia bármely pozíciójában lévő bármely $\mathbf{x} \in \mathbb{R}^d$ bemenetre,
megköveteljük, hogy $\textrm{sublayer}(\mathbf{x}) \in \mathbb{R}^d$ teljesüljön, hogy
a $\mathbf{x} + \textrm{sublayer}(\mathbf{x}) \in \mathbb{R}^d$ maradék összeköttetés kivitelezhető legyen.
Ezt az összeadást a maradék összeköttetésből
rétegnormalizáció követi azonnal :cite:`Ba.Kiros.Hinton.2016`.
Ennek eredményeként a Transformer kódoló $d$-dimenziós vektoros reprezentációt ad ki
a bemeneti szekvencia minden pozíciójára.

A Transformer dekódoló szintén több azonos réteg egymásra rakása
maradék összeköttetésekkel és rétegnormalizációval.
A kódolóban leírt két alrétegen túl a dekódoló
egy harmadik alréteget szúr be, amelyet
kódoló–dekódoló figyelemnek neveznek,
e kettő közé.
A kódoló–dekódoló figyelemben,
a lekérdezések a
dekódoló önfigyelmi alrétegének kimenetéből,
a kulcsok és értékek pedig
a Transformer kódoló kimenetéből származnak.
A dekódoló önfigyelemben,
a lekérdezések, kulcsok és értékek mind az
előző dekódolóréteg kimenetéből jönnek.
Azonban a dekódoló minden pozíciójának
csak a dekódoló összes pozíciójára szabad figyelni
az adott pozícióig.
Ez a *maszkolású* figyelem
megőrzi az autoregresszív tulajdonságot,
biztosítva, hogy az előrejelzés csak
a már generált kimeneti tokeneektől függjön.


Már leírtuk és implementáltuk
a skálázott dot product-okon alapuló többfejű figyelmet
a :numref:`sec_multihead-attention`-ban
és a pozícióenkódolást a :numref:`subsec_positional-encoding`-ban.
A következőkben a Transformer modell
többi részét fogjuk implementálni.

## [**Pozíciószerinti előreirányított hálózatok**]
:label:`subsec_positionwise-ffn`

A pozíciószerinti előre irányított hálózat
ugyanazt az MLP-t használva átalakítja
a szekvencia összes pozíciójának representációját.
Ezért nevezzük *pozíciószerintinek*.
Az alábbi implementációban
az `X` bemenet
(batch-méret, időlépések száma vagy szekvenciahossz tokenekben,
rejtett egységek száma vagy jellemződimenziók)
alakkal rendelkezik,
és egy kétrétegű MLP által átalakítva
(batch-méret, időlépések száma, `ffn_num_outputs`)
alakú kimeneti tenzort eredményez.

```{.python .input}
%%tab mxnet
class PositionWiseFFN(nn.Block):  #@save
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Dense(ffn_num_hiddens, flatten=False,
                               activation='relu')
        self.dense2 = nn.Dense(ffn_num_outputs, flatten=False)

    def forward(self, X):
        return self.dense2(self.dense1(X))
```

```{.python .input}
%%tab pytorch
class PositionWiseFFN(nn.Module):  #@save
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
```

```{.python .input}
%%tab tensorflow
class PositionWiseFFN(tf.keras.layers.Layer):  #@save
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)

    def call(self, X):
        return self.dense2(self.relu(self.dense1(X)))
```

```{.python .input}
%%tab jax
class PositionWiseFFN(nn.Module):  #@save
    """The positionwise feed-forward network."""
    ffn_num_hiddens: int
    ffn_num_outputs: int

    def setup(self):
        self.dense1 = nn.Dense(self.ffn_num_hiddens)
        self.dense2 = nn.Dense(self.ffn_num_outputs)

    def __call__(self, X):
        return self.dense2(nn.relu(self.dense1(X)))
```

A következő példa megmutatja, hogy [**egy tenzor legbelső dimenziója megváltozik**] a
pozíciószerinti előre irányított hálózatban a kimenetek számára.
Mivel ugyanaz az MLP transzformál
az összes pozícióban,
ha az összes pozíción lévő bemenetek azonosak,
akkor a kimeneteik is azonosak lesznek.

```{.python .input}
%%tab mxnet
ffn = PositionWiseFFN(4, 8)
ffn.initialize()
ffn(np.ones((2, 3, 4)))[0]
```

```{.python .input}
%%tab pytorch
ffn = PositionWiseFFN(4, 8)
ffn.eval()
ffn(d2l.ones((2, 3, 4)))[0]
```

```{.python .input}
%%tab tensorflow
ffn = PositionWiseFFN(4, 8)
ffn(tf.ones((2, 3, 4)))[0]
```

```{.python .input}
%%tab jax
ffn = PositionWiseFFN(4, 8)
ffn.init_with_output(d2l.get_key(), jnp.ones((2, 3, 4)))[0][0]
```

## Maradék Összeköttetés és rétegnormalizáció

Most koncentráljunk a :numref:`fig_transformer`-ben lévő „add & norm" komponensre.
Ahogy az e szakasz elején leírtuk,
ez egy maradék összeköttetés, amelyet
azonnal rétegnormalizáció követ.
Mindkettő kulcsfontosságú a hatékony mély architektúrákhoz.

A :numref:`sec_batch_norm`-ban
megmagyaráztuk, hogy a batchnormalizáció hogyan
tölti fel és skálázza újra a mini-batch-en belüli példányokat.
Ahogy a :numref:`subsec_layer-normalization-in-bn`-ban tárgyaltuk,
a rétegnormalizáció ugyanolyan, mint a batchnormalizáció,
kivéve, hogy az előbbi
a jellemződimenziók mentén normalizál,
így skálaindependens és batch-méret-független előnyöket élvez.
Kiterjedt alkalmazásai ellenére
a számítógépes látásban,
a batchnormalizáció
általában empirikusan
kevésbé hatékony, mint a rétegnormalizáció
a természetes nyelvfeldolgozási feladatokban,
ahol a bemenetek gyakran
változó hosszúságú szekvenciák.

A következő kódrészlet
[**összehasonlítja a normalizálást különböző dimenziók mentén
rétegnormalizációval és batchnormalizációval**].

```{.python .input}
%%tab mxnet
ln = nn.LayerNorm()
ln.initialize()
bn = nn.BatchNorm()
bn.initialize()
X = d2l.tensor([[1, 2], [2, 3]])
# Számítsuk ki az átlagot és a varianciát X-ből tanítási módban
with autograd.record():
    print('layer norm:', ln(X), '\nbatch norm:', bn(X))
```

```{.python .input}
%%tab pytorch
ln = nn.LayerNorm(2)
bn = nn.LazyBatchNorm1d()
X = d2l.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# Számítsuk ki az átlagot és a varianciát X-ből tanítási módban
print('layer norm:', ln(X), '\nbatch norm:', bn(X))
```

```{.python .input}
%%tab tensorflow
ln = tf.keras.layers.LayerNormalization()
bn = tf.keras.layers.BatchNormalization()
X = tf.constant([[1, 2], [2, 3]], dtype=tf.float32)
print('layer norm:', ln(X), '\nbatch norm:', bn(X, training=True))
```

```{.python .input}
%%tab jax
ln = nn.LayerNorm()
bn = nn.BatchNorm()
X = d2l.tensor([[1, 2], [2, 3]], dtype=d2l.float32)
# Számítsuk ki az átlagot és a varianciát X-ből tanítási módban
print('layer norm:', ln.init_with_output(d2l.get_key(), X)[0],
      '\nbatch norm:', bn.init_with_output(d2l.get_key(), X,
                                           use_running_average=False)[0])
```

Most implementálhatjuk az `AddNorm` osztályt
[**maradék összeköttetéssel, amelyet rétegnormalizáció követ**].
Dropout-ot is alkalmazunk regularizációhoz.

```{.python .input}
%%tab mxnet
class AddNorm(nn.Block):  #@save
    """The residual connection followed by layer normalization."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm()

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
```

```{.python .input}
%%tab pytorch
class AddNorm(nn.Module):  #@save
    """The residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
```

```{.python .input}
%%tab tensorflow
class AddNorm(tf.keras.layers.Layer):  #@save
    """The residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(norm_shape)

    def call(self, X, Y, **kwargs):
        return self.ln(self.dropout(Y, **kwargs) + X)
```

```{.python .input}
%%tab jax
class AddNorm(nn.Module):  #@save
    """The residual connection followed by layer normalization."""
    dropout: int

    @nn.compact
    def __call__(self, X, Y, training=False):
        return nn.LayerNorm()(
            nn.Dropout(self.dropout)(Y, deterministic=not training) + X)
```

A maradék összeköttetés megköveteli,
hogy a két bemenet azonos alakú legyen,
hogy [**a kimeneti tenzor is azonos alakú legyen az összeadás után**].

```{.python .input}
%%tab mxnet
add_norm = AddNorm(0.5)
add_norm.initialize()
shape = (2, 3, 4)
d2l.check_shape(add_norm(d2l.ones(shape), d2l.ones(shape)), shape)
```

```{.python .input}
%%tab pytorch
add_norm = AddNorm(4, 0.5)
shape = (2, 3, 4)
d2l.check_shape(add_norm(d2l.ones(shape), d2l.ones(shape)), shape)
```

```{.python .input}
%%tab tensorflow
# A normalized_shape értéke: [i for i in range(len(input.shape))][1:]
add_norm = AddNorm([1, 2], 0.5)
shape = (2, 3, 4)
d2l.check_shape(add_norm(tf.ones(shape), tf.ones(shape), training=False),
                shape)
```

```{.python .input}
%%tab jax
add_norm = AddNorm(0.5)
shape = (2, 3, 4)
output, _ = add_norm.init_with_output(d2l.get_key(), d2l.ones(shape),
                                      d2l.ones(shape))
d2l.check_shape(output, shape)
```

## Kódoló
:label:`subsec_transformer-encoder`

A Transformer kódoló összeállításához szükséges összes alapvető komponenssel,
kezdjük [**a kódolón belüli egyetlen réteg**] implementálásával.
A következő `TransformerEncoderBlock` osztály
két alréteget tartalmaz: többfejű önfigyelmi pooling-ot és pozíciószerinti előre irányított hálózatokat,
ahol mindkét alréteg körül maradék összeköttetést alkalmaznak, amelyet rétegnormalizáció követ.

```{.python .input}
%%tab mxnet
class TransformerEncoderBlock(nn.Block):  #@save
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(
            num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
```

```{.python .input}
%%tab pytorch
class TransformerEncoderBlock(nn.Module):  #@save
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
```

```{.python .input}
%%tab tensorflow
class TransformerEncoderBlock(tf.keras.layers.Layer):  #@save
    """The Transformer encoder block."""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, dropout, bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def call(self, X, valid_lens, **kwargs):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens, **kwargs),
                          **kwargs)
        return self.addnorm2(Y, self.ffn(Y), **kwargs)
```

```{.python .input}
%%tab jax
class TransformerEncoderBlock(nn.Module):  #@save
    """The Transformer encoder block."""
    num_hiddens: int
    ffn_num_hiddens: int
    num_heads: int
    dropout: float
    use_bias: bool = False

    def setup(self):
        self.attention = d2l.MultiHeadAttention(self.num_hiddens, self.num_heads,
                                                self.dropout, self.use_bias)
        self.addnorm1 = AddNorm(self.dropout)
        self.ffn = PositionWiseFFN(self.ffn_num_hiddens, self.num_hiddens)
        self.addnorm2 = AddNorm(self.dropout)

    def __call__(self, X, valid_lens, training=False):
        output, attention_weights = self.attention(X, X, X, valid_lens,
                                                   training=training)
        Y = self.addnorm1(X, output, training=training)
        return self.addnorm2(Y, self.ffn(Y), training=training), attention_weights
```

Ahogy látjuk,
[**a Transformer kódolóban egyetlen réteg sem változtatja meg a bemenet alakját.**]

```{.python .input}
%%tab mxnet
X = d2l.ones((2, 100, 24))
valid_lens = d2l.tensor([3, 2])
encoder_blk = TransformerEncoderBlock(24, 48, 8, 0.5)
encoder_blk.initialize()
d2l.check_shape(encoder_blk(X, valid_lens), X.shape)
```

```{.python .input}
%%tab pytorch
X = d2l.ones((2, 100, 24))
valid_lens = d2l.tensor([3, 2])
encoder_blk = TransformerEncoderBlock(24, 48, 8, 0.5)
encoder_blk.eval()
d2l.check_shape(encoder_blk(X, valid_lens), X.shape)
```

```{.python .input}
%%tab tensorflow
X = tf.ones((2, 100, 24))
valid_lens = tf.constant([3, 2])
norm_shape = [i for i in range(len(X.shape))][1:]
encoder_blk = TransformerEncoderBlock(24, 24, 24, 24, norm_shape, 48, 8, 0.5)
d2l.check_shape(encoder_blk(X, valid_lens, training=False), X.shape)
```

```{.python .input}
%%tab jax
X = jnp.ones((2, 100, 24))
valid_lens = jnp.array([3, 2])
encoder_blk = TransformerEncoderBlock(24, 48, 8, 0.5)
(output, _), _ = encoder_blk.init_with_output(d2l.get_key(), X, valid_lens,
                                              training=False)
d2l.check_shape(output, X.shape)
```

A következő [**Transformer kódoló**] implementációban
`num_blks` példányt egymásra rakunk a fenti `TransformerEncoderBlock` osztályokból.
Mivel rögzített pozícióenkódolást használunk,
amelynek értékei mindig $-1$ és $1$ között vannak,
a tanítható bemeneti beágyazások értékeit
megszorozzuk a beágyazási dimenzió négyzetgyökével,
hogy újraskálázzuk, mielőtt összeadjuk a bemeneti beágyazást és a pozícióenkódolást.

```{.python .input}
%%tab mxnet
class TransformerEncoder(d2l.Encoder):  #@save
    """The Transformer encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for _ in range(num_blks):
            self.blks.add(TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))
        self.initialize()

    def forward(self, X, valid_lens):
        # Mivel a pozícióenkódolás értékei -1 és 1 közé esnek, a beágyazások
        # értékeit a beágyazási dimenzió négyzetgyökével szorozzuk meg, hogy
        # újraskálázzuk őket az összegzés előtt
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

```{.python .input}
%%tab pytorch
class TransformerEncoder(d2l.Encoder):  #@save
    """The Transformer encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):
        # Mivel a pozícióenkódolás értékei -1 és 1 közé esnek, a beágyazások
        # értékeit a beágyazási dimenzió négyzetgyökével szorozzuk meg, hogy
        # újraskálázzuk őket az összegzés előtt
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

```{.python .input}
%%tab tensorflow
class TransformerEncoder(d2l.Encoder):  #@save
    """The Transformer encoder."""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_hiddens, num_heads,
                 num_blks, dropout, bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = [TransformerEncoderBlock(
            key_size, query_size, value_size, num_hiddens, norm_shape,
            ffn_num_hiddens, num_heads, dropout, bias) for _ in range(
            num_blks)]

    def call(self, X, valid_lens, **kwargs):
        # Mivel a pozícióenkódolás értékei -1 és 1 közé esnek, a beágyazások
        # értékeit a beágyazási dimenzió négyzetgyökével szorozzuk meg, hogy
        # újraskálázzuk őket az összegzés előtt
        X = self.pos_encoding(self.embedding(X) * tf.math.sqrt(
            tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens, **kwargs)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

```{.python .input}
%%tab jax
class TransformerEncoder(d2l.Encoder):  #@save
    """The Transformer encoder."""
    vocab_size: int
    num_hiddens:int
    ffn_num_hiddens: int
    num_heads: int
    num_blks: int
    dropout: float
    use_bias: bool = False

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(self.num_hiddens, self.dropout)
        self.blks = [TransformerEncoderBlock(self.num_hiddens,
                                             self.ffn_num_hiddens,
                                             self.num_heads,
                                             self.dropout, self.use_bias)
                     for _ in range(self.num_blks)]

    def __call__(self, X, valid_lens, training=False):
        # Mivel a pozícióenkódolás értékei -1 és 1 közé esnek, a beágyazások
        # értékeit a beágyazási dimenzió négyzetgyökével szorozzuk meg, hogy
        # újraskálázzuk őket az összegzés előtt
        X = self.embedding(X) * math.sqrt(self.num_hiddens)
        X = self.pos_encoding(X, training=training)
        attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X, attention_w = blk(X, valid_lens, training=training)
            attention_weights[i] = attention_w
        # Flax sow API is used to capture intermediate variables
        self.sow('intermediates', 'enc_attention_weights', attention_weights)
        return X
```

Az alábbiakban hiperparamétereket adunk meg [**kétrétegű Transformer kódoló létrehozásához**].
A Transformer kódoló kimenetének alakja (batch-méret, időlépések száma, `num_hiddens`).

```{.python .input}
%%tab mxnet
encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
d2l.check_shape(encoder(np.ones((2, 100)), valid_lens), (2, 100, 24))
```

```{.python .input}
%%tab pytorch
encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
d2l.check_shape(encoder(d2l.ones((2, 100), dtype=torch.long), valid_lens),
                (2, 100, 24))
```

```{.python .input}
%%tab tensorflow
encoder = TransformerEncoder(200, 24, 24, 24, 24, [1, 2], 48, 8, 2, 0.5)
d2l.check_shape(encoder(tf.ones((2, 100)), valid_lens, training=False),
                (2, 100, 24))
```

```{.python .input}
%%tab jax
encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
d2l.check_shape(encoder.init_with_output(d2l.get_key(),
                                         jnp.ones((2, 100), dtype=jnp.int32),
                                         valid_lens)[0],
                (2, 100, 24))
```

## Dekódoló

Ahogy a :numref:`fig_transformer`-ben látható,
[**a Transformer dekódoló
több azonos rétegből áll**].
Minden réteg a következő
`TransformerDecoderBlock` osztályban van implementálva,
amely három alréteget tartalmaz:
dekódoló önfigyelem,
kódoló–dekódoló figyelem,
és pozíciószerinti előre irányított hálózatok.
Ezek az alrétegek
maradék összeköttetést alkalmaznak körülöttük,
amelyet rétegnormalizáció követ.


Ahogy korábban leírtuk ebben a szakaszban,
a maszkolású többfejű dekódoló önfigyelemben
(az első alréteg),
a lekérdezések, kulcsok és értékek
mind az előző dekódolóréteg kimenetéből jönnek.
A szekvencia-szekvencia modellek tanulásakor
a kimeneti szekvencia összes pozíciójának (időlépéseinek) tokenjei
ismertek.
Azonban
előrejelzés közben
a kimeneti szekvencia tokenről tokenre generálódik;
így
bármely dekódolási időlépésnél
csak a generált tokenek
használhatók a dekódoló önfigyelemben.
Az autoregresszió megőrzésére a dekódolóban,
a maszkolású önfigyelme
`dec_valid_lens`-t ad meg, hogy
bármely lekérdezés
csak a lekérdezés pozíciójáig
figyeljen a dekódoló összes pozíciójára.

```{.python .input}
%%tab mxnet
class TransformerDecoderBlock(nn.Block):
    # The i-th block in the Transformer decoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm1 = AddNorm(dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm2 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so state[2][self.i] is None as initialized. When
        # decoding any output sequence token by token during prediction,
        # state[2][self.i] contains representations of the decoded output at
        # the i-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = np.concatenate((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        if autograd.is_training():
            batch_size, num_steps, _ = X.shape
            # A dec_valid_lens alakja: (batch_size, num_steps), ahol minden
            # sor [1, 2, ..., num_steps]
            dec_valid_lens = np.tile(np.arange(1, num_steps + 1, ctx=X.ctx),
                                     (batch_size, 1))
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Kódoló-dekódoló figyelem. Az enc_outputs alakja:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
```

```{.python .input}
%%tab pytorch
class TransformerDecoderBlock(nn.Module):
    # The i-th block in the Transformer decoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so state[2][self.i] is None as initialized. When
        # decoding any output sequence token by token during prediction,
        # state[2][self.i] contains representations of the decoded output at
        # the i-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # A dec_valid_lens alakja: (batch_size, num_steps), ahol minden
            # sor [1, 2, ..., num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Kódoló-dekódoló figyelem. Az enc_outputs alakja:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
```

```{.python .input}
%%tab tensorflow
class TransformerDecoderBlock(tf.keras.layers.Layer):
    # The i-th block in the Transformer decoder
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def call(self, X, state, **kwargs):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so state[2][self.i] is None as initialized. When
        # decoding any output sequence token by token during prediction,
        # state[2][self.i] contains representations of the decoded output at
        # the i-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = tf.concat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if kwargs["training"]:
            batch_size, num_steps, _ = X.shape
            # A dec_valid_lens alakja: (batch_size, num_steps), ahol minden
            # sor [1, 2, ..., num_steps]
            dec_valid_lens = tf.repeat(
                tf.reshape(tf.range(1, num_steps + 1),
                           shape=(-1, num_steps)), repeats=batch_size, axis=0)
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens,
                             **kwargs)
        Y = self.addnorm1(X, X2, **kwargs)
        # Kódoló-dekódoló figyelem. Az enc_outputs alakja:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens,
                             **kwargs)
        Z = self.addnorm2(Y, Y2, **kwargs)
        return self.addnorm3(Z, self.ffn(Z), **kwargs), state
```

```{.python .input}
%%tab jax
class TransformerDecoderBlock(nn.Module):
    # The i-th block in the Transformer decoder
    num_hiddens: int
    ffn_num_hiddens: int
    num_heads: int
    dropout: float
    i: int

    def setup(self):
        self.attention1 = d2l.MultiHeadAttention(self.num_hiddens,
                                                 self.num_heads,
                                                 self.dropout)
        self.addnorm1 = AddNorm(self.dropout)
        self.attention2 = d2l.MultiHeadAttention(self.num_hiddens,
                                                 self.num_heads,
                                                 self.dropout)
        self.addnorm2 = AddNorm(self.dropout)
        self.ffn = PositionWiseFFN(self.ffn_num_hiddens, self.num_hiddens)
        self.addnorm3 = AddNorm(self.dropout)

    def __call__(self, X, state, training=False):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so state[2][self.i] is None as initialized. When
        # decoding any output sequence token by token during prediction,
        # state[2][self.i] contains representations of the decoded output at
        # the i-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = jnp.concatenate((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if training:
            batch_size, num_steps, _ = X.shape
            # A dec_valid_lens alakja: (batch_size, num_steps), ahol minden
            # sor [1, 2, ..., num_steps]
            dec_valid_lens = jnp.tile(jnp.arange(1, num_steps + 1),
                                      (batch_size, 1))
        else:
            dec_valid_lens = None
        # Self-attention
        X2, attention_w1 = self.attention1(X, key_values, key_values,
                                           dec_valid_lens, training=training)
        Y = self.addnorm1(X, X2, training=training)
        # Kódoló-dekódoló figyelem. Az enc_outputs alakja:
        # (batch_size, num_steps, num_hiddens)
        Y2, attention_w2 = self.attention2(Y, enc_outputs, enc_outputs,
                                           enc_valid_lens, training=training)
        Z = self.addnorm2(Y, Y2, training=training)
        return self.addnorm3(Z, self.ffn(Z), training=training), state, attention_w1, attention_w2
```

A kódoló–dekódoló figyelemben végzett skálázott dot product műveletek
és a maradék összeköttetések összeadási műveleteinek elősegítéséhez,
[**a dekódoló jellemződimenziója (`num_hiddens`) azonos a kódolóéval.**]

```{.python .input}
%%tab mxnet
decoder_blk = TransformerDecoderBlock(24, 48, 8, 0.5, 0)
decoder_blk.initialize()
X = np.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
d2l.check_shape(decoder_blk(X, state)[0], X.shape)
```

```{.python .input}
%%tab pytorch
decoder_blk = TransformerDecoderBlock(24, 48, 8, 0.5, 0)
X = d2l.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
d2l.check_shape(decoder_blk(X, state)[0], X.shape)
```

```{.python .input}
%%tab tensorflow
decoder_blk = TransformerDecoderBlock(24, 24, 24, 24, [1, 2], 48, 8, 0.5, 0)
X = tf.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
d2l.check_shape(decoder_blk(X, state, training=False)[0], X.shape)
```

```{.python .input}
%%tab jax
decoder_blk = TransformerDecoderBlock(24, 48, 8, 0.5, 0)
X = d2l.ones((2, 100, 24))
state = [encoder_blk.init_with_output(d2l.get_key(), X, valid_lens)[0][0],
         valid_lens, [None]]
d2l.check_shape(decoder_blk.init_with_output(d2l.get_key(), X, state)[0][0],
                X.shape)
```

Most [**felépítjük a teljes Transformer dekódolót**]
`num_blks` `TransformerDecoderBlock` példányból.
Végül
egy teljes összeköttetésű réteg számítja ki az előrejelzést
az összes lehetséges `vocab_size` kimeneti tokenre.
Mind a dekódoló önfigyelmi súlyai,
mind a kódoló–dekódoló figyelemsúlyok
kerülnek tárolásra a későbbi vizualizációhoz.

```{.python .input}
%%tab mxnet
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add(TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.Dense(vocab_size, flatten=False)
        self.initialize()

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # A dekóder önfigyelmi súlyai
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # A kódoló-dekódoló figyelmi súlyai
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab pytorch
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.LazyLinear(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # A dekóder önfigyelmi súlyai
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # A kódoló-dekódoló figyelmi súlyai
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab tensorflow
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = [TransformerDecoderBlock(
            key_size, query_size, value_size, num_hiddens, norm_shape,
            ffn_num_hiddens, num_heads, dropout, i)
                     for i in range(num_blks)]
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def call(self, X, state, **kwargs):
        X = self.pos_encoding(self.embedding(X) * tf.math.sqrt(
            tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        # 2 attention layers in decoder
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state, **kwargs)
            # A dekóder önfigyelmi súlyai
            self._attention_weights[0][i] = (
                blk.attention1.attention.attention_weights)
            # A kódoló-dekódoló figyelmi súlyai
            self._attention_weights[1][i] = (
                blk.attention2.attention.attention_weights)
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab jax
class TransformerDecoder(nn.Module):
    vocab_size: int
    num_hiddens: int
    ffn_num_hiddens: int
    num_heads: int
    num_blks: int
    dropout: float

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(self.num_hiddens,
                                                   self.dropout)
        self.blks = [TransformerDecoderBlock(self.num_hiddens,
                                             self.ffn_num_hiddens,
                                             self.num_heads, self.dropout, i)
                     for i in range(self.num_blks)]
        self.dense = nn.Dense(self.vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def __call__(self, X, state, training=False):
        X = self.embedding(X) * jnp.sqrt(jnp.float32(self.num_hiddens))
        X = self.pos_encoding(X, training=training)
        attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state, attention_w1, attention_w2 = blk(X, state,
                                                       training=training)
            # A dekóder önfigyelmi súlyai
            attention_weights[0][i] = attention_w1
            # A kódoló-dekódoló figyelmi súlyai
            attention_weights[1][i] = attention_w2
        # Flax sow API is used to capture intermediate variables
        self.sow('intermediates', 'dec_attention_weights', attention_weights)
        return self.dense(X), state
```

## [**Tanítás**]

Példányosítsunk egy kódoló–dekódoló modellt
a Transformer architektúrát követve.
Itt megadjuk, hogy
mind a Transformer kódolónak, mind a Transformer dekódolónak
két rétege van, 4-fejű figyelemmel.
A :numref:`sec_seq2seq_training`-hoz hasonlóan,
a Transformer modellt szekvencia-szekvencia tanulásra tanítjuk
az angol–francia gépi fordítási adathalmazon.

```{.python .input}
%%tab all
data = d2l.MTFraEng(batch_size=128)
num_hiddens, num_blks, dropout = 256, 2, 0.2
ffn_num_hiddens, num_heads = 64, 4
if tab.selected('tensorflow'):
    key_size, query_size, value_size = 256, 256, 256
    norm_shape = [2]
if tab.selected('pytorch', 'mxnet', 'jax'):
    encoder = TransformerEncoder(
        len(data.src_vocab), num_hiddens, ffn_num_hiddens, num_heads,
        num_blks, dropout)
    decoder = TransformerDecoder(
        len(data.tgt_vocab), num_hiddens, ffn_num_hiddens, num_heads,
        num_blks, dropout)
if tab.selected('mxnet', 'pytorch'):
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.001)
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
if tab.selected('jax'):
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.001, training=True)
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        encoder = TransformerEncoder(
            len(data.src_vocab), key_size, query_size, value_size, num_hiddens,
            norm_shape, ffn_num_hiddens, num_heads, num_blks, dropout)
        decoder = TransformerDecoder(
            len(data.tgt_vocab), key_size, query_size, value_size, num_hiddens,
            norm_shape, ffn_num_hiddens, num_heads, num_blks, dropout)
        model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                            lr=0.001)
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1)
trainer.fit(model, data)
```

A tanítás után
a Transformer modellt [**néhány angol mondat**] franciára fordítására használjuk, és kiszámítjuk a BLEU-pontszámokat.

```{.python .input}
%%tab all
engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds, _ = model.predict_step(
        data.build(engs, fras), d2l.try_gpu(), data.num_steps)
if tab.selected('jax'):
    preds, _ = model.predict_step(
        trainer.state.params, data.build(engs, fras), data.num_steps)
for en, fr, p in zip(engs, fras, preds):
    translation = []
    for token in data.tgt_vocab.to_tokens(p):
        if token == '<eos>':
            break
        translation.append(token)
    print(f'{en} => {translation}, bleu,'
          f'{d2l.bleu(" ".join(translation), fr, k=2):.3f}')
```

[**Vizualizáljuk a Transformer figyelemsúlyait**] az utolsó angol mondat franciára fordításakor.
A kódoló önfigyelmi súlyainak alakja
(kódolórétegek száma, figyelemfejek száma, `num_steps` vagy lekérdezések száma, `num_steps` vagy kulcs-érték párok száma).

```{.python .input}
%%tab pytorch, mxnet, tensorflow
_, dec_attention_weights = model.predict_step(
    data.build([engs[-1]], [fras[-1]]), d2l.try_gpu(), data.num_steps, True)
enc_attention_weights = d2l.concat(model.encoder.attention_weights, 0)
shape = (num_blks, num_heads, -1, data.num_steps)
enc_attention_weights = d2l.reshape(enc_attention_weights, shape)
d2l.check_shape(enc_attention_weights,
                (num_blks, num_heads, data.num_steps, data.num_steps))
```

```{.python .input}
%%tab jax
_, (dec_attention_weights, enc_attention_weights) = model.predict_step(
    trainer.state.params, data.build([engs[-1]], [fras[-1]]),
    data.num_steps, True)
enc_attention_weights = d2l.concat(enc_attention_weights, 0)
shape = (num_blks, num_heads, -1, data.num_steps)
enc_attention_weights = d2l.reshape(enc_attention_weights, shape)
d2l.check_shape(enc_attention_weights,
                (num_blks, num_heads, data.num_steps, data.num_steps))
```

A kódoló önfigyelemben
a lekérdezések és kulcsok egyaránt az ugyanazon bemeneti szekvenciából jönnek.
Mivel a kitöltő tokenek nem hordoznak jelentést,
a bemeneti szekvencia megadott érvényes hosszával
egyetlen lekérdezés sem figyel a kitöltő tokenek pozícióira.
Az alábbiakban
a kétrétegű többfejű figyelemsúlyok
soronként kerülnek bemutatásra.
Minden fej függetlenül figyel
a lekérdezések, kulcsok és értékek külön reprezentációs altere alapján.

```{.python .input}
%%tab mxnet, tensorflow, jax
d2l.show_heatmaps(
    enc_attention_weights, xlabel='Kulcspozíciók', ylabel='Lekérdezési pozíciók',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
```

```{.python .input}
%%tab pytorch
d2l.show_heatmaps(
    enc_attention_weights.cpu(), xlabel='Kulcspozíciók',
    ylabel='Lekérdezési pozíciók', titles=['Fej %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))
```

[**A dekódoló önfigyelmi súlyok és a kódoló–dekódoló figyelemsúlyok vizualizálásához
több adatmanipulációra van szükség.**]
Például
a maszkolású figyelemsúlyokat nullával töltjük fel.
Vegyük észre, hogy
mind a dekódoló önfigyelmi súlyok,
mind a kódoló–dekódoló figyelemsúlyok
azonos lekérdezésekkel rendelkeznek:
a szekvencia-kezdő token, amelyet
a kimeneti tokenek és esetleg
szekvencia-vége tokenek követnek.

```{.python .input}
%%tab mxnet
dec_attention_weights_2d = [d2l.tensor(head[0]).tolist()
                            for step in dec_attention_weights
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = d2l.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = d2l.reshape(dec_attention_weights_filled, (
    -1, 2, num_blks, num_heads, data.num_steps))
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.transpose(1, 2, 3, 0, 4)
```

```{.python .input}
%%tab pytorch
dec_attention_weights_2d = [head[0].tolist()
                            for step in dec_attention_weights
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = d2l.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
shape = (-1, 2, num_blks, num_heads, data.num_steps)
dec_attention_weights = d2l.reshape(dec_attention_weights_filled, shape)
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.permute(1, 2, 3, 0, 4)
```

```{.python .input}
%%tab tensorflow
dec_attention_weights_2d = [head[0] for step in dec_attention_weights
                            for attn in step
                            for blk in attn for head in blk]
dec_attention_weights_filled = tf.convert_to_tensor(
    np.asarray(pd.DataFrame(dec_attention_weights_2d).fillna(
        0.0).values).astype(np.float32))
dec_attention_weights = tf.reshape(dec_attention_weights_filled, shape=(
    -1, 2, num_blks, num_heads, data.num_steps))
dec_self_attention_weights, dec_inter_attention_weights = tf.transpose(
    dec_attention_weights, perm=(1, 2, 3, 0, 4))
```

```{.python .input}
%%tab jax
dec_attention_weights_2d = [head[0].tolist() for step in dec_attention_weights
                            for attn in step
                            for blk in attn for head in blk]
dec_attention_weights_filled = d2l.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = dec_attention_weights_filled.reshape(
    (-1, 2, num_blks, num_heads, data.num_steps))
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.transpose(1, 2, 3, 0, 4)
```

```{.python .input}
%%tab all
d2l.check_shape(dec_self_attention_weights,
                (num_blks, num_heads, data.num_steps, data.num_steps))
d2l.check_shape(dec_inter_attention_weights,
                (num_blks, num_heads, data.num_steps, data.num_steps))
```

A dekódoló önfigyelemének autoregresszív tulajdonsága miatt
egyetlen lekérdezés sem figyel a lekérdezés pozíciója utáni kulcs–érték párokra.

```{.python .input}
%%tab all
d2l.show_heatmaps(
    dec_self_attention_weights[:, :, :, :],
    xlabel='Kulcspozíciók', ylabel='Lekérdezési pozíciók',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
```

A kódoló önfigyelemhez hasonlóan,
a bemeneti szekvencia megadott érvényes hosszával,
[**a kimeneti szekvenciából egyetlen lekérdezés sem
figyel a bemeneti szekvencia kitöltő tokenjeihez.**]

```{.python .input}
%%tab all
d2l.show_heatmaps(
    dec_inter_attention_weights, xlabel='Kulcspozíciók',
    ylabel='Lekérdezési pozíciók', titles=['Fej %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))
```

Bár a Transformer architektúrát
eredetileg szekvencia-szekvencia tanulásra javasolták,
ahogy a könyv többi részében felfedezzük,
akár a Transformer kódolót,
akár a Transformer dekódolót
gyakran egyenként is alkalmazzák
különböző mélytanulási feladatokhoz.

## Összefoglalás

A Transformer a kódoló–dekódoló architektúra egy példánya,
bár a kódolót vagy a dekódolót a gyakorlatban egyenként is lehet alkalmazni.
A Transformer architektúrában a többfejű önfigyelmet
a bemeneti szekvencia és a kimeneti szekvencia reprezentálásához használják,
bár a dekódolónak meg kell őriznie az autoregresszív tulajdonságot egy maszkolású verzión keresztül.
Mind a maradék összeköttetések, mind a rétegnormalizáció a Transformerben
fontosak egy nagyon mély modell tanításához.
A pozíciószerinti előre irányított hálózat a Transformer modellben
ugyanazt az MLP-t használva transzformálja a szekvencia összes pozíciójának reprezentációját.


## Feladatok

1. Tanítj mélyebb Transformert a kísérletekben. Hogyan befolyásolja ez a tanítási sebességet és a fordítási teljesítményt?
1. Jó ötlet-e a skálázott dot product figyelmet additív figyelemre cserélni a Transformerben? Miért?
1. A nyelvi modellezéshez a Transformer kódolót, a dekódolót vagy mindkettőt kellene-e használni? Hogyan terveznéd meg ezt a módszert?
1. Milyen kihívásokkal szembesülhetnek a Transformerek, ha a bemeneti szekvenciák nagyon hosszúak? Miért?
1. Hogyan javítanád a Transformerek számítási és memóriahatékonyságát? Tipp: hivatkozhatsz :citet:`Tay.Dehghani.Bahri.ea.2020` áttekintő cikkére.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/348)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1066)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3871)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18031)
:end_tab:
