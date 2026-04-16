```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

#  Sorozatból Sorozatba Irányuló Tanulás Gépi Fordításhoz
:label:`sec_seq2seq`

Az úgynevezett sorozatból sorozatba irányuló problémáknál, mint amilyen a gépi fordítás
(ahogy azt a :numref:`sec_machine_translation` fejezetben tárgyaltuk),
ahol a bemenetek és kimenetek egyaránt
változó hosszúságú, nem igazított sorozatokból állnak,
általában kódoló–dekódoló architektúrákat alkalmazunk
(:numref:`sec_encoder-decoder`).
Ebben a részben
bemutatjuk egy kódoló–dekódoló architektúra alkalmazását,
ahol mind a kódoló, mind a dekódoló
RNN-ként van implementálva,
a gépi fordítás feladatára
:cite:`Sutskever.Vinyals.Le.2014,Cho.Van-Merrienboer.Gulcehre.ea.2014`.

Ebben a kódoló RNN változó hosszúságú sorozatot vesz bemenetként,
és rögzített alakú rejtett állapottá alakítja.
Később, a :numref:`chap_attention-and-transformers` fejezetben
bevezető figyelmi mechanizmusokat,
amelyek lehetővé teszik a kódolt bemenetek elérését
anélkül, hogy a teljes bemenetet
egyetlen rögzített hosszúságú reprezentációba kellene tömöríteni.

Ezután a kimeneti sorozat generálásához,
tokenenként,
a dekódoló modell,
amely egy külön RNN-ből áll,
minden egymást követő cél tokent megjósol
mind a bemeneti sorozat,
mind a kimenetben már megjósolt tokenek alapján.
A tanítás során a dekódolót jellemzően
a korábbi tokenekre kondicionálják
a hivatalos "ground truth" címkékből.
A teszt idején azonban minden dekódoló kimenetet
a már megjósolt tokenekre kell kondicionálni.
Vegyük figyelembe, hogy ha figyelmen kívül hagyjuk a kódolót,
a sorozatból sorozatba irányuló architektúra dekódolója
pontosan úgy viselkedik, mint egy normális nyelvi modell.
A :numref:`fig_seq2seq` ábra szemlélteti,
hogyan használható két RNN
sorozatból sorozatba irányuló tanuláshoz
gépi fordításban.


![Sorozatból sorozatba irányuló tanulás egy RNN kódolóval és egy RNN dekódolóval.](../img/seq2seq.svg)
:label:`fig_seq2seq`

A :numref:`fig_seq2seq` ábrán
a speciális "&lt;eos&gt;" token
jelzi a sorozat végét.
A modell leállíthatja az előrejelzéseket,
miután ezt a tokent generálta.
Az RNN dekódoló kezdeti időlépésénél
két speciális tervezési döntést kell figyelembe venni:
Először, minden bemenetet egy speciális
sorozat-kezdő "&lt;bos&gt;" tokennel kezdünk.
Másodszor, a kódoló végső rejtett állapotát
a dekódolóba táplálhatjuk
minden egyes dekódolási időlépésnél :cite:`Cho.Van-Merrienboer.Gulcehre.ea.2014`.
Néhány más tervezési megközelítésben,
például :citet:`Sutskever.Vinyals.Le.2014` esetében,
az RNN kódoló végső rejtett állapotát
csak az első dekódolási lépésnél használják
a dekódoló rejtett állapotának inicializálására.

```{.python .input}
%%tab mxnet
import collections
from d2l import mxnet as d2l
import math
from mxnet import np, npx, init, gluon, autograd
from mxnet.gluon import nn, rnn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import collections
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input}
%%tab jax
import collections
from d2l import jax as d2l
from flax import linen as nn
from functools import partial
import jax
from jax import numpy as jnp
import math
import optax
```

## Teacher Forcing (tanárkényszeres tanítás)

Miközben a kódoló futtatása a bemeneti sorozaton
viszonylag egyszerű,
a dekódoló bemenetének és kimenetének kezelése több gondosságot igényel.
A leggyakoribb megközelítést néha *teacher forcing*-nak nevezzük.
Ebben az esetben az eredeti célsorozat (token-címkék)
bemenetként kerül a dekódolóba.
Konkrétabban,
a speciális sorozat-kezdő token
és az eredeti célsorozat,
az utolsó token kizárásával,
összefűzve kerül a dekódoló bemenetébe,
míg a dekódoló kimenete (tanítási címkék)
az eredeti célsorozat,
egy tokennel eltolva:
"&lt;bos&gt;", "Ils", "regardent", "." $\rightarrow$
"Ils", "regardent", ".", "&lt;eos&gt;" (:numref:`fig_seq2seq`).

A :numref:`subsec_loading-seq-fixed-len` fejezetben lévő implementációnk
előkészítette a tanítási adatokat a teacher forcing számára,
ahol az önfelügyelt tanuláshoz szükséges token-eltolás
hasonló a :numref:`sec_language-model` fejezetbeli
nyelvi modellek tanításához.
Alternatív megközelítés az,
hogy az előző időlépés *megjósolt* tokenjét
adjuk be az aktuális dekódoló bemenetként.


A következőkben részletesebben elmagyarázzuk a :numref:`fig_seq2seq` ábrán látható tervet.
Ezt a modellt gépi fordításhoz tanítjuk majd
az angol–francia adathalmazon, amelyet a
:numref:`sec_machine_translation` fejezetben mutattunk be.

## Kódoló

Felidézzük, hogy a kódoló változó hosszúságú bemeneti sorozatot
rögzített alakú *kontextusvariábissá* $\mathbf{c}$ alakítja (lásd :numref:`fig_seq2seq`).


Vizsgáljunk egy egyelemű sorozat-példát (batch méret 1).
Tegyük fel, hogy a bemeneti sorozat $x_1, \ldots, x_T$,
ahol $x_t$ a $t$-edik token.
A $t$ időlépésnél az RNN
az $x_t$-hez tartozó $\mathbf{x}_t$ jellemzővektort
és az előző időlépés rejtett állapotát $\mathbf{h}_{t-1}$
az aktuális rejtett állapottá $\mathbf{h}_t$ alakítja.
Egy $f$ függvénnyel kifejezhetjük
az RNN rekurrens rétegének transzformációját:

$$\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1}). $$

Általánosan, a kódoló az összes időlépés rejtett állapotait
kontextusvariábissá alakítja egy testreszabott $q$ függvényen keresztül:

$$\mathbf{c} =  q(\mathbf{h}_1, \ldots, \mathbf{h}_T).$$

Például a :numref:`fig_seq2seq` ábrán,
a kontextusvariábis csupán a kódoló RNN-nek a bemeneti sorozat
utolsó tokenjének feldolgozása utáni
rejtett állapota $\mathbf{h}_T$.

Ebben a példában egyirányú RNN-t használtunk
a kódoló tervezéséhez,
ahol a rejtett állapot csak a rejtett állapot időlépésénél
és az azt megelőző bemeneti alsorozattól függ.
Kétirányú RNN-ek segítségével is építhetünk kódolókat.
Ebben az esetben egy rejtett állapot az időlépés előtti és utáni alsorozattól is függ
(beleértve a bemenetet az aktuális időlépésnél),
amely az egész sorozat információját kódolja.


Most **implementáljuk az RNN kódolót**.
Vegyük figyelembe, hogy *beágyazási réteget* alkalmazunk,
hogy megkapjuk a bemeneti sorozat minden tokenjének jellemzővektorát.
A beágyazási réteg súlya egy mátrix,
amelynek sorainak száma megfelel
a bemeneti szókincs méretének (`vocab_size`),
az oszlopok száma pedig
a jellemzővektor dimenziójának (`embed_size`).
Bármely bemeneti token $i$ indexéhez,
a beágyazási réteg lekéri a súlymátrix $i$-edik sorát
(0-tól kezdve),
hogy visszaadja a jellemzővektorát.
Ebben a kódolót többrétegű GRU-val implementáljuk.

```{.python .input}
%%tab mxnet
class Seq2SeqEncoder(d2l.Encoder):  #@save
    """RNN kódoló sorozatból sorozatba irányuló tanuláshoz."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout)
        self.initialize(init.Xavier())
            
    def forward(self, X, *args):
        # X alakja: (batch_size, num_steps)
        embs = self.embedding(d2l.transpose(X))
        # embs alakja: (num_steps, batch_size, embed_size)    
        outputs, state = self.rnn(embs)
        # outputs alakja: (num_steps, batch_size, num_hiddens)
        # state alakja: (num_layers, batch_size, num_hiddens)
        return outputs, state
```

```{.python .input}
%%tab pytorch
def init_seq2seq(module):  #@save
    """Súlyok inicializálása sorozatból sorozatba irányuló tanuláshoz."""
    if type(module) == nn.Linear:
         nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])
```

```{.python .input}
%%tab pytorch
class Seq2SeqEncoder(d2l.Encoder):  #@save
    """RNN kódoló sorozatból sorozatba irányuló tanuláshoz."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(embed_size, num_hiddens, num_layers, dropout)
        self.apply(init_seq2seq)
            
    def forward(self, X, *args):
        # X alakja: (batch_size, num_steps)
        embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int64))
        # embs alakja: (num_steps, batch_size, embed_size)
        outputs, state = self.rnn(embs)
        # outputs alakja: (num_steps, batch_size, num_hiddens)
        # state alakja: (num_layers, batch_size, num_hiddens)
        return outputs, state
```

```{.python .input}
%%tab tensorflow
class Seq2SeqEncoder(d2l.Encoder):  #@save
    """RNN kódoló sorozatból sorozatba irányuló tanuláshoz."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout)
            
    def call(self, X, *args):
        # X alakja: (batch_size, num_steps)
        embs = self.embedding(d2l.transpose(X))
        # embs alakja: (num_steps, batch_size, embed_size)    
        outputs, state = self.rnn(embs)
        # outputs alakja: (num_steps, batch_size, num_hiddens)
        # state alakja: (num_layers, batch_size, num_hiddens)
        return outputs, state
```

```{.python .input}
%%tab jax
class Seq2SeqEncoder(d2l.Encoder):  #@save
    """RNN kódoló sorozatból sorozatba irányuló tanuláshoz."""
    vocab_size: int
    embed_size: int
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.embed_size)
        self.rnn = d2l.GRU(self.num_hiddens, self.num_layers, self.dropout)

    def __call__(self, X, *args, training=False):
        # X alakja: (batch_size, num_steps)
        embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int32))
        # embs alakja: (num_steps, batch_size, embed_size)
        outputs, state = self.rnn(embs, training=training)
        # outputs alakja: (num_steps, batch_size, num_hiddens)
        # state alakja: (num_layers, batch_size, num_hiddens)
        return outputs, state
```

Konkrét példával
**szemléltetjük a fenti kódoló-implementációt.**
Alább egy kétrétegű GRU kódolót hozunk létre,
amelynek 16 rejtett egysége van.
Adott egy szekvencia-bemenet minibatch `X`
(batch méret $=4$; időlépések száma $=9$),
a végső réteg rejtett állapotai
az összes időlépésnél
(`enc_outputs` amelyet a kódoló rekurrens rétegei adnak vissza)
egy (időlépések száma, batch méret, rejtett egységek száma) alakú tenzor.

```{.python .input}
%%tab all
vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
batch_size, num_steps = 4, 9
encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
X = d2l.zeros((batch_size, num_steps))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    enc_outputs, enc_state = encoder(X)
if tab.selected('jax'):
    (enc_outputs, enc_state), _ = encoder.init_with_output(d2l.get_key(), X)

d2l.check_shape(enc_outputs, (num_steps, batch_size, num_hiddens))
```

Mivel itt GRU-t alkalmazunk,
a többrétegű rejtett állapotok alakja
az utolsó időlépésnél
(rejtett rétegek száma, batch méret, rejtett egységek száma).

```{.python .input}
%%tab all
if tab.selected('mxnet', 'pytorch', 'jax'):
    d2l.check_shape(enc_state, (num_layers, batch_size, num_hiddens))
if tab.selected('tensorflow'):
    d2l.check_len(enc_state, num_layers)
    d2l.check_shape(enc_state[0], (batch_size, num_hiddens))
```

## **Dekódoló**
:label:`sec_seq2seq_decoder`

Adott egy $y_1, y_2, \ldots, y_{T'}$ célkimeneti sorozat
minden $t'$ időlépésnél
(az $t^\prime$ jelölést használjuk, hogy megkülönböztessük a bemeneti sorozat időlépéseitől),
a dekódoló becsült valószínűséget rendel
minden lehetséges tokenhez, amely a $y_{t'+1}$ lépésnél jelenhet meg,
kondicionálva a cél korábbi tokenjeire
$y_1, \ldots, y_{t'}$
és a kontextusvariábisra
$\mathbf{c}$, azaz $P(y_{t'+1} \mid y_1, \ldots, y_{t'}, \mathbf{c})$.

A $t^\prime+1$ következő token előrejelzéséhez a célsorozatban,
az RNN dekódoló az előző lépés cél tokenjét $y_{t^\prime}$,
az előző időlépés RNN rejtett állapotát $\mathbf{s}_{t^\prime-1}$,
és a kontextusvariábist $\mathbf{c}$ veszi bemenetnek,
és az aktuális időlépés rejtett állapotává $\mathbf{s}_{t^\prime}$ alakítja.
Egy $g$ függvénnyel kifejezhetjük
a dekódoló rejtett rétegének transzformációját:

$$\mathbf{s}_{t^\prime} = g(y_{t^\prime-1}, \mathbf{c}, \mathbf{s}_{t^\prime-1}).$$
:eqlabel:`eq_seq2seq_s_t`

A dekódoló rejtett állapotának megszerzése után
egy kimeneti réteget és a softmax műveletet alkalmazhatjuk
a $p(y_{t^{\prime}+1} \mid y_1, \ldots, y_{t^\prime}, \mathbf{c})$ prediktív eloszlás kiszámításához
a következő $t^\prime+1$ kimeneti tokenre vonatkozóan.

A :numref:`fig_seq2seq` ábrát követve,
a dekódoló alábbi implementálásánál
közvetlenül a kódoló végső időlépésének rejtett állapotát használjuk
a dekódoló rejtett állapotának inicializálásához.
Ez megköveteli, hogy az RNN kódolónak és az RNN dekódolónak
azonos számú rétege és rejtett egysége legyen.
A kódolt bemeneti sorozat információjának további beépítéséhez
a kontextusvariábist összefűzzük
a dekódoló bemenetével az összes időlépésnél.
A kimeneti token valószínűség-eloszlásának előrejelzéséhez
egy teljesen összekötött réteget alkalmazunk,
hogy az RNN dekódoló végső rétegének rejtett állapotát transzformáljuk.

```{.python .input}
%%tab mxnet
class Seq2SeqDecoder(d2l.Decoder):
    """RNN dekódoló sorozatból sorozatba irányuló tanuláshoz."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)
        self.initialize(init.Xavier())
            
    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs 

    def forward(self, X, state):
        # X alakja: (batch_size, num_steps)
        # embs alakja: (num_steps, batch_size, embed_size)
        embs = self.embedding(d2l.transpose(X))
        enc_output, hidden_state = state
        # context alakja: (batch_size, num_hiddens)
        context = enc_output[-1]
        # context szórása (num_steps, batch_size, num_hiddens) alakra
        context = np.tile(context, (embs.shape[0], 1, 1))
        # Összefűzés a jellemződimenziónál
        embs_and_context = d2l.concat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        outputs = d2l.swapaxes(self.dense(outputs), 0, 1)
        # outputs alakja: (batch_size, num_steps, vocab_size)
        # hidden_state alakja: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]
```

```{.python .input}
%%tab pytorch
class Seq2SeqDecoder(d2l.Decoder):
    """RNN dekódoló sorozatból sorozatba irányuló tanuláshoz."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(embed_size+num_hiddens, num_hiddens,
                           num_layers, dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(init_seq2seq)
            
    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def forward(self, X, state):
        # X alakja: (batch_size, num_steps)
        # embs alakja: (num_steps, batch_size, embed_size)
        embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int32))
        enc_output, hidden_state = state
        # context alakja: (batch_size, num_hiddens)
        context = enc_output[-1]
        # context szórása (num_steps, batch_size, num_hiddens) alakra
        context = context.repeat(embs.shape[0], 1, 1)
        # Összefűzés a jellemződimenziónál
        embs_and_context = d2l.concat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        outputs = d2l.swapaxes(self.dense(outputs), 0, 1)
        # outputs alakja: (batch_size, num_steps, vocab_size)
        # hidden_state alakja: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]
```

```{.python .input}
%%tab tensorflow
class Seq2SeqDecoder(d2l.Decoder):
    """RNN dekódoló sorozatból sorozatba irányuló tanuláshoz."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout)
        self.dense = tf.keras.layers.Dense(vocab_size)
            
    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def call(self, X, state):
        # X alakja: (batch_size, num_steps)
        # embs alakja: (num_steps, batch_size, embed_size)
        embs = self.embedding(d2l.transpose(X))
        enc_output, hidden_state = state
        # context alakja: (batch_size, num_hiddens)
        context = enc_output[-1]
        # context szórása (num_steps, batch_size, num_hiddens) alakra
        context = tf.tile(tf.expand_dims(context, 0), (embs.shape[0], 1, 1))
        # Összefűzés a jellemződimenziónál
        embs_and_context = d2l.concat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        outputs = d2l.transpose(self.dense(outputs), (1, 0, 2))
        # outputs alakja: (batch_size, num_steps, vocab_size)
        # hidden_state alakja: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]
```

```{.python .input}
%%tab jax
class Seq2SeqDecoder(d2l.Decoder):
    """RNN dekódoló sorozatból sorozatba irányuló tanuláshoz."""
    vocab_size: int
    embed_size: int
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.embed_size)
        self.rnn = d2l.GRU(self.num_hiddens, self.num_layers, self.dropout)
        self.dense = nn.Dense(self.vocab_size)

    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def __call__(self, X, state, training=False):
        # X alakja: (batch_size, num_steps)
        # embs alakja: (num_steps, batch_size, embed_size)
        embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int32))
        enc_output, hidden_state = state
        # context alakja: (batch_size, num_hiddens)
        context = enc_output[-1]
        # context szórása (num_steps, batch_size, num_hiddens) alakra
        context = jnp.tile(context, (embs.shape[0], 1, 1))
        # Összefűzés a jellemződimenziónál
        embs_and_context = d2l.concat((embs, context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state,
                                         training=training)
        outputs = d2l.swapaxes(self.dense(outputs), 0, 1)
        # outputs alakja: (batch_size, num_steps, vocab_size)
        # hidden_state alakja: (num_layers, batch_size, num_hiddens)
        return outputs, [enc_output, hidden_state]
```

A **implementált dekódoló szemléltetéséhez**
alább példányosítjuk azt a fent említett kódolóból vett azonos hiperparaméterekkel.
Ahogy látható, a dekódoló kimenetének alakja (batch méret, időlépések száma, szókincs mérete),
ahol a tenzor utolsó dimenziója tárolja a megjósolt token-eloszlást.

```{.python .input}
%%tab all
decoder = Seq2SeqDecoder(vocab_size, embed_size, num_hiddens, num_layers)
if tab.selected('mxnet', 'pytorch', 'tensorflow'):
    state = decoder.init_state(encoder(X))
    dec_outputs, state = decoder(X, state)
if tab.selected('jax'):
    state = decoder.init_state(encoder.init_with_output(d2l.get_key(), X)[0])
    (dec_outputs, state), _ = decoder.init_with_output(d2l.get_key(), X,
                                                       state)


d2l.check_shape(dec_outputs, (batch_size, num_steps, vocab_size))
if tab.selected('mxnet', 'pytorch', 'jax'):
    d2l.check_shape(state[1], (num_layers, batch_size, num_hiddens))
if tab.selected('tensorflow'):
    d2l.check_len(state[1], num_layers)
    d2l.check_shape(state[1][0], (batch_size, num_hiddens))
```

A fenti RNN kódoló–dekódoló modell rétegei
a :numref:`fig_seq2seq_details` ábrán összegezve láthatók.

![Rétegek egy RNN kódoló–dekódoló modellben.](../img/seq2seq-details.svg)
:label:`fig_seq2seq_details`



## Kódoló–Dekódoló Sorozatból Sorozatba Irányuló Tanuláshoz


Mindezt kódban összefoglalva a következőt kapjuk:

```{.python .input}
%%tab pytorch, tensorflow, mxnet
class Seq2Seq(d2l.EncoderDecoder):  #@save
    """RNN kódoló–dekódoló sorozatból sorozatba irányuló tanuláshoz."""
    def __init__(self, encoder, decoder, tgt_pad, lr):
        super().__init__(encoder, decoder)
        self.save_hyperparameters()
        
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        
    def configure_optimizers(self):
        # Adam optimalizáló kerül alkalmazásra
        if tab.selected('mxnet'):
            return gluon.Trainer(self.parameters(), 'adam',
                                 {'learning_rate': self.lr})
        if tab.selected('pytorch'):
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        if tab.selected('tensorflow'):
            return tf.keras.optimizers.Adam(learning_rate=self.lr)
```

```{.python .input}
%%tab jax
class Seq2Seq(d2l.EncoderDecoder):  #@save
    """RNN kódoló–dekódoló sorozatból sorozatba irányuló tanuláshoz."""
    encoder: nn.Module
    decoder: nn.Module
    tgt_pad: int
    lr: float

    def validation_step(self, params, batch, state):
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        # Adam optimalizáló kerül alkalmazásra
        return optax.adam(learning_rate=self.lr)
```

## Veszteségfüggvény Maszkolással

Minden időlépésnél a dekódoló
valószínűség-eloszlást jósol a kimeneti tokenekre vonatkozóan.
A nyelvi modellezéshez hasonlóan
softmax-ot alkalmazhatunk az eloszlás eléréséhez,
és keresztentrópia-veszteséget számíthatunk az optimalizáláshoz.
Felidézzük a :numref:`sec_machine_translation` fejezetből,
hogy a speciális kitöltési tokenek
a sorozatok végéhez lesznek fűzve,
így a különböző hosszúságú sorozatok
hatékonyan betölthetők
azonos alakú minibatchekbe.
Azonban a kitöltési tokenek előrejelzését
ki kell zárni a veszteség-számításból.
Ebből a célból
**lényegtelen bejegyzéseket nullával maszkolhatjuk**,
így bármely lényegtelen előrejelzés
nullával szorozva nullává válik.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(Seq2Seq)
def loss(self, Y_hat, Y):
    l = super(Seq2Seq, self).loss(Y_hat, Y, averaged=False)
    mask = d2l.astype(d2l.reshape(Y, -1) != self.tgt_pad, d2l.float32)
    return d2l.reduce_sum(l * mask) / d2l.reduce_sum(mask)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(Seq2Seq)
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=False):
    Y_hat = state.apply_fn({'params': params}, *X,
                           rngs={'dropout': state.dropout_rng})
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    l = fn(Y_hat, Y)
    mask = d2l.astype(d2l.reshape(Y, -1) != self.tgt_pad, d2l.float32)
    return d2l.reduce_sum(l * mask) / d2l.reduce_sum(mask), {}
```

## **Tanítás**
:label:`sec_seq2seq_training`

Most **létrehozhatunk és taníthatunk egy RNN kódoló–dekódoló modellt**
sorozatból sorozatba irányuló tanuláshoz a gépi fordítás adathalmazán.

```{.python .input}
%%tab all
data = d2l.MTFraEng(batch_size=128) 
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
if tab.selected('mxnet', 'pytorch', 'jax'):
    encoder = Seq2SeqEncoder(
        len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(
        len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
if tab.selected('mxnet', 'pytorch'):
    model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                    lr=0.005)
if tab.selected('jax'):
    model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                    lr=0.005, training=True)
if tab.selected('mxnet', 'pytorch', 'jax'):
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        encoder = Seq2SeqEncoder(
            len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
        decoder = Seq2SeqDecoder(
            len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
        model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.005)
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1)
trainer.fit(model, data)
```

## **Előrejelzés**

A kimeneti sorozat előrejelzéséhez
minden lépésnél
az előző időlépés megjósolt tokenjét adjuk be a dekódolóba bemenetként.
Egy egyszerű stratégia az, hogy minden lépésnél
azt a tokent vesszük, amelyhez a dekódoló
a legmagasabb valószínűséget rendelte.
Ahogy a tanítás során, a kezdeti időlépésnél
a sorozat-kezdő ("&lt;bos&gt;") tokent
adjuk be a dekódolóba.
Ez az előrejelzési folyamat
a :numref:`fig_seq2seq_predict` ábrán látható.
Amikor a sorozat-vége ("&lt;eos&gt;") tokent jósolja meg,
a kimeneti sorozat előrejelzése befejeződik.


![A kimeneti sorozat tokenenként való előrejelzése egy RNN kódoló–dekódolóval.](../img/seq2seq-predict.svg)
:label:`fig_seq2seq_predict`

A következő részben kifinomultabb stratégiákat mutatunk be
beam search alapján (:numref:`sec_beam-search`).

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(d2l.EncoderDecoder)  #@save
def predict_step(self, batch, device, num_steps,
                 save_attention_weights=False):
    if tab.selected('mxnet', 'pytorch'):
        batch = [d2l.to(a, device) for a in batch]
    src, tgt, src_valid_len, _ = batch
    if tab.selected('mxnet', 'pytorch'):
        enc_all_outputs = self.encoder(src, src_valid_len)
    if tab.selected('tensorflow'):
        enc_all_outputs = self.encoder(src, src_valid_len, training=False)
    dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
    outputs, attention_weights = [d2l.expand_dims(tgt[:, 0], 1), ], []
    for _ in range(num_steps):
        if tab.selected('mxnet', 'pytorch'):
            Y, dec_state = self.decoder(outputs[-1], dec_state)
        if tab.selected('tensorflow'):
            Y, dec_state = self.decoder(outputs[-1], dec_state, training=False)
        outputs.append(d2l.argmax(Y, 2))
        # Figyelmi súlyok mentése (később tárgyaljuk)
        if save_attention_weights:
            attention_weights.append(self.decoder.attention_weights)
    return d2l.concat(outputs[1:], 1), attention_weights
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.EncoderDecoder)  #@save
def predict_step(self, params, batch, num_steps,
                 save_attention_weights=False):
    src, tgt, src_valid_len, _ = batch
    enc_all_outputs, inter_enc_vars = self.encoder.apply(
        {'params': params['encoder']}, src, src_valid_len, training=False,
        mutable='intermediates')
    # Kódoló figyelmi súlyainak mentése, ha az inter_enc_vars
    # kódoló figyelmi súlyokat tartalmaz (később tárgyaljuk)
    enc_attention_weights = []
    if bool(inter_enc_vars) and save_attention_weights:
        # Kódoló figyelmi súlyok az intermediates gyűjteményben tárolva
        enc_attention_weights = inter_enc_vars[
            'intermediates']['enc_attention_weights'][0]

    dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
    outputs, attention_weights = [d2l.expand_dims(tgt[:,0], 1), ], []
    for _ in range(num_steps):
        (Y, dec_state), inter_dec_vars = self.decoder.apply(
            {'params': params['decoder']}, outputs[-1], dec_state,
            training=False, mutable='intermediates')
        outputs.append(d2l.argmax(Y, 2))
        # Figyelmi súlyok mentése (később tárgyaljuk)
        if save_attention_weights:
            # Dekódoló figyelmi súlyok az intermediates gyűjteményben tárolva
            dec_attention_weights = inter_dec_vars[
                'intermediates']['dec_attention_weights'][0]
            attention_weights.append(dec_attention_weights)
    return d2l.concat(outputs[1:], 1), (attention_weights,
                                        enc_attention_weights)
```

## Az Előrejelzett Sorozatok Kiértékelése

Az előrejelzett sorozatot kiértékelhetjük
összehasonlítva a célsorozattal (a ground truth).
De pontosan mi a megfelelő mérőszám
két sorozat hasonlóságának összehasonlításához?


A BLEU (Bilingual Evaluation Understudy),
amelyet eredetileg a gépi fordítás eredményeinek kiértékelésére javasoltak :cite:`Papineni.Roukos.Ward.ea.2002`,
széles körben alkalmazzák
a különböző alkalmazásokhoz való kimeneti sorozatok minőségének mérésére.
Alapelvként, bármely $n$-gram (:numref:`subsec_markov-models-and-n-grams`) esetén az előrejelzett sorozatban, a BLEU kiértékeli, hogy ez az $n$-gram megjelenik-e
a célsorozatban.

Jelöljük $p_n$-nel egy $n$-gram pontosságát,
amelyet az előrejelzett és célsorozatokban egyező $n$-gramok száma
és az előrejelzett sorozatban lévő $n$-gramok számának arányaként definiálunk.
Magyarázatképpen, adott egy $A$, $B$, $C$, $D$, $E$, $F$ célsorozat
és egy $A$, $B$, $B$, $C$, $D$ előrejelzett sorozat,
kapjuk, hogy $p_1 = 4/5$, $p_2 = 3/4$, $p_3 = 1/3$ és $p_4 = 0$.
Legyen $\textrm{len}_{\textrm{label}}$ és $\textrm{len}_{\textrm{pred}}$
a célsorozat és az előrejelzett sorozat tokenjeinek száma.
A BLEU definíciója ezután:

$$ \exp\left(\min\left(0, 1 - \frac{\textrm{len}_{\textrm{label}}}{\textrm{len}_{\textrm{pred}}}\right)\right) \prod_{n=1}^k p_n^{1/2^n},$$
:eqlabel:`eq_bleu`

ahol $k$ a leghosszabb $n$-gram az egyeztetéshez.

A :eqref:`eq_bleu` képletbeli BLEU definíciója alapján,
ha az előrejelzett sorozat megegyezik a célsorozattal, a BLEU értéke 1.
Ráadásul,
mivel a hosszabb $n$-gramok egyeztetése nehezebb,
a BLEU nagyobb súlyt rendel
a hosszabb $n$-gram egyezéshez.
Konkrétan, ha $p_n$ rögzített,
$p_n^{1/2^n}$ növekszik, ahogy $n$ nő (az eredeti cikk $p_n^{1/n}$-t használ).
Ráadásul,
mivel a rövidebb sorozatok előrejelzése
általában magasabb $p_n$ értéket eredményez,
a szorzati tagot megelőző együttható a :eqref:`eq_bleu` képletben
bünteti a rövidebb előrejelzett sorozatokat.
Például, ha $k=2$,
adott a $A$, $B$, $C$, $D$, $E$, $F$ célsorozat és az $A$, $B$ előrejelzett sorozat,
bár $p_1 = p_2 = 1$, a büntető tényező $\exp(1-6/2) \approx 0.14$ csökkenti a BLEU értékét.

**Implementáljuk a BLEU mértéket** az alábbi módon.

```{.python .input}
%%tab all
def bleu(pred_seq, label_seq, k):  #@save
    """BLEU-pontszám kiszámítása."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
```

Végül
a tanított RNN kódoló–dekódolót használjuk
**néhány angol mondat franciára fordítására**
és az eredmények BLEU értékének kiszámítására.

```{.python .input}
%%tab all
engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds, _ = model.predict_step(
        data.build(engs, fras), d2l.try_gpu(), data.num_steps)
if tab.selected('jax'):
    preds, _ = model.predict_step(trainer.state.params, data.build(engs, fras),
                                  data.num_steps)
for en, fr, p in zip(engs, fras, preds):
    translation = []
    for token in data.tgt_vocab.to_tokens(p):
        if token == '<eos>':
            break
        translation.append(token)        
    print(f'{en} => {translation}, bleu,'
          f'{bleu(" ".join(translation), fr, k=2):.3f}')
```

## Összefoglalás

A kódoló–dekódoló architektúra tervét követve, két RNN-t alkalmazhatunk sorozatból sorozatba irányuló tanulás modelljének tervezéséhez.
A kódoló–dekódoló tanítás során a teacher forcing megközelítés az eredeti kimeneti sorozatokat (az előrejelzésekkel szemben) adja be a dekódolóba.
A kódoló és a dekódoló implementálásakor többrétegű RNN-eket alkalmazhatunk.
Maszkok segítségével kizárhatjuk a lényegtelen számításokat, például a veszteség számításakor.
A kimeneti sorozatok kiértékelésére
a BLEU egy népszerű mérőszám, amely $n$-gramokat egyeztet az előrejelzett sorozat és a célsorozat között.


## Feladatok

1. Módosíthatod a hiperparamétereket a fordítási eredmények javítása érdekében?
1. Futtasd újra a kísérletet maszkok nélkül a veszteség-számításban. Milyen eredményeket figyelsz meg? Miért?
1. Ha a kódoló és a dekódoló különbözik a rétegek számában vagy a rejtett egységek számában, hogyan inicializálható a dekódoló rejtett állapota?
1. A tanítás során cseréld fel a teacher forcing-ot azzal, hogy az előző időlépés előrejelzését adod be a dekódolónak. Hogyan befolyásolja ez a teljesítményt?
1. Futtasd újra a kísérletet a GRU LSTM-re cserélésével.
1. Vannak más módszerek a dekódoló kimeneti rétegének tervezésére?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/345)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1062)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3865)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18022)
:end_tab:
