```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# A Bahdanau-féle figyelemmechanizmus
:label:`sec_seq2seq_attention`

Amikor a :numref:`sec_seq2seq`-ben a gépi fordítással találkoztunk,
kódoló–dekódoló architektúrát terveztünk a szekvencia-szekvencia tanuláshoz
két RNN alapján :cite:`Sutskever.Vinyals.Le.2014`.
Konkrétan az RNN kódoló egy változó hosszúságú szekvenciát
egy *rögzített alakú* kontextusvariábissá alakít.
Ezután az RNN dekódoló tokenről tokenre generálja a kimeneti (cél) szekvenciát
a generált tokenek és a kontextusvariábis alapján.

Idézzük fel a :numref:`fig_seq2seq_details`-t, amelyet megismétlünk (:numref:`fig_s2s_attention_state`) néhány további részlettel. Hagyományosan egy RNN-ben a forrásbeli szekvencia összes releváns információját a kódoló lefordítja valamilyen belső *rögzített dimenziójú* állapotreprezentációba. Pontosan ez az állapot, amelyet a dekódoló a lefordított szekvencia generálásának teljes és kizárólagos információforrásként használ. Más szóval, a szekvencia-szekvencia mechanizmus a közbülső állapotot a bemenetként szolgált sztring elégséges statisztikájaként kezeli.

![Szekvencia-szekvencia modell. A kódoló által generált állapot az egyetlen információrész, amelyet a kódoló és a dekódoló megoszt.](../img/seq2seq-state.svg)
:label:`fig_s2s_attention_state`

Bár ez rövid szekvenciák esetén teljesen ésszerű, egyértelmű, hogy hosszúak esetén nem kivitelezhető, mint például egy könyvfejezet vagy akár csak egy nagyon hosszú mondat. Végül is nem sokára egyszerűen nem lesz elegendő „hely" a közbülső reprezentációban a forrásszekvenciában fontos dolgok tárolásához. Következésképpen a dekódoló nem tudja majd lefordítani a hosszú és összetett mondatokat. Az elsők egyike, aki ezzel szembesült, :citet:`Graves.2013` volt, aki RNN-t próbált tervezni kézírásos szöveg generálásához. Mivel a forrásbeli szövegnek tetszőleges hossza van, differenciálható figyelemmodellt terveztek
a szöveg karaktereinek igazításához a sokkal hosszabb toll nyomához,
ahol az igazítás csak egy irányba halad. Ez viszont a beszédfelismerés dekódolási algoritmusaira épít, pl. rejtett Markov modellekre :cite:`rabiner1993fundamentals`.

Az igazítás megtanulásának ötletétől ihletve,
:citet:`Bahdanau.Cho.Bengio.2014` egy differenciálható figyelemmodellt javasolt
az egyirányú igazítás korlátozása *nélkül*.
Egy token előrejelzésekor,
ha nem minden bemeneti token releváns,
a modell csak a bemeneti szekvencia
az aktuális előrejelzés szempontjából relevánsnak ítélt részeire
igazodik (vagy figyel). Ezt ezután az állapot frissítéséhez használják, mielőtt a következő tokent generálnák. Bár leírásában elég ártalmatlannak tűnik, ez a *Bahdanau figyelemmechanizmus* vitathatatlanul a mélytanulás elmúlt évtizedének egyik legbefolyásosabb ötletévé vált, amelyből Transformerek :cite:`Vaswani.Shazeer.Parmar.ea.2017` és sok kapcsolódó új architektúra születtek.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import init, np, npx
from mxnet.gluon import rnn, nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
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

A :numref:`sec_seq2seq` szekvencia-szekvencia architektúrája által bevezetett jelölésrendszert követjük, különösen a :eqref:`eq_seq2seq_s_t`-t.
Az alapötlet az, hogy ahelyett, hogy az állapotot megtartanánk,
azaz a forrásmondat $\mathbf{c}$ kontextusváltozóját rögzítetten,
dinamikusan frissítjük azt, mind az eredeti szöveg (kódoló rejtett állapotok $\mathbf{h}_{t}$), mind a már generált szöveg (dekódoló rejtett állapotok $\mathbf{s}_{t'-1}$) függvényeként. Ez adja a $\mathbf{c}_{t'}$-t, amely minden $t'$ dekódolási időlépés után frissül. Tételezzük fel, hogy a bemeneti szekvencia $T$ hosszúságú. Ebben az esetben a kontextusváltozó a figyelempooling kimenete:

$$\mathbf{c}_{t'} = \sum_{t=1}^{T} \alpha(\mathbf{s}_{t' - 1}, \mathbf{h}_{t}) \mathbf{h}_{t}.$$

A $\mathbf{s}_{t' - 1}$-et lekérdezésként,
és a $\mathbf{h}_{t}$-t mind kulcsként, mind értékként használjuk. Vegyük észre, hogy $\mathbf{c}_{t'}$-t ezután az $\mathbf{s}_{t'}$ állapot generálására és új token generálására használják: lásd :eqref:`eq_seq2seq_s_t`-t. Különösen az $\alpha$ figyelemsúly a :eqref:`eq_attn-scoring-alpha`-ban leírt módon kerül kiszámításra,
a :eqref:`eq_additive-attn` által definiált additív figyelempuntozási függvénnyel.
Ez az RNN kódoló–dekódoló architektúra
figyelemmel a :numref:`fig_s2s_attention_details`-ben látható. Vegyük észre, hogy ezt a modellt később módosították, hogy a már generált tokeneket a dekódolóban további kontextusként tartalmazza (azaz a figyelemösszeg nem áll meg $T$-nél, hanem $t'-1$-ig halad). Például lásd :citet:`chan2015listen`-t e stratégia leírásáért, a beszédfelismerésre alkalmazva.

![Rétegek egy RNN kódoló–dekódoló modellben a Bahdanau figyelemmechanizmussal.](../img/seq2seq-details-attention.svg)
:label:`fig_s2s_attention_details`

## A dekódoló definiálása figyelemmel

Az RNN kódoló–dekódoló megvalósításához figyelemmel,
csak a dekódolót kell újradefiniálni (a generált szimbólumok kihagyása a figyelemfüggvényből egyszerűsíti a tervezést). Kezdjük [**a figyelemalapú dekódolók alapinterfészével**] az `AttentionDecoder` osztály definiálásával.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class AttentionDecoder(d2l.Decoder):  #@save
    """A figyelemalapú dekódolók alapinterfésze."""
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError
```

A `Seq2SeqAttentionDecoder` osztályban [**implementálni kell az RNN dekódolót**].
A dekódoló állapota a következőkkel inicializálódik:
(i) a kódoló utolsó rétegének összes időlépésen átívelő rejtett állapotai, amelyek kulcsként és értékként szolgálnak a figyelemhez;
(ii) a kódoló rejtett állapota az utolsó időlépésnél az összes rétegen keresztül, amely a dekódoló rejtett állapotának inicializálására szolgál;
és (iii) a kódoló érvényes hossza, hogy kizárjuk a kitöltő tokeneket a figyelempooling-ból.
Minden dekódolási időlépésnél a dekódoló végső rétegének rejtett állapotát, amelyet az előző időlépésnél kaptak, a figyelemmechanizmus lekérdezéseként használják.
A figyelemmechanizmus és a bemeneti beágyazás kimenete össze van fűzve, hogy az RNN dekódoló bemeneteként szolgáljon.

```{.python .input}
%%tab mxnet
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)
        self.initialize(init.Xavier())

    def init_state(self, enc_outputs, enc_valid_lens):
        # outputs alakja: (num_steps, batch_size, num_hiddens).
        # hidden_state alakja: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.swapaxes(0, 1), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs alakja: (batch_size, num_steps, num_hiddens).
        # hidden_state alakja: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # A kimeneti X alakja: (num_steps, batch_size, embed_size)
        X = self.embedding(X).swapaxes(0, 1)
        outputs, self._attention_weights = [], []
        for x in X:
            # query alakja: (batch_size, 1, num_hiddens)
            query = np.expand_dims(hidden_state[-1], axis=1)
            # context alakja: (batch_size, 1, num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Összefűzés a jellemződimenzió mentén
            x = np.concatenate((context, np.expand_dims(x, axis=1)), axis=-1)
            # Reshape x as (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.swapaxes(0, 1), hidden_state)
            hidden_state = hidden_state[0]
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully connected layer transformation, shape of outputs:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(np.concatenate(outputs, axis=0))
        return outputs.swapaxes(0, 1), [enc_outputs, hidden_state,
                                        enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab pytorch
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(d2l.init_seq2seq)

    def init_state(self, enc_outputs, enc_valid_lens):
        # outputs alakja: (num_steps, batch_size, num_hiddens).
        # hidden_state alakja: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs alakja: (batch_size, num_steps, num_hiddens).
        # hidden_state alakja: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # A kimeneti X alakja: (num_steps, batch_size, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # query alakja: (batch_size, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # context alakja: (batch_size, 1, num_hiddens)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # Összefűzés a jellemződimenzió mentén
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # Reshape x as (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # After fully connected layer transformation, shape of outputs:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab tensorflow
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, num_hiddens,
                                               num_hiddens, dropout)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.rnn = tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
             for _ in range(num_layers)]), return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        # outputs alakja: (batch_size, num_steps, num_hiddens).
        # A hidden_state lista hossza num_layers; elemeinek alakja
        # (batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (tf.transpose(outputs, (1, 0, 2)), hidden_state,
                enc_valid_lens)

    def call(self, X, state, **kwargs):
        # enc_outputs alakja: (batch_size, num_steps, num_hiddens)
        # A hidden_state lista hossza num_layers; elemeinek alakja
        # (batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # A kimeneti X alakja: (num_steps, batch_size, embed_size)
        X = self.embedding(X)  # A bemeneti X alakja: (batch_size, num_steps)
        X = tf.transpose(X, perm=(1, 0, 2))
        outputs, self._attention_weights = [], []
        for x in X:
            # query alakja: (batch_size, 1, num_hiddens)
            query = tf.expand_dims(hidden_state[-1], axis=1)
            # context alakja: (batch_size, 1, num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs,
                                     enc_valid_lens, **kwargs)
            # Összefűzés a jellemződimenzió mentén
            x = tf.concat((context, tf.expand_dims(x, axis=1)), axis=-1)
            out = self.rnn(x, hidden_state, **kwargs)
            hidden_state = out[1:]
            outputs.append(out[0])
            self._attention_weights.append(self.attention.attention_weights)
        # After fully connected layer transformation, shape of outputs:
        # (batch_size, num_steps, vocab_size)
        outputs = self.dense(tf.concat(outputs, axis=1))
        return outputs, [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab jax
class Seq2SeqAttentionDecoder(nn.Module):
    vocab_size: int
    embed_size: int
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    def setup(self):
        self.attention = d2l.AdditiveAttention(self.num_hiddens, self.dropout)
        self.embedding = nn.Embed(self.vocab_size, self.embed_size)
        self.dense = nn.Dense(self.vocab_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout=self.dropout)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs alakja: (num_steps, batch_size, num_hiddens).
        # hidden_state alakja: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        # A figyelemsúlyok az állapot részeként térnek vissza; kezdetben None
        return (outputs.transpose(1, 0, 2), hidden_state, enc_valid_lens)

    @nn.compact
    def __call__(self, X, state, training=False):
        # enc_outputs alakja: (batch_size, num_steps, num_hiddens).
        # hidden_state alakja: (num_layers, batch_size, num_hiddens)
        # Az állapotban lévő figyelemértéket figyelmen kívül hagyjuk
        enc_outputs, hidden_state, enc_valid_lens = state
        # A kimeneti X alakja: (num_steps, batch_size, embed_size)
        X = self.embedding(X).transpose(1, 0, 2)
        outputs, attention_weights = [], []
        for x in X:
            # query alakja: (batch_size, 1, num_hiddens)
            query = jnp.expand_dims(hidden_state[-1], axis=1)
            # context alakja: (batch_size, 1, num_hiddens)
            context, attention_w = self.attention(query, enc_outputs,
                                                  enc_outputs, enc_valid_lens,
                                                  training=training)
            # Összefűzés a jellemződimenzió mentén
            x = jnp.concatenate((context, jnp.expand_dims(x, axis=1)), axis=-1)
            # Reshape x as (1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.transpose(1, 0, 2), hidden_state,
                                         training=training)
            outputs.append(out)
            attention_weights.append(attention_w)

        # A Flax sow API köztes változók rögzítésére szolgál
        self.sow('intermediates', 'dec_attention_weights', attention_weights)

        # After fully connected layer transformation, shape of outputs:
        # (num_steps, batch_size, vocab_size)
        outputs = self.dense(jnp.concatenate(outputs, axis=0))
        return outputs.transpose(1, 0, 2), [enc_outputs, hidden_state,
                                            enc_valid_lens]
```

A következőkben [**teszteljük az implementált dekódolót**] figyelemmel,
négy szekvenciából álló mini-batch-csel, amelyek mindegyike hét időlépés hosszú.

```{.python .input}
%%tab all
vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
batch_size, num_steps = 4, 7
encoder = d2l.Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
decoder = Seq2SeqAttentionDecoder(vocab_size, embed_size, num_hiddens,
                                  num_layers)
if tab.selected('mxnet'):
    X = d2l.zeros((batch_size, num_steps))
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
if tab.selected('pytorch'):
    X = d2l.zeros((batch_size, num_steps), dtype=torch.long)
    state = decoder.init_state(encoder(X), None)
    output, state = decoder(X, state)
if tab.selected('tensorflow'):
    X = tf.zeros((batch_size, num_steps))
    state = decoder.init_state(encoder(X, training=False), None)
    output, state = decoder(X, state, training=False)
if tab.selected('jax'):
    X = jnp.zeros((batch_size, num_steps), dtype=jnp.int32)
    state = decoder.init_state(encoder.init_with_output(d2l.get_key(),
                                                        X, training=False)[0],
                               None)
    (output, state), _ = decoder.init_with_output(d2l.get_key(), X,
                                                  state, training=False)
d2l.check_shape(output, (batch_size, num_steps, vocab_size))
d2l.check_shape(state[0], (batch_size, num_steps, num_hiddens))
d2l.check_shape(state[1][0], (batch_size, num_hiddens))
```

## [**Tanítás**]

Most, hogy meghatároztuk az új dekódolót, hasonlóan járhatunk el, mint a :numref:`sec_seq2seq_training`-ban:
meghatározzuk a hiperparamétereket, példányosítunk
egy szabályos kódolót és egy figyelmes dekódolót,
és betanítjuk ezt a modellt a gépi fordításhoz.

```{.python .input}
%%tab all
data = d2l.MTFraEng(batch_size=128)
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2
if tab.selected('mxnet', 'pytorch', 'jax'):
    encoder = d2l.Seq2SeqEncoder(
        len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(
        len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
if tab.selected('mxnet', 'pytorch'):
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.005)
if tab.selected('jax'):
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.005, training=True)
if tab.selected('mxnet', 'pytorch', 'jax'):
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        encoder = d2l.Seq2SeqEncoder(
            len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
        decoder = Seq2SeqAttentionDecoder(
            len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
        model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                            lr=0.005)
    trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1)
trainer.fit(model, data)
```

A modell betanítása után
[**néhány angol mondat fordítására**] használjuk
franciára, és kiszámítjuk a BLEU-pontszámaikat.

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

Vizualizáljuk [**a figyelemsúlyokat**]
az utolsó angol mondat fordításakor.
Látjuk, hogy minden lekérdezés nem egyenletes súlyokat rendel
a kulcs–érték párokhoz.
Ez megmutatja, hogy minden dekódolási lépésnél
a bemeneti szekvenciák különböző részei
szelektíven aggregálódnak a figyelempooling-ban.

```{.python .input}
%%tab all
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    _, dec_attention_weights = model.predict_step(
        data.build([engs[-1]], [fras[-1]]), d2l.try_gpu(), data.num_steps, True)
if tab.selected('jax'):
    _, (dec_attention_weights, _) = model.predict_step(
        trainer.state.params, data.build([engs[-1]], [fras[-1]]),
        data.num_steps, True)
attention_weights = d2l.concat(
    [step[0][0][0] for step in dec_attention_weights], 0)
attention_weights = d2l.reshape(attention_weights, (1, 1, -1, data.num_steps))
```

```{.python .input}
%%tab mxnet
# Plusz egy az end-of-sequence token miatt
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1],
    xlabel='Kulcspozíciók', ylabel='Lekérdezési pozíciók')
```

```{.python .input}
%%tab pytorch
# Plusz egy az end-of-sequence token miatt
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='Kulcspozíciók', ylabel='Lekérdezési pozíciók')
```

```{.python .input}
%%tab tensorflow
# Plusz egy az end-of-sequence token miatt
d2l.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1],
                  xlabel='Kulcspozíciók', ylabel='Lekérdezési pozíciók')
```

```{.python .input}
%%tab jax
# Plusz egy az end-of-sequence token miatt
d2l.show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1],
                  xlabel='Kulcspozíciók', ylabel='Lekérdezési pozíciók')
```

## Összefoglalás

Egy token előrejelzésekor, ha nem minden bemeneti token releváns, a Bahdanau figyelemmechanizmussal ellátott RNN kódoló–dekódoló szelektíven aggregálja a bemeneti szekvencia különböző részeit. Ezt úgy érik el, hogy az állapotot (kontextusváltozót) az additív figyelempooling kimenetének tekintik.
Az RNN kódoló–dekódolóban a Bahdanau figyelemmechanizmus a dekódoló rejtett állapotát az előző időlépésnél lekérdezésként, a kódoló rejtett állapotait az összes időlépésnél kulcsként és értékként is kezeli.


## Feladatok

1. Cseréld le a GRU-t LSTM-re a kísérletben.
1. Módosítsd a kísérletet az additív figyelempuntozási függvény skálázott skaláris szorzat-ra való cseréléséhez. Hogyan befolyásolja ez a tanítási hatékonyságot?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/347)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1065)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3868)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18028)
:end_tab:
