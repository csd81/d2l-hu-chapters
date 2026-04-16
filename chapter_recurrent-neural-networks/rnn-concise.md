# Rekurrens Neurális Hálózatok Tömör Implementálása
:label:`sec_rnn-concise`

Az alapoktól készült implementációink legtöbbjéhez hasonlóan
a :numref:`sec_rnn-scratch` fejezet arra lett tervezve,
hogy betekintést nyújtson minden komponens működésébe.
De ha naponta használsz RNN-eket
vagy gyártásra kész kódot írsz,
inkább olyan könyvtárakra akarsz majd támaszkodni,
amelyek csökkentik mind az implementálási időt
(a közönséges modellek és függvények könyvtári kódjának biztosításával),
mind a számítási időt
(ezen könyvtári implementációk maximális optimalizálásával).
Ez a szakasz megmutatja, hogyan lehet ugyanazt a nyelvmodellt hatékonyabban implementálni
a deep learning keretrendszered által biztosított
magas szintű API segítségével.
Mint korábban, az *Az időgép* adathalmaz betöltésével kezdjük.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
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
```

## **A modell definiálása**

Az alábbi osztályt definiáljuk
a magas szintű API-k által implementált
RNN segítségével.

:begin_tab:`mxnet`
Konkrétan, a rejtett állapot inicializálásához
a `begin_state` tagmetódust hívjuk meg.
Ez egy listát ad vissza, amely tartalmaz
egy kezdeti rejtett állapotot
a mini-batch minden példájára,
amelynek alakja
(rejtett rétegek száma, batch méret, rejtett egységek száma).
Néhány később bevezetett modellnél
(pl. long short-term memory),
ez a lista más információkat is tartalmaz.
:end_tab:

:begin_tab:`jax`
A Flax jelenleg nem biztosít RNNCell-t a vanilla RNN-ek tömör implementálásához.
Az RNN-ek fejlettebb változatai, mint az LSTM-ek és GRU-k
elérhetők a Flax `linen` API-ban.
:end_tab:

```{.python .input}
%%tab mxnet
class RNN(d2l.Module):  #@save
    """Magas szintű API-kkal implementált RNN modell."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()        
        self.rnn = rnn.RNN(num_hiddens)
        
    def forward(self, inputs, H=None):
        if H is None:
            H, = self.rnn.begin_state(inputs.shape[1], ctx=inputs.ctx)
        outputs, (H, ) = self.rnn(inputs, (H, ))
        return outputs, H
```

```{.python .input}
%%tab pytorch
class RNN(d2l.Module):  #@save
    """Magas szintű API-kkal implementált RNN modell."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)
        
    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
```

```{.python .input}
%%tab tensorflow
class RNN(d2l.Module):  #@save
    """Magas szintű API-kkal implementált RNN modell."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()            
        self.rnn = tf.keras.layers.SimpleRNN(
            num_hiddens, return_sequences=True, return_state=True,
            time_major=True)
        
    def forward(self, inputs, H=None):
        outputs, H = self.rnn(inputs, H)
        return outputs, H
```

```{.python .input}
%%tab jax
class RNN(nn.Module):  #@save
    """Magas szintű API-kkal implementált RNN modell."""
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H=None):
        raise NotImplementedError
```

A :numref:`sec_rnn-scratch` fejezetből örökölve az `RNNLMScratch` osztályt,
a következő `RNNLM` osztály egy teljes RNN-alapú nyelvmodellt definiál.
Megjegyezzük, hogy külön teljesen összekötött kimeneti réteget kell létrehoznunk.

```{.python .input}
%%tab pytorch
class RNNLM(d2l.RNNLMScratch):  #@save
    """Magas szintű API-kkal implementált RNN-alapú nyelvmodell."""
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)
        
    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)
```

```{.python .input}
%%tab mxnet, tensorflow
class RNNLM(d2l.RNNLMScratch):  #@save
    """Magas szintű API-kkal implementált RNN-alapú nyelvmodell."""
    def init_params(self):
        if tab.selected('mxnet'):
            self.linear = nn.Dense(self.vocab_size, flatten=False)
            self.initialize()
        if tab.selected('tensorflow'):
            self.linear = tf.keras.layers.Dense(self.vocab_size)
        
    def output_layer(self, hiddens):
        if tab.selected('mxnet'):
            return d2l.swapaxes(self.linear(hiddens), 0, 1)        
        if tab.selected('tensorflow'):
            return d2l.transpose(self.linear(hiddens), (1, 0, 2))
```

```{.python .input}
%%tab jax
class RNNLM(d2l.RNNLMScratch):  #@save
    """Magas szintű API-kkal implementált RNN-alapú nyelvmodell."""
    training: bool = True

    def setup(self):
        self.linear = nn.Dense(self.vocab_size)

    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)

    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state, self.training)
        return self.output_layer(rnn_outputs)
```

## Tanítás és előrejelzés

A modell tanítása előtt **végezzünk előrejelzést
véletlenszerű súlyokkal inicializált modellel.**
Mivel a hálózatot még nem tanítottuk,
értelmetlen előrejelzéseket fog generálni.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'tensorflow'):
    rnn = RNN(num_hiddens=32)
if tab.selected('pytorch'):
    rnn = RNN(num_inputs=len(data.vocab), num_hiddens=32)
model = RNNLM(rnn, vocab_size=len(data.vocab), lr=1)
model.predict('it has', 20, data.vocab)
```

Ezután **tanítjuk a modellünket a magas szintű API kihasználásával**.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
if tab.selected('mxnet', 'pytorch'):
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

A :numref:`sec_rnn-scratch` fejezethez képest
ez a modell összehasonlítható perplexitást ér el,
de az optimalizált implementációknak köszönhetően gyorsabban fut.
Mint korábban, generálhatunk megjósolt tokeneket
a megadott előtag karakterlánc követésével.

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

## Összefoglalás

A deep learning keretrendszerek magas szintű API-jai biztosítják a standard RNN-ek implementációit.
Ezek a könyvtárak segítenek elkerülni, hogy időt pazaroljunk a standard modellek újraimplementálásával.
Sőt,
a keretrendszer implementációi általában nagymértékben optimalizáltak,
ami jelentős (számítási) teljesítménynövekedést eredményez
az alapoktól készült implementációkhoz képest.

## Feladatok

1. Túlillesztheted-e az RNN modellt a magas szintű API-k segítségével?
1. Implementáld a :numref:`sec_sequence` autoregresszív modelljét RNN segítségével!

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/335)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1053)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2211)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18015)
:end_tab:
