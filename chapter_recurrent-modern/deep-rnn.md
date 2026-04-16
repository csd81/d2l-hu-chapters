# Mély Rekurrens Neurális Hálózatok

:label:`sec_deep_rnn`

Eddig arra összpontosítottunk, hogy sorozatbemenetből,
egyetlen rejtett RNN-rétegből
és egy kimeneti rétegből álló hálózatokat definiáljunk.
Bár csak egyetlen rejtett réteg van
a bármely időlépésnél lévő bemenet
és a megfelelő kimenet között,
ezek a hálózatok bizonyos értelemben mélyek.
Az első időlépés bemenetei befolyásolhatják
a végső $T$ időlépésnél lévő kimeneteket
(sokszor 100 vagy 1000 lépéssel később).
Ezek a bemenetek a rekurrens rétegen keresztül $T$-szeres alkalmazáson mennek át,
mielőtt elérik a végső kimenetet.
Azonban sokszor azt is szeretnénk megőrizni, hogy
képesek legyünk összetett kapcsolatokat kifejezni
egy adott időlépés bemenetei
és az ugyanazon időlépés kimenetei között.
Ezért olyan RNN-eket is szokás építeni, amelyek
nemcsak az időirányban mélyek,
hanem a bemenet-kimenet irányban is.
Ez pontosan a mélységnek az a fogalma,
amellyel már találkoztunk
az MLP-k és mély CNN-ek fejlesztése során.


Ennek a fajta mély RNN-nek az építésének szokásos módszere
meglepően egyszerű: egymásra halmozzuk az RNN-eket.
Adott egy $T$ hosszúságú sorozat, az első RNN
szintén $T$ hosszúságú kimenet-sorozatot állít elő.
Ezek alkotják a következő RNN-réteg bemeneteit.
Ebben a rövid részben ezt a tervezési mintát szemléltetjük,
és bemutatunk egy egyszerű példát az ilyen halmozott RNN-ek kódolására.
Alább, a :numref:`fig_deep_rnn` ábrán,
egy $L$ rejtett réteggel rendelkező mély RNN-t illusztrálunk.
Minden rejtett állapot szekvenciális bemeneten működik
és szekvenciális kimenetet állít elő.
Ráadásul bármely RNN-cella (fehér doboz a :numref:`fig_deep_rnn` ábrán) minden időlépésnél
függ mind az ugyanazon réteg előző időlépésbeli értékétől,
mind az előző réteg ugyanazon időlépésbeli értékétől.

![Mély RNN architektúrája.](../img/deep-rnn.svg)
:label:`fig_deep_rnn`

Formálisan, tegyük fel, hogy a $t$ időlépésnél van egy mini-batch bemenetünk
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$
(példák száma $=n$; az egyes példákban lévő bemenetek száma $=d$).
Ugyanannál az időlépésnél,
legyen az $l$-edik rejtett réteg ($l=1,\ldots,L$) rejtett állapota $\mathbf{H}_t^{(l)} \in \mathbb{R}^{n \times h}$
(rejtett egységek száma $=h$),
és a kimeneti réteg változója legyen
$\mathbf{O}_t \in \mathbb{R}^{n \times q}$
(kimenetek száma: $q$).
A $\mathbf{H}_t^{(0)} = \mathbf{X}_t$ beállítással,
a $\phi_l$ aktivációs függvényt alkalmazó
$l$-edik rejtett réteg rejtett állapota
a következőképpen számítható:

$$\mathbf{H}_t^{(l)} = \phi_l(\mathbf{H}_t^{(l-1)} \mathbf{W}_{\textrm{xh}}^{(l)} + \mathbf{H}_{t-1}^{(l)} \mathbf{W}_{\textrm{hh}}^{(l)}  + \mathbf{b}_\textrm{h}^{(l)}),$$
:eqlabel:`eq_deep_rnn_H`

ahol a $\mathbf{W}_{\textrm{xh}}^{(l)} \in \mathbb{R}^{h \times h}$ és $\mathbf{W}_{\textrm{hh}}^{(l)} \in \mathbb{R}^{h \times h}$ súlyok, valamint
a $\mathbf{b}_\textrm{h}^{(l)} \in \mathbb{R}^{1 \times h}$ eltolás
az $l$-edik rejtett réteg modell paraméterei.

Végül a kimeneti réteg kiszámítása
csak a végső $L$-edik rejtett réteg rejtett állapotán alapul:

$$\mathbf{O}_t = \mathbf{H}_t^{(L)} \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q},$$

ahol a $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$ súly
és a $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$ eltolás
a kimeneti réteg modell paraméterei.

Akárcsak az MLP-knél, a rejtett rétegek száma $L$
és a rejtett egységek száma $h$ olyan hiperparaméterek,
amelyeket finomhangolhatunk.
A szokásos RNN-réteg szélességek ($h$) a $(64, 2056)$ tartományba esnek,
a szokásos mélységek ($L$) pedig a $(1, 8)$ tartományba.
Ezen felül könnyen kaphatunk mély kapuzott RNN-t
azáltal, hogy a rejtett állapot kiszámítását a :eqref:`eq_deep_rnn_H` képletben
LSTM vagy GRU kiszámítással váltjuk fel.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn
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
import jax
from jax import numpy as jnp
```

## Implementáció Alapoktól

Egy többrétegű RNN alapoktól való implementálásához
minden réteget egy saját tanulható paraméterekkel rendelkező `RNNScratch` példányként kezelhetünk.

```{.python .input}
%%tab mxnet, tensorflow
class StackedRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.rnns = [d2l.RNNScratch(num_inputs if i==0 else num_hiddens,
                                    num_hiddens, sigma)
                     for i in range(num_layers)]
```

```{.python .input}
%%tab pytorch
class StackedRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.rnns = nn.Sequential(*[d2l.RNNScratch(
            num_inputs if i==0 else num_hiddens, num_hiddens, sigma)
                                    for i in range(num_layers)])
```

```{.python .input}
%%tab jax
class StackedRNNScratch(d2l.Module):
    num_inputs: int
    num_hiddens: int
    num_layers: int
    sigma: float = 0.01

    def setup(self):
        self.rnns = [d2l.RNNScratch(self.num_inputs if i==0 else self.num_hiddens,
                                    self.num_hiddens, self.sigma)
                     for i in range(self.num_layers)]
```

A többrétegű előreterjesztés
egyszerűen rétegről rétegre végez előreterjesztést.

```{.python .input}
%%tab all
@d2l.add_to_class(StackedRNNScratch)
def forward(self, inputs, Hs=None):
    outputs = inputs
    if Hs is None: Hs = [None] * self.num_layers
    for i in range(self.num_layers):
        outputs, Hs[i] = self.rnns[i](outputs, Hs[i])
        outputs = d2l.stack(outputs, 0)
    return outputs, Hs
```

Példaként tanítsunk egy mély GRU modellt
a *The Time Machine* adathalmazon (ugyanúgy, mint a :numref:`sec_rnn-scratch` fejezetben).
Az egyszerűség kedvéért a rétegek számát 2-re állítjuk.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    rnn_block = StackedRNNScratch(num_inputs=len(data.vocab),
                                  num_hiddens=32, num_layers=2)
    model = d2l.RNNLMScratch(rnn_block, vocab_size=len(data.vocab), lr=2)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        rnn_block = StackedRNNScratch(num_inputs=len(data.vocab),
                                  num_hiddens=32, num_layers=2)
        model = d2l.RNNLMScratch(rnn_block, vocab_size=len(data.vocab), lr=2)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

## Tömör Implementáció

:begin_tab:`pytorch, mxnet, tensorflow`
Szerencsére az RNN több rétegének implementálásához szükséges logisztikai részletek nagy része
könnyen elérhető a magas szintű API-kban.
A tömör implementációnk ilyen beépített funkciókat fog használni.
A kód általánosítja a :numref:`sec_gru` fejezetben korábban használtat,
lehetővé téve a rétegek számának explicit megadását
ahelyett, hogy az egyrétegű alapértelmezést választanánk.
:end_tab:

:begin_tab:`jax`
A Flax minimális megközelítést alkalmaz az RNN-ek implementálásakor.
A rétegek számának definiálása egy RNN-ben vagy dropout alkalmazása
nem elérhető alapból.
A tömör implementációnk az összes beépített funkciót felhasználja, és
`num_layers` és `dropout` funkciókat ad hozzá.
A kód általánosítja a :numref:`sec_gru` fejezetben korábban használtat,
lehetővé téve a rétegek számának explicit megadását
ahelyett, hogy az egyrétegű alapértelmezést választanánk.
:end_tab:

```{.python .input}
%%tab mxnet
class GRU(d2l.RNN):  #@save
    """A többrétegű GRU modell."""
    def __init__(self, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
```

```{.python .input}
%%tab pytorch
class GRU(d2l.RNN):  #@save
    """A többrétegű GRU modell."""
    def __init__(self, num_inputs, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers,
                          dropout=dropout)
```

```{.python .input}
%%tab tensorflow
class GRU(d2l.RNN):  #@save
    """A többrétegű GRU modell."""
    def __init__(self, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        gru_cells = [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
                     for _ in range(num_layers)]
        self.rnn = tf.keras.layers.RNN(gru_cells, return_sequences=True,
                                       return_state=True, time_major=True)

    def forward(self, X, state=None):
        outputs, *state = self.rnn(X, state)
        return outputs, state
```

```{.python .input}
%%tab jax
class GRU(d2l.RNN):  #@save
    """A többrétegű GRU modell."""
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    @nn.compact
    def __call__(self, X, state=None, training=False):
        outputs = X
        new_state = []
        if state is None:
            batch_size = X.shape[1]
            state = [nn.GRUCell.initialize_carry(jax.random.PRNGKey(0),
                    (batch_size,), self.num_hiddens)] * self.num_layers

        GRU = nn.scan(nn.GRUCell, variable_broadcast="params",
                      in_axes=0, out_axes=0, split_rngs={"params": False})

        # Dropout réteget adunk minden GRU réteg után, kivéve az utolsót
        for i in range(self.num_layers - 1):
            layer_i_state, X = GRU()(state[i], outputs)
            new_state.append(layer_i_state)
            X = nn.Dropout(self.dropout, deterministic=not training)(X)

        # Az utolsó GRU réteg dropout nélkül
        out_state, X = GRU()(state[-1], X)
        new_state.append(out_state)
        return X, jnp.array(new_state)
```

Az architektúrával kapcsolatos döntések, például a hiperparaméterek megválasztása,
nagyon hasonlóak a :numref:`sec_gru` fejezetéhez.
Ugyanannyi bemenetet és kimenetet választunk,
amennyit különböző tokenek vannak, azaz `vocab_size`.
A rejtett egységek száma még mindig 32.
Az egyetlen különbség az, hogy most
(**egy nemtriviális számú rejtett réteget választunk
a `num_layers` értékének megadásával.**)

```{.python .input}
%%tab mxnet
gru = GRU(num_hiddens=32, num_layers=2)
model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)

# A futás több mint 1 órát vesz igénybe (MXNet javítás folyamatban)
# trainer.fit(model, data)
# model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab pytorch, tensorflow, jax
if tab.selected('tensorflow', 'jax'):
    gru = GRU(num_hiddens=32, num_layers=2)
if tab.selected('pytorch'):
    gru = GRU(num_inputs=len(data.vocab), num_hiddens=32, num_layers=2)
if tab.selected('pytorch', 'jax'):
    model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)
trainer.fit(model, data)
```

```{.python .input}
%%tab pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

```{.python .input}
%%tab jax
model.predict('it has', 20, data.vocab, trainer.state.params)
```

## Összefoglalás

A mély RNN-ekben a rejtett állapot információja
az aktuális réteg következő időlépésére
és a következő réteg aktuális időlépésére kerül átadásra.
A mély RNN-eknek számos különböző változata létezik, például LSTM-ek, GRU-k vagy sima RNN-ek.
Kényelmesen, ezek a modellek mind elérhetők
a mélytanulási keretrendszerek magas szintű API-jainak részeként.
A modellek inicializálása gondosságot igényel.
Összességében a mély RNN-ek jelentős munkát igényelnek
(például tanulási ráta és vágás)
a megfelelő konvergencia biztosításához.

## Feladatok

1. Cseréld fel a GRU-t LSTM-re, és hasonlítsd össze a pontosságot és a tanítási sebességet.
1. Növeld a tanítási adatokat, hogy több könyvet tartalmazzon. Mennyire alacsonyra lehet vinni a perplexitást?
1. Érdemes lenne különböző szerzők forrásait kombinálni szöveg modellezésekor? Miért jó ez az ötlet? Mi mehet rosszul?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/340)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1058)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3862)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18018)
:end_tab:
