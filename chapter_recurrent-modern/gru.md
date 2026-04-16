# Kapuzott Rekurrens Egységek (GRU)
:label:`sec_gru`


Ahogy az RNN-ek, és különösen az LSTM-architektúra (:numref:`sec_lstm`)
rohamosan népszerűvé vált a 2010-es évek során,
számos kutató elkezdett egyszerűsített architektúrákkal kísérletezni,
abban a reményben, hogy megőrizzék a belső állapot
és a szorzó kapumechanizmusok beépítésének kulcsgondolatát,
de ezúttal a számítás felgyorsítását célozva.
A kapuzott rekurrens egység (GRU) :cite:`Cho.Van-Merrienboer.Bahdanau.ea.2014`
az LSTM memóriasejt egy egyszerűsített változatát kínálja,
amely gyakran hasonló teljesítményt ér el,
de azzal az előnnyel, hogy gyorsabban számítható :cite:`Chung.Gulcehre.Cho.ea.2014`.

```{.python .input  n=5}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

```{.python .input  n=6}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn
npx.set_np()
```

```{.python .input  n=7}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=8}
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

## Visszaállító kapu és Frissítő kapu

Az LSTM három kapuját itt kettő váltja fel:
a *visszaállító kapu* és a *frissítő kapu*.
Az LSTM-ekhez hasonlóan ezek a kapuk sigmoid aktivációt kapnak,
így értékeik a $(0, 1)$ intervallumba esnek.
Intuitívan a visszaállító kapu azt szabályozza, hogy az előző állapotból
mennyit szeretnénk még megőrizni.
Hasonlóképpen, a frissítő kapu lehetővé teszi annak szabályozását,
hogy az új állapot mennyiben csak másolata a réginek.
A :numref:`fig_gru_1` ábra szemlélteti a GRU-ban a visszaállító és frissítő kapuk bemeneteit,
az aktuális időlépés bemenete
és az előző időlépés rejtett állapota alapján.
A kapuk kimeneteit két teljesen összekötött réteg adja
sigmoid aktivációs függvénnyel.

![A visszaállító kapu és a frissítő kapu kiszámítása egy GRU modellben.](../img/gru-1.svg)
:label:`fig_gru_1`

Matematikailag, egy adott $t$ időlépésnél,
tegyük fel, hogy a bemenet egy mini-batch
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$
(példák száma $=n$; bemenetek száma $=d$),
az előző időlépés rejtett állapota pedig
$\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$
(rejtett egységek száma $=h$).
Ekkor a visszaállító kapu $\mathbf{R}_t \in \mathbb{R}^{n \times h}$
és a frissítő kapu $\mathbf{Z}_t \in \mathbb{R}^{n \times h}$ kiszámítása a következőképpen történik:

$$
\begin{aligned}
\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xr}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hr}} + \mathbf{b}_\textrm{r}),\\
\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xz}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hz}} + \mathbf{b}_\textrm{z}),
\end{aligned}
$$

ahol $\mathbf{W}_{\textrm{xr}}, \mathbf{W}_{\textrm{xz}} \in \mathbb{R}^{d \times h}$
és $\mathbf{W}_{\textrm{hr}}, \mathbf{W}_{\textrm{hz}} \in \mathbb{R}^{h \times h}$
súlyparaméterek, és $\mathbf{b}_\textrm{r}, \mathbf{b}_\textrm{z} \in \mathbb{R}^{1 \times h}$
eltolási paraméterek.


## Jelölt Rejtett Állapot

Következő lépésként integráljuk a visszaállító kaput $\mathbf{R}_t$
a szokásos frissítési mechanizmusba
a :eqref:`rnn_h_with_state` képletből,
ami a következő
*jelölt rejtett állapothoz*
$\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$ vezet a $t$ időlépésnél:

$$\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{\textrm{hh}} + \mathbf{b}_\textrm{h}),$$
:eqlabel:`gru_tilde_H`

ahol $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}$ és $\mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$
súlyparaméterek,
$\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$
az eltolás,
és a $\odot$ szimbólum a Hadamard (elemenként vett) szorzat operátora.
Ebben tanh aktivációs függvényt alkalmazunk.

Az eredmény csupán *jelölt*, hiszen még be kell építenünk a frissítő kapu hatását.
Összehasonlítva a :eqref:`rnn_h_with_state` képlettel,
az előző állapotok hatása
most csökkenthető az
$\mathbf{R}_t$ és $\mathbf{H}_{t-1}$ elemenként vett szorzatával
a :eqref:`gru_tilde_H` képletben.
Amikor a visszaállító kapu $\mathbf{R}_t$ elemei közel vannak az 1-hez,
visszakapjuk a :eqref:`rnn_h_with_state` képletben szereplő sima RNN-t.
Ha a visszaállító kapu $\mathbf{R}_t$ összes eleme közel van a 0-hoz,
a jelölt rejtett állapot egy MLP eredménye $\mathbf{X}_t$ bemenettel.
A meglévő rejtett állapot így *visszaáll* az alapértékekre.

A :numref:`fig_gru_2` ábra szemlélteti a visszaállító kapu alkalmazása utáni számítási folyamatot.

![A jelölt rejtett állapot kiszámítása egy GRU modellben.](../img/gru-2.svg)
:label:`fig_gru_2`


## Rejtett Állapot

Végül be kell építenünk a frissítő kapu $\mathbf{Z}_t$ hatását.
Ez határozza meg, hogy az új rejtett állapot $\mathbf{H}_t \in \mathbb{R}^{n \times h}$
mennyiben egyezik meg a régi állapottal $\mathbf{H}_{t-1}$, és mennyiben közelít
az új jelölt állapothoz $\tilde{\mathbf{H}}_t$.
A frissítő kapu $\mathbf{Z}_t$ felhasználható erre a célra,
egyszerűen $\mathbf{H}_{t-1}$ és $\tilde{\mathbf{H}}_t$ elemenként vett konvex kombinációjával.
Ez adja a GRU végső frissítési egyenletét:

$$\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.$$


Amikor a frissítő kapu $\mathbf{Z}_t$ közel van az 1-hez,
egyszerűen megtartjuk a régi állapotot.
Ebben az esetben az $\mathbf{X}_t$-ből érkező információt figyelmen kívül hagyjuk,
ami hatékonyan átugorja a $t$ időlépést a függőségi láncban.
Ezzel szemben, ha $\mathbf{Z}_t$ közel van a 0-hoz,
az új látens állapot $\mathbf{H}_t$ megközelíti a jelölt látens állapotot $\tilde{\mathbf{H}}_t$.
A :numref:`fig_gru_3` ábra a frissítő kapu alkalmazása utáni számítási folyamatot mutatja.

![A rejtett állapot kiszámítása egy GRU modellben.](../img/gru-3.svg)
:label:`fig_gru_3`


Összefoglalva, a GRU-knak a következő két megkülönböztető tulajdonsága van:

* A visszaállító kapuk segítenek rövid távú függőségek rögzítésében a sorozatokban.
* A frissítő kapuk segítenek hosszú távú függőségek rögzítésében a sorozatokban.

## Implementáció Alapoktól

A GRU modell jobb megértéséhez implementáljuk azt alapoktól.

### (**Modell Paramétereinek Inicializálása**)

Az első lépés a modell paramétereinek inicializálása.
A súlyokat Gauss-eloszlásból vesszük
`sigma` szórással, és az eltolást 0-ra állítjuk.
A `num_hiddens` hiperparaméter határozza meg a rejtett egységek számát.
Inicializáljuk az összes súlyt és eltolást, amelyek
a frissítő kapuhoz, a visszaállító kapuhoz és a jelölt rejtett állapothoz kapcsolódnak.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class GRUScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        
        if tab.selected('mxnet'):
            init_weight = lambda *shape: d2l.randn(*shape) * sigma
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              d2l.zeros(num_hiddens))            
        if tab.selected('pytorch'):
            init_weight = lambda *shape: nn.Parameter(d2l.randn(*shape) * sigma)
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              nn.Parameter(d2l.zeros(num_hiddens)))
        if tab.selected('tensorflow'):
            init_weight = lambda *shape: tf.Variable(d2l.normal(shape) * sigma)
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              tf.Variable(d2l.zeros(num_hiddens)))            
            
        self.W_xz, self.W_hz, self.b_z = triple()  # Frissítő kapu
        self.W_xr, self.W_hr, self.b_r = triple()  # Visszaállító kapu
        self.W_xh, self.W_hh, self.b_h = triple()  # Jelölt rejtett állapot        
```

```{.python .input}
%%tab jax
class GRUScratch(d2l.Module):
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        init_weight = lambda name, shape: self.param(name,
                                                     nn.initializers.normal(self.sigma),
                                                     shape)
        triple = lambda name : (
            init_weight(f'W_x{name}', (self.num_inputs, self.num_hiddens)),
            init_weight(f'W_h{name}', (self.num_hiddens, self.num_hiddens)),
            self.param(f'b_{name}', nn.initializers.zeros, (self.num_hiddens)))

        self.W_xz, self.W_hz, self.b_z = triple('z')  # Frissítő kapu
        self.W_xr, self.W_hr, self.b_r = triple('r')  # Visszaállító kapu
        self.W_xh, self.W_hh, self.b_h = triple('h')  # Jelölt rejtett állapot
```

### A Modell Definiálása

Most készen állunk a **GRU előre irányú számítás definiálására**.
Szerkezete megegyezik az alap RNN-cella szerkezetével,
kivéve, hogy a frissítési egyenletek összetettebbek.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(GRUScratch)
def forward(self, inputs, H=None):
    if H is None:
        # Kezdeti állapot alakja: (batch_size, num_hiddens)
        if tab.selected('mxnet'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          ctx=inputs.ctx)
        if tab.selected('pytorch'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          device=inputs.device)
        if tab.selected('tensorflow'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens))
    outputs = []
    for X in inputs:
        Z = d2l.sigmoid(d2l.matmul(X, self.W_xz) +
                        d2l.matmul(H, self.W_hz) + self.b_z)
        R = d2l.sigmoid(d2l.matmul(X, self.W_xr) + 
                        d2l.matmul(H, self.W_hr) + self.b_r)
        H_tilde = d2l.tanh(d2l.matmul(X, self.W_xh) + 
                           d2l.matmul(R * H, self.W_hh) + self.b_h)
        H = Z * H + (1 - Z) * H_tilde
        outputs.append(H)
    return outputs, H
```

```{.python .input}
%%tab jax
@d2l.add_to_class(GRUScratch)
def forward(self, inputs, H=None):
    # A lax.scan primitívet használjuk a bemenetek feletti ciklus helyett,
    # mivel a scan időt takarít meg a jit fordítás során
    def scan_fn(H, X):
        Z = d2l.sigmoid(d2l.matmul(X, self.W_xz) + d2l.matmul(H, self.W_hz) +
                        self.b_z)
        R = d2l.sigmoid(d2l.matmul(X, self.W_xr) +
                        d2l.matmul(H, self.W_hr) + self.b_r)
        H_tilde = d2l.tanh(d2l.matmul(X, self.W_xh) +
                           d2l.matmul(R * H, self.W_hh) + self.b_h)
        H = Z * H + (1 - Z) * H_tilde
        return H, H  # carry és y visszaadása

    if H is None:
        batch_size = inputs.shape[1]
        carry = jnp.zeros((batch_size, self.num_hiddens))
    else:
        carry = H

    # a scan a scan_fn-t, a kezdeti carry állapotot és a vezető tengellyel rendelkező xs-t veszi
    carry, outputs = jax.lax.scan(scan_fn, carry, inputs)
    return outputs, carry
```

### Tanítás

A *The Time Machine* adathalmazon való **Tanítás**
pontosan ugyanúgy működik, mint a :numref:`sec_rnn-scratch` fejezetben.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    gru = GRUScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = d2l.RNNLMScratch(gru, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        gru = GRUScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = d2l.RNNLMScratch(gru, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1)
trainer.fit(model, data)
```

## **Tömör Implementáció**

A magas szintű API-kban közvetlenül létrehozhatunk egy GRU modellt.
Ez magába foglalja az összes fent explicit módon megadott konfigurációs részletet.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class GRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.rnn = rnn.GRU(num_hiddens)
        if tab.selected('pytorch'):
            self.rnn = nn.GRU(num_inputs, num_hiddens)
        if tab.selected('tensorflow'):
            self.rnn = tf.keras.layers.GRU(num_hiddens, return_sequences=True, 
                                           return_state=True)
```

```{.python .input}
%%tab jax
class GRU(d2l.RNN):
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H=None, training=False):
        if H is None:
            batch_size = inputs.shape[1]
            H = nn.GRUCell.initialize_carry(jax.random.PRNGKey(0),
                                            (batch_size,), self.num_hiddens)

        GRU = nn.scan(nn.GRUCell, variable_broadcast="params",
                      in_axes=0, out_axes=0, split_rngs={"params": False})

        H, outputs = GRU()(H, inputs)
        return outputs, H
```

A kód lényegesen gyorsabb a tanítás során, mivel lefordított operátorokat használ
a Python helyett.

```{.python .input}
%%tab all
if tab.selected('mxnet', 'pytorch', 'tensorflow'):
    gru = GRU(num_inputs=len(data.vocab), num_hiddens=32)
if tab.selected('jax'):
    gru = GRU(num_hiddens=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=4)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=4)
trainer.fit(model, data)
```

A tanítás után kiíratjuk a perplexitást a tanítóhalmazon
és az adott előtagot követő előrejelzett sorozatot.

```{.python .input}
%%tab mxnet, pytorch
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

Az LSTM-ekhez képest a GRU-k hasonló teljesítményt érnek el, de számítási szempontból könnyebbek.
Általánosságban, az egyszerű RNN-ekhez képest a kapuzott RNN-ek, például az LSTM-ek és a GRU-k,
jobban képesek rögzíteni a nagy időlépés-távolságú sorozatokban lévő függőségeket.
A GRU-k alap RNN-eket tartalmaznak szélső esetként, amikor a visszaállító kapu be van kapcsolva.
Részsorozatokat is átugrorhatnak a frissítő kapu bekapcsolásával.


## Feladatok

1. Tegyük fel, hogy csak a $t'$ időlépésnél lévő bemenetet szeretnénk felhasználni a $t > t'$ időlépésnél lévő kimenet előrejelzéséhez. Mik lennének a visszaállító és frissítő kapuk legjobb értékei minden időlépésnél?
1. Állítsd be a hiperparamétereket, és elemezd azok hatását a futási időre, a perplexitásra és a kimeneti sorozatra.
1. Hasonlítsd össze az `rnn.RNN` és `rnn.GRU` implementációk futási idejét, perplexitását és kimeneti karakterláncait egymással.
1. Mi történik, ha csak a GRU egyes részeit implementálod, pl. csak visszaállító kapuval vagy csak frissítő kapuval?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/342)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1056)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3860)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18017)
:end_tab:
