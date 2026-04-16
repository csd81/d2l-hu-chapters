# Hosszú Rövid Távú Memória (LSTM)
:label:`sec_lstm`


Röviddel azután, hogy az első Elman-stílusú RNN-eket betanították visszaterjesztés segítségével
:cite:`elman1990finding`, a hosszú távú függőségek tanulásának problémái
(az eltűnő és robbanó gradiensek miatt)
hangsúlyossá váltak, amelyeket Bengio és Hochreiter tárgyalt
:cite:`bengio1994learning,Hochreiter.Bengio.Frasconi.ea.2001`.
Hochreiter már 1991-ben megfogalmazta ezt a problémát
mesterszakos diplomadolgozatában, de az eredmények nem váltak széles körben ismertté,
mivel a dolgozat németül íródott.
Bár a gradiensvágás segít a robbanó gradiensgel,
az eltűnő gradiensek kezelése láthatólag kifinomultabb megoldást igényel.
Az eltűnő gradiensek kezelésének egyik első és legsikeresebb technikája
a hosszú rövid távú memória (LSTM) modell formájában jelent meg
:citet:`Hochreiter.Schmidhuber.1997` jóvoltából.
Az LSTM-ek standard rekurrens neurális hálózatokra emlékeztetnek,
de itt minden hagyományos rekurrens csomópontot
egy *memóriasejt* vált fel.
Minden memóriasejt tartalmaz egy *belső állapotot*,
azaz egy önmagával összekötött, 1-es rögzített súlyú rekurrens éllel rendelkező csomópontot,
amely biztosítja, hogy a gradiens sok időlépésen át haladhasson
eltűnés vagy robbanás nélkül.

A "hosszú rövid távú memória" kifejezés a következő intuícióból ered.
Az egyszerű rekurrens neurális hálózatok
*hosszú távú memóriával* rendelkeznek súlyok formájában.
A súlyok lassan változnak a tanítás során,
általános tudást kódolva az adatokról.
*Rövid távú memóriájuk* is van
múló aktivációk formájában,
amelyek minden csomópontból a következőkbe áramlanak.
Az LSTM modell bevezet egy közbülső tárolási típust a memóriasejten keresztül.
A memóriasejt egy összetett egység,
amelyet egyszerűbb csomópontokból építenek fel
meghatározott összekötési mintázattal,
szorzó csomópontok újszerű beépítésével.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
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

## Kapuzott Memóriasejt

Minden memóriasejt fel van szerelve egy *belső állapottal*
és számos szorzó kapuval, amelyek meghatározzák, hogy
(i) egy adott bemenet hatással legyen-e a belső állapotra (*bemeneti kapu*),
(ii) a belső állapotot töröljük-e $0$-ra (*felejtési kapu*),
és (iii) egy adott neuron belső állapota
befolyásolhatja-e a sejt kimenetét (*kimeneti kapu*).


### Kapuzott Rejtett Állapot

A sima RNN-ek és az LSTM-ek közötti legfontosabb különbség az,
hogy az utóbbiak támogatják a rejtett állapot kapuzását.
Ez azt jelenti, hogy dedikált mechanizmusokkal rendelkezünk arra,
hogy mikor kell a rejtett állapotot *frissíteni*,
és mikor kell *visszaállítani*.
Ezeket a mechanizmusokat tanuljuk, és a fent felsorolt problémákat kezelik.
Például, ha az első token nagy fontossággal bír,
megtanuljuk, hogy ne frissítsük a rejtett állapotot az első megfigyelés után.
Hasonlóképpen, megtanuljuk átugorni a lényegtelen ideiglenes megfigyeléseket.
Végül megtanuljuk szükség szerint visszaállítani a látens állapotot.
Ezt alább részletesen tárgyaljuk.

### Bemeneti Kapu, Felejtési Kapu és Kimeneti Kapu

Az LSTM kapuiba betáplált adatok az aktuális időlépés bemenete
és az előző időlépés rejtett állapota,
ahogy a :numref:`fig_lstm_0` ábra szemlélteti.
Sigmoid aktivációs függvényű három teljesen összekötött réteg
számítja ki a bemeneti, felejtési és kimeneti kapuk értékeit.
A sigmoid aktiváció eredményeként
mindhárom kapu értékei a $(0, 1)$ tartományban vannak.
Emellett szükségünk van egy *bemeneti csomópontra*,
amelyet jellemzően *tanh* aktivációs függvénnyel számítanak.
Intuitívan a *bemeneti kapu* meghatározza, hogy a bemeneti csomópont értékéből
mennyit kell hozzáadni az aktuális memóriasejt belső állapotához.
A *felejtési kapu* meghatározza, hogy megőrizzük-e
a memória aktuális értékét, vagy töröljük.
A *kimeneti kapu* meghatározza, hogy a memóriasejt
befolyásolja-e az aktuális időlépés kimenetét.


![A bemeneti kapu, a felejtési kapu és a kimeneti kapu kiszámítása egy LSTM modellben.](../img/lstm-0.svg)
:label:`fig_lstm_0`

Matematikailag, tegyük fel, hogy $h$ rejtett egység van,
a batch mérete $n$, és a bemenetek száma $d$.
Ekkor a bemenet $\mathbf{X}_t \in \mathbb{R}^{n \times d}$
és az előző időlépés rejtett állapota
$\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$.
A $t$ időlépésnél a kapuk a következőképpen vannak definiálva:
a bemeneti kapu $\mathbf{I}_t \in \mathbb{R}^{n \times h}$,
a felejtési kapu $\mathbf{F}_t \in \mathbb{R}^{n \times h}$,
és a kimeneti kapu $\mathbf{O}_t \in \mathbb{R}^{n \times h}$.
Kiszámításuk a következőképpen történik:

$$
\begin{aligned}
\mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xi}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hi}} + \mathbf{b}_\textrm{i}),\\
\mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xf}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hf}} + \mathbf{b}_\textrm{f}),\\
\mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xo}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{ho}} + \mathbf{b}_\textrm{o}),
\end{aligned}
$$

ahol $\mathbf{W}_{\textrm{xi}}, \mathbf{W}_{\textrm{xf}}, \mathbf{W}_{\textrm{xo}} \in \mathbb{R}^{d \times h}$ és $\mathbf{W}_{\textrm{hi}}, \mathbf{W}_{\textrm{hf}}, \mathbf{W}_{\textrm{ho}} \in \mathbb{R}^{h \times h}$ súlyparaméterek,
és $\mathbf{b}_\textrm{i}, \mathbf{b}_\textrm{f}, \mathbf{b}_\textrm{o} \in \mathbb{R}^{1 \times h}$ eltolási paraméterek.
Vegyük figyelembe, hogy az összeg során broadcasting
(lásd :numref:`subsec_broadcasting`)
aktiválódik.
Sigmoid függvényeket alkalmazunk
(ahogy a :numref:`sec_mlp` fejezetben bevezettük),
hogy a bemeneti értékeket a $(0, 1)$ intervallumra leképezzük.


### Bemeneti Csomópont

Következőként tervezzük meg a memóriasejtet.
Mivel még nem specifikáltuk a különböző kapuk működését,
először bevezetjük a *bemeneti csomópontot*
$\tilde{\mathbf{C}}_t \in \mathbb{R}^{n \times h}$.
Kiszámítása hasonló a fent leírt három kapuéhoz,
de $(-1, 1)$ értéktartományú $\tanh$ aktivációs függvényt használ.
Ez a következő egyenlethez vezet a $t$ időlépésnél:

$$\tilde{\mathbf{C}}_t = \textrm{tanh}(\mathbf{X}_t \mathbf{W}_{\textrm{xc}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hc}} + \mathbf{b}_\textrm{c}),$$

ahol $\mathbf{W}_{\textrm{xc}} \in \mathbb{R}^{d \times h}$ és $\mathbf{W}_{\textrm{hc}} \in \mathbb{R}^{h \times h}$ súlyparaméterek és $\mathbf{b}_\textrm{c} \in \mathbb{R}^{1 \times h}$ eltolási paraméter.

A bemeneti csomópont rövid szemléltetése látható a :numref:`fig_lstm_1` ábrán.

![A bemeneti csomópont kiszámítása egy LSTM modellben.](../img/lstm-1.svg)
:label:`fig_lstm_1`


### Memóriasejt Belső Állapota

Az LSTM-ekben a bemeneti kapu $\mathbf{I}_t$ szabályozza,
hogy mennyit veszünk figyelembe az új adatból $\tilde{\mathbf{C}}_t$ révén,
a felejtési kapu $\mathbf{F}_t$ pedig azt szabályozza,
hogy a régi sejt belső állapotából $\mathbf{C}_{t-1} \in \mathbb{R}^{n \times h}$ mennyit őrzünk meg.
A Hadamard (elemenként vett) szorzat operátor $\odot$ alkalmazásával
a következő frissítési egyenlethez jutunk:

$$\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t.$$

Ha a felejtési kapu mindig 1 és a bemeneti kapu mindig 0,
a memóriasejt belső állapota $\mathbf{C}_{t-1}$
örökre változatlan marad,
és minden következő időlépésre változatlanul átadódik.
A bemeneti és felejtési kapuk azonban rugalmasságot adnak a modellnek,
hogy megtanulja, mikor kell ezt az értéket változatlanul hagyni,
és mikor kell módosítani a következő bemenetek hatására.
A gyakorlatban ez a tervezés enyhíti az eltűnő gradiens problémát,
olyan modellek létrehozásával, amelyeket jóval könnyebb tanítani,
különösen hosszú sorozathosszúságú adathalmaz esetén.

Így jutunk a :numref:`fig_lstm_2` ábrán látható folyamatábrához.

![A memóriasejt belső állapotának kiszámítása egy LSTM modellben.](../img/lstm-2.svg)

:label:`fig_lstm_2`


### Rejtett Állapot

Végül meg kell határoznunk, hogyan számítsuk ki a memóriasejt kimenetét,
azaz a rejtett állapotot $\mathbf{H}_t \in \mathbb{R}^{n \times h}$, ahogy más rétegek látják.
Ez az a pont, ahol a kimeneti kapu lép működésbe.
Az LSTM-ekben először a $\tanh$-t alkalmazzuk a memóriasejt belső állapotára,
majd egy újabb pontonkénti szorzást végzünk,
ezúttal a kimeneti kapuval.
Ez biztosítja, hogy $\mathbf{H}_t$ értékei
mindig a $(-1, 1)$ intervallumban legyenek:

$$\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t).$$


Amikor a kimeneti kapu közel van az 1-hez,
a memóriasejt belső állapota akadálytalanul befolyásolja a következő rétegeket,
míg ha a kimeneti kapu értékei közel vannak a 0-hoz,
az aktuális memória nem befolyásolja a hálózat többi rétegét
az aktuális időlépésben.
Vegyük figyelembe, hogy egy memóriasejt
sok időlépésen át gyűjthet információt a hálózat többi részének befolyásolása nélkül
(mindaddig, amíg a kimeneti kapu értékei közel vannak a 0-hoz),
majd hirtelen befolyásolhatja a hálózatot egy következő időlépésnél,
amint a kimeneti kapu a 0-hoz közeli értékekről az 1-hez közeli értékekre vált.
A :numref:`fig_lstm_3` ábra grafikusan szemlélteti az adatfolyamot.

![A rejtett állapot kiszámítása egy LSTM modellben.](../img/lstm-3.svg)
:label:`fig_lstm_3`



## Implementáció Alapoktól

Most implementáljunk egy LSTM-et alapoktól.
Ahogy a :numref:`sec_rnn-scratch` fejezetben végzett kísérletekben,
először betöltjük *The Time Machine* adathalmazt.

### **Modell Paramétereinek Inicializálása**

Ezután meg kell határoznunk és inicializálnunk a modell paramétereit.
Ahogy korábban, a `num_hiddens` hiperparaméter
határozza meg a rejtett egységek számát.
A súlyokat Gauss-eloszlásból inicializáljuk
0.01-es szórással,
és az eltolásokat 0-ra állítjuk.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LSTMScratch(d2l.Module):
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

        self.W_xi, self.W_hi, self.b_i = triple()  # Bemeneti kapu
        self.W_xf, self.W_hf, self.b_f = triple()  # Felejtési kapu
        self.W_xo, self.W_ho, self.b_o = triple()  # Kimeneti kapu
        self.W_xc, self.W_hc, self.b_c = triple()  # Bemeneti csomópont
```

```{.python .input}
%%tab jax
class LSTMScratch(d2l.Module):
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

        self.W_xi, self.W_hi, self.b_i = triple('i')  # Bemeneti kapu
        self.W_xf, self.W_hf, self.b_f = triple('f')  # Felejtési kapu
        self.W_xo, self.W_ho, self.b_o = triple('o')  # Kimeneti kapu
        self.W_xc, self.W_hc, self.b_c = triple('c')  # Bemeneti csomópont
```

:begin_tab:`pytorch, mxnet, tensorflow`
**A tényleges modell** a fent leírtak szerint van definiálva,
három kapuból és egy bemeneti csomópontból áll.
Vegyük figyelembe, hogy csak a rejtett állapot kerül átadásra a kimeneti rétegnek.
:end_tab:

:begin_tab:`jax`
**A tényleges modell** a fent leírtak szerint van definiálva,
három kapuból és egy bemeneti csomópontból áll.
Vegyük figyelembe, hogy csak a rejtett állapot kerül átadásra a kimeneti rétegnek.
A `forward` metódusban lévő hosszú for-ciklus rendkívül hosszú
JIT fordítási időt eredményez az első futásnál. Ennek megoldásaként,
ahelyett, hogy for-ciklussal frissítenénk az állapotot minden időlépésnél,
a JAX `jax.lax.scan` segédfüggvényét alkalmazzuk ugyanazon viselkedés eléréséhez.
Ez egy kezdeti `carry` állapotot és egy `inputs` tömböt vesz be, amelyet
a vezető tengelyén végigpásztáz. A `scan` transzformáció végül
visszaadja a végső állapotot és az összerakott kimeneteket.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LSTMScratch)
def forward(self, inputs, H_C=None):
    if H_C is None:
        # Kezdeti állapot alakja: (batch_size, num_hiddens)
        if tab.selected('mxnet'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          ctx=inputs.ctx)
            C = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          ctx=inputs.ctx)
        if tab.selected('pytorch'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          device=inputs.device)
            C = d2l.zeros((inputs.shape[1], self.num_hiddens),
                          device=inputs.device)
        if tab.selected('tensorflow'):
            H = d2l.zeros((inputs.shape[1], self.num_hiddens))
            C = d2l.zeros((inputs.shape[1], self.num_hiddens))
    else:
        H, C = H_C
    outputs = []
    for X in inputs:
        I = d2l.sigmoid(d2l.matmul(X, self.W_xi) +
                        d2l.matmul(H, self.W_hi) + self.b_i)
        F = d2l.sigmoid(d2l.matmul(X, self.W_xf) +
                        d2l.matmul(H, self.W_hf) + self.b_f)
        O = d2l.sigmoid(d2l.matmul(X, self.W_xo) +
                        d2l.matmul(H, self.W_ho) + self.b_o)
        C_tilde = d2l.tanh(d2l.matmul(X, self.W_xc) +
                           d2l.matmul(H, self.W_hc) + self.b_c)
        C = F * C + I * C_tilde
        H = O * d2l.tanh(C)
        outputs.append(H)
    return outputs, (H, C)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LSTMScratch)
def forward(self, inputs, H_C=None):
    # A lax.scan primitívet használjuk a bemenetek
    # végigiterálása helyett, mert scan megtakarít jit fordítási időt.
    def scan_fn(carry, X):
        H, C = carry
        I = d2l.sigmoid(d2l.matmul(X, self.W_xi) + (
            d2l.matmul(H, self.W_hi)) + self.b_i)
        F = d2l.sigmoid(d2l.matmul(X, self.W_xf) +
                        d2l.matmul(H, self.W_hf) + self.b_f)
        O = d2l.sigmoid(d2l.matmul(X, self.W_xo) +
                        d2l.matmul(H, self.W_ho) + self.b_o)
        C_tilde = d2l.tanh(d2l.matmul(X, self.W_xc) +
                           d2l.matmul(H, self.W_hc) + self.b_c)
        C = F * C + I * C_tilde
        H = O * d2l.tanh(C)
        return (H, C), H  # carry és y visszaadása

    if H_C is None:
        batch_size = inputs.shape[1]
        carry = jnp.zeros((batch_size, self.num_hiddens)), \
                jnp.zeros((batch_size, self.num_hiddens))
    else:
        carry = H_C

    # a scan a scan_fn-t, a kezdeti carry állapotot és xs-t veszi át, amelynek vezető tengelyét pásztázza
    carry, outputs = jax.lax.scan(scan_fn, carry, inputs)
    return outputs, carry
```

### **Tanítás** és Előrejelzés

Tanítsunk egy LSTM modellt az `RNNLMScratch` osztály példányosításával a :numref:`sec_rnn-scratch` fejezetből.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = d2l.RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        lstm = LSTMScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = d2l.RNNLMScratch(lstm, vocab_size=len(data.vocab), lr=4)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1)
trainer.fit(model, data)
```

## **Tömör Implementáció**

A magas szintű API-k segítségével
közvetlenül létrehozhatunk egy LSTM modellt.
Ez magába foglalja az összes konfigurációs részletet,
amelyeket fent explicit módon megadtunk.
A kód lényegesen gyorsabb, mivel fordított operátorokat használ a Python helyett
számos olyan részlethez, amelyeket korábban részletesen leírtunk.

```{.python .input}
%%tab mxnet
class LSTM(d2l.RNN):
    def __init__(self, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = rnn.LSTM(num_hiddens)

    def forward(self, inputs, H_C=None):
        if H_C is None: H_C = self.rnn.begin_state(
            inputs.shape[1], ctx=inputs.ctx)
        return self.rnn(inputs, H_C)
```

```{.python .input}
%%tab pytorch
class LSTM(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.LSTM(num_inputs, num_hiddens)

    def forward(self, inputs, H_C=None):
        return self.rnn(inputs, H_C)
```

```{.python .input}
%%tab tensorflow
class LSTM(d2l.RNN):
    def __init__(self, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = tf.keras.layers.LSTM(
                num_hiddens, return_sequences=True,
                return_state=True, time_major=True)

    def forward(self, inputs, H_C=None):
        outputs, *H_C = self.rnn(inputs, H_C)
        return outputs, H_C
```

```{.python .input}
%%tab jax
class LSTM(d2l.RNN):
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H_C=None, training=False):
        if H_C is None:
            batch_size = inputs.shape[1]
            H_C = nn.OptimizedLSTMCell.initialize_carry(jax.random.PRNGKey(0),
                                                        (batch_size,),
                                                        self.num_hiddens)

        LSTM = nn.scan(nn.OptimizedLSTMCell, variable_broadcast="params",
                       in_axes=0, out_axes=0, split_rngs={"params": False})

        H_C, outputs = LSTM()(H_C, inputs)
        return outputs, H_C
```

```{.python .input}
%%tab all
if tab.selected('pytorch'):
    lstm = LSTM(num_inputs=len(data.vocab), num_hiddens=32)
if tab.selected('mxnet', 'tensorflow', 'jax'):
    lstm = LSTM(num_hiddens=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    model = d2l.RNNLM(lstm, vocab_size=len(data.vocab), lr=4)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        model = d2l.RNNLM(lstm, vocab_size=len(data.vocab), lr=4)
trainer.fit(model, data)
```

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

Az LSTM-ek az archetipikus látens változós autoregresszív modell nemtriviális állapotvezérléssel.
Az évek során számos változatát javasolták, pl. több réteg, reziduális összekötések, különböző regularizációs típusok. Azonban az LSTM-ek és más sorozatmodellek (pl. GRU-k) tanítása meglehetősen költséges a sorozat hosszú távú függőségei miatt.
Később alternatív modellekkel, például Transformerekkel fogunk találkozni, amelyek bizonyos esetekben alkalmazhatók.


## Összefoglalás

Bár az LSTM-eket 1997-ben publikálták,
a 2000-es évek közepén előrejelzési versenyeken aratott győzelmekkel kerültek reflektorfénybe,
és 2011-től 2017-ig, a Transformer modellek megjelenéséig
a sorozattanulás domináns modelljeivé váltak.
Még a Transformer modellek is köszönnek néhány kulcsgondolatot
az LSTM által bevezetett architektúra-tervezési innovációknak.


Az LSTM-eknek háromféle kapujuk van:
bemeneti kapuk, felejtési kapuk és kimeneti kapuk,
amelyek szabályozzák az információáramlást.
Az LSTM rejtett rétegének kimenete tartalmazza a rejtett állapotot és a memóriasejt belső állapotát.
Csak a rejtett állapot kerül átadásra a kimeneti rétegnek,
míg a memóriasejt belső állapota teljesen belső marad.
Az LSTM-ek enyhíthetik az eltűnő és robbanó gradienseket.



## Feladatok

1. Állítsd be a hiperparamétereket, és elemezd azok hatását a futási időre, a perplexitásra és a kimeneti sorozatra.
1. Hogyan kellene módosítani a modellt, hogy helyes szavakat generáljon karaktersorozatok helyett?
1. Hasonlítsd össze a GRU-k, LSTM-ek és sima RNN-ek számítási költségét adott rejtett dimenzió esetén. Különös figyelmet fordíts a tanítási és következtetési költségre.
1. Mivel a jelölt memóriasejt biztosítja, hogy az értéktartomány $-1$ és $1$ között legyen a $\tanh$ függvény alkalmazásával, miért kell a rejtett állapotnak ismét $\tanh$ függvényt alkalmaznia, hogy biztosítsa a kimeneti értéktartomány $-1$ és $1$ közé esését?
1. Implementálj egy LSTM modellt idősorelem-előrejelzéshez karaktersorozat-előrejelzés helyett.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/343)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1057)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3861)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18016)
:end_tab:
