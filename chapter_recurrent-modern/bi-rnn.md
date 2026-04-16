# Kétirányú Rekurrens Neurális Hálózatok
:label:`sec_bi_rnn`

Eddig a sorozattanulási feladatok munkamintájában a nyelvi modellezés volt,
ahol a cél az volt, hogy az összes előző token alapján megjósoljuk a következő tokent egy sorozatban.
Ebben a forgatókönyvben csak a bal oldali kontextusra szeretnénk kondicionálni,
és így a standard RNN egyirányú láncolása megfelelőnek tűnik.
Azonban sok más sorozattanulási feladatkörnyezet létezik,
ahol teljesen indokolt minden időlépés előrejelzését
mind a bal oldali, mind a jobb oldali kontextusra kondicionálni.
Gondoljunk például a szófajtázásra (part of speech detection).
Miért ne vennénk figyelembe a kontextust mindkét irányból
egy adott szóhoz tartozó szófajta meghatározásakor?

Egy másik gyakori feladat – amely hasznos előtanítási gyakorlatként szolgálhat
egy tényleges célfeladaton való finomhangolás előtt –
az, hogy véletlenszerű tokeneket maszkolunk ki egy szöveges dokumentumban,
majd egy sorozatmodellt tanítunk a hiányzó tokenek értékeinek előrejelzésére.
Vegyük figyelembe, hogy attól függően, mi jön az üres hely után,
a hiányzó token valószínű értéke drámaian változik:

* Én `___` vagyok.
* Én `___` éhes vagyok.
* Én `___` éhes vagyok, és megenném az egész disznót.

Az első mondatban a "boldog" valószínű jelöltnek tűnik.
A "nem" és a "nagyon" szavak a második mondatban ésszerűnek tűnnek,
de a "nem" összeegyeztethetetlen a harmadik mondattal.


Szerencsére egy egyszerű technika bármely egyirányú RNN-t
kétirányú RNN-né alakítja :cite:`Schuster.Paliwal.1997`.
Egyszerűen két egyirányú RNN-réteget implementálunk,
amelyek egymással ellentétes irányban vannak összefűzve,
és ugyanazon bemeneten működnek (:numref:`fig_birnn`).
Az első RNN-rétegnél az első bemenet $\mathbf{x}_1$
és az utolsó bemenet $\mathbf{x}_T$,
de a második RNN-rétegnél az első bemenet $\mathbf{x}_T$
és az utolsó bemenet $\mathbf{x}_1$.
A kétirányú RNN-réteg kimenetének előállításához
egyszerűen összefűzzük a két alapul szolgáló egyirányú RNN-réteg megfelelő kimeneteit.


![Kétirányú RNN architektúrája.](../img/birnn.svg)
:label:`fig_birnn`


Formálisan bármely $t$ időlépésnél
egy $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ minibatch bemenetet vizsgálunk
(példák száma $=n$; az egyes példákban lévő bemenetek száma $=d$),
és legyen a rejtett réteg aktivációs függvénye $\phi$.
A kétirányú architektúrában
az előre és hátra irányú rejtett állapotok erre az időlépésre vonatkozóan
$\overrightarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$
és $\overleftarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$, ahol $h$ a rejtett egységek száma.
Az előre és hátra irányú rejtett állapot frissítései a következők:


$$
\begin{aligned}
\overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{\textrm{hh}}^{(f)}  + \mathbf{b}_\textrm{h}^{(f)}),\\
\overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{\textrm{hh}}^{(b)}  + \mathbf{b}_\textrm{h}^{(b)}),
\end{aligned}
$$

ahol a $\mathbf{W}_{\textrm{xh}}^{(f)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{\textrm{hh}}^{(f)} \in \mathbb{R}^{h \times h}, \mathbf{W}_{\textrm{xh}}^{(b)} \in \mathbb{R}^{d \times h}, \textrm{ és } \mathbf{W}_{\textrm{hh}}^{(b)} \in \mathbb{R}^{h \times h}$ súlyok, és a $\mathbf{b}_\textrm{h}^{(f)} \in \mathbb{R}^{1 \times h}$ és $\mathbf{b}_\textrm{h}^{(b)} \in \mathbb{R}^{1 \times h}$ torzítások mind a modell paraméterei.

Ezután összefűzzük az előre és hátra irányú rejtett állapotokat:
$\overrightarrow{\mathbf{H}}_t$ és $\overleftarrow{\mathbf{H}}_t$
alkotja a rejtett állapotot $\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$, amelyet a kimeneti rétegbe táplálunk.
Több rejtett réteggel rendelkező mély kétirányú RNN-ekben
ez az információ *bemenetként* kerül át a következő kétirányú rétegre.
Végül a kimeneti réteg kiszámítja a kimenetet
$\mathbf{O}_t \in \mathbb{R}^{n \times q}$ (kimenetek száma $=q$):

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}.$$

Itt a $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{2h \times q}$ súlymátrix
és a $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$ torzítás
a kimeneti réteg modell paraméterei.
Bár technikailag a két iránynak különböző számú rejtett egysége lehet,
ezt a tervezési döntést ritkán hozzák meg a gyakorlatban.
Most bemutatunk egy egyszerű kétirányú RNN-implementációt.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import npx, np
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
from jax import numpy as jnp
```

## Implementáció Alapoktól

Ha kétirányú RNN-t szeretnénk alapoktól implementálni,
két egyirányú `RNNScratch` példányt vehetünk fel
külön tanulható paraméterekkel.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class BiRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.f_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.b_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.num_hiddens *= 2  # A kimeneti dimenzió megduplázódik
```

```{.python .input}
%%tab jax
class BiRNNScratch(d2l.Module):
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        self.f_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.b_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.num_hiddens *= 2  # A kimeneti dimenzió megduplázódik
```

Az előre és hátra irányú RNN-ek állapotait külön frissítjük,
míg e két RNN kimeneteit összefűzzük.

```{.python .input}
%%tab all
@d2l.add_to_class(BiRNNScratch)
def forward(self, inputs, Hs=None):
    f_H, b_H = Hs if Hs is not None else (None, None)
    f_outputs, f_H = self.f_rnn(inputs, f_H)
    b_outputs, b_H = self.b_rnn(reversed(inputs), b_H)
    outputs = [d2l.concat((f, b), -1) for f, b in zip(
        f_outputs, reversed(b_outputs))]
    return outputs, (f_H, b_H)
```

## Tömör Implementáció

:begin_tab:`pytorch, mxnet, tensorflow`
A magas szintű API-k segítségével
tömörebben implementálhatunk kétirányú RNN-eket.
Példaként egy GRU modellt használunk.
:end_tab:

:begin_tab:`jax`
A Flax API nem kínál RNN-rétegeket, ezért nincs
`bidirectional` argumentum. Az alapoktól való implementálásban bemutatott módon
manuálisan kell megfordítani a bemeneteket,
ha kétirányú rétegre van szükség.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
class BiGRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.rnn = rnn.GRU(num_hiddens, bidirectional=True)
        if tab.selected('pytorch'):
            self.rnn = nn.GRU(num_inputs, num_hiddens, bidirectional=True)
        self.num_hiddens *= 2
```

## Összefoglalás

A kétirányú RNN-ekben minden időlépés rejtett állapotát egyidejűleg határozza meg az aktuális időlépés előtti és utáni adat. A kétirányú RNN-ek leginkább sorozatkódoláshoz és kétirányú kontextust figyelembe vevő megfigyelés-becsléshez hasznosak. A kétirányú RNN-ek tanítása nagyon költséges a hosszú gradiens-láncok miatt.

## Feladatok

1. Ha a különböző irányok eltérő számú rejtett egységet használnak, hogyan változik meg $\mathbf{H}_t$ alakja?
1. Tervezz kétirányú RNN-t több rejtett réteggel.
1. A poliszémia (többértelműség) közönséges a természetes nyelvekben. Például a "bank" szónak különböző jelentései vannak a "mentem a bankba pénzt befizetni" és a "mentem a folyópartra leülni" kontextusokban. Hogyan tervezhetnénk neurális hálózati modellt, amely adott kontextus-sorozat és szó esetén visszaadja a szó helyes kontextusban lévő vektoros reprezentációját? Milyen típusú neurális hálózat-architektúrák a legmegfelelőbbek a poliszémia kezelésére?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/339)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1059)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18019)
:end_tab:
