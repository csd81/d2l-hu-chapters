# Rekurrens Neurális Hálózatok
:label:`sec_rnn`


A :numref:`sec_language-model` fejezetben leírtuk a Markov-modelleket és $n$-gramokat a nyelvmodellezéshez, ahol a $t$ időlépésnél lévő $x_t$ token feltételes valószínűsége csak az előző $n-1$ tokentől függ.
Ha be szeretnénk vonni a $t-(n-1)$ időlépésnél korábbi tokenek lehetséges hatását $x_t$-re,
növelnünk kell $n$-t.
A modell paramétereinek száma azonban ezzel exponenciálisan nőne, mivel egy $\mathcal{V}$ szókincskészlethez $|\mathcal{V}|^n$ számot kell tárolnunk.
Ezért a $P(x_t \mid x_{t-1}, \ldots, x_{t-n+1})$ modellezése helyett célszerűbb egy látens változómodellt alkalmazni:

$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),$$

ahol $h_{t-1}$ egy *rejtett állapot*, amely a sorozat információját tárolja a $t-1$ időlépésig.
Általánosan,
a $t$ időlépésnél lévő rejtett állapot kiszámítható mind az aktuális $x_{t}$ bemenet, mind az előző $h_{t-1}$ rejtett állapot alapján:

$$h_t = f(x_{t}, h_{t-1}).$$
:eqlabel:`eq_ht_xt`

Egy kellően erős $f$ függvény esetén a :eqref:`eq_ht_xt`-ban a látens változómodell nem közelítés. Végül is $h_t$ egyszerűen tárolhatja az összes eddigi megfigyelt adatot.
Ez azonban potenciálisan drágává teheti mind a számítást, mind a tárolást.

Emlékezzünk arra, hogy a :numref:`chap_perceptrons` fejezetben rejtett egységekkel rendelkező rejtett rétegeket tárgyaltunk.
Figyelemre méltó, hogy
a rejtett rétegek és a rejtett állapotok két nagyon különböző fogalomra utalnak.
A rejtett rétegek, ahogy korábban kifejtettük, olyan rétegek, amelyek a bemenet és kimenet közötti úton nem láthatók.
A rejtett állapotok technikailag szólva *bemenetei* annak, amit egy adott lépésben csinálunk,
és csak a korábbi időlépések adatainak megtekintésével számíthatók ki.

A *rekurrens neurális hálózatok* (RNN-ek) rejtett állapotokkal rendelkező neurális hálózatok. Az RNN modell bevezetése előtt először újra megvizsgáljuk a :numref:`sec_mlp` fejezetben bemutatott MLP modellt.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
```

## Neurális hálózatok rejtett állapot nélkül

Nézzük meg egy egyrétegű rejtett réteggel rendelkező MLP-t.
Legyen a rejtett réteg aktivációs függvénye $\phi$.
Egy $n$ batch méretű és $d$ bemenetű példák $\mathbf{X} \in \mathbb{R}^{n \times d}$ minibatch-jéhez a rejtett réteg kimenete $\mathbf{H} \in \mathbb{R}^{n \times h}$ a következőképpen számítható:

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{\textrm{xh}} + \mathbf{b}_\textrm{h}).$$
:eqlabel:`rnn_h_without_state`

A :eqref:`rnn_h_without_state`-ban megvan a $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}$ súlyparaméter, a $\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$ torzítás paraméter, és a rejtett egységek $h$ száma a rejtett réteghez.
Ezzel felvértezve az összeadásnál broadcasting-et alkalmazunk (ld. :numref:`subsec_broadcasting`).
Ezután a rejtett réteg kimenete $\mathbf{H}$ a kimeneti réteg bemeneteként kerül felhasználásra, amelyet a következő adja meg:

$$\mathbf{O} = \mathbf{H} \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q},$$

ahol $\mathbf{O} \in \mathbb{R}^{n \times q}$ a kimeneti változó, $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$ a súlyparaméter, és $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$ a kimeneti réteg torzítás paramétere. Ha osztályozási problémáról van szó, a $\mathrm{softmax}(\mathbf{O})$ segítségével kiszámíthatjuk a kimeneti kategóriák valószínűségeloszlását.

Ez teljesen analóg a :numref:`sec_sequence` fejezetben korábban megoldott regressziós problémával, ezért kihagyjuk a részleteket.
Elég annyit megjegyezni, hogy véletlenszerűen vehetünk jellemző-címke párokat, és az automatikus differenciálás és stochastic gradient descent segítségével megtaníthatjuk a hálózat paramétereit.

## Rekurrens Neurális Hálózatok rejtett állapotokkal
:label:`subsec_rnn_w_hidden_states`

A dolgok teljesen másak, ha rejtett állapotaink vannak. Nézzük meg a struktúrát részletesebben.

Tegyük fel, hogy a $t$ időlépésnél
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$ bemenetekből álló minibatch-ünk van.
Más szóval,
$n$ sorozatpélda minibatch-jénél,
a $\mathbf{X}_t$ minden sora megfelel egy példának a $t$ időlépésnél a sorozatból.
Ezután
jelöljük $\mathbf{H}_t  \in \mathbb{R}^{n \times h}$-val a $t$ időlépés rejtett réteg kimenetét.
Az MLP-vel ellentétben itt elmentjük az előző időlépés $\mathbf{H}_{t-1}$ rejtett réteg kimenetét, és bevezetünk egy új $\mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$ súlyparamétert, hogy leírjuk, hogyan kell az előző időlépés rejtett réteg kimenetét az aktuális időlépésben használni. Konkrétan, az aktuális időlépés rejtett réteg kimenetének kiszámítása az aktuális időlépés bemenete és az előző időlépés rejtett réteg kimenete alapján történik:

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}  + \mathbf{b}_\textrm{h}).$$
:eqlabel:`rnn_h_with_state`

A :eqref:`rnn_h_without_state`-val összehasonlítva a :eqref:`rnn_h_with_state` egy további $\mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}$ tagot ad hozzá, ezáltal
megvalósítva a :eqref:`eq_ht_xt`-t.
A szomszédos időlépések $\mathbf{H}_t$ és $\mathbf{H}_{t-1}$ rejtett réteg kimenetei közötti kapcsolatból
tudjuk, hogy ezek a változók megragadták és megőrizték a sorozat historikus információját az aktuális időlépésükig, akárcsak a neurális hálózat aktuális időlépésének állapota vagy memóriája. Ezért az ilyen rejtett réteg kimenetet *rejtett állapotnak* nevezzük.
Mivel a rejtett állapot az aktuális időlépésnél az előző időlépés azonos definícióját használja, a :eqref:`rnn_h_with_state` számítása *rekurrens*. Ezért, ahogy mondtuk, a rekurrens számításon alapuló rejtett állapotokkal rendelkező neurális hálózatokat
*rekurrens neurális hálózatoknak* nevezzük.
Azokat a rétegeket, amelyek a :eqref:`rnn_h_with_state` számítását végzik
az RNN-ekben,
*rekurrens rétegeknek* nevezzük.


Az RNN-ek felépítésének számos különböző módja van.
Azok, amelyek a :eqref:`rnn_h_with_state` által meghatározott rejtett állapottal rendelkeznek, nagyon elterjedtek.
A $t$ időlépésnél
a kimeneti réteg kimenete hasonló az MLP számításához:

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}.$$

Az RNN paraméterei
magukban foglalják a rejtett réteg $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}, \mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$ súlyait
és $\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$ torzítását,
valamint a kimeneti réteg $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$ súlyait
és $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$ torzítását.
Érdemes megemlíteni, hogy
még különböző időlépéseknél is
az RNN-ek mindig ugyanezeket a modellparamétereket használják.
Ezért egy RNN paraméterezési költsége
nem növekszik az időlépések számának növekedésével.

A :numref:`fig_rnn` szemlélteti egy RNN számítási logikáját három egymást követő időlépésnél.
Bármely $t$ időlépésnél
a rejtett állapot számítása kezelhető:
(i) az aktuális $t$ időlépés $\mathbf{X}_t$ bemenete és az előző $t-1$ időlépés $\mathbf{H}_{t-1}$ rejtett állapota összefűzéseként;
(ii) az összefűzés eredményének egy teljesen összekötött rétegbe adásaként a $\phi$ aktivációs függvénnyel.
Az ilyen teljesen összekötött réteg kimenete az aktuális $t$ időlépés $\mathbf{H}_t$ rejtett állapota.
Ebben az esetben
a modell paraméterei a $\mathbf{W}_{\textrm{xh}}$ és $\mathbf{W}_{\textrm{hh}}$ összefűzése, és egy $\mathbf{b}_\textrm{h}$ torzítás, mindkettő a :eqref:`rnn_h_with_state`-ből.
Az aktuális $t$ időlépés rejtett állapota, $\mathbf{H}_t$, részt vesz a következő $t+1$ időlépés $\mathbf{H}_{t+1}$ rejtett állapotának kiszámításában.
Sőt, $\mathbf{H}_t$-t
a teljesen összekötött kimeneti rétegbe is
betáplálják a kimenet
$\mathbf{O}_t$ kiszámításához az aktuális $t$ időlépésnél.

![Egy RNN rejtett állapottal.](../img/rnn.svg)
:label:`fig_rnn`

Éppen megemlítettük, hogy a rejtett állapothoz tartozó $\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}$ számítása ekvivalens a
$\mathbf{X}_t$ és $\mathbf{H}_{t-1}$ összefűzésének
és a
$\mathbf{W}_{\textrm{xh}}$ és $\mathbf{W}_{\textrm{hh}}$ összefűzésének mátrix-szorzatával.
Bár ez matematikailag igazolható,
az alábbiakban csak egy egyszerű kódrészletet mutatunk be szemléltetésként.
Kezdetnek
definiáljuk az `X`, `W_xh`, `H` és `W_hh` mátrixokat, amelyek alakjai rendre (3, 1), (1, 4), (3, 4) és (4, 4).
Az `X`-t megszorozva `W_xh`-val, a `H`-t megszorozva `W_hh`-val, majd összeadva a két szorzatot,
egy (3, 4) alakú mátrixot kapunk.

```{.python .input}
%%tab mxnet, pytorch
X, W_xh = d2l.randn(3, 1), d2l.randn(1, 4)
H, W_hh = d2l.randn(3, 4), d2l.randn(4, 4)
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
%%tab tensorflow
X, W_xh = d2l.normal((3, 1)), d2l.normal((1, 4))
H, W_hh = d2l.normal((3, 4)), d2l.normal((4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
%%tab jax
X, W_xh = jax.random.normal(d2l.get_key(), (3, 1)), jax.random.normal(
                                                        d2l.get_key(), (1, 4))
H, W_hh = jax.random.normal(d2l.get_key(), (3, 4)), jax.random.normal(
                                                        d2l.get_key(), (4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

Most összefűzzük az `X` és `H` mátrixokat
oszlopok mentén (1-es tengelyen),
és a `W_xh` és `W_hh` mátrixokat sorok mentén (0-s tengelyen).
Ez a két összefűzés
(3, 5) és (5, 4) alakú mátrixokat eredményez rendre.
Megszorozva ezt a két összefűzött mátrixot,
ugyanazt a (3, 4) alakú kimeneti mátrixot kapjuk,
mint fentebb.

```{.python .input}
%%tab all
d2l.matmul(d2l.concat((X, H), 1), d2l.concat((W_xh, W_hh), 0))
```

## RNN-alapú karakter szintű nyelvmodellek

Emlékezzünk arra, hogy a :numref:`sec_language-model` fejezetbeli nyelvmodellezésnél
a következő tokent a jelenlegi és múltbeli tokenek alapján kívánjuk megjósolni;
ezért az eredeti sorozatot egy tokennel eltoljuk
célokként (címkékként).
:citet:`Bengio.Ducharme.Vincent.ea.2003` először javasolta
neurális hálózat alkalmazását a nyelvmodellezéshez.
Az alábbiakban bemutatjuk, hogyan alkalmazhatók az RNN-ek egy nyelvmodell felépítéséhez.
Legyen a minibatch mérete egy, és a szöveg sorozata "machine".
A tanítás egyszerűsítése érdekében a következő szakaszokban
a szöveget szavak helyett karakterekre tokenizáljuk,
és egy *karakter szintű nyelvmodellt* veszünk figyelembe.
A :numref:`fig_rnn_train` bemutatja, hogyan lehet megjósolni a következő karaktert az aktuális és korábbi karakterek alapján egy RNN-nel a karakter szintű nyelvmodellezéshez.

![Egy karakter szintű nyelvmodell RNN alapján. A bemeneti és célsorozatok rendre "machin" és "achine".](../img/rnn-train.svg)
:label:`fig_rnn_train`

A tanítási folyamat során
softmax műveletet futtatunk a kimeneti réteg kimenetén minden időlépésnél, majd a keresztentrópia-veszteséget használjuk a modell kimenete és a cél közötti hiba kiszámításához.
A rejtett állapot rekurrens számítása miatt a rejtett rétegben a :numref:`fig_rnn_train`-ban látható 3. időlépés kimenete $\mathbf{O}_3$ a "m", "a" és "c" szöveges sorozattól függ. Mivel a sorozat következő karaktere a tanítási adatokban "h", a 3. időlépés vesztesége a "m", "a", "c" jellemzősorozat alapján generált következő karakter valószínűségeloszlásától és az ezen időlépés "h" céljától fog függeni.

A gyakorlatban minden tokent egy $d$-dimenziós vektor képviseli, és egy $n>1$ batch méretet alkalmazunk. Ezért a $t$ időlépésnél lévő $\mathbf X_t$ bemenet egy $n\times d$ mátrix lesz, ami azonos azzal, amit a :numref:`subsec_rnn_w_hidden_states` fejezetben tárgyaltunk.

A következő szakaszokban RNN-eket implementálunk
karakter szintű nyelvmodellek számára.


## Összefoglalás

Azt a neurális hálózatot, amely rekurrens számítást alkalmaz a rejtett állapotokhoz, rekurrens neurális hálózatnak (RNN) nevezzük.
Az RNN rejtett állapota megragadhatja a sorozat historikus információját az aktuális időlépésig. A rekurrens számítással az RNN modellparaméterek száma nem növekszik az időlépések számának növekedésével. Ami az alkalmazásokat illeti, egy RNN karakter szintű nyelvmodellek létrehozásához is használható.


## Feladatok

1. Ha RNN-t használunk egy szöveges sorozat következő karakterének megjóslásához, milyen szükséges dimenzióval kell rendelkeznie bármely kimenetnek?
1. Miért tudnak az RNN-ek kifejezni egy token feltételes valószínűségét egy adott időlépésnél a szöveges sorozat összes korábbi tokenje alapján?
1. Mi történik a gradienssel, ha visszaterjesztünk egy hosszú sorozaton?
1. Milyen problémák kapcsolódnak az ebben a szakaszban leírt nyelvmodellhez?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/337)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1050)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1051)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/180013)
:end_tab:
