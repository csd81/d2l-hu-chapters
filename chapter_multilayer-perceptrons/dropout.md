```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Dropout
:label:`sec_dropout`


Gondoljuk át röviden, mit várunk
egy jó prediktív modelltől.
Azt szeretnénk, hogy nem látott adatokon is jól teljesítsen.
A klasszikus általánosítási elmélet
azt sugallja, hogy a tanítási és teszt teljesítmény közötti rés csökkentéséhez
egyszerű modellre kellene törekednünk.
Az egyszerűség megjelenhet
kevés dimenzió formájában.
Ezt vizsgáltuk a lineáris modellek
monomiális bázisfüggvényeinek tárgyalásakor
a :numref:`sec_generalization_basics` részben.
Emellett, ahogy a súlybomlás
($\ell_2$ regularizáció) tárgyalásakor láttuk a :numref:`sec_weight_decay` részben,
a paraméterek (inverz) normája szintén
hasznos egyszerűségi mérőszám.
Az egyszerűség egy másik hasznos fogalma a simaság,
azaz hogy a függvény ne legyen érzékeny
a bemeneteinek kis változásaira.
Például, amikor képeket osztályozunk,
arra számítunk, hogy valamennyi véletlenszerű zajt adni
a pixelekhez nagyrészt ártalmatlan lenne.

:citet:`Bishop.1995` formalizálta
ezt az ötletet, amikor bebizonyította, hogy a bemeneti zajjal való tanítás
egyenértékű a Tikhonov regularizációval.
Ez a munka egyértelmű matematikai kapcsolatot teremtett
aközött a követelmény között, hogy egy függvény sima legyen (és így egyszerű),
és aközött a követelmény között, hogy ellenálló legyen
a bemeneti perturbációkkal szemben.

Majd :citet:`Srivastava.Hinton.Krizhevsky.ea.2014`
egy okos ötletet dolgozott ki arra, hogyan alkalmazza Bishop ötletét
a hálózat belső rétegeire is.
Ötletüket, amelyet *dropoutnak* hívnak, az az elképzelés jellemez,
hogy zajt injektálnak az előre irányú terjesztés során
minden belső réteg kiszámításakor,
és ez a technika standard módszerré vált
a neurális hálózatok tanításában.
A módszert *dropoutnak* nevezzük, mert szó szerint
*kiejtünk* néhány neuront tanítás közben.
A tanítás során minden iterációban
a standard dropout abból áll, hogy nullázzuk
az egyes rétegek csomópontjainak egy bizonyos töredékét
a következő réteg kiszámítása előtt.

Pontosság kedvéért meg kell jegyezni,
hogy saját narratívánkat alkalmazzuk Bishop és a dropout közötti kapcsolatban.
A dropoutról szóló eredeti cikk
meglepő analógiával, az ivaros szaporodással magyarázza az intuíciót.
A szerzők azzal érvelnek, hogy a neurális hálózat túlillesztése
olyan állapotban jellemzett, amelyben
minden réteg az előző réteg aktivációinak egy specifikus
mintájától függ,
ezt a feltételt *ko-adaptációnak* nevezve.
A dropout, állítják, megtöri a ko-adaptációt,
ahogyan az ivaros szaporodás is megtöri
a ko-adaptált géneket.
Bár ez az elméleti indoklás biztosan vitatható,
maga a dropout technika tartósnak bizonyult,
és a dropout különböző formái implementálva vannak
a legtöbb mély tanulási könyvtárban.


A legfontosabb kihívás az, hogyan injektáljuk a zajt.
Az egyik ötlet az, hogy *torzítatlan* módon injektáljuk,
úgy, hogy minden réteg várható értéke — miközben
a többit rögzítjük — egyenlő legyen azzal az értékkel, amelyet zaj nélkül vett volna fel.
Bishop munkájában Gauss-zajt adott hozzá
egy lineáris modell bemeneteihez.
Minden tanítási iterációnál zajt adott hozzá,
amelyet nullás átlagú eloszlásból mintavételeztek
$\epsilon \sim \mathcal{N}(0,\sigma^2)$, a $\mathbf{x}$ bemenethez,
egy perturbált $\mathbf{x}' = \mathbf{x} + \epsilon$ pontot adva.
Várható értékben $E[\mathbf{x}'] = \mathbf{x}$.

A standard dropout regularizációban
nullázzuk a csomópontok egy töredékét az egyes rétegekben,
majd *detorzítjuk* az egyes rétegeket normalizálással
az megtartott (nem kiesett) csomópontok töredékével.
Más szóval,
$p$ *dropout valószínűséggel*,
minden $h$ köztes aktivációt
egy $h'$ véletlen változóval helyettesítünk az alábbiak szerint:

$$
\begin{aligned}
h' =
\begin{cases}
    0 & \textrm{ valószínűséggel } p \\
    \frac{h}{1-p} & \textrm{ egyébként}
\end{cases}
\end{aligned}
$$

Tervezés szerint a várható érték változatlan marad, azaz $E[h'] = h$.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
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
from functools import partial
import jax
from jax import numpy as jnp
import optax
```

## Dropout a gyakorlatban

Idézzük fel az MLP-t egy rejtett réteggel és öt rejtett egységgel
a :numref:`fig_mlp` ábrából.
Amikor dropoutot alkalmazunk egy rejtett rétegre,
minden rejtett egységet $p$ valószínűséggel nullázva,
az eredmény úgy tekinthető, mint egy hálózat,
amely csak az eredeti neuronok egy részét tartalmazza.
A :numref:`fig_dropout2` ábrán $h_2$ és $h_5$ el van távolítva.
Ennek következtében a kimenetek kiszámítása
már nem függ $h_2$-től vagy $h_5$-től,
és a megfelelő gradienseik is eltűnnek
a visszaterjesztés végrehajtásakor.
Így a kimeneti réteg kiszámítása
nem lehet túlzottan függő a $h_1, \ldots, h_5$ bármely elemétől.

![MLP dropout előtt és után.](../img/dropout2.svg)
:label:`fig_dropout2`

Általában letiltjuk a dropoutot teszt idején.
Egy tanított modell és egy új példány esetén
nem ejtünk ki egyetlen csomópontot sem,
és így nincs szükség normalizálásra.
Vannak azonban kivételek:
néhány kutató teszt idején is alkalmaz dropoutot heurisztikaként
a neurális hálózat előrejelzéseinek *bizonytalanságának* becslésére:
ha az előrejelzések egyeznek sok különböző dropout kimenet esetén,
akkor azt mondhatjuk, hogy a hálózat magabiztosabb.

## Implementálás nulláról

Egyetlen réteg dropout függvényének implementálásához
annyi mintát kell húznunk
egy Bernoulli (bináris) véletlen változóból,
amennyit a rétegünk dimenziói száma,
ahol a véletlen változó $1$ (megtartás) értéket vesz fel
$1-p$ valószínűséggel és $0$ (kiesés) értéket $p$ valószínűséggel.
Ennek egyik egyszerű megvalósítása az $U[0, 1]$-es egyenletes eloszlásból való mintavételezés.
Aztán megtarthatjuk azokat a csomópontokat, amelyeknél a megfelelő minta nagyobb $p$-nél,
a többit ejtve.

A következő kódban [**egy `dropout_layer` függvényt implementálunk,
amely az `X` tenzor bemenet elemeit `dropout` valószínűséggel ejti ki**],
a maradékot a fentebb leírtak szerint újraskálázva:
a túlélőket `1.0-dropout`-tal osztva.

```{.python .input}
%%tab mxnet
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return np.zeros_like(X)
    mask = np.random.uniform(0, 1, X.shape) > dropout
    return mask.astype(np.float32) * X / (1.0 - dropout)
```

```{.python .input}
%%tab pytorch
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
```

```{.python .input}
%%tab tensorflow
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return tf.zeros_like(X)
    mask = tf.random.uniform(
        shape=tf.shape(X), minval=0, maxval=1) < 1 - dropout
    return tf.cast(mask, dtype=tf.float32) * X / (1.0 - dropout)
```

```{.python .input}
%%tab jax
def dropout_layer(X, dropout, key=d2l.get_key()):
    assert 0 <= dropout <= 1
    if dropout == 1: return jnp.zeros_like(X)
    mask = jax.random.uniform(key, X.shape) > dropout
    return jnp.asarray(mask, dtype=jnp.float32) * X / (1.0 - dropout)
```

[**A `dropout_layer` függvényt néhány példán tesztelhetjük**].
A következő kódsorokban
a `X` bemenetünket a dropout műveleten vesszük át,
0, 0,5 és 1 valószínűséggel rendre.

```{.python .input}
%%tab all
if tab.selected('mxnet'):
    X = np.arange(16).reshape(2, 8)
if tab.selected('pytorch'):
    X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
if tab.selected('tensorflow'):
    X = tf.reshape(tf.range(16, dtype=tf.float32), (2, 8))
if tab.selected('jax'):
    X = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
print('dropout_p = 0:', dropout_layer(X, 0))
print('dropout_p = 0.5:', dropout_layer(X, 0.5))
print('dropout_p = 1:', dropout_layer(X, 1))
```

### A modell definiálása

Az alábbi modell dropoutot alkalmaz minden rejtett réteg kimenetére
(az aktivációs függvény után).
Az egyes rétegek dropout valószínűségét külön-külön állíthatjuk be.
Általánosan elterjedt megközelítés, hogy
alacsonyabb dropout valószínűséget állítunk be a bemeneti réteghez közelebb.
Biztosítjuk, hogy a dropout csak tanítás során aktív.

```{.python .input}
%%tab mxnet
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.Dense(num_hiddens_1, activation='relu')
        self.lin2 = nn.Dense(num_hiddens_2, activation='relu')
        self.lin3 = nn.Dense(num_outputs)
        self.initialize()

    def forward(self, X):
        H1 = self.lin1(X)
        if autograd.is_training():
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.lin2(H1)
        if autograd.is_training():
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab pytorch
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:  
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab tensorflow
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = tf.keras.layers.Dense(num_hiddens_1, activation='relu')
        self.lin2 = tf.keras.layers.Dense(num_hiddens_2, activation='relu')
        self.lin3 = tf.keras.layers.Dense(num_outputs)

    def forward(self, X):
        H1 = self.lin1(tf.reshape(X, (X.shape[0], -1)))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.lin2(H1)
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab jax
class DropoutMLPScratch(d2l.Classifier):
    num_hiddens_1: int
    num_hiddens_2: int
    num_outputs: int
    dropout_1: float
    dropout_2: float
    lr: float
    training: bool = True

    def setup(self):
        self.lin1 = nn.Dense(self.num_hiddens_1)
        self.lin2 = nn.Dense(self.num_hiddens_2)
        self.lin3 = nn.Dense(self.num_outputs)
        self.relu = nn.relu

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape(X.shape[0], -1)))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

### [**Tanítás**]

Az alábbiakban hasonló az MLP-k korábban leírt tanításához.

```{.python .input}
%%tab all
hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256,
           'dropout_1':0.5, 'dropout_2':0.5, 'lr':0.1}
model = DropoutMLPScratch(**hparams)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## [**Tömör implementáció**]

A magas szintű API-kkal mindössze annyit kell tennünk, hogy egy `Dropout` réteget adunk
minden teljesen összekötött réteg után,
átadva a dropout valószínűséget
egyetlen argumentumként a konstruktornak.
Tanítás közben a `Dropout` réteg véletlenszerűen
kiejti az előző réteg kimeneteit
(vagy egyenértékűen a következő réteg bemeneteit)
a megadott dropout valószínűség szerint.
Ha nem tanítási módban van,
a `Dropout` réteg egyszerűen átengedi az adatokat teszt közben.

```{.python .input}
%%tab mxnet
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens_1, activation="relu"),
                     nn.Dropout(dropout_1),
                     nn.Dense(num_hiddens_2, activation="relu"),
                     nn.Dropout(dropout_2),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```{.python .input}
%%tab pytorch
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(), 
            nn.Dropout(dropout_1), nn.LazyLinear(num_hiddens_2), nn.ReLU(), 
            nn.Dropout(dropout_2), nn.LazyLinear(num_outputs))
```

```{.python .input}
%%tab tensorflow
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens_1, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_1),
            tf.keras.layers.Dense(num_hiddens_2, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_2),
            tf.keras.layers.Dense(num_outputs)])
```

```{.python .input}
%%tab jax
class DropoutMLP(d2l.Classifier):
    num_hiddens_1: int
    num_hiddens_2: int
    num_outputs: int
    dropout_1: float
    dropout_2: float
    lr: float
    training: bool = True

    @nn.compact
    def __call__(self, X):
        x = nn.relu(nn.Dense(self.num_hiddens_1)(X.reshape((X.shape[0], -1))))
        x = nn.Dropout(self.dropout_1, deterministic=not self.training)(x)
        x = nn.relu(nn.Dense(self.num_hiddens_2)(x))
        x = nn.Dropout(self.dropout_2, deterministic=not self.training)(x)
        return nn.Dense(self.num_outputs)(x)
```

:begin_tab:`jax`
Vegyük észre, hogy újra kell definiálnunk a veszteségfüggvényt, mivel egy
dropout réteggel rendelkező hálózatnak PRNGKey-re van szüksége a `Module.apply()` használatakor,
és ennek az RNG vetőmagnak explicit módon `dropout` nevet kell kapnia. Ezt a kulcsot
a `dropout` réteg használja a Flax-ban a véletlenszerű dropout maszk belső generálásához.
Fontos, hogy minden egyes epoch esetén egyedi `dropout_rng` kulcsot használjunk a tanítási ciklusban,
különben a generált dropout maszk nem lesz sztochasztikus és különböző az egyes epoch futások között.
Ez a `dropout_rng` tárolható a
`TrainState` objektumban (a :numref:`oo-design-training` részben definiált `d2l.Trainer` osztályban)
attribútumként, és minden egyes epochban egy új `dropout_rng`-vel cseréljük ki.
Ezt már kezeltük a :numref:`sec_linear_scratch` részben definiált `fit_epoch` metódussal.
:end_tab:

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    Y_hat = state.apply_fn({'params': params}, *X,
                           mutable=False,  # Később lesz használva (pl. batch norm)
                           rngs={'dropout': state.dropout_rng})
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    # A visszaadott üres szótár egy helyfoglaló a segédadatok számára,
    # amelyet később fogunk használni (pl. batch norm esetén)
    return (fn(Y_hat, Y).mean(), {}) if averaged else (fn(Y_hat, Y), {})
```

Ezután [**tanítjuk a modellt**].

```{.python .input}
%%tab all
model = DropoutMLP(**hparams)
trainer.fit(model, data)
```

## Összefoglalás

A dimenziók számának és a súlyvektor méretének szabályozásán túl a dropout egy újabb eszköz a túlillesztés elkerülésére. Az eszközöket gyakran együttesen alkalmazzák.
Vegyük észre, hogy a dropout
csak tanítás során alkalmazott:
egy $h$ aktivációt egy $h$ várható értékű véletlen változóval helyettesít.


## Feladatok

1. Mi történik, ha megváltoztatjuk az első és második réteg dropout valószínűségeit? Különösen, mi történik, ha felcseréljük mindkét réteg valószínűségét? Tervezz egy kísérletet e kérdések megválaszolására, írd le az eredményeket számszerűen, és foglald össze a kvalitatív következtetéseket.
1. Növeld az epochok számát, és hasonlítsd össze a dropout alkalmazásával kapott eredményeket azokkal, amelyeket anélkül kaptál.
1. Mi az aktivációk varianciája minden rejtett rétegben, ha alkalmazzuk a dropoutot, és ha nem? Rajzolj egy ábrát, amely megmutatja, hogyan fejlődik ez a mennyiség mindkét modell esetén az idő múlásával.
1. Miért nem alkalmazzák általában a dropoutot teszt idején?
1. A jelen részben lévő modellt példaként használva, hasonlítsd össze a dropout és a súlybomlás hatásait. Mi történik, ha a dropout és a súlybomlás egyidejűleg alkalmazzák? Additívak-e az eredmények? Vannak-e csökkent hozamok (vagy rosszabbak)? Kioltják-e egymást?
1. Mi történik, ha a dropout-ot a súlymátrix egyes súlyaira alkalmazzuk az aktivációk helyett?
1. Találj ki egy másik technikát véletlenszerű zaj injektálására minden rétegben, amely különbözik a standard dropout technikától. Tudsz-e olyan módszert kidolgozni, amely felülmúlja a dropoutot a Fashion-MNIST adathalmazon (rögzített architektúra esetén)?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/100)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/101)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/261)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17987)
:end_tab:
