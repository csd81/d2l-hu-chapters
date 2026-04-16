```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Softmax regresszió tömör implementációja
:label:`sec_softmax_concise`



A magas szintű mély tanulási keretrendszerek
éppúgy megkönnyítik a softmax regresszió implementálását,
mint ahogy a lineáris regresszióét is megkönnyítették
(lásd :numref:`sec_linear_concise`).

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
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
from functools import partial
import jax
from jax import numpy as jnp
import optax
```

## A modell meghatározása

A :numref:`sec_linear_concise` szakaszhoz hasonlóan
a teljesen összekötött réteget a beépített réteg segítségével hozzuk létre.
A beépített `__call__` metódus ezután meghívja a `forward` metódust,
valahányszor a hálózatot egy bemenetre kell alkalmazni.

:begin_tab:`mxnet`
Bár az `X` bemenet negyedrendű tenzor,
a beépített `Dense` réteg automatikusan másodrendű tenzorrá alakítja `X`-et,
az első tengely dimenzióit érintetlenül hagyva.
:end_tab:

:begin_tab:`pytorch`
Egy `Flatten` réteget használunk a negyedrendű `X` tenzor másodrendűvé alakítására,
az első tengely dimenzióit érintetlenül hagyva.

:end_tab:

:begin_tab:`tensorflow`
Egy `Flatten` réteget használunk a negyedrendű `X` tenzor átalakítására,
az első tengely dimenzióit érintetlenül hagyva.
:end_tab:

:begin_tab:`jax`
A Flax lehetővé teszi a felhasználók számára, hogy a hálózati osztályt kompaktabb formában írják meg
a `@nn.compact` dekorátor segítségével. A `@nn.compact` használatával
az összes hálózati logika egyetlen „előre irányú átmeneti" metódusba írható,
anélkül, hogy a dataclass-ban definiálni kellene a szokásos `setup` metódust.
:end_tab:

```{.python .input}
%%tab pytorch
class SoftmaxRegression(d2l.Classifier):  #@save
    """A softmax regressziós modell."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))

    def forward(self, X):
        return self.net(X)
```

```{.python .input}
%%tab mxnet, tensorflow
class SoftmaxRegression(d2l.Classifier):  #@save
    """A softmax regressziós modell."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(num_outputs)
            self.net.initialize()
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            self.net.add(tf.keras.layers.Flatten())
            self.net.add(tf.keras.layers.Dense(num_outputs))

    def forward(self, X):
        return self.net(X)
```

```{.python .input}
%%tab jax
class SoftmaxRegression(d2l.Classifier):  #@save
    num_outputs: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        X = nn.Dense(self.num_outputs)(X)
        return X
```

## A Softmax újratekintve
:label:`subsec_softmax-implementation-revisited`

A :numref:`sec_softmax_scratch` szakaszban kiszámítottuk a modell kimenetét
és alkalmaztuk a keresztentrópia-veszteséget. Bár ez matematikailag teljesen helyes,
számítástechnikai szempontból kockázatos, mivel az exponenciálásban numerikus alulcsordulás és túlcsordulás léphet fel.

Emlékeztetőül: a softmax függvény a valószínűségeket a
$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$ formulával számítja.
Ha egyes $o_k$ értékek nagyon nagyok, vagyis nagyon pozitívak,
akkor $\exp(o_k)$ meghaladhatja bizonyos adattípusoknál a maximálisan ábrázolható számot. Ezt *túlcsordulásnak* nevezzük. Hasonlóan,
ha minden argumentum nagyon nagy negatív szám, *alulcsordulás* következik be.
Például az egyszeres pontosságú lebegőpontos számok kb. $10^{-38}$-tól $10^{38}$-ig fedik a tartományt. Emiatt ha $\mathbf{o}$ legnagyobb eleme a $[-90, 90]$ intervallumon kívül esik, az eredmény nem lesz stabil.
Ennek megoldása az $\bar{o} \stackrel{\textrm{def}}{=} \max_k o_k$ kivonása minden elemből:

$$
\hat y_j = \frac{\exp o_j}{\sum_k \exp o_k} =
\frac{\exp(o_j - \bar{o}) \exp \bar{o}}{\sum_k \exp (o_k - \bar{o}) \exp \bar{o}} =
\frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})}.
$$

A konstrukcióból következik, hogy minden $j$-re $o_j - \bar{o} \leq 0$. Így egy $q$-osztályos
osztályozási feladatnál a nevező az $[1, q]$ intervallumba esik. Ráadásul a
számláló soha nem haladja meg az $1$-et, megakadályozva a numerikus túlcsordulást. Numerikus alulcsordulás csak
akkor következik be, ha $\exp(o_j - \bar{o})$ numerikusan $0$-nak értékelődik ki. Mégis, néhány lépéssel
arrébb bajba kerülhetünk, amikor $\log \hat{y}_j$-t $\log 0$-ként akarjuk kiszámítani.
Különösen a visszaterjesztésnél
találhatjuk magunkat egy teli képernyőnyi
rettegett `NaN` (Not a Number) eredménnyel szembe.

Szerencsére megmentéseinkre van az a tény, hogy
bár exponenciális függvényeket számítunk,
végül azok logaritmusát kívánjuk venni
(a keresztentrópia-veszteség kiszámításánál).
A softmax és a keresztentrópia összekapcsolásával
teljesen elkerülhetjük a numerikus stabilitási problémákat:

$$
\log \hat{y}_j =
\log \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})} =
o_j - \bar{o} - \log \sum_k \exp (o_k - \bar{o}).
$$

Ez egyszerre kerüli el a túlcsordulást és az alulcsordulást.
A hagyományos softmax függvényt kéznél tartjuk,
ha a modell kimeneti valószínűségeit kiértékelnénk.
De a softmax valószínűségek helyett egyszerűen
[**a logitokat adjuk át, és a softmax-ot és annak logaritmusát
egyszerre számítjuk ki a keresztentrópia-veszteség függvényén belül,**]
amely okos trükköket alkalmaz, mint a [„LogSumExp trükk"](https://en.wikipedia.org/wiki/LogSumExp).

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    if tab.selected('mxnet'):
        fn = gluon.loss.SoftmaxCrossEntropyLoss()
        l = fn(Y_hat, Y)
        return l.mean() if averaged else l
    if tab.selected('pytorch'):
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return fn(Y, Y_hat)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    # Később lesz felhasználva (pl. batch norm esetén)
    Y_hat = state.apply_fn({'params': params}, *X,
                           mutable=False, rngs=None)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    # A visszaadott üres szótár egy helyfoglaló a segédadatok számára,
    # amelyet később használunk fel (pl. batch norm esetén)
    return (fn(Y_hat, Y).mean(), {}) if averaged else (fn(Y_hat, Y), {})
```

## Tanítás

Ezután tanítjuk a modellünket. A Fashion-MNIST képeket 784 dimenziós jellemzővektorokká kiterítve használjuk.

```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

Ahogy korábban, ez az algoritmus egy ésszerűen pontos megoldáshoz konvergál,
bár ezúttal kevesebb kódsorra volt szükség.


## Összefoglalás

A magas szintű API-k nagyon kényelmesek, mivel elrejtik a felhasználók elől a potenciálisan veszélyes aspektusokat, mint például a numerikus stabilitás. Emellett lehetővé teszik a modellek tömör, néhány sornyi kóddal való megtervezését. Ez egyszerre áldás és átok. A nyilvánvaló előny, hogy a dolgok nagyon hozzáférhetővé válnak, még olyan mérnökök számára is, akik soha egyetlen statisztika órát sem végeztek (valójában ők a könyv célközönségének egy részét alkotják). De az éles sarkok elrejtése is jár árakkal: csökkenti az ösztönzést arra, hogy saját kezűleg adjunk hozzá új és különböző komponenseket, mivel kevés az izommemória a megvalósításhoz. Ráadásul megnehezíti a hibák *javítását*, valahányszor egy keretrendszer védőpárnája nem fedi le az összes szélső esetet. Ez szintén az ismeretlenség következménye.

Ezért erősen ösztönzünk arra, hogy a következő implementációk *mindkét* változatát — a csupasz és az elegáns verziót is — átnézd. Bár a könnyű érthetőségre helyezzük a hangsúlyt, az implementációk általában mégis meglehetősen jó teljesítményűek (a konvolúciók itt a nagy kivétel). Az a szándékunk, hogy ezekre építkezve valami újat alkoss, amit egyetlen keretrendszer sem tud megadni neked.


## Feladatok

1. A mély tanulás számos különböző számformátumot alkalmaz, köztük FP64 dupla pontosságú (rendkívül ritkán használt), FP32 egyszeres pontosságú, BFLOAT16 (tömörített ábrázolásokhoz jó), FP16 (nagyon instabil), TF32 (az NVIDIA új formátuma) és INT8 formátumokat. Számítsd ki az exponenciális függvény legkisebb és legnagyobb argumentumát, amelyre az eredmény nem okoz numerikus alulcsordulást vagy túlcsordulást!
1. Az INT8 egy nagyon korlátozott formátum, amely $1$-től $255$-ig terjedő nem nulla számokat tartalmaz. Hogyan bővíthetnéd dinamikus tartományát anélkül, hogy több bitet használnál? A szokásos szorzás és összeadás továbbra is működik?
1. Növeld az epokszámot a tanításban! Miért csökkenhet a validációs pontosság egy idő után? Hogyan lehetne ezt megoldani?
1. Mi történik, ha növeled a tanulási rátát? Hasonlítsd össze a veszteséggörbéket különböző tanulási ráták esetén! Melyik működik jobban? Mikor?

:begin_tab:`mxnet`
[Megbeszélések](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[Megbeszélések](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[Megbeszélések](https://discuss.d2l.ai/t/260)
:end_tab:

:begin_tab:`jax`
[Megbeszélések](https://discuss.d2l.ai/t/17983)
:end_tab:
