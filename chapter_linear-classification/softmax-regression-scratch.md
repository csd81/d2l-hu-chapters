```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Softmax regresszió implementálása nulláról
:label:`sec_softmax_scratch`

Mivel a softmax regresszió annyira alapvető,
úgy gondoljuk, hogy fontos tudnod, hogyan implementálható saját magadtól.
Itt csupán a modell softmax-specifikus részeire szorítkozunk,
és a többi komponenst — beleértve a tanítási ciklust is —
a lineáris regresszió szakaszából vesszük át.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
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
from flax import linen as nn
import jax
from jax import numpy as jnp
from functools import partial
```

## A Softmax

Kezdjük a legfontosabb résszel:
a skalárokból valószínűségekbe való leképezéssel.
Emlékeztetőül idézzük fel a tensor meghatározott dimenziói mentén végzett összegzési műveletet,
amelyet a :numref:`subsec_lin-alg-reduction` és :numref:`subsec_lin-alg-non-reduction` szakaszokban tárgyaltunk.
[**Adott egy `X` mátrix, amelynek összes elemét összeadhatjuk (alapértelmezetten) vagy csak az azonos tengely mentén lévő elemeket.**]
Az `axis` változóval sor- és oszlopösszegeket számíthatunk:

```{.python .input}
%%tab all
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

A softmax kiszámítása három lépésből áll:
(i) minden elem hatványozása;
(ii) soronkénti összeg a normalizálási állandó kiszámításához minden példányhoz;
(iii) minden sor osztása a normalizálási állandóval,
biztosítva, hogy az eredmény összege 1 legyen:

(**
$$\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$$
**)

A nevező (logaritmusa) (log) *partíciós függvénynek* nevezik.
Ezt a [statisztikus fizikában](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics))
vezették be, ahol termodinamikai rendszer összes lehetséges állapotán összegzik.
Az implementáció egyszerű:

```{.python .input}
%%tab all
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # A broadcasting mechanizmus itt kerül alkalmazásra
```

Bármely `X` bemenetre [**minden elemet nemnegatív számmá alakítunk.
Minden sor összege 1,**]
ahogyan azt a valószínűségektől elvárjuk. Figyelmeztetés: a fenti kód *nem* robusztus nagyon nagy vagy nagyon kis argumentumok esetén. Bár elegendő illusztrációnak, komoly célra *ne* használd szó szerint. A mély tanulási keretrendszerek beépített védelemmel rendelkeznek, és a továbbiakban a beépített softmax-ot fogjuk alkalmazni.

```{.python .input}
%%tab mxnet
X = d2l.rand(2, 5)
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
%%tab tensorflow, pytorch
X = d2l.rand((2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
%%tab jax
X = jax.random.uniform(jax.random.PRNGKey(d2l.get_seed()), (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

## A modell

Most már minden szükséges elemünk megvan
[**a softmax regressziós modell implementálásához.**]
A lineáris regressziós példánkhoz hasonlóan
minden példányt rögzített hosszúságú vektorral ábrázolunk.
Mivel az adatok $28 \times 28$ pixeles képekből állnak,
[**minden képet kiterítünk,
784 hosszúságú vektorrá alakítva őket.**]
A következő fejezetekben bemutatjuk
a konvolúciós neurális hálózatokat,
amelyek kielégítőbben kiaknázzák a térbeli struktúrát.


A softmax regresszióban
a hálózat kimeneteinek száma
egyenlő kell legyen az osztályok számával.
(**Mivel adathalmazunknak 10 osztálya van,
hálózatunk kimeneti dimenziója 10.**) Következésképpen a súlyok egy $784 \times 10$-es mátrixot alkotnak,
plusz egy $1 \times 10$-es sorvektor a torzításokhoz.
A lineáris regresszióhoz hasonlóan
a `W` súlyokat Gauss-zajjal inicializáljuk.
A torzítások nullára vannak inicializálva.

```{.python .input}
%%tab mxnet
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = np.random.normal(0, sigma, (num_inputs, num_outputs))
        self.b = np.zeros(num_outputs)
        self.W.attach_grad()
        self.b.attach_grad()

    def collect_params(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab pytorch
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab tensorflow
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = tf.random.normal((num_inputs, num_outputs), 0, sigma)
        self.b = tf.zeros(num_outputs)
        self.W = tf.Variable(self.W)
        self.b = tf.Variable(self.b)
```

```{.python .input}
%%tab jax
class SoftmaxRegressionScratch(d2l.Classifier):
    num_inputs: int
    num_outputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.W = self.param('W', nn.initializers.normal(self.sigma),
                            (self.num_inputs, self.num_outputs))
        self.b = self.param('b', nn.initializers.zeros, self.num_outputs)
```

Az alábbi kód meghatározza, hogyan képez le a hálózat minden bemenetet egy kimenetre.
Megjegyezzük, hogy a batch minden $28 \times 28$ pixeles képét
vektorrá alakítjuk `reshape` segítségével, mielőtt az adatokat a modellen átvezetjük.

```{.python .input}
%%tab all
@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.W.shape[0]))
    return softmax(d2l.matmul(X, self.W) + self.b)
```

## A keresztentrópia-veszteség

Ezután implementálni kell a keresztentrópia-veszteség függvényt
(amelyet a :numref:`subsec_softmax-regression-loss-func` szakasz mutatott be).
Ez talán a leggyakrabban használt veszteségfüggvény az egész mély tanulásban.
Manapság a mély tanulás alkalmazásai közül sokkal több
sorolható osztályozási feladatnak,
mint regressziósnak.

Emlékeztetőül: a keresztentrópia a valódi címkéhez rendelt becsült valószínűség
negatív log-valószínűségét veszi.
A hatékonyság érdekében kerüljük a Python for-ciklusokat és indexelést alkalmazunk helyette.
Különösen a $\mathbf{y}$-ban szereplő egy-forró kódolás
lehetővé teszi, hogy kiválogassuk a $\hat{\mathbf{y}}$-ból a megfelelő tagokat.

Ennek szemléltetésére [**mintaadatokat hozunk létre: `y_hat`-ban 2 példány 3 osztály feletti becsült valószínűséggel, és a hozzájuk tartozó `y` címkékkel.**]
A helyes címkék rendre $0$ és $2$ (azaz az első és a harmadik osztály).
[**A `y`-t a `y_hat`-ban lévő valószínűségek indexeként használva**]
hatékonyan kiválaszthatjuk a szükséges tagokat.

```{.python .input}
%%tab mxnet, pytorch, jax
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
%%tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

:begin_tab:`pytorch, mxnet, tensorflow`
Most (**implementálhatjuk a keresztentrópia-veszteség függvényt**) a kiválasztott valószínűségek logaritmusainak átlagolásával.
:end_tab:

:begin_tab:`jax`
Most (**implementálhatjuk a keresztentrópia-veszteség függvényt**) a kiválasztott valószínűségek logaritmusainak átlagolásával.

Megjegyezzük, hogy a `jax.jit` használatához a JAX implementációk felgyorsítása érdekében,
és annak biztosításához, hogy a `loss` tiszta függvény legyen, a `cross_entropy` függvényt
a `loss` belsejében definiáljuk újra, hogy elkerüljük bármely globális változó vagy függvény használatát,
amely a `loss` függvényt tisztátalanná tehetné.
Az érdeklődőket a `jax.jit`-ről és a tiszta függvényekről szóló [JAX dokumentációhoz](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions) irányítjuk.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch, jax
def cross_entropy(y_hat, y):
    return -d2l.reduce_mean(d2l.log(y_hat[list(range(len(y_hat))), y]))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab tensorflow
def cross_entropy(y_hat, y):
    return -tf.reduce_mean(tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(SoftmaxRegressionScratch)
@partial(jax.jit, static_argnums=(0))
def loss(self, params, X, y, state):
    def cross_entropy(y_hat, y):
        return -d2l.reduce_mean(d2l.log(y_hat[list(range(len(y_hat))), y]))
    y_hat = state.apply_fn({'params': params}, *X)
    # A visszaadott üres szótár a segédadatok helyfoglalója,
    # amelyet később használunk majd (pl. batch normalizációhoz)
    return cross_entropy(y_hat, y), {}
```

## Tanítás

Újra felhasználjuk a :numref:`sec_linear_scratch` szakaszban definiált `fit` metódust
[**a modell tanításához 10 epokon keresztül.**]
Megjegyezzük, hogy az epokszám (`max_epochs`),
a minibatch mérete (`batch_size`),
és a tanulási ráta (`lr`)
mind beállítható hiperparaméterek.
Ez azt jelenti, hogy bár ezek az értékek nem
kerülnek megtanulásra az elsődleges tanítási ciklusban,
mégis befolyásolják modellünk teljesítményét
mind a tanítás, mind az általánosítási teljesítmény szempontjából.
A gyakorlatban ezeket az értékeket
az adatok *validációs* felosztása alapján érdemes megválasztani,
majd végül a *tesztelési* felosztáson kiértékelni a végső modellt.
Ahogy a :numref:`subsec_generalization-model-selection` szakaszban tárgyaltuk,
a Fashion-MNIST tesztadatait validációs halmaznak tekintjük,
és ezen a felosztáson jelentjük a validációs veszteséget és pontosságot.

```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## Jóslás

Most, hogy a tanítás befejeződött,
modellünk készen áll [**néhány kép osztályozására.**]

```{.python .input}
%%tab all
X, y = next(iter(data.val_dataloader()))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds = d2l.argmax(model(X), axis=1)
if tab.selected('jax'):
    preds = d2l.argmax(model.apply({'params': trainer.state.params}, X), axis=1)
preds.shape
```

A leginkább a *tévesen* osztályozott képek érdekelnek minket. Ezeket megjelenítjük
a tényleges és a modell által jósolt címkék összehasonlításával
(az első szövegsor a valós, a második a becsült címke).

```{.python .input}
%%tab all
wrong = d2l.astype(preds, y.dtype) != y
X, y, preds = X[wrong], y[wrong], preds[wrong]
labels = [a+'\n'+b for a, b in zip(
    data.text_labels(y), data.text_labels(preds))]
data.visualize([X, y], labels=labels)
```

## Összefoglalás

Mostanra már van némi tapasztalatunk
lineáris regressziós és osztályozási feladatok megoldásában.
Ezzel elértünk az 1960–1970-es évek statisztikai modellezési csúcsteljesítményéig.
A következő szakaszban megmutatjuk, hogyan lehet mély tanulási keretrendszerekkel
sokkal hatékonyabban implementálni ezt a modellt.

## Feladatok

1. Ebben a szakaszban a softmax függvényt közvetlenül a matematikai definíció alapján implementáltuk. Ahogy a :numref:`sec_softmax` szakaszban tárgyaltuk, ez numerikus instabilitáshoz vezethet.
    1. Teszteld, hogy a `softmax` helyesen működik-e, ha egy bemeneti értéke $100$!
    1. Teszteld, hogy a `softmax` helyesen működik-e, ha az összes bemenet legnagyobb értéke kisebb, mint $-100$!
    1. Valósíts meg egy javítást, amely az argumentum legnagyobb eleméhez viszonyított értéket figyeli!
1. Implementálj egy `cross_entropy` függvényt, amely a $\sum_i y_i \log \hat{y}_i$ keresztentrópia-veszteség definícióját követi!
    1. Próbáld ki a szakasz kódpéldájában!
    1. Miért gondolod, hogy lassabban fut?
    1. Érdemes-e használni? Mikor lenne értelme?
    1. Mire kell ügyelni? Tipp: gondolj a logaritmus értelmezési tartományára!
1. Mindig jó ötlet a legvalószínűbb címkét visszaadni? Például megtennéd-e ezt orvosi diagnózisnál? Hogyan próbálnád megoldani ezt a problémát?
1. Tegyük fel, hogy softmax regresszió segítségével akarjuk megjósolni a következő szót bizonyos jellemzők alapján. Milyen problémák adódhatnak egy nagy szókészletnél?
1. Kísérletezz az e szakaszbeli kód hiperparamétereivel! Különösen:
    1. Ábrázold, hogyan változik a validációs veszteség a tanulási ráta változásával!
    1. Változik-e a validációs és a tanítási veszteség a minibatch méretének változásával? Mekkora vagy milyen kicsi kell legyen, mielőtt hatást tapasztalsz?


:begin_tab:`mxnet`
[Megbeszélések](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[Megbeszélések](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[Megbeszélések](https://discuss.d2l.ai/t/225)
:end_tab:

:begin_tab:`jax`
[Megbeszélések](https://discuss.d2l.ai/t/17982)
:end_tab:
