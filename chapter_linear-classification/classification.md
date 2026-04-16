```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Az alap osztályozási modell
:label:`sec_classification`

Talán észrevetted, hogy a nulláról való implementáció és a keretrendszer funkcionalitását felhasználó tömör implementáció meglehetősen hasonló volt a regresszió esetén. Ugyanez igaz az osztályozásra is. Mivel a könyv számos modellje foglalkozik osztályozással, érdemes olyan funkcionalitásokat hozzáadni, amelyek kifejezetten ezt a beállítást támogatják. Ez a szakasz egy alaposztályt biztosít az osztályozási modellekhez, hogy egyszerűsítse a jövőbeli kódot.

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
from functools import partial
from jax import numpy as jnp
import jax
import optax
```

## A `Classifier` osztály

:begin_tab:`pytorch, mxnet, tensorflow`
Az alábbiakban definiáljuk a `Classifier` osztályt. A `validation_step`-ben jelentjük mind a veszteség értékét, mind az osztályozás pontosságát egy validációs batch-en. Minden `num_val_batches` batch-nél rajzolunk egy frissítést. Ennek az az előnye, hogy az átlagos veszteséget és pontosságot a teljes validációs adaton generálja. Ezek az átlagok nem pontosan helyesek, ha az utolsó batch kevesebb példányt tartalmaz, de ezt a kisebb különbséget figyelmen kívül hagyjuk az egyszerűbb kód érdekében.
:end_tab:


:begin_tab:`jax`
Az alábbiakban definiáljuk a `Classifier` osztályt. A `validation_step`-ben jelentjük mind a veszteség értékét, mind az osztályozás pontosságát egy validációs batch-en. Minden `num_val_batches` batch-nél rajzolunk egy frissítést. Ennek az az előnye, hogy az átlagos veszteséget és pontosságot a teljes validációs adaton generálja. Ezek az átlagok nem pontosan helyesek, ha az utolsó batch kevesebb példányt tartalmaz, de ezt a kisebb különbséget figyelmen kívül hagyjuk az egyszerűbb kód érdekében.

Újra is definiáljuk a `training_step` metódust JAX esetén, mivel az összes modell, amely majd a `Classifier` alosztálya lesz, egy olyan veszteséggel rendelkezik, amely segédadatokat is visszaad.
Ezek a segédadatok felhasználhatók batch normalizációval rendelkező modellekhez
(amelyeket a :numref:`sec_batch_norm` szakasz magyaráz el), míg minden más esetben
a veszteséget egy helyőrzőt (üres szótárat) is visszaadó módon tesszük,
a segédadatokat képviselve.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class Classifier(d2l.Module):  #@save
    """Az osztályozási modellek alaposztálya."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
```

```{.python .input}
%%tab jax
class Classifier(d2l.Module):  #@save
    """Az osztályozási modellek alaposztálya."""
    def training_step(self, params, batch, state):
        # Az érték egy tuple, mivel a BatchNorm rétegeket tartalmazó modelleknél
        # a veszteségfüggvénynek segédadatokat is kell visszaadnia
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot("loss", l, train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        # A visszaadott második értéket eldobjuk. BatchNorm rétegeket tartalmazó
        # modellek tanításához használatos, mivel a loss segédadatokat is visszaad
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('loss', l, train=False)
        self.plot('acc', self.accuracy(params, batch[:-1], batch[-1], state),
                  train=False)
```

Alapértelmezésként sztochasztikus gradienscsökkenés optimalizálót használunk, amelyet minibatch-eken alkalmazunk, ahogy a lineáris regresszió kontextusában is tettük.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    params = self.parameters()
    if isinstance(params, list):
        return d2l.SGD(params, self.lr)
    return gluon.Trainer(params, 'sgd', {'learning_rate': self.lr})
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(self.lr)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return optax.sgd(self.lr)
```

## Pontosság

Adott a `y_hat` becsült valószínűségi eloszlás,
általában a legmagasabb becsült valószínűségű osztályt választjuk,
ha kemény jóslatot kell adnunk.
Valóban számos alkalmazás megköveteli a döntést.
Például a Gmailnek egy e-mailt „Elsődleges", „Közösségi", „Frissítések", „Fórumok" vagy „Spam" kategóriába kell sorolnia.
Belsőleg valószínűségeket becsülhet,
de a nap végén az osztályok egyikét kell kiválasztania.

Ha a jóslatok egyeznek az `y` felirat osztállyal, helyesek.
Az osztályozási pontosság az összes helyesen osztályozott jóslat aránya.
Bár nehézkes közvetlenül optimalizálni a pontosságot (nem differenciálható),
ez sokszor a legfontosabb teljesítménymérce. Referenciahalmazokon sokszor *ez a* releváns mennyiség. Ezért szinte mindig közöljük, amikor osztályozókat tanítunk.

A pontosság a következőképpen számítható ki.
Először, ha `y_hat` egy mátrix,
feltételezzük, hogy a második dimenzió az egyes osztályok jóslati pontszámait tárolja.
Az `argmax` segítségével megkapjuk a becsült osztályt az egyes sorok legnagyobb bejegyzésének indexeként.
Majd [**összehasonlítjuk a becsült osztályt az `y` valódi értékkel elemenként.**]
Mivel az `==` egyenlőség operátor érzékeny az adattípusokra,
az `y_hat` adattípusát az `y`-hoz igazítjuk.
Az eredmény egy 0 (hamis) és 1 (igaz) elemeket tartalmazó tenzor.
Az összeg a helyes jóslatok számát adja.

```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(Classifier)  #@save
def accuracy(self, Y_hat, Y, averaged=True):
    """A helyes jóslatok számának kiszámítása."""
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=9}
%%tab jax
@d2l.add_to_class(Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def accuracy(self, params, X, Y, state, averaged=True):
    """A helyes jóslatok számának kiszámítása."""
    Y_hat = state.apply_fn({'params': params,
                            'batch_stats': state.batch_stats},  # BatchNorm Only
                           *X)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=10}
%%tab mxnet

@d2l.add_to_class(d2l.Module)  #@save
def get_scratch_params(self):
    params = []
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            params.append(a)
        if isinstance(a, d2l.Module):
            params.extend(a.get_scratch_params())
    return params

@d2l.add_to_class(d2l.Module)  #@save
def parameters(self):
    params = self.collect_params()
    return params if isinstance(params, gluon.parameter.ParameterDict) and len(
        params.keys()) else self.get_scratch_params()
```

## Összefoglalás

Az osztályozás elég általános probléma ahhoz, hogy saját segédfüggvényeket érdemel. Az osztályozásban központi fontosságú az osztályozó *pontossága*. Megjegyezzük, hogy bár elsősorban a pontosság érdekel minket, az osztályozókat különféle egyéb célok optimalizálására tanítjuk statisztikai és számítástechnikai okokból. Azonban függetlenül attól, hogy a tanítás során melyik veszteségfüggvényt minimalizáltuk, hasznos egy segédmetódus az osztályozó pontosságának empirikus értékelésére.


## Feladatok

1. Jelöld $L_\textrm{v}$-vel a validációs veszteséget, és legyen $L_\textrm{v}^\textrm{q}$ a gyors és durva becslése, amelyet ebben a szakaszban a veszteségfüggvény átlagolásával számítottunk. Végül jelöld $l_\textrm{v}^\textrm{b}$-vel az utolsó minibatch-en számolt veszteséget. Fejezd ki $L_\textrm{v}$-t $L_\textrm{v}^\textrm{q}$, $l_\textrm{v}^\textrm{b}$ és a minta- és minibatch-méretek segítségével!
1. Mutasd meg, hogy a gyors és durva $L_\textrm{v}^\textrm{q}$ becslés torzítatlan! Vagyis mutasd meg, hogy $E[L_\textrm{v}] = E[L_\textrm{v}^\textrm{q}]$! Miért akarnád mégis inkább $L_\textrm{v}$-t használni?
1. Adott egy többosztályos osztályozási veszteség, ahol $l(y,y')$ jelöli az $y'$ becslésekor kapott büntetést, ha $y$-t látunk, és adott $p(y \mid x)$ valószínűség esetén, fogalmazd meg az optimális $y'$ kiválasztásának szabályát! Tipp: a várható veszteséget $l$ és $p(y \mid x)$ segítségével fejezd ki!

:begin_tab:`mxnet`
[Megbeszélések](https://discuss.d2l.ai/t/6808)
:end_tab:

:begin_tab:`pytorch`
[Megbeszélések](https://discuss.d2l.ai/t/6809)
:end_tab:

:begin_tab:`tensorflow`
[Megbeszélések](https://discuss.d2l.ai/t/6810)
:end_tab:

:begin_tab:`jax`
[Megbeszélések](https://discuss.d2l.ai/t/17981)
:end_tab:
