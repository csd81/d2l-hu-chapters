```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Egyéni rétegek

A deep learning sikerének egyik tényezője
a széles körű rétegkészlet elérhetősége,
amelyek kreatív módokon kombinálhatók
a különféle feladatokhoz megfelelő
architektúrák tervezéséhez.
Például a kutatók kifejezetten képek, szövegek kezelésére,
szekvenciális adatokon való végigiterálásra
és
dinamikus programozáshoz találtak fel rétegeket.
Előbb vagy utóbb szükséged lesz
egy olyan rétegre, amely még nem létezik a deep learning keretrendszerben.
Ilyen esetekben egyéni réteget kell készíteni.
Ebben a részben megmutatjuk, hogyan.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
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
import jax
from jax import numpy as jnp
```

## (**Paraméter nélküli rétegek**)

Kezdjük azzal, hogy konstruálunk egy egyéni réteget,
amelynek nincsenek saját paraméterei.
Ez ismerősnek tűnhet, ha emlékszel a
:numref:`sec_model_construction` részben lévő modulok bevezetésére.
Az alábbi `CenteredLayer` osztály egyszerűen
kivonja az átlagot a bemenetéből.
Az elkészítéséhez egyszerűen örökölnünk kell
az alapréteg osztályból, és implementálnunk kell az előreterjedési függvényt.

```{.python .input}
%%tab mxnet
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab pytorch
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab tensorflow
class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, X):
        return X - tf.reduce_mean(X)
```

```{.python .input}
%%tab jax
class CenteredLayer(nn.Module):
    def __call__(self, X):
        return X - X.mean()
```

Ellenőrizzük, hogy a rétegünk a szándékoltnak megfelelően működik-e, ha adatokat adunk át rajta.

```{.python .input}
%%tab all
layer = CenteredLayer()
layer(d2l.tensor([1.0, 2, 3, 4, 5]))
```

Most már [**beépíthetjük a rétegünket komponensként
összetettebb modellek konstruálásában.**]

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
```

```{.python .input}
%%tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(128), CenteredLayer()])
```

Egy extra ellenőrzésként véletlenszerű adatokat
küldhetünk a hálózaton keresztül, és ellenőrizhetjük, hogy az átlag valóban 0-e.
Mivel lebegőpontos számokkal dolgozunk,
előfordulhat, hogy még mindig látunk egy nagyon kis nullától eltérő számot
a kvantálás miatt.

:begin_tab:`jax`
Itt az `init_with_output` metódust használjuk, amely mind a hálózat kimenetét, mind a paramétereket visszaadja. Ebben az esetben csak a kimenetre összpontosítunk.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet
Y = net(d2l.rand(4, 8))
Y.mean()
```

```{.python .input}
%%tab tensorflow
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

```{.python .input}
%%tab jax
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (4, 8)))
Y.mean()
```

## [**Paraméteres rétegek**]

Most, hogy tudjuk, hogyan definiáljunk egyszerű rétegeket,
lépjünk tovább a tanítással hangolható paraméteres rétegek definiálásához.
Beépített függvényeket használhatunk paraméterek létrehozásához,
amelyek alapvető köztevékenységi funkcionalitást biztosítanak.
Különösen irányítják a hozzáférést, az inicializálást,
a megosztást, a mentést és a modell paramétereinek betöltését.
Így, egyéb előnyök mellett, nem kell egyéni szerializálási rutinokat írnunk
minden egyéni réteghez.

Most implementáljuk a teljesen összefüggő réteg saját változatát.
Emlékezzünk, hogy ez a réteg két paramétert igényel,
egyet a súly és egyet a torzítás megjelenítéséhez.
Ebben az implementációban a ReLU aktivációt beépítjük alapértelmezettként.
Ez a réteg két bemeneti argumentumot igényel: `in_units` és `units`, amelyek
a bemenetek és kimenetek számát jelölik.

```{.python .input}
%%tab mxnet
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(
            ctx=x.ctx)
        return npx.relu(linear)
```

```{.python .input}
%%tab pytorch
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
        
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

```{.python .input}
%%tab tensorflow
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)
```

```{.python .input}
%%tab jax
class MyDense(nn.Module):
    in_units: int
    units: int

    def setup(self):
        self.weight = self.param('weight', nn.initializers.normal(stddev=1),
                                 (self.in_units, self.units))
        self.bias = self.param('bias', nn.initializers.zeros, self.units)

    def __call__(self, X):
        linear = jnp.matmul(X, self.weight) + self.bias
        return nn.relu(linear)
```

:begin_tab:`mxnet, tensorflow, jax`
Ezután példányosítjuk a `MyDense` osztályt
és elérjük a modell paramétereit.
:end_tab:

:begin_tab:`pytorch`
Ezután példányosítjuk a `MyLinear` osztályt
és elérjük a modell paramétereit.
:end_tab:

```{.python .input}
%%tab mxnet
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input}
%%tab pytorch
linear = MyLinear(5, 3)
linear.weight
```

```{.python .input}
%%tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

```{.python .input}
%%tab jax
dense = MyDense(5, 3)
params = dense.init(d2l.get_key(), jnp.zeros((3, 5)))
params
```

[**Közvetlenül végezhetünk előreterjedési számításokat egyéni rétegekkel.**]

```{.python .input}
%%tab mxnet
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input}
%%tab pytorch
linear(torch.rand(2, 5))
```

```{.python .input}
%%tab tensorflow
dense(tf.random.uniform((2, 5)))
```

```{.python .input}
%%tab jax
dense.apply(params, jax.random.uniform(d2l.get_key(),
                                       (2, 5)))
```

(**Modelleket is konstruálhatunk egyéni rétegek segítségével.**) Ha ez megvan, ugyanúgy használhatjuk, mint a beépített teljesen összefüggő réteget.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

```{.python .input}
%%tab jax
net = nn.Sequential([MyDense(64, 8), MyDense(8, 1)])
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (2, 64)))
Y
```

## Összefoglalás

Az alapréteg osztályon keresztül egyéni rétegeket tervezhetünk. Ez lehetővé teszi, hogy rugalmas új rétegeket definiáljunk, amelyek eltérően viselkednek a könyvtárban meglévő rétegektől.
Ha egyszer definiáltuk, az egyéni rétegek tetszőleges kontextusokban és architektúrákban meghívhatók.
A rétegek helyi paraméterekkel rendelkezhetnek, amelyeket beépített függvényeken keresztül lehet létrehozni.


## Feladatok

1. Tervezz olyan réteget, amely bemenetet fogad és tenzor-redukciót számít,
   azaz visszaadja a $y_k = \sum_{i, j} W_{ijk} x_i x_j$ értéket.
1. Tervezz olyan réteget, amely visszaadja az adatok Fourier-együtthatóinak vezető felét.

:begin_tab:`mxnet`
[Megbeszélések](https://discuss.d2l.ai/t/58)
:end_tab:

:begin_tab:`pytorch`
[Megbeszélések](https://discuss.d2l.ai/t/59)
:end_tab:

:begin_tab:`tensorflow`
[Megbeszélések](https://discuss.d2l.ai/t/279)
:end_tab:

:begin_tab:`jax`
[Megbeszélések](https://discuss.d2l.ai/t/17993)
:end_tab:
