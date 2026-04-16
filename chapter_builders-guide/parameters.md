```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Paraméterkezelés

Miután kiválasztottuk az architektúrát
és beállítottuk a hiperparamétereket,
a tanítási ciklushoz lépünk,
ahol a célunk olyan paraméterértékek megtalálása,
amelyek minimalizálják a veszteségfüggvényt.
A tanítás után szükségünk lesz ezekre a paraméterekre
a jövőbeli előrejelzések készítéséhez.
Emellett néha ki szeretnénk vonni a paramétereket,
például hogy más kontextusban újra felhasználjuk őket,
hogy a modellt lemezre mentsük, hogy
más szoftverben is futtatható legyen,
vagy vizsgálat céljából, remélve
tudományos megértés megszerzését.

Az esetek többségében figyelmen kívül hagyhatjuk
a paraméterek deklarálásának és kezelésének
aprólékos részleteit, a deep learning keretrendszerekre bízva
a nehéz munkát.
Azonban, amikor elhagyjuk
a szabványos rétegekkel rendelkező halmozott architektúrákat,
néha szükség lesz elmélyülni
a paraméterek deklarálásában és kezelésében.
Ebben a részben a következőket tárgyaljuk:

* Paraméterek elérése hibakereséshez, diagnosztikához és vizualizációkhoz.
* Paraméterek megosztása különböző modellkomponensek között.

```{.python .input}
%%tab mxnet
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

(**Kezdjük egy egy rejtett réteggel rendelkező MLP vizsgálatával.**)

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Az alapértelmezett inicializálási módszer használata

X = np.random.uniform(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(8),
                    nn.ReLU(),
                    nn.LazyLinear(1))

X = torch.rand(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X).shape
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(8), nn.relu, nn.Dense(1)])

X = jax.random.uniform(d2l.get_key(), (2, 4))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

## [**Paraméterek elérése**]
:label:`subsec_param-access`

Kezdjük azzal, hogyan lehet elérni a paramétereket
az általunk már ismert modellekből.

:begin_tab:`mxnet, pytorch, tensorflow`
Amikor egy modellt a `Sequential` osztályon keresztül definiálunk,
először elérhetjük bármely réteget az indexeléssel,
mintha a modell egy lista lenne.
Minden réteg paraméterei kényelmesen
az attribútumában találhatók.
:end_tab:

:begin_tab:`jax`
A Flax és a JAX szétválasztja a modellt és a paramétereket, ahogyan
azt a korábban definiált modellekben megfigyelhettük.
Amikor egy modellt a `Sequential` osztályon keresztül definiálunk,
először inicializálni kell a hálózatot a
paraméterszótár generálásához. Bármely réteg paramétereit
elérhetjük ezen szótár kulcsain keresztül.
:end_tab:

A második teljesen összefüggő réteg paramétereit a következőképpen vizsgálhatjuk meg.

```{.python .input}
%%tab mxnet
net[1].params
```

```{.python .input}
%%tab pytorch
net[2].state_dict()
```

```{.python .input}
%%tab tensorflow
net.layers[2].weights
```

```{.python .input}
%%tab jax
params['params']['layers_2']
```

Láthatjuk, hogy ez a teljesen összefüggő réteg
két paramétert tartalmaz,
amelyek az adott réteg
súlyainak és eltolásainak felelnek meg.


### [**Célzott paraméterek**]

Vegyük észre, hogy minden paramétert
a paraméter osztály egy példánya képvisel.
Ahhoz, hogy bármit hasznos dolgot tegyünk a paraméterekkel,
először el kell érnünk a mögöttes numerikus értékeket.
Erre több módszer is létezik.
Néhány egyszerűbb, míg mások általánosabbak.
A következő kód kiveszi az eltolást
a neurális hálózat második rétegéből, amely egy paraméterosztály-példányt ad vissza, és
tovább éri el az adott paraméter értékét.

```{.python .input}
%%tab mxnet
type(net[1].bias), net[1].bias.data()
```

```{.python .input}
%%tab pytorch
type(net[2].bias), net[2].bias.data
```

```{.python .input}
%%tab tensorflow
type(net.layers[2].weights[1]), tf.convert_to_tensor(net.layers[2].weights[1])
```

```{.python .input}
%%tab jax
bias = params['params']['layers_2']['bias']
type(bias), bias
```

:begin_tab:`mxnet,pytorch`
A paraméterek összetett objektumok,
amelyek értékeket, gradienseket
és további információkat tartalmaznak.
Ezért kell az értéket kifejezetten kérnünk.

Az értéken kívül minden paraméter lehetővé teszi a gradient elérését is. Mivel ennél a hálózatnál még nem indítottuk el a visszaterjedést, az kezdeti állapotban van.
:end_tab:

:begin_tab:`jax`
A többi keretrendszertől eltérően a JAX nem követi nyomon a gradienseket a
neurális hálózat paraméterei felett, ehelyett a paraméterek és a hálózat szétválasztottak.
Lehetővé teszi a felhasználó számára, hogy számításukat egy
Python függvényként fejezzék ki, és ugyanarra a célra a `grad` transzformációt alkalmazzák.
:end_tab:

```{.python .input}
%%tab mxnet
net[1].weight.grad()
```

```{.python .input}
%%tab pytorch
net[2].weight.grad == None
```

### [**Összes paraméter egyszerre**]

Amikor az összes paraméteren műveletet kell végezni,
ezek egyenkénti elérése fárasztóvá válhat.
A helyzet különösen nehézkessé válhat,
ha összetettebb, pl. beágyazott modulokkal dolgozunk,
mivel rekurzívan kellene végigmenni
az egész fán, hogy kinyerjük
minden almodul paramétereit. Az alábbiakban bemutatjuk az összes réteg paramétereihez való hozzáférést.

```{.python .input}
%%tab mxnet
net.collect_params()
```

```{.python .input}
%%tab pytorch
[(name, param.shape) for name, param in net.named_parameters()]
```

```{.python .input}
%%tab tensorflow
net.get_weights()
```

```{.python .input}
%%tab jax
jax.tree_util.tree_map(lambda x: x.shape, params)
```

## [**Kötött paraméterek**]

Sokszor szeretnénk megosztani a paramétereket több réteg között.
Nézzük meg, hogyan tehetjük ezt elegánsan.
A következőkben kiosztunk egy teljesen összefüggő réteget,
majd a paramétereit kifejezetten egy másik réteg
paramétereinek beállítására használjuk.
Ehhez szükség van az előreterjedés futtatására
`net(X)`, mielőtt hozzáférnénk a paraméterekhez.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
# A megosztott rétegnek nevet kell adni, hogy hivatkozhassunk a
# paramétereire
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))

net(X)
# Ellenőrizzük, hogy a paraméterek azonosak-e
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Meggyőződünk róla, hogy ténylegesen ugyanaz az objektum, nem csupán
# azonos értékű
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.python .input}
%%tab pytorch
# A megosztott rétegnek nevet kell adni, hogy hivatkozhassunk a
# paramétereire
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.LazyLinear(1))

net(X)
# Ellenőrizzük, hogy a paraméterek azonosak-e
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Meggyőződünk róla, hogy ténylegesen ugyanaz az objektum, nem csupán
# azonos értékű
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```{.python .input}
%%tab tensorflow
# A tf.keras kissé eltérően viselkedik: automatikusan eltávolítja
# a duplikált réteget
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(X)
# Ellenőrizzük, hogy a paraméterek eltérők-e
print(len(net.layers) == 3)
```

```{.python .input}
%%tab jax
# A megosztott rétegnek nevet kell adni, hogy hivatkozhassunk a
# paramétereire
shared = nn.Dense(8)
net = nn.Sequential([nn.Dense(8), nn.relu,
                     shared, nn.relu,
                     shared, nn.relu,
                     nn.Dense(1)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)

# Ellenőrizzük, hogy a paraméterek eltérők-e
print(len(params['params']) == 3)
```

Ez a példa azt mutatja, hogy a második és harmadik réteg paraméterei kötöttek.
Nem csupán egyenlők, hanem pontosan ugyanaz a tenzor képviseli őket.
Így, ha megváltoztatjuk az egyik paramétert,
a másik is változik.

:begin_tab:`mxnet, pytorch, tensorflow`
Felmerülhet a kérdés,
ha a paraméterek kötöttek,
mi történik a gradiensekkel?
Mivel a modell paraméterei gradienseket tartalmaznak,
a második rejtett réteg
és a harmadik rejtett réteg gradienseit
összeadják a visszaterjedés során.
:end_tab:


## Összefoglalás

Több módunk van a modell paramétereinek elérésére és kötésére.


## Feladatok

1. Használd a :numref:`sec_model_construction` részben definiált `NestMLP` modellt, és érd el a különböző rétegek paramétereit.
1. Készíts egy megosztott paraméteres réteget tartalmazó MLP-t, és tanítsd. A tanítási folyamat során figyeld meg a modell paramétereit és az egyes rétegek gradienseit.
1. Miért jó ötlet a paramétermegosztás?

:begin_tab:`mxnet`
[Megbeszélések](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[Megbeszélések](https://discuss.d2l.ai/t/57)
:end_tab:

:begin_tab:`tensorflow`
[Megbeszélések](https://discuss.d2l.ai/t/269)
:end_tab:

:begin_tab:`jax`
[Megbeszélések](https://discuss.d2l.ai/t/17990)
:end_tab:
