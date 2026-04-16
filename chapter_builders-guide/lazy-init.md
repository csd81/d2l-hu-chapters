```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Lusta inicializálás
:label:`sec_lazy_init`

Eddig úgy tűnhet, hogy megúsztuk
a hálózataink beállításában tanúsított hanyagságot.
Pontosabban, a következő nem intuitív dolgokat tettük,
amelyek nem is tűnhetnének működőképesnek:

* Definiáltuk a hálózati architektúrákat
  a bemeneti dimenzionalitás megadása nélkül.
* Rétegeket adtunk hozzá az előző réteg
  kimeneti dimenziójának megadása nélkül.
* Sőt, "inicializáltuk" ezeket a paramétereket
  mielőtt elegendő információt adtunk volna meg annak meghatározásához,
  hogy hány paramétert kell tartalmazniuk a modelljeinknek.

Talán meglepő, hogy a kódunk egyáltalán fut.
Hiszen a mélytanulás keretrendszernek nem volt módja megtudni,
hogy mi lenne egy hálózat bemeneti dimenzionalitása.
A trükk itt az, hogy a keretrendszer *halasztja az inicializálást*,
megvárja, amíg először átadunk adatot a modellen,
hogy menet közben meghatározza az egyes rétegek méretét.


Később, konvolúciós neurális hálózatokkal dolgozva,
ez a technika még kényelmesebbé válik,
mivel a bemeneti dimenzionalitás
(pl. egy kép felbontása)
befolyásolja az egyes következő rétegek
dimenzionalitását.
Ezért a paraméterek beállításának lehetősége
anélkül, hogy a kód írásakor tudnánk
a dimenzió értékét,
nagymértékben egyszerűsítheti a feladatot
a modelljeink meghatározásában és az ezt követő módosításában.
Következőként mélyebben beleásunk az inicializálás mechanizmusába.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
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
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

Kezdjük egy MLP példányosításával.

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(10)])
```

Ezen a ponton a hálózatnak nem lehet tudnia
a bemeneti réteg súlyainak dimenzióit,
mivel a bemeneti dimenzió ismeretlen marad.

:begin_tab:`mxnet, pytorch, tensorflow`
Következésképpen a keretrendszer még nem inicializált egyetlen paramétert sem.
Ezt azzal erősítjük meg, hogy megpróbálunk hozzáférni az alábbi paraméterekhez.
:end_tab:

:begin_tab:`jax`
Amint azt :numref:`subsec_param-access` részben megemlítettük, a paraméterek és a hálózat definíciója szétválasztott
a Jax-ban és a Flax-ban, és a felhasználó kezeli mindkettőt manuálisan. A Flax modellek állapot nélküliek,
ezért nincs `parameters` attribútumuk.
:end_tab:

```{.python .input}
%%tab mxnet
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
%%tab pytorch
net[0].weight
```

```{.python .input}
%%tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
Vegyük észre, hogy bár a paraméterobjektumok léteznek,
minden réteg bemeneti dimenziója -1-ként van felsorolva.
Az MXNet a speciális -1 értéket használja jelezve,
hogy a paraméter dimenziója ismeretlen marad.
Ezen a ponton a `net[0].weight.data()` elérésére tett kísérletek
futásidejű hibát váltanának ki, jelezve, hogy a hálózatot
inicializálni kell, mielőtt a paraméterekhez hozzá lehetne férni.
Nézzük meg, mi történik, ha megpróbáljuk inicializálni
a paramétereket az `initialize` metóduson keresztül.
:end_tab:

:begin_tab:`tensorflow`
Vegyük észre, hogy minden réteg objektum létezik, de a súlyok üresek.
A `net.get_weights()` használata hibát dobna, mivel a súlyok
még nem lettek inicializálva.
:end_tab:

```{.python .input}
%%tab mxnet
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
Ahogy láthatjuk, semmi sem változott.
Amikor a bemeneti dimenziók ismeretlenek,
az inicializáló hívások nem valóban inicializálják a paramétereket.
Ehelyett ez a hívás regisztrálja az MXNet-nél, hogy szeretnénk
(és opcionálisan, milyen eloszlás szerint)
inicializálni a paramétereket.
:end_tab:

Következőként adjunk át adatokat a hálózaton keresztül,
hogy a keretrendszer végre inicializálja a paramétereket.

```{.python .input}
%%tab mxnet
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
%%tab pytorch
X = torch.rand(2, 20)
net(X)

net[0].weight.shape
```

```{.python .input}
%%tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

```{.python .input}
%%tab jax
params = net.init(d2l.get_key(), jnp.zeros((2, 20)))
jax.tree_util.tree_map(lambda x: x.shape, params).tree_flatten_with_keys()
```

Amint megismerjük a bemeneti dimenzionalitást,
20,
a keretrendszer meg tudja határozni az első réteg súlymátrixának alakzatát a 20 érték behelyettesítésével.
Miután felismerte az első réteg alakzatát, a keretrendszer folytatja
a második réteggel,
és így tovább a számítási gráfon keresztül,
amíg az összes alakzat ismertté válik.
Vegyük észre, hogy ebben az esetben
csak az első réteg igényel lusta inicializálást,
de a keretrendszer szekvenciálisan inicializál.
Miután az összes paraméter alakzata ismert,
a keretrendszer végre tudja inicializálni a paramétereket.

:begin_tab:`pytorch`
A következő metódus
dummy bemeneteket
ad át a hálózaton
próbafuttatáshoz,
hogy meghatározza az összes paraméter alakzatát,
majd inicializálja a paramétereket.
Később fogjuk használni, amikor az alapértelmezett véletlenszerű inicializálások nem kívánatosak.
:end_tab:

:begin_tab:`jax`
A Flax-ban a paraméter inicializálás mindig manuálisan történik, és a felhasználó kezeli.
A következő metódus dummy bemenetet és egy kulcsszótárat vesz argumentumként.
Ez a kulcsszótár tartalmazza a modell paramétereinek inicializálásához szükséges rngs-eket
és a dropout rng-t a dropout maszkjának generálásához azokhoz a modellekhez, amelyek
dropout rétegeket tartalmaznak. A dropout-ról részletesebben :numref:`sec_dropout` részben lesz szó.
Végül a metódus inicializálja a modellt és visszaadja a paramétereket.
Az előző részekben is ezt alkalmaztuk a háttérben.
:end_tab:

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, dummy_input, key):
    params = self.init(key, *dummy_input)  # dummy_input tuple kicsomagolva
    return params
```

## Összefoglalás

A lusta inicializálás kényelmes lehet, lehetővé téve a keretrendszer számára, hogy automatikusan következtessen a paraméterek alakzataira, megkönnyítve az architektúrák módosítását és kiküszöbölve a hibák egyik leggyakoribb forrását.
Átadhatunk adatokat a modellen keresztül, hogy a keretrendszer végre inicializálja a paramétereket.


## Feladatok

1. Mi történik, ha megadod a bemeneti dimenziókat az első rétegnek, de a következő rétegeknek nem? Azonnal inicializálódnak?
1. Mi történik, ha nem egyező dimenziókat adsz meg?
1. Mit kellene tenned, ha változó dimenzionalitású bemeneted van? Tipp: nézd meg a paraméter kötést.

:begin_tab:`mxnet`
[Megbeszélések](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`pytorch`
[Megbeszélések](https://discuss.d2l.ai/t/8092)
:end_tab:

:begin_tab:`tensorflow`
[Megbeszélések](https://discuss.d2l.ai/t/281)
:end_tab:

:begin_tab:`jax`
[Megbeszélések](https://discuss.d2l.ai/t/17992)
:end_tab:
