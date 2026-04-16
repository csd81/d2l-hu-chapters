```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Paraméter-inicializálás

Most, hogy tudjuk, hogyan kell elérni a paramétereket,
nézzük meg, hogyan kell azokat megfelelően inicializálni.
A megfelelő inicializálás szükségességét :numref:`sec_numerical_stability` részben tárgyaltuk.
A deep learning keretrendszer alapértelmezett véletlenszerű inicializálást biztosít a rétegeihez.
Azonban sokszor szeretnénk a súlyainkat
különféle más protokollok szerint inicializálni. A keretrendszer a leggyakrabban
használt protokollokat biztosítja, és lehetővé teszi egyéni inicializáló létrehozását is.

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

:begin_tab:`mxnet`
Alapértelmezés szerint az MXNet az egyenletes $U(-0.07, 0.07)$ eloszlásból véletlenszerűen húzva inicializálja a súlyparamétereket,
az eltolás paramétereket nullára állítva.
Az MXNet `init` modulja számos
előre beállított inicializálási módszert biztosít.
:end_tab:

:begin_tab:`pytorch`
Alapértelmezés szerint a PyTorch egyenletesen inicializálja a súly- és eltolás mátrixokat
olyan tartományból húzva, amelyet a bemeneti és kimeneti dimenzió szerint számít.
A PyTorch `nn.init` modulja számos
előre beállított inicializálási módszert biztosít.
:end_tab:

:begin_tab:`tensorflow`
Alapértelmezés szerint a Keras egyenletesen inicializálja a súlymátrixokat olyan tartományból húzva, amelyet a bemeneti és kimeneti dimenzió szerint számít, az eltolás paraméterek mind nullára vannak állítva.
A TensorFlow számos inicializálási módszert biztosít mind a gyökérmodulban, mind a `keras.initializers` modulban.
:end_tab:

:begin_tab:`jax`
Alapértelmezés szerint a Flax a `jax.nn.initializers.lecun_normal` segítségével inicializálja a súlyokat,
azaz 0 középpontú csonkított normális eloszlásból húz mintákat,
ahol a szórás $1 / \textrm{fan}_{\textrm{in}}$ négyzetgyöke,
ahol a `fan_in` a súlytenzor bemeneti egységeinek száma. Az eltolás
paraméterek mind nullára vannak állítva.
A Jax `nn.initializers` modulja számos
előre beállított inicializálási módszert biztosít.
:end_tab:

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
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
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

## [**Beépített inicializálás**]

Kezdjük a beépített inicializálók meghívásával.
Az alábbi kód az összes súlyparamétert
Gauss-eloszlású véletlen változóként inicializálja
0.01 szórással, míg az eltolás paramétereket nullára állítja.

```{.python .input}
%%tab mxnet
# A force_reinit biztosítja, hogy a paraméterek frissen inicializálódjanak
# még akkor is, ha korábban már inicializálva voltak
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input}
%%tab jax
weight_init = nn.initializers.normal(0.01)
bias_init = nn.initializers.zeros

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
layer_0 = params['params']['layers_0']
layer_0['kernel'][:, 0], layer_0['bias'][0]
```

Az összes paramétert inicializálhatjuk egy adott konstans értékre is
(például 1).

```{.python .input}
%%tab mxnet
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)

net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input}
%%tab jax
weight_init = nn.initializers.constant(1)

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
layer_0 = params['params']['layers_0']
layer_0['kernel'][:, 0], layer_0['bias'][0]
```

[**Különböző inicializálókat is alkalmazhatunk bizonyos blokkokhoz.**]
Például az alábbiakban az első réteget
az Xavier inicializálóval inicializáljuk,
a második réteget pedig
42-es konstans értékre inicializáljuk.

```{.python .input}
%%tab mxnet
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input}
%%tab pytorch
def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(42)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(8, kernel_init=nn.initializers.xavier_uniform(),
                              bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=nn.initializers.constant(42),
                              bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
params['params']['layers_0']['kernel'][:, 0], params['params']['layers_2']['kernel']
```

### [**Egyéni inicializálás**]

Néha szükségünk van olyan inicializálási módszerekre,
amelyeket a deep learning keretrendszer nem biztosít.
Az alábbi példában egy inicializálót definiálunk
bármely $w$ súlyparaméterhez a következő furcsa eloszlás alkalmazásával:

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \textrm{ valószínűséggel } \frac{1}{4} \\
            0    & \textrm{ valószínűséggel } \frac{1}{2} \\
        U(-10, -5) & \textrm{ valószínűséggel } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`mxnet`
Itt az `Initializer` osztály alosztályát definiáljuk.
Általában csak az `_init_weight` függvényt kell implementálnunk,
amely tenzor argumentumot (`data`) vesz fel
és hozzárendeli a kívánt inicializált értékeket.
:end_tab:

:begin_tab:`pytorch`
Ismét implementálunk egy `my_init` függvényt a `net`-re való alkalmazáshoz.
:end_tab:

:begin_tab:`tensorflow`
Itt az `Initializer` alosztályát definiáljuk és implementáljuk a `__call__`
függvényt, amely adott alakzat és adattípus esetén visszaadja a kívánt tenzort.
:end_tab:

:begin_tab:`jax`
A Jax inicializálási függvények argumentumként fogadják a `PRNGKey`-t, az `shape`-et és a
`dtype`-ot. Itt implementáljuk a `my_init` függvényt, amely adott alakzat és adattípus esetén visszaadja a kívánt tenzort.
:end_tab:

```{.python .input}
%%tab mxnet
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

```{.python .input}
%%tab pytorch
def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input}
%%tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data=tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor=(tf.abs(data) >= 5)
        factor=tf.cast(factor, tf.float32)
        return data * factor

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.layers[1].weights[0])
```

```{.python .input}
%%tab jax
def my_init(key, shape, dtype=jnp.float_):
    data = jax.random.uniform(key, shape, minval=-10, maxval=10)
    return data * (jnp.abs(data) >= 5)

net = nn.Sequential([nn.Dense(8, kernel_init=my_init), nn.relu, nn.Dense(1)])
params = net.init(d2l.get_key(), X)
print(params['params']['layers_0']['kernel'][:, :2])
```

:begin_tab:`mxnet, pytorch, tensorflow`
Vegyük észre, hogy mindig lehetőségünk van
közvetlenül beállítani a paramétereket.
:end_tab:

:begin_tab:`jax`
A paraméterek inicializálásakor JAX-ban és Flax-ban a visszaadott
paraméterszótárnak `flax.core.frozen_dict.FrozenDict` típusa van. A Jax ökoszisztémában nem tanácsos közvetlenül megváltoztatni egy tömb értékeit, ezért az adattípusok általában megváltoztathatatlanok. Változtatáshoz a `params.unfreeze()` metódust lehet használni.
:end_tab:

```{.python .input}
%%tab mxnet
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

```{.python .input}
%%tab tensorflow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

## Összefoglalás

A paramétereket beépített és egyéni inicializálókkal inicializálhatjuk.

## Feladatok

Keresd meg az online dokumentációt a további beépített inicializálókhoz.

:begin_tab:`mxnet`
[Megbeszélések](https://discuss.d2l.ai/t/8089)
:end_tab:

:begin_tab:`pytorch`
[Megbeszélések](https://discuss.d2l.ai/t/8090)
:end_tab:

:begin_tab:`tensorflow`
[Megbeszélések](https://discuss.d2l.ai/t/8091)
:end_tab:

:begin_tab:`jax`
[Megbeszélések](https://discuss.d2l.ai/t/17991)
:end_tab:
