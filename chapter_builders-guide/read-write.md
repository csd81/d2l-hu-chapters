```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Fájl I/O

Eddig tárgyaltuk az adatok feldolgozásának módját, valamint azt, hogyan
kell deep learning modelleket építeni, tanítani és tesztelni.
Azonban egy ponton remélhetőleg eléggé meg leszünk elégedve
a tanult modellekkel ahhoz, hogy
az eredményeket el akarjuk menteni a különféle kontextusokban
való későbbi felhasználásra
(talán még telepítési előrejelzések készítéséhez is).
Emellett, ha hosszú tanítási folyamatot futtatunk,
a bevett gyakorlat az, hogy rendszeres időközönként közbülső eredményeket mentünk el (ellenőrzőpontok),
hogy ne veszítsük el a több napos számítást,
ha véletlenül kirántjuk a szerver tápkábelét.
Tehát itt az ideje megtanulni, hogyan kell betölteni és tárolni
mind az egyes súlyvektorokat, mind a teljes modelleket.
Ez a rész mindkét problémával foglalkozik.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
import numpy as np
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import flax
from flax import linen as nn
from flax.training import checkpoints
import jax
from jax import numpy as jnp
```

## (**Tenzorok betöltése és mentése**)

Az egyes tenzorokhoz közvetlenül
meghívhatjuk a `load` és `save` függvényeket,
hogy rendre olvassuk és írjuk őket.
Mindkét függvény megköveteli, hogy adjunk meg egy nevet,
a `save` pedig bemenetként a mentendő változót igényli.

```{.python .input}
%%tab mxnet
x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input}
%%tab pytorch
x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input}
%%tab tensorflow
x = tf.range(4)
np.save('x-file.npy', x)
```

```{.python .input}
%%tab jax
x = jnp.arange(4)
jnp.save('x-file.npy', x)
```

Most visszaolvashatjuk az adatokat a tárolt fájlból a memóriába.

```{.python .input}
%%tab mxnet
x2 = npx.load('x-file')
x2
```

```{.python .input}
%%tab pytorch
x2 = torch.load('x-file')
x2
```

```{.python .input}
%%tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

```{.python .input}
%%tab jax
x2 = jnp.load('x-file.npy', allow_pickle=True)
x2
```

[**Tárolhatjuk tenzorok listáját és visszaolvashatjuk őket a memóriába.**]

```{.python .input}
%%tab mxnet
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

```{.python .input}
%%tab jax
y = jnp.zeros(4)
jnp.save('xy-files.npy', [x, y])
x2, y2 = jnp.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

Sőt, [**írhatunk és olvashatunk egy szótárat, amely
karakterláncokból tenzorokba képez.**]
Ez kényelmes, ha
egy modell összes súlyát szeretnénk olvasni vagy írni.

```{.python .input}
%%tab mxnet
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input}
%%tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.python .input}
%%tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

```{.python .input}
%%tab jax
mydict = {'x': x, 'y': y}
jnp.save('mydict.npy', mydict)
mydict2 = jnp.load('mydict.npy', allow_pickle=True)
mydict2
```

## [**Modell paramétereinek betöltése és mentése**]

Az egyes súlyvektorok (vagy más tenzorok) mentése hasznos,
de nagyon fárasztóvá válik, ha el akarjuk menteni
(és később betölteni) egy egész modellt.
Végül is előfordulhat, hogy
száz paramétercsoport van elszórva az egész modellben.
Ezért a deep learning keretrendszer beépített funkcionalitást biztosít
teljes hálózatok betöltéséhez és mentéséhez.
Egy fontos részlet, amelyre érdemes felfigyelni, az az, hogy ez
a modell *paramétereit* menti, nem az egész modellt.
Ha például van egy 3 rétegű MLP-nk,
az architektúrát külön kell megadni.
Ennek oka, hogy maguk a modellek tetszőleges kódot tartalmazhatnak,
így nem szerializálhatók olyan természetesen.
Ezért egy modell visszaállításához szükség van
az architektúra kódban való generálására,
majd a paraméterek lemezről való betöltésére.
(**Kezdjük az ismerős MLP-nkkel.**)

```{.python .input}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

```{.python .input}
%%tab jax
class MLP(nn.Module):
    def setup(self):
        self.hidden = nn.Dense(256)
        self.output = nn.Dense(10)

    def __call__(self, x):
        return self.output(nn.relu(self.hidden(x)))

net = MLP()
X = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (2, 20))
Y, params = net.init_with_output(jax.random.PRNGKey(d2l.get_seed()), X)
```

Ezután [**fájlként tároljuk a modell paramétereit**] "mlp.params" névvel.

```{.python .input}
%%tab mxnet
net.save_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input}
%%tab tensorflow
net.save_weights('mlp.params')
```

```{.python .input}
%%tab jax
checkpoints.save_checkpoint('ckpt_dir', params, step=1, overwrite=True)
```

A modell visszaállításához példányosítjuk az eredeti MLP modell
klónját.
A modell paramétereinek véletlenszerű inicializálása helyett
[**közvetlenül olvassuk be a fájlban tárolt paramétereket**].

```{.python .input}
%%tab mxnet
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

```{.python .input}
%%tab tensorflow
clone = MLP()
clone.load_weights('mlp.params')
```

```{.python .input}
%%tab jax
clone = MLP()
cloned_params = flax.core.freeze(checkpoints.restore_checkpoint('ckpt_dir',
                                                                target=None))
```

Mivel mindkét példánynak azonos modell paraméterei vannak,
az `X` ugyanazon bemenetének számítási eredménye megegyező kell legyen.
Ellenőrizzük ezt.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
%%tab jax
Y_clone = clone.apply(cloned_params, X)
Y_clone == Y
```

## Összefoglalás

A `save` és `load` függvények tenzor objektumok fájl I/O műveleteinek elvégzésére használhatók.
Egy hálózat paramétereinek teljes készletét elmenthetjük és betölthetjük egy paraméterszótáron keresztül.
Az architektúra mentése kódban kell elvégezni, nem paraméterekben.

## Feladatok

1. Még ha nincs is szükség a tanított modellek másik eszközre való telepítésére, mik a modell paramétereinek tárolásának gyakorlati előnyei?
1. Tegyük fel, hogy egy hálózat részeit szeretnénk újrafelhasználni, és beépíteni egy eltérő architektúrájú hálózatba. Hogyan használnád például egy korábbi hálózat első két rétegét egy új hálózatban?
1. Hogyan mentenéd a hálózat architektúráját és paramétereit? Milyen korlátozásokat szabnál az architektúrára?

:begin_tab:`mxnet`
[Megbeszélések](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[Megbeszélések](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[Megbeszélések](https://discuss.d2l.ai/t/327)
:end_tab:

:begin_tab:`jax`
[Megbeszélések](https://discuss.d2l.ai/t/17994)
:end_tab:
