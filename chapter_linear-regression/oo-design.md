```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Objektumorientált tervezés az implementációhoz
:label:`sec_oo-design`

A lineáris regresszió bevezetőjében
végigjártuk a különböző összetevőket,
beleértve
az adatokat, a modellt, a veszteségfüggvényt
és az optimalizálási algoritmust.
Valóban,
a lineáris regresszió
az egyik legegyszerűbb gépi tanulási modell.
A tanítása
azonban sok olyan összetevőt használ, amelyeket a könyv többi modellje is igényel.
Ezért,
mielőtt belemennénk az implementáció részleteibe,
érdemes megtervezni néhány
API-t, amelyeket végig használunk.
Ha a deep learning összetevőit
objektumként kezeljük,
elkezdhetjük ezeknek az objektumoknak
és kölcsönhatásaiknak az osztályait definiálni.
Ez az objektumorientált tervezési megközelítés
az implementációhoz
nagyban leegyszerűsíti a bemutatást,
és akár saját projektjeidben is érdemes lehet alkalmazni.


Az olyan nyílt forráskódú könyvtárak által inspirálva, mint a [PyTorch Lightning](https://www.pytorchlightning.ai/),
magas szinten három osztályt szeretnénk:
(i) a `Module` tartalmazza a modelleket, a veszteségeket és az optimalizálási módszereket;
(ii) a `DataModule` adatbetöltőket biztosít a tanításhoz és a validációhoz;
(iii) mindkét osztályt a `Trainer` osztály kombinálja, amely lehetővé teszi,
hogy különböző hardverplatformokon tanítsuk a modelleket.
A könyv legtöbb kódja a `Module` és a `DataModule` osztályokat alakítja. A `Trainer` osztályt csak akkor érintjük, amikor a GPU-kat, CPU-kat, párhuzamos tanítást és optimalizálási algoritmusokat tárgyaljuk.

```{.python .input}
%%tab mxnet
import time
import numpy as np
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input}
%%tab pytorch
import time
import numpy as np
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import time
import numpy as np
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from dataclasses import field
from d2l import jax as d2l
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp
import numpy as np
import jax
import time
from typing import Any
```

## Segédeszközök
:label:`oo-design-utilities`

Néhány segédeszközre van szükségünk az objektumorientált programozás leegyszerűsítéséhez Jupyter notebookban. Az egyik kihívás az, hogy az osztálydefiníciók jellemzően meglehetősen hosszú kódblokkokból állnak. A notebook olvashatóságához rövid kódtöredékek szükségesek, magyarázatokkal tarkítva – ez a követelmény összeegyeztethetetlen a Python könyvtárakban szokásos programozási stílussal. Az első
segédfüggvény lehetővé teszi számunkra, hogy függvényeket metódusként regisztráljunk egy osztályba *az osztály létrehozása után*. Valójában ezt még *az osztály példányainak létrehozása után* is megtehetjük! Ez lehetővé teszi számunkra, hogy egy osztály implementációját több kódblokkra osszuk fel.

```{.python .input}
%%tab all
def add_to_class(Class):  #@save
    """Függvények metódusként való regisztrálása egy létrehozott osztályba."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
```

Nézzük meg gyorsan, hogyan használható. Tervezzük implementálni az `A` osztályt egy `do` metódussal. Ahelyett, hogy mind az `A`, mind a `do` kódja ugyanabban a kódblokkban lenne, először deklarálhatjuk az `A` osztályt és létrehozhatunk egy `a` példányt.

```{.python .input}
%%tab all
class A:
    def __init__(self):
        self.b = 1

a = A()
```

Ezután a `do` metódust szokás szerint definiáljuk, de nem az `A` osztály hatókörében. Ehelyett ezt a metódust az `add_to_class` dekorátorral díszítjük, az `A` osztályt argumentumként megadva. Ily módon a metódus képes hozzáférni az `A` tagváltozóihoz, csakúgy, mintha az `A` definíciójának részét képezné. Nézzük meg, mi történik, ha meghívjuk az `a` példányra.

```{.python .input}
%%tab all
@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)

a.do()
```

A második egy segédosztály, amely egy osztály `__init__` metódusában szereplő összes argumentumot osztályattribútumként menti el. Ez lehetővé teszi számunkra, hogy az osztálykonstruktor szignatúráját implicit módon bővítsük, anélkül hogy további kódot kellene írnunk.

```{.python .input}
%%tab all
class HyperParameters:  #@save
    """A hiperparaméterek alap osztálya."""
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
```

Az implementációját a :numref:`sec_utils` részbe halasztjuk. A használatához definiáljuk az osztályunkat, amely a `HyperParameters`-ből örököl, és meghívja a `save_hyperparameters` metódust az `__init__` metódusban.

```{.python .input}
%%tab all
# A d2l-ben mentett, teljesen implementált HyperParameters osztály hívása
class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)
```

Az utolsó segédeszköz lehetővé teszi számunkra, hogy interaktívan ábrázoljuk a kísérlet előrehaladását, miközben az folyamatban van. A sokkal erőteljesebb (és összetettebb) [TensorBoard](https://www.tensorflow.org/tensorboard) iránti tiszteletből `ProgressBoard`-nak nevezzük el. Az implementáció a :numref:`sec_utils` részbe van halasztva. Egyelőre nézzük meg a gyakorlatban.

A `draw` metódus egy `(x, y)` pontot rajzol az ábrán, a jelmagyarázatban megadott `label` felirattal. Az opcionális `every_n` simítja a vonalat, mivel az ábrán csak $1/n$ pontot jelenít meg. Értékeik az eredeti ábrán szereplő $n$ szomszédos pont átlagából adódnak.

```{.python .input}
%%tab all
class ProgressBoard(d2l.HyperParameters):  #@save
    """A tábla, amely animációban ábrázolja az adatpontokat."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented
```

A következő példában a `sin` és `cos` függvényeket különböző simítással ábrázoljuk. Ha lefuttatod ezt a kódblokkot, animációban láthatod, ahogy a vonalak növekednek.

```{.python .input}
%%tab all
board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)
```

## Modellek
:label:`subsec_oo-design-models`

A `Module` osztály az alap osztálya az összes általunk implementált modellnek. Legalább három metódusra van szükségünk. Az első, az `__init__`, tárolja a tanulható paramétereket; a `training_step` metódus egy adatköteget fogad el, és visszaadja a veszteség értékét; végül a `configure_optimizers` visszaadja az optimalizálási módszert, vagy azok listáját, amelyet a tanulható paraméterek frissítésére használnak. Opcionálisan definiálhatjuk a `validation_step` metódust a kiértékelési mértékek bejelentésére.
Néha a kimenet kiszámítását egy külön `forward` metódusba helyezzük, hogy újrafelhasználhatóbbá tegyük.

:begin_tab:`jax`
A Python 3.7-ben bevezetett [dataclasses](https://docs.python.org/3/library/dataclasses.html)
segítségével a `@dataclass` dekorátorral ellátott osztályok automatikusan kapnak
varázslatos metódusokat, mint az `__init__` és a `__repr__`. A tagváltozókat
típusjegyzetekkel definiáljuk. Az összes Flax modul Python 3.7 dataclass.
:end_tab:

```{.python .input}
%%tab pytorch
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """A modellek alap osztálya."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Egy pont ábrázolása animációban."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```

```{.python .input}
%%tab mxnet, tensorflow, jax
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """A modellek alap osztálya."""
    if tab.selected('mxnet', 'tensorflow'):
        def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
            super().__init__()
            self.save_hyperparameters()
            self.board = ProgressBoard()
        if tab.selected('tensorflow'):
            self.training = None

    if tab.selected('jax'):
        # Python dataclass használatakor nincs szükség save_hyperparam-ra
        plot_train_per_epoch: int = field(default=2, init=False)
        plot_valid_per_epoch: int = field(default=1, init=False)
        # default_factory biztosítja, hogy minden futtatáskor új ábrák keletkezzenek
        board: ProgressBoard = field(default_factory=lambda: ProgressBoard(),
                                     init=False)

    def loss(self, y_hat, y):
        raise NotImplementedError

    if tab.selected('mxnet', 'tensorflow'):
        def forward(self, X):
            assert hasattr(self, 'net'), 'Neural network is defined'
            return self.net(X)

    if tab.selected('tensorflow'):
        def call(self, X, *args, **kwargs):
            if kwargs and "training" in kwargs:
                self.training = kwargs['training']
            return self.forward(X, *args)

    if tab.selected('jax'):
        # A JAX és a Flax nem rendelkezik forward metódushoz hasonló szintaxissal.
        # A Flax a setup és a beépített __call__ varázslatos metódusokat használja
        # az előre menethez. Az egységesség kedvéért itt hozzáadjuk
        def forward(self, X, *args, **kwargs):
            assert hasattr(self, 'net'), 'Neural network is defined'
            return self.net(X, *args, **kwargs)

        def __call__(self, X, *args, **kwargs):
            return self.forward(X, *args, **kwargs)

    def plot(self, key, value, train):
        """Egy pont ábrázolása animációban."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        if tab.selected('mxnet', 'tensorflow'):
            self.board.draw(x, d2l.numpy(value), (
                'train_' if train else 'val_') + key, every_n=int(n))
        if tab.selected('jax'):
            self.board.draw(x, d2l.to(value, d2l.cpu()),
                            ('train_' if train else 'val_') + key,
                            every_n=int(n))

    if tab.selected('mxnet', 'tensorflow'):
        def training_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            self.plot('loss', l, train=True)
            return l

        def validation_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            self.plot('loss', l, train=False)

    if tab.selected('jax'):
        def training_step(self, params, batch, state):
            l, grads = jax.value_and_grad(self.loss)(params, batch[:-1],
                                                     batch[-1], state)
            self.plot("loss", l, train=True)
            return l, grads

        def validation_step(self, params, batch, state):
            l = self.loss(params, batch[:-1], batch[-1], state)
            self.plot('loss', l, train=False)
        
        def apply_init(self, dummy_input, key):
            """Később kerül meghatározásra: :numref:`sec_lazy_init`"""
            raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
```

:begin_tab:`mxnet`
Vedd észre, hogy a `Module` az `nn.Block` alosztálya, amely a Gluon neurális hálózatainak alap osztálya.
Kényelmes funkciókat biztosít a neurális hálózatok kezeléséhez. Például, ha definiálunk egy `forward` metódust, mint a `forward(self, X)`, akkor az `a` példányra ezt a metódust az `a(X)` hívással érjük el. Ez azért működik, mert a beépített `__call__` metódus hívja meg a `forward` metódust. Az `nn.Block`-ról további részleteket és példákat találsz a :numref:`sec_model_construction` részben.
:end_tab:

:begin_tab:`pytorch`
Vedd észre, hogy a `Module` az `nn.Module` alosztálya, amely a PyTorch neurális hálózatainak alap osztálya.
Kényelmes funkciókat biztosít a neurális hálózatok kezeléséhez. Például, ha definiálunk egy `forward` metódust, mint a `forward(self, X)`, akkor az `a` példányra ezt a metódust az `a(X)` hívással érjük el. Ez azért működik, mert a beépített `__call__` metódus hívja meg a `forward` metódust. Az `nn.Module`-ról további részleteket és példákat találsz a :numref:`sec_model_construction` részben.
:end_tab:

:begin_tab:`tensorflow`
Vedd észre, hogy a `Module` a `tf.keras.Model` alosztálya, amely a TensorFlow neurális hálózatainak alap osztálya.
Kényelmes funkciókat biztosít a neurális hálózatok kezeléséhez. Például meghívja a `call` metódust a beépített `__call__` metódusból. Itt a `call`-t átirányítjuk a `forward` metódushoz, és az argumentumait osztályattribútumként mentjük. Ezt azért tesszük, hogy a kódunk jobban hasonlítson más keretrendszer implementációkhoz.
:end_tab:

:begin_tab:`jax`
Vedd észre, hogy a `Module` a `linen.Module` alosztálya, amely a Flax neurális hálózatainak alap osztálya.
Kényelmes funkciókat biztosít a neurális hálózatok kezeléséhez. Például kezeli a modell paramétereit, biztosítja az `nn.compact` dekorátort a kód egyszerűsítéséhez, meghívja a `__call__` metódust és még sok mást.
Itt a `__call__`-t is átirányítjuk a `forward` metódushoz. Ezt azért tesszük, hogy a kódunk jobban hasonlítson más keretrendszer implementációkhoz.
:end_tab:

##  Adatok
:label:`oo-design-data`

A `DataModule` osztály az adatok alap osztálya. Meglehetősen gyakran az `__init__` metódust használják az adatok előkészítésére. Ez magában foglalja a letöltést és az előfeldolgozást is, ha szükséges. A `train_dataloader` visszaadja az adatbetöltőt a tanítóhalmazhoz. Az adatbetöltő egy (Python) generátor, amely minden egyes alkalmazáskor egy adatköteget ad vissza. Ezt a köteget ezután a `Module` `training_step` metódusába adjuk be a veszteség kiszámításához. Van egy opcionális `val_dataloader` is a validációs halmaz betöltőjének visszaadásához. Ugyanúgy viselkedik, azzal a különbséggel, hogy a `Module` `validation_step` metódusához ad adatkötegeket.

```{.python .input}
%%tab all
class DataModule(d2l.HyperParameters):  #@save
    """Az adatok alap osztálya."""
    if tab.selected('mxnet', 'pytorch'):
        def __init__(self, root='../data', num_workers=4):
            self.save_hyperparameters()

    if tab.selected('tensorflow', 'jax'):
        def __init__(self, root='../data'):
            self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
```

## Tanítás
:label:`oo-design-training`

:begin_tab:`pytorch, mxnet, tensorflow`
A `Trainer` osztály a `Module` osztályban lévő tanulható paramétereket tanítja a `DataModule`-ban megadott adatokkal. A legfontosabb metódus a `fit`, amely két argumentumot fogad el: `model`, a `Module` egy példánya, és `data`, a `DataModule` egy példánya. Ezután az egész adathalmazon `max_epochs` alkalommal iterál a modell tanításához. Mint korábban, ennek a metódusnak az implementációját későbbi fejezetekre halasztjuk.
:end_tab:

:begin_tab:`jax`
A `Trainer` osztály a `params` tanulható paramétereket tanítja a `DataModule`-ban megadott adatokkal. A legfontosabb metódus a `fit`, amely három argumentumot fogad el: `model`, a `Module` egy példánya, `data`, a `DataModule` egy példánya, és `key`, egy JAX `PRNGKeyArray`. A `key` argumentumot opcionálissá tesszük az interfész egyszerűsítése érdekében, de JAX-ban és Flax-ban ajánlott mindig megadni és egy gyökérkulccsal inicializálni a modell paramétereit. Ezután az egész adathalmazon `max_epochs` alkalommal iterál a modell tanításához. Mint korábban, ennek a metódusnak az implementációját későbbi fejezetekre halasztjuk.
:end_tab:

```{.python .input}
%%tab all
class Trainer(d2l.HyperParameters):  #@save
    """Adatokkal modellek tanítására szolgáló alap osztály."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        def fit(self, model, data):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    if tab.selected('jax'):
        def fit(self, model, data, key=None):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()

            if key is None:
                root_key = d2l.get_key()
            else:
                root_key = key
            params_key, dropout_key = jax.random.split(root_key)
            key = {'params': params_key, 'dropout': dropout_key}

            dummy_input = next(iter(self.train_dataloader))[:-1]
            variables = model.apply_init(dummy_input, key=key)
            params = variables['params']

            if 'batch_stats' in variables.keys():
                # A batch_stats értéket később használjuk (pl. batchnormalizációhoz)
                batch_stats = variables['batch_stats']
            else:
                batch_stats = {}

            # A Flax optax-ot használ a háttérben egyetlen TrainState állapotobjektumhoz.
            # Erről bővebben a dropout és a batchnormalizáció szakaszban lesz szó
            class TrainState(train_state.TrainState):
                batch_stats: Any
                dropout_rng: jax.random.PRNGKeyArray

            self.state = TrainState.create(apply_fn=model.apply,
                                           params=params,
                                           batch_stats=batch_stats,
                                           dropout_rng=dropout_key,
                                           tx=model.configure_optimizers())
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
```

## Összefoglalás

A jövőbeli deep learning implementációnk objektumorientált tervezésének kiemelése érdekében
a fenti osztályok egyszerűen bemutatják, hogyan tárolják az objektumok az adatokat
és hogyan hatnak kölcsön egymással.
Ezeket az osztályok implementációit folyamatosan gazdagítjuk majd,
például az `@add_to_class` segítségével,
a könyv hátralévő részében.
Ezenkívül
ezek a teljesen implementált osztályok
el vannak mentve a [D2L könyvtárban](https://github.com/d2l-ai/d2l-en/tree/master/d2l),
egy *könnyűsúlyú eszközkészletben*, amely megkönnyíti a deep learning strukturált modellezését.
Különösen megkönnyíti számos összetevő újrafelhasználását projektek között
anélkül, hogy sokat kellene változtatni rajta. Például csak az optimalizálót, csak a modellt, csak az adathalmazt cserélhetjük ki, stb.;
ez a modularitási fok az egész könyvben megtérül a tömörség és egyszerűség szempontjából (ezért adtuk hozzá), és ugyanezt teheti a saját projektjeidnél is.


## Feladatok

1. Keresd meg a fenti osztályok teljes implementációit, amelyek a [D2L könyvtárban](https://github.com/d2l-ai/d2l-en/tree/master/d2l) vannak elmentve. Erősen ajánljuk, hogy az implementációt részletesen nézd meg, miután már jobban megismerted a deep learning modellezést.
1. Távolítsd el a `save_hyperparameters` utasítást a `B` osztályból. Tudod-e még mindig kiírni a `self.a` és `self.b` értékeket? Opcionális: ha belemélyedtél a `HyperParameters` osztály teljes implementációjába, meg tudod magyarázni, miért?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/6645)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/6646)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/6647)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17974)
:end_tab:
