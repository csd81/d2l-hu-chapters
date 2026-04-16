```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Lineáris regresszió implementációja nulláról
:label:`sec_linear_scratch`

Most már készen állunk arra, hogy végigmenjünk
a lineáris regresszió egy teljesen működő implementációján.
Ebben a részben
(**az egész módszert nulláról implementáljuk,
beleértve (i) a modellt; (ii) a veszteségfüggvényt;
(iii) egy mini-batch sztochasztikus gradienscsökkenés optimalizálót;
és (iv) a tanítási függvényt,
amely összefűzi ezeket az összetevőket.**)
Végül futtatjuk a szintetikus adatgenerátorunkat
a :numref:`sec_synthetic-regression-data` részből,
és alkalmazzuk a modellünket
a kapott adathalmazra.
Bár a modern deep learning keretrendszerek
szinte mindezt automatizálni tudják,
az implementáció nulláról az egyetlen módja annak,
hogy valóban megbizonyosodj arról, hogy tudod, mit csinálsz.
Ezenkívül, amikor eljön az ideje a modellek testreszabásának,
saját rétegek vagy veszteségfüggvények definiálásakor,
a háttérben zajló folyamatok megértése nagyon hasznos lesz.
Ebben a részben csak tenzorokat és automatikus differenciálást használunk.
Később bemutatjuk a tömörebb implementációt,
kihasználva a deep learning keretrendszerek lehetőségeit,
miközben megtartjuk az alábbi struktúrát.

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
```

## A modell definiálása

[**Mielőtt mini-batch SGD segítségével elkezdhetnénk optimalizálni a modell paramétereit**],
(**először szükségünk van néhány paraméterre.**)
Az alábbiakban a súlyokat 0 várható értékű és 0.01 szórású normális eloszlásból
vett véletlen számokkal inicializáljuk.
A 0.01 mágikus szám a gyakorlatban általában jól működik,
de a `sigma` argumentumon keresztül megadható más érték is.
Ezenkívül az eltolást 0-ra állítjuk.
Vegyük észre, hogy az objektumorientált tervezés érdekében
a kódot a `d2l.Module` alosztályának `__init__` metódusába adjuk hozzá (amelyet a :numref:`subsec_oo-design-models` részben mutattunk be).

```{.python .input  n=6}
%%tab pytorch, mxnet, tensorflow
class LinearRegressionScratch(d2l.Module):  #@save
    """Nulláról implementált lineáris regressziós modell."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1))
            self.b = d2l.zeros(1)
            self.w.attach_grad()
            self.b.attach_grad()
        if tab.selected('pytorch'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1), requires_grad=True)
            self.b = d2l.zeros(1, requires_grad=True)
        if tab.selected('tensorflow'):
            w = tf.random.normal((num_inputs, 1), mean=0, stddev=0.01)
            b = tf.zeros(1)
            self.w = tf.Variable(w, trainable=True)
            self.b = tf.Variable(b, trainable=True)
```

```{.python .input  n=7}
%%tab jax
class LinearRegressionScratch(d2l.Module):  #@save
    """Nulláról implementált lineáris regressziós modell."""
    num_inputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.w = self.param('w', nn.initializers.normal(self.sigma),
                            (self.num_inputs, 1))
        self.b = self.param('b', nn.initializers.zeros, (1))
```

Ezután [**definiálnunk kell a modellünket,
meghatározva a bemenete és paraméterei, valamint a kimenete közötti kapcsolatot.**]
Ugyanazt a jelölésrendszert használva, mint a :eqref:`eq_linreg-y-vec` részben,
a lineáris modellünkhöz egyszerűen vesszük a $\mathbf{X}$ bemeneti jellemzők
és a $\mathbf{w}$ modellsúlyok mátrix-vektor szorzatát,
és hozzáadjuk a $b$ eltolást minden példányhoz.
A $\mathbf{Xw}$ szorzat egy vektor, és $b$ egy skalár.
A broadcasting mechanizmusnak köszönhetően
(lásd: :numref:`subsec_broadcasting`),
amikor egy vektort és egy skalárát összeadunk,
a skalár a vektor minden eleméhez hozzáadódik.
A kapott `forward` metódust
a `LinearRegressionScratch` osztályba regisztrálják
az `add_to_class` segítségével (amelyet a :numref:`oo-design-utilities` részben mutattunk be).

```{.python .input  n=8}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    return d2l.matmul(X, self.w) + self.b
```

## A veszteségfüggvény definiálása

Mivel [**a modellünk frissítéséhez szükség van
a veszteségfüggvény gradiensének kiszámítására,**]
(**először definiálnunk kell a veszteségfüggvényt.**)
Itt a :eqref:`eq_mse` négyzetesen összegzett veszteségfüggvényt használjuk.
Az implementációban az igazi értéket (`y`) a becsült érték alakjára (`y_hat`) kell transzformálni.
A következő metódus által visszaadott eredmény
szintén ugyanolyan alakú lesz, mint a `y_hat`.
A mini-batch összes példányára átlagolt veszteségértéket is visszaadjuk.

```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l = (y_hat - y) ** 2 / 2
    return d2l.reduce_mean(l)
```

```{.python .input  n=10}
%%tab jax
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, params, X, y, state):
    y_hat = state.apply_fn({'params': params}, *X)  # X kicsomagolva egy tuple-ből
    l = (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
    return d2l.reduce_mean(l)
```

## Az optimalizálási algoritmus definiálása

Ahogyan a :numref:`sec_linear_regression` részben tárgyaltuk,
a lineáris regressziónak zárt formájú megoldása van.
Azonban itt az a célunk, hogy bemutassuk,
hogyan tanítsunk általánosabb neurális hálózatokat,
és ehhez szükséges, hogy megismerd a mini-batch SGD-t.
Ezért ezt az alkalmat kihasználjuk
az SGD első működő példájának bemutatására.
Minden lépésben, az adathalmazunkból
véletlenszerűen vett mini-batch segítségével
becsüljük a veszteség gradiensét
a paraméterekre vonatkozóan.
Ezután frissítjük a paramétereket
abba az irányba, amely csökkentheti a veszteséget.

A következő kód alkalmazza a frissítést,
adott paraméterek és egy `lr` tanulási ráta esetén.
Mivel a veszteségünket a mini-batch átlagaként számítjuk,
nem kell a tanulási rátát a kötegmérethez igazítani.
A következő fejezetekben megvizsgáljuk,
hogyan kell a tanulási rátát beállítani
nagyon nagy mini-batch-ek esetén,
ahogyan azok az elosztott, nagy léptékű tanítás során felmerülnek.
Egyelőre figyelmen kívül hagyhatjuk ezt a függőséget.

:begin_tab:`mxnet`
Definiáljuk az `SGD` osztályunkat,
amely a `d2l.HyperParameters` alosztálya (amelyet a :numref:`oo-design-utilities` részben mutattunk be),
hasonló API-val rendelkezik,
mint a beépített SGD optimalizáló.
A paramétereket a `step` metódusban frissítjük.
Elfogad egy `batch_size` argumentumot, amelyet figyelmen kívül lehet hagyni.
:end_tab:

:begin_tab:`pytorch`
Definiáljuk az `SGD` osztályunkat,
amely a `d2l.HyperParameters` alosztálya (amelyet a :numref:`oo-design-utilities` részben mutattunk be),
hasonló API-val rendelkezik,
mint a beépített SGD optimalizáló.
A paramétereket a `step` metódusban frissítjük.
A `zero_grad` metódus minden gradienst 0-ra állít,
amelyet egy visszaterjesztési lépés előtt kell futtatni.
:end_tab:

:begin_tab:`tensorflow`
Definiáljuk az `SGD` osztályunkat,
amely a `d2l.HyperParameters` alosztálya (amelyet a :numref:`oo-design-utilities` részben mutattunk be),
hasonló API-val rendelkezik,
mint a beépített SGD optimalizáló.
A paramétereket az `apply_gradients` metódusban frissítjük.
Elfogad egy paraméter-gradiens párokat tartalmazó listát.
:end_tab:

```{.python .input  n=11}
%%tab mxnet, pytorch
class SGD(d2l.HyperParameters):  #@save
    """Minibatch sztochasztikus gradiens módszer."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    if tab.selected('mxnet'):
        def step(self, _):
            for param in self.params:
                param -= self.lr * param.grad

    if tab.selected('pytorch'):
        def step(self):
            for param in self.params:
                param -= self.lr * param.grad

        def zero_grad(self):
            for param in self.params:
                if param.grad is not None:
                    param.grad.zero_()
```

```{.python .input  n=12}
%%tab tensorflow
class SGD(d2l.HyperParameters):  #@save
    """Minibatch sztochasztikus gradiens módszer."""
    def __init__(self, lr):
        self.save_hyperparameters()

    def apply_gradients(self, grads_and_vars):
        for grad, param in grads_and_vars:
            param.assign_sub(self.lr * grad)
```

```{.python .input  n=13}
%%tab jax
class SGD(d2l.HyperParameters):  #@save
    """Minibatch sztochasztikus gradiens módszer."""
    # Az Optax fő transzformációja a GradientTransformation,
    # amelyet két metódus definiál: az init és az update.
    # Az init inicializálja az állapotot, az update pedig transzformálja a gradienseket.
    # https://github.com/deepmind/optax/blob/master/optax/_src/transform.py
    def __init__(self, lr):
        self.save_hyperparameters()

    def init(self, params):
        # Nem használt paraméterek törlése
        del params
        return optax.EmptyState

    def update(self, updates, state, params=None):
        del params
        # Amikor a state.apply_gradients metódust hívják meg a flax
        # train_state objektumának frissítéséhez, az belsőleg meghívja
        # az optax.apply_updates metódust, hozzáadva a paramétereket
        # az alább definiált frissítési egyenlethez.
        updates = jax.tree_util.tree_map(lambda g: -self.lr * g, updates)
        return updates, state

    def __call__():
        return optax.GradientTransformation(self.init, self.update)
```

Ezután definiáljuk a `configure_optimizers` metódust, amely az `SGD` osztály egy példányát adja vissza.

```{.python .input  n=14}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet') or tab.selected('pytorch'):
        return SGD([self.w, self.b], self.lr)
    if tab.selected('tensorflow', 'jax'):
        return SGD(self.lr)
```

## Tanítás

Most, hogy minden összetevő a helyén van
(paraméterek, veszteségfüggvény, modell és optimalizáló),
készen állunk arra, hogy [**implementáljuk a fő tanítási ciklust.**]
Nagyon fontos, hogy teljesen megértsd ezt a kódot,
mivel hasonló tanítási ciklusokat fogsz alkalmazni
minden más deep learning modellnél,
amelyet ez a könyv tárgyal.
Minden *korszakban* végigmegyünk
az egész tanítási adathalmazon,
minden példányon egyszer áthaladva
(feltéve, hogy a példányok száma
osztható a kötegmérettel).
Minden *iterációban* megragadunk egy mini-batch tanítási példányt,
és kiszámítjuk a veszteségét a modell `training_step` metódusán keresztül.
Ezután kiszámítjuk a gradienseket minden paraméterrel szemben.
Végül meghívjuk az optimalizálási algoritmust
a modell paramétereinek frissítéséhez.
Összefoglalva, a következő ciklust hajtjuk végre:

* Paraméterek inicializálása $(\mathbf{w}, b)$
* Ismételd amíg kész nem vagy
    * Gradient kiszámítása $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * Paraméterek frissítése $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$

Emlékeztetőül: a :numref:``sec_synthetic-regression-data`` részben
generált szintetikus regressziós adathalmaz
nem biztosít validációs adathalmazt.
A legtöbb esetben azonban
validációs adathalmazra lesz szükségünk
a modell minőségének mérésére.
Itt minden korszakban egyszer végigmegyünk a validációs adatbetöltőn
a modell teljesítményének mérésére.
Az objektumorientált tervezésünket követve,
a `prepare_batch` és `fit_epoch` metódusokat
a `d2l.Trainer` osztályba regisztráljuk
(amelyet a :numref:`oo-design-training` részben mutattunk be).

```{.python .input  n=15}
%%tab all    
@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch
```

```{.python .input  n=16}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.train()        
    for batch in self.train_dataloader:        
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # Később tárgyaljuk
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():            
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=17}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    for batch in self.train_dataloader:
        with autograd.record():
            loss = self.model.training_step(self.prepare_batch(batch))
        loss.backward()
        if self.gradient_clip_val > 0:
            self.clip_gradients(self.gradient_clip_val, self.model)
        self.optim.step(1)
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=18}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    for batch in self.train_dataloader:            
        with tf.GradientTape() as tape:
            loss = self.model.training_step(self.prepare_batch(batch))
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.gradient_clip_val > 0:
            grads = self.clip_gradients(self.gradient_clip_val, grads)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=19}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    if self.state.batch_stats:
        # A módosítható állapotokat később fogjuk használni (pl. batchnormalizációhoz)
        for batch in self.train_dataloader:
            (_, mutated_vars), grads = self.model.training_step(self.state.params,
                                                           self.prepare_batch(batch),
                                                           self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # Dropout rétegek nélküli modelleknél figyelmen kívül hagyható
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.state = self.state.replace(batch_stats=mutated_vars['batch_stats'])
            self.train_batch_idx += 1
    else:
        for batch in self.train_dataloader:
            _, grads = self.model.training_step(self.state.params,
                                                self.prepare_batch(batch),
                                                self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # Dropout rétegek nélküli modelleknél figyelmen kívül hagyható
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.train_batch_idx += 1

    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:
        self.model.validation_step(self.state.params,
                                   self.prepare_batch(batch),
                                   self.state)
        self.val_batch_idx += 1
```

Már majdnem készen állunk a modell tanítására,
de először szükségünk van néhány tanítási adatra.
Itt a `SyntheticRegressionData` osztályt használjuk,
és megadjuk az igazi paramétereket.
Ezután `lr=0.03` tanulási rátával tanítjuk a modellünket,
és `max_epochs=3`-t állítunk be.
Megjegyezzük, hogy általában mind a korszakok száma,
mind a tanulási ráta hiperparaméter.
Általában a hiperparaméterek beállítása nehézkes,
és általában háromirányú felosztást szeretnénk alkalmazni:
az egyik halmazt tanításra,
a másikat hiperparaméter-kiválasztásra,
a harmadikat pedig a végső kiértékelésre fenntartva.
Ezeket a részleteket egyelőre kihagyjuk, de később visszatérünk rájuk.

```{.python .input  n=20}
%%tab all
model = LinearRegressionScratch(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

Mivel mi magunk szintetizáltuk az adathalmazt,
pontosan tudjuk, mik az igazi paraméterek.
Így [**kiértékelhetjük a tanítási sikerességünket
az igazi paraméterek
és a tanítási ciklusunkon keresztül tanultak összehasonlításával.**]
Valóban nagyon közel esnek egymáshoz.

```{.python .input  n=21}
%%tab pytorch
with torch.no_grad():
    print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
    print(f'error in estimating b: {data.b - model.b}')
```

```{.python .input  n=22}
%%tab mxnet, tensorflow
print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
print(f'error in estimating b: {data.b - model.b}')
```

```{.python .input  n=23}
%%tab jax
params = trainer.state.params
print(f"error in estimating w: {data.w - d2l.reshape(params['w'], data.w.shape)}")
print(f"error in estimating b: {data.b - params['b']}")
```

Nem szabad természetesnek venni azt a képességet,
hogy pontosan visszanyerjük az igazi paramétereket.
Általában mély modelleknél egyedi megoldások
a paraméterekre nem léteznek,
és még lineáris modelleknél is
a paraméterek pontos visszanyerése
csak akkor lehetséges, ha egyetlen jellemző sem
lineárisan függő a többitől.
Azonban a gépi tanulásban
általában kevésbé foglalkoztat minket az igazi mögöttes paraméterek visszanyerése,
inkább azokkal a paraméterekkel foglalkozunk,
amelyek nagyon pontos előrejelzéshez vezetnek :cite:`Vapnik.1992`.
Szerencsére, még nehéz optimalizálási problémákon is,
a sztochasztikus gradienscsökkenés gyakran találhat kiváló megoldásokat,
részben annak köszönhetően, hogy mély hálózatoknál
a paraméterek sok olyan konfigurációja létezik,
amelyek nagyon pontos előrejelzéshez vezet.


## Összefoglalás

Ebben a részben jelentős lépést tettünk
a deep learning rendszerek tervezése felé,
egy teljesen funkcionális
neurális hálózati modell és tanítási ciklus implementálásával.
Ebben a folyamatban adatbetöltőt,
modellt, veszteségfüggvényt, optimalizálási eljárást,
valamint vizualizálási és monitoring eszközt építettünk.
Mindezt egy Python objektum összeállításával tettük,
amely tartalmaz minden releváns összetevőt egy modell tanításához.
Bár ez még nem professzionális minőségű implementáció,
teljesen funkcionális, és az ehhez hasonló kód
már segíthet a kisebb problémák gyors megoldásában.
A következő részekben látni fogjuk, hogyan tehető ez
*tömörebb* (a boilerplate kód elkerülésével)
és *hatékonyabb* (a GPU-k teljes potenciáljának kihasználásával).



## Feladatok

1. Mi történne, ha a súlyokat nullára inicializálnánk? Az algoritmus még mindig működne? Mi van, ha a paramétereket $1000$, nem $0.01$ szórással inicializálnánk?
1. Tegyük fel, hogy [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm) vagy, és megpróbálsz kidolgozni egy modellt az ellenállásra, amely a feszültséget és az áramot hozza összefüggésbe. Használhatsz-e automatikus differenciálást a modelled paramétereinek megtanulásához?
1. Használhatod-e a [Planck-törvényt](https://en.wikipedia.org/wiki/Planck%27s_law) egy tárgy hőmérsékletének meghatározásához spektrális energiasűrűség alapján? Referencia: egy fekete test által kibocsátott sugárzás $B$ spektrális sűrűsége: $B(\lambda, T) = \frac{2 hc^2}{\lambda^5} \cdot \left(\exp \frac{h c}{\lambda k T} - 1\right)^{-1}$. Itt $\lambda$ a hullámhossz, $T$ a hőmérséklet, $c$ a fénysebesség, $h$ a Planck-állandó, és $k$ a Boltzmann-állandó. Különböző $\lambda$ hullámhosszakra mérted az energiát, és most a spektrális sűrűséggörbét kell illesztened a Planck-törvényhez.
1. Milyen problémák merülhetnek fel, ha a veszteség második deriváltjait szeretnéd kiszámítani? Hogyan javítanád ezeket?
1. Miért szükséges a `reshape` metódus a `loss` függvényben?
1. Kísérletezz különböző tanulási rátákkel, hogy megtudd, milyen gyorsan csökken a veszteségfüggvény értéke. Csökkenthető-e a hiba a tanítási korszakok számának növelésével?
1. Ha a példányok száma nem osztható a kötegmérettel, mi történik a `data_iter`-rel egy korszak végén?
1. Próbálj meg egy másik veszteségfüggvényt implementálni, mint például az abszolút értékű veszteség `(y_hat - d2l.reshape(y, y_hat.shape)).abs().sum()`.
    1. Ellenőrizd, mi történik normál adatoknál.
    1. Ellenőrizd, hogy van-e viselkedésbeli különbség, ha aktívan megzavarsz néhány bejegyzést, például $y_5 = 10000$, a $\mathbf{y}$-ból.
    1. Tudsz-e olcsó megoldást kitalálni, amely ötvözi a négyzetesen összegzett veszteség és az abszolút értékű veszteség legjobb aspektusait? Tipp: hogyan kerülheted el a valóban nagy gradiens értékeket?
1. Miért kell újrakeverni az adathalmazt? Tudsz-e olyan esetet tervezni, ahol egy rosszindulatúan összerakott adathalmaz egyébként eltörné az optimalizálási algoritmust?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/201)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17976)
:end_tab:
