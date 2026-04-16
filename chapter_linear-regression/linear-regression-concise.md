```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# A lineáris regresszió tömör implementációja
:label:`sec_linear_concise`

A deep learning az elmúlt évtizedben valamiféle kambriumi robbanást élt meg.
A technikák, alkalmazások és algoritmusok száma messze meghaladja
az előző évtizedek fejlődését.
Ez több tényező szerencsés kombinációjának köszönhető,
amelyek egyike a számos nyílt forráskódú deep learning keretrendszer
által kínált hatékony, ingyenes eszközök.
Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010`,
DistBelief :cite:`Dean.Corrado.Monga.ea.2012`,
és Caffe :cite:`Jia.Shelhamer.Donahue.ea.2014`
valószínűleg az ilyen modellek
első generációját képviseli,
amelyek széles körben elterjedtek.
Az olyan korábbi (úttörő) munkákhoz képest, mint
az SN2 (Simulateur Neuristique) :cite:`Bottou.Le-Cun.1988`,
amely Lisp-szerű programozási élményt nyújtott,
a modern keretrendszerek automatikus differenciálást
és a Python kényelmét kínálják.
Ezek a keretrendszerek lehetővé teszik számunkra,
hogy automatizáljuk és modularizáljuk
a gradient alapú tanulási algoritmusok implementálásának ismétlődő munkáját.

A :numref:`sec_linear_scratch` részben csak
(i) tenzorokat támaszkodtunk az adattároláshoz és lineáris algebrához;
és (ii) automatikus differenciálást a gradiensek kiszámításához.
A gyakorlatban, mivel az adatiterátorok, veszteségfüggvények, optimalizálók
és neurális hálózati rétegek
olyan közönségesek, a modern könyvtárak ezeket az összetevőket is implementálják nekünk.
Ebben a részben (**bemutatjuk, hogyan implementálható
a lineáris regresszió modell**) a :numref:`sec_linear_scratch` részből
(**tömören a deep learning keretrendszerek magas szintű API-jaival**).

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
```

## A modell definiálása

Amikor a lineáris regressziót nulláról implementáltuk
a :numref:`sec_linear_scratch` részben,
a modell paramétereit explicit módon definiáltuk,
és a kimenetek előállítását szolgáló számításokat
alapvető lineáris algebra műveletekkel kódoltuk.
Ezt *tudnod kell* megcsinálni.
De amint a modelljeid összetettebbé válnak,
és amint ezt szinte minden nap kell elvégezned,
örülni fogsz a segítségnek.
A helyzet hasonló ahhoz, mintha nulláról kódolnád a saját blogodat.
Egyszer-kétszer megcsinálni kifizetődő és tanulságos,
de rossz webfejlesztő lennél,
ha egy hónapot töltesz a kerék újrafeltalálásával.

Standard műveleteknél
[**használhatjuk a keretrendszer előre definiált rétegeit,**]
amelyek lehetővé teszik számunkra, hogy a modell felépítéséhez
használt rétegekre összpontosítsunk,
ahelyett hogy az implementációjukkal aggódnánk.
Emlékezz az egyrétegű hálózat architektúrájára,
ahogyan a :numref:`fig_single_neuron` részben leírtuk.
A réteg neve *teljesen összekötött*,
mivel minden bemenetét az összes kimenetéhez csatlakoztatja
egy mátrix-vektor szorzás útján.

:begin_tab:`mxnet`
Gluon-ban a teljesen összekötött réteg a `Dense` osztályban van definiálva.
Mivel csak egyetlen skaláros kimenetet szeretnénk generálni,
ezt a számot 1-re állítjuk.
Érdemes megjegyezni, hogy kényelem szempontjából
a Gluon nem igényli, hogy minden réteghez megadjuk
a bemeneti alakot.
Ezért nem kell megmondanunk a Gluon-nak,
hogy hány bemenet kerül ebbe a lineáris rétegbe.
Amikor először adunk át adatokat a modellünkön,
pl. amikor később végrehajtjuk a `net(X)`-et,
a Gluon automatikusan következteti ki minden réteg bemenetének számát, és
így példányosítja a helyes modellt.
Ezt részletesebben is leírjuk később.
:end_tab:

:begin_tab:`pytorch`
PyTorch-ban a teljesen összekötött réteg a `Linear` és `LazyLinear` osztályokban van definiálva (az 1.8.0-s verziótól érhető el).
Az utóbbi lehetővé teszi a felhasználóknak, hogy *csupán*
a kimeneti dimenziót adják meg,
míg az előbbi emellett azt is megkérdezi,
hogy hány bemenet kerül ebbe a rétegbe.
A bemeneti alakok megadása kényelmetlen és nem triviális számításokat igényelhet
(például konvolúciós rétegekben).
Ezért az egyszerűség kedvéért ilyen "lusta" rétegeket fogunk használni,
amikor csak lehetséges.
:end_tab:

:begin_tab:`tensorflow`
Keras-ban a teljesen összekötött réteg a `Dense` osztályban van definiálva.
Mivel csak egyetlen skaláros kimenetet szeretnénk generálni,
ezt a számot 1-re állítjuk.
Érdemes megjegyezni, hogy kényelem szempontjából
a Keras nem igényli, hogy minden réteghez megadjuk
a bemeneti alakot.
Nem kell megmondanunk a Keras-nak,
hogy hány bemenet kerül ebbe a lineáris rétegbe.
Amikor először próbálunk adatokat átadni a modellünkön,
pl. amikor később végrehajtjuk a `net(X)`-et,
a Keras automatikusan következteti ki
minden réteg bemenetének számát.
Ezt részletesebben is leírjuk később.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LinearRegression(d2l.Module):  #@save
    """A lineáris regresszió modell magas szintű API-okkal implementálva."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(1)
            self.net.initialize(init.Normal(sigma=0.01))
        if tab.selected('tensorflow'):
            initializer = tf.initializers.RandomNormal(stddev=0.01)
            self.net = tf.keras.layers.Dense(1, kernel_initializer=initializer)
        if tab.selected('pytorch'):
            self.net = nn.LazyLinear(1)
            self.net.weight.data.normal_(0, 0.01)
            self.net.bias.data.fill_(0)
```

```{.python .input}
%%tab jax
class LinearRegression(d2l.Module):  #@save
    """A lineáris regresszió modell magas szintű API-okkal implementálva."""
    lr: float

    def setup(self):
        self.net = nn.Dense(1, kernel_init=nn.initializers.normal(0.01))
```

A `forward` metódusban egyszerűen meghívjuk az előre definiált rétegek beépített `__call__` metódusát a kimenetek kiszámításához.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

## A veszteségfüggvény definiálása

:begin_tab:`mxnet`
A `loss` modul számos hasznos veszteségfüggvényt definiál.
A sebesség és kényelem érdekében nem implementáljuk a sajátunkat,
hanem a beépített `loss.L2Loss`-t választjuk.
Mivel az általa visszaadott `loss`
minden példány négyzetes hibája,
a `mean`-t használjuk a veszteség átlagolásához a mini-batch felett.
:end_tab:

:begin_tab:`pytorch`
[**Az `MSELoss` osztály kiszámítja az átlagos négyzethibát (a :eqref:`eq_mse` $1/2$ tényezője nélkül).**]
Alapértelmezés szerint az `MSELoss` visszaadja a példányok átlagos veszteségét.
Gyorsabb (és könnyebben használható), mint a sajátunkat implementálni.
:end_tab:

:begin_tab:`tensorflow`
A `MeanSquaredError` osztály kiszámítja az átlagos négyzethibát (a :eqref:`eq_mse` $1/2$ tényezője nélkül).
Alapértelmezés szerint a példányok átlagos veszteségét adja vissza.
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    if tab.selected('mxnet'):
        fn = gluon.loss.L2Loss()
        return fn(y_hat, y).mean()
    if tab.selected('pytorch'):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, params, X, y, state):
    y_hat = state.apply_fn({'params': params}, *X)
    return d2l.reduce_mean(optax.l2_loss(y_hat, y))
```

## Az optimalizálási algoritmus definiálása

:begin_tab:`mxnet`
A mini-batch SGD egy standard eszköz
a neurális hálózatok optimalizálásához,
így a Gluon az algoritmus számos variációja mellett
a `Trainer` osztályán keresztül támogatja.
Megjegyezzük, hogy a Gluon `Trainer` osztálya
az optimalizálási algoritmust jelenti,
míg a :numref:`sec_oo-design` részben létrehozott `Trainer` osztályunk
a tanítási metódust tartalmazza,
azaz ismételten meghívja az optimalizálót
a modell paramétereinek frissítéséhez.
Amikor példányosítjuk a `Trainer`-t,
megadjuk az optimalizálandó paramétereket,
amelyek a `net.collect_params()` segítségével nyerhetők ki a `net` modellünkből,
az általunk kívánt optimalizálási algoritmust (`sgd`),
és az optimalizálási algoritmusunkhoz szükséges hiperparaméterek szótárát.
:end_tab:

:begin_tab:`pytorch`
A mini-batch SGD egy standard eszköz
a neurális hálózatok optimalizálásához,
így a PyTorch az `optim` modulban az algoritmus számos variációja mellett támogatja azt.
Amikor (**példányosítunk egy `SGD` példányt,**)
megadjuk az optimalizálandó paramétereket,
amelyek a modellünkből a `self.parameters()` segítségével nyerhetők ki,
és a tanulási rátát (`self.lr`)
amelyet az optimalizálási algoritmusunk igényel.
:end_tab:

:begin_tab:`tensorflow`
A mini-batch SGD egy standard eszköz
a neurális hálózatok optimalizálásához,
így a Keras az `optimizers` modulban az algoritmus számos variációja mellett támogatja azt.
:end_tab:

```{.python .input}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet'):
        return gluon.Trainer(self.collect_params(),
                             'sgd', {'learning_rate': self.lr})
    if tab.selected('pytorch'):
        return torch.optim.SGD(self.parameters(), self.lr)
    if tab.selected('tensorflow'):
        return tf.keras.optimizers.SGD(self.lr)
    if tab.selected('jax'):
        return optax.sgd(self.lr)
```

## Tanítás

Talán észrevetted, hogy a modellünk kifejezése
egy deep learning keretrendszer magas szintű API-jai segítségével
kevesebb kódsort igényel.
Nem kellett egyedileg lefoglalni a paramétereket,
definiálni a veszteségfüggvényt, vagy implementálni a mini-batch SGD-t.
Amint sokkal összetettebb modellekkel kezdünk dolgozni,
a magas szintű API előnyei jelentősen növekednek.

Most, hogy minden alapelem a helyén van,
[**a tanítási ciklus maga ugyanaz,
mint amelyet nulláról implementáltunk.**]
Tehát egyszerűen meghívjuk a `fit` metódust (amelyet a :numref:`oo-design-training` részben mutattunk be),
amely a `fit_epoch` metódus implementációjára támaszkodik
a :numref:`sec_linear_scratch` részből,
a modellünk tanításához.

```{.python .input}
%%tab all
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

Az alábbiakban
[**összehasonlítjuk a véges adatokon való tanítással megtanult
modell paramétereket
és a tényleges paramétereket,**]
amelyek az adathalmazunkat generálták.
A paraméterek eléréséhez
hozzáférünk a szükséges réteg súlyaihoz és eltolásához.
Ahogy a nulláról való implementációban is láttuk,
vegyük észre, hogy a becsült paramétereink
közel vannak a valódi megfelelőikhez.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    if tab.selected('mxnet'):
        return (self.net.weight.data(), self.net.bias.data())
    if tab.selected('pytorch'):
        return (self.net.weight.data, self.net.bias.data)
    if tab.selected('tensorflow'):
        return (self.get_weights()[0], self.get_weights()[1])

w, b = model.get_w_b()
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self, state):
    net = state.params['net']
    return net['kernel'], net['bias']

w, b = model.get_w_b(trainer.state)
```

```{.python .input}
print(f'hiba a w becslésében: {data.w - d2l.reshape(w, data.w.shape)}')
print(f'hiba a b becslésében: {data.b - b}')
```

## Összefoglalás

Ez a rész tartalmazza egy mély hálózat (ebben a könyvben)
az első implementációját,
amely kihasználja a modern deep learning keretrendszerek
által nyújtott kényelmet,
mint például MXNet :cite:`Chen.Li.Li.ea.2015`,
JAX :cite:`Frostig.Johnson.Leary.2018`,
PyTorch :cite:`Paszke.Gross.Massa.ea.2019`,
és TensorFlow :cite:`Abadi.Barham.Chen.ea.2016`.
A keretrendszer alapértelmezéseit alkalmaztuk az adatok betöltéséhez, egy réteg,
egy veszteségfüggvény, egy optimalizáló és egy tanítási ciklus definiálásához.
Amikor a keretrendszer biztosítja az összes szükséges funkciót,
általában jó ötlet ezeket használni,
mivel ezen összetevők könyvtári implementációi
általában erősen optimalizáltak a teljesítményre
és megfelelően teszteltek a megbízhatóság szempontjából.
Ugyanakkor próbáld meg ne felejteni,
hogy ezeket a modulokat *közvetlenül is lehet* implementálni.
Ez különösen fontos azon törekvő kutatóknak,
akik a modellfejlesztés élvonalán szeretnének élni,
ahol olyan új összetevőket fogsz kitalálni,
amelyek valószínűleg nem létezhetnek egyetlen jelenlegi könyvtárban sem.

:begin_tab:`mxnet`
Gluon-ban a `data` modul adatfeldolgozó eszközöket biztosít,
az `nn` modul nagy számú neurális hálózati réteget definiál,
és a `loss` modul számos általános veszteségfüggvényt definiál.
Ezenkívül az `initializer` hozzáférést biztosít
a paraméter-inicializálás számos lehetőségéhez.
A felhasználó kényelmére
a dimenzionalitás és a tárhely automatikusan következtethető ki.
Ennek a lusta inicializálásnak az a következménye,
hogy nem szabad megkísérelni hozzáférni a paraméterekhez
mielőtt azok példányosítva (és inicializálva) lettek volna.
:end_tab:

:begin_tab:`pytorch`
PyTorch-ban a `data` modul adatfeldolgozó eszközöket biztosít,
az `nn` modul nagy számú neurális hálózati réteget és általános veszteségfüggvényt definiál.
A paramétereket `_`-re végződő metódusokkal inicializálhatjuk
az értékek lecserélésével.
Megjegyezzük, hogy meg kell adnunk a hálózat bemeneti dimenzióit.
Bár ez most triviális, jelentős tovagyűrűző hatásai lehetnek,
ha összetett, sok rétegből álló hálózatokat szeretnénk tervezni.
Ezek a hálózatok paramétrezéséhez gondos megfontolás szükséges
a hordozhatóság érdekében.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow-ban a `data` modul adatfeldolgozó eszközöket biztosít,
a `keras` modul nagy számú neurális hálózati réteget és általános veszteségfüggvényt definiál.
Ezenkívül az `initializers` modul különböző módszereket biztosít a modell paramétereinek inicializálásához.
A hálózatok dimenzionalitása és tárhelye automatikusan következtethető ki
(de vigyázz, ne kíséreld meg hozzáférni a paraméterekhez mielőtt azok inicializálva lettek volna).
:end_tab:

## Feladatok

1. Hogyan kellene megváltoztatnod a tanulási rátát, ha a mini-batch feletti összesített veszteséget
   a mini-batch veszteségének átlagával helyettesítenéd?
1. Tekintsd át a keretrendszer dokumentációját, hogy melyik veszteségfüggvények állnak rendelkezésre. Különösen
   cseréld le a négyzethibát Huber robusztus veszteségfüggvényére. Azaz, használd a veszteségfüggvényt:
   $$l(y,y') = \begin{cases}|y-y'| -\frac{\sigma}{2} & \textrm{ ha } |y-y'| > \sigma \\ \frac{1}{2 \sigma} (y-y')^2 & \textrm{ egyébként}\end{cases}$$
1. Hogyan éred el a modell súlyainak gradiensét?
1. Mi a hatása a megoldásra, ha megváltoztatod a tanulási rátát és a korszakok számát? Folyamatosan javul-e?
1. Hogyan változik a megoldás, ahogy változtatod a generált adatok mennyiségét?
    1. Ábrázold a becslési hibát $\hat{\mathbf{w}} - \mathbf{w}$ és $\hat{b} - b$ az adatok mennyiségének függvényében. Tipp: logaritmikusan növeld az adatok mennyiségét, nem lineárisan, azaz 5, 10, 20, 50, ..., 10000, nem pedig 1000, 2000, ..., 10000.
    2. Miért megfelelő a tippben szereplő javaslat?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/204)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17977)
:end_tab:
