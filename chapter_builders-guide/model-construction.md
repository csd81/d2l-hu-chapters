```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Rétegek és modulok
:label:`sec_model_construction`

Amikor először mutattuk be a neurális hálózatokat,
egyetlen kimenettel rendelkező lineáris modellekre összpontosítottunk.
Ott az egész modell csupán egyetlen neuronból állt.
Fontos megjegyezni, hogy egyetlen neuron
(i) bemenetek egy halmazát fogadja;
(ii) megfelelő skaláris kimenetet generál;
és (iii) rendelkezik a kapcsolódó paraméterek egy halmazával, amelyek frissíthetők
az érdeklődési célfüggvény optimalizálása érdekében.
Majd, amikor több kimenetű hálózatokban gondolkodtunk,
vektorizált aritmetikát alkalmaztunk
a neuronok egy egész rétegének jellemzéséhez.
Ahogy az egyes neuronok esetén,
a rétegek is (i) bemenetek egy halmazát fogadják,
(ii) megfelelő kimeneteket generálnak,
és (iii) hangolható paraméterek egy halmaza írja le őket.
Amikor a softmax regresszión dolgoztunk,
egyetlen réteg maga volt a modell.
Ugyanakkor, még az MLP-k bevezetésekor is
gondolhattunk a modellre úgy,
mint amely megőrzi ezt az alapstruktúrát.

Érdekes módon az MLP-k esetén
mind az egész modell, mind az alkotórétegek
ugyanezt a struktúrát osztják.
Az egész modell nyers bemeneteket fogad (a jellemzőket),
kimeneteket generál (az előrejelzéseket),
és paraméterekkel rendelkezik
(az összes alkotóréteg kombinált paraméterei).
Ugyanígy, minden egyes réteg bemeneteket vesz fel
(az előző réteg által szállított),
kimeneteket generál (a következő réteg bemenetei),
és hangolható paraméterek egy halmazával rendelkezik, amelyek frissülnek
a következő rétegből visszafelé áramló jel szerint.


Bár azt gondolhatnánk, hogy a neuronok, rétegek és modellek
elegendő absztrakciót adnak a munkánkhoz,
kiderül, hogy sokszor célszerűbb
olyan komponensekről beszélni, amelyek
nagyobbak egy egyes rétegtől,
de kisebbek az egész modellnél.
Például a ResNet-152 architektúra,
amely rendkívül népszerű a számítógépes látásban,
több száz réteget tartalmaz.
Ezek a rétegek *rétegcsoportok* ismétlődő mintázataiból állnak. Egy ilyen hálózat rétegenként való implementálása fárasztóvá válhat.
Ez az aggodalom nem csupán elméleti – ilyen
tervezési minták a gyakorlatban is általánosak.
A fentebb említett ResNet architektúra
megnyerte a 2015-ös ImageNet és COCO számítógépes látás versenyeket
mind a felismerés, mind az érzékelés kategóriában :cite:`He.Zhang.Ren.ea.2016`
és számos látási feladat esetén alapvető architektúra maradt.
Hasonló architektúrák, amelyekben a rétegek
különféle ismétlődő mintázatokban vannak elrendezve,
ma már más területeken is mindenütt jelen vannak,
beleértve a természetes nyelvfeldolgozást és a beszédfeldolgozást.

Ezen összetett hálózatok implementálásához
bevezetjük a neurális hálózat *modul* fogalmát.
Egy modul leírhat egyetlen réteget,
több rétegből álló komponenst,
vagy magát az egész modellt!
A modulabsztrakció egyik előnye,
hogy ezek nagyobb egységekbe kombinálhatók,
sokszor rekurzívan. Ezt szemlélteti :numref:`fig_blocks`. Az igény szerint tetszőleges bonyolultságú modulokat generáló kód definiálásával
meglepően tömör kódot írhatunk,
és mégis összetett neurális hálózatokat implementálhatunk.

![Több réteg modulokba kombinálódik, ismétlődő mintázatokat alkotva a nagyobb modellekben.](../img/blocks.svg)
:label:`fig_blocks`


Programozási szempontból a modult egy *osztály* képviseli.
Bármely alosztályának definiálnia kell egy előreterjedési metódust,
amely a bemenetet kimenetté alakítja,
és el kell tárolnia a szükséges paramétereket.
Vegyük észre, hogy néhány modulnak egyáltalán nincsenek paraméterei.
Végül a modulnak rendelkeznie kell egy visszaterjedési metódussal
a gradiensek kiszámításához.
Szerencsére az automatikus differenciálás által biztosított
háttérbeli mágia miatt
(amelyet :numref:`sec_autograd` részben mutattunk be),
saját modulunk definiálásakor
csak a paraméterekkel és
az előreterjedési metódussal kell foglalkoznunk.

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
```

```{.python .input}
%%tab jax
from typing import List
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

[**Kezdjük azzal, hogy visszatérünk az MLP-k implementálásához
használt kódhoz**]
(:numref:`sec_mlp`).
A következő kód egy hálózatot generál
256 egységgel és ReLU aktivációval rendelkező
teljesen összefüggő rejtett réteggel,
amelyet egy tíz egységgel rendelkező teljesen összefüggő kimeneti réteg követ
(aktivációs függvény nélkül).

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X).shape
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

X = torch.rand(2, 20)
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

X = tf.random.uniform((2, 20))
net(X).shape
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(10)])

# get_key egy d2l-ben elmentett függvény, amely jax.random.PRNGKey(random_seed)-et ad vissza
X = jax.random.uniform(d2l.get_key(), (2, 20))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

:begin_tab:`mxnet`
Ebben a példában a modellünket
egy `nn.Sequential` példányosításával építettük fel,
a visszaadott objektumot a `net` változóhoz rendelve.
Ezután ismételten meghívjuk az `add` metódusát,
rétegeket hozzáfűzve abban a sorrendben,
ahogy végrehajtásra kerülnek.
Röviden, az `nn.Sequential` egy speciális `Block` típust definiál,
a Gluon-ban a *modult* megjelenítő osztályt.
Megőriz egy rendezett listát az alkotó `Block`-okról.
Az `add` metódus egyszerűen megkönnyíti
minden egymást követő `Block` listához való hozzáadását.
Vegyük észre, hogy minden réteg a `Dense` osztály egy példánya,
amely maga is a `Block` alosztálya.
Az előreterjedési (`forward`) metódus is feltűnően egyszerű:
láncolja a lista minden `Block`-ját egymáshoz,
mindegyik kimenetét a következő bemenetéül adva át.
Vegyük észre, hogy mostanáig a modelleinket
a `net(X)` konstrukcióval hívtuk meg a kimenetük megszerzéséhez.
Ez valójában csak a `net.forward(X)` rövidítése,
egy ügyes Python-trükk, amely a
`Block` osztály `__call__` metódusán keresztül érhető el.
:end_tab:

:begin_tab:`pytorch`
Ebben a példában a modellünket
egy `nn.Sequential` példányosításával építettük fel, a rétegekkel abban a sorrendben argumentumként átadva,
ahogy végrehajtásra kerülnek.
Röviden, (**az `nn.Sequential` egy speciális `Module` típust definiál**),
a PyTorch-ban a modult megjelenítő osztályt.
Megőriz egy rendezett listát az alkotó `Module`-okről.
Vegyük észre, hogy mindkét teljesen összefüggő réteg a `Linear` osztály egy példánya,
amely maga is a `Module` alosztálya.
Az előreterjedési (`forward`) metódus is feltűnően egyszerű:
láncolja a lista minden modulját egymáshoz,
mindegyik kimenetét a következő bemenetéül adva át.
Vegyük észre, hogy mostanáig a modelleinket
a `net(X)` konstrukcióval hívtuk meg a kimenetük megszerzéséhez.
Ez valójában csak a `net.__call__(X)` rövidítése.
:end_tab:

:begin_tab:`tensorflow`
Ebben a példában a modellünket
egy `keras.models.Sequential` példányosításával építettük fel, a rétegekkel abban a sorrendben argumentumként átadva,
ahogy végrehajtásra kerülnek.
Röviden, a `Sequential` egy speciális `keras.Model` típust definiál,
a Keras-ban a modult megjelenítő osztályt.
Megőriz egy rendezett listát az alkotó `Model`-ekről.
Vegyük észre, hogy mindkét teljesen összefüggő réteg a `Dense` osztály egy példánya,
amely maga is a `Model` alosztálya.
Az előreterjedési (`call`) metódus is feltűnően egyszerű:
láncolja a lista minden modulját egymáshoz,
mindegyik kimenetét a következő bemenetéül adva át.
Vegyük észre, hogy mostanáig a modelleinket
a `net(X)` konstrukcióval hívtuk meg a kimenetük megszerzéséhez.
Ez valójában csak a `net.call(X)` rövidítése,
egy ügyes Python-trükk, amely a
modul osztály `__call__` metódusán keresztül érhető el.
:end_tab:

## [**Egyéni modul**]

Talán a legegyszerűbb módja annak, hogy megértsük, hogyan működik egy modul,
ha mi magunk implementálunk egyet.
Mielőtt ezt megtennénk,
röviden összefoglaljuk az alapvető funkcionalitást,
amelyet minden modulnak biztosítania kell:


1. Bemeneti adatok fogadása argumentumként az előreterjedési metódusban.
1. Kimenet generálása az előreterjedési metódus által visszaadott értékkel. Vegyük észre, hogy a kimenet eltérő alakzatú lehet a bemenettől. Például a fenti modellünkben az első teljesen összefüggő réteg tetszőleges dimenziójú bemenetet fogad, de 256 dimenziójú kimenetet ad vissza.
1. A kimenet gradiensének kiszámítása a bemenethez képest, amely a visszaterjedési metóduson keresztül érhető el. Általában ez automatikusan történik.
1. A szükséges paraméterek tárolása és hozzáférhetővé tétele
   az előreterjedési számítás végrehajtásához.
1. A modell paramétereinek inicializálása szükség esetén.


A következő kódrészletben
nulláról kódolunk egy modult,
amely egy rejtett réteggel rendelkező MLP-nek felel meg
256 rejtett egységgel
és 10 dimenziós kimeneti réteggel.
Vegyük észre, hogy az alábbi `MLP` osztály örökli a modult megjelenítő osztályt.
Nagymértékben támaszkodunk a szülőosztály metódusaira,
csak a saját konstruktorunkat (Pythonban az `__init__` metódust) és az előreterjedési metódust adjuk meg.

```{.python .input}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self):
        # Az MLP szülőosztálya, az nn.Block konstruktorának meghívása
        # a szükséges inicializálás elvégzéséhez
        super().__init__()
        self.hidden = nn.Dense(256, activation='relu')
        self.out = nn.Dense(10)

    # A modell előreterjedésének definiálása, azaz hogyan adjuk vissza
    # a szükséges modellkimenetet az X bemenet alapján
    def forward(self, X):
        return self.out(self.hidden(X))
```

```{.python .input}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        # Az nn.Module szülőosztály konstruktorának meghívása
        # a szükséges inicializálás elvégzéséhez
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.out = nn.LazyLinear(10)

    # A modell előreterjedésének definiálása, azaz hogyan adjuk vissza
    # a szükséges modellkimenetet az X bemenet alapján
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
```

```{.python .input}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        # A tf.keras.Model szülőosztály konstruktorának meghívása
        # a szükséges inicializálás elvégzéséhez
        super().__init__()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    # A modell előreterjedésének definiálása, azaz hogyan adjuk vissza
    # a szükséges modellkimenetet az X bemenet alapján
    def call(self, X):
        return self.out(self.hidden((X)))
```

```{.python .input}
%%tab jax
class MLP(nn.Module):
    def setup(self):
        # A rétegek definiálása
        self.hidden = nn.Dense(256)
        self.out = nn.Dense(10)

    # A modell előreterjedésének definiálása, azaz hogyan adjuk vissza
    # a szükséges modellkimenetet az X bemenet alapján
    def __call__(self, X):
        return self.out(nn.relu(self.hidden(X)))
```

Először az előreterjedési metódusra összpontosítsunk.
Vegyük észre, hogy bemenetként fogadja az `X`-et,
kiszámítja az alkalmazott aktivációs függvénnyel ellátott rejtett reprezentációt,
és kimeneti logitokat ad vissza.
Ebben az `MLP` implementációban
mindkét réteg példányváltozó.
Hogy miért ésszerű ez, képzeljük el
két MLP példányosítását, `net1` és `net2`,
amelyeket különböző adatokon tanítunk.
Természetesen elvárjuk tőlük,
hogy két különböző tanított modellt képviseljenek.

[**Az MLP rétegeinek példányosítása**]
a konstruktorban történik
(**és ezeket a rétegeket ezt követően hívjuk meg**)
az előreterjedési metódus minden egyes meghívásakor.
Vegyünk észre néhány kulcsfontosságú részletet.
Először is, a testreszabott `__init__` metódusunk
meghívja a szülőosztály `__init__` metódusát
a `super().__init__()` segítségével,
megkímélve minket attól, hogy megismételjük
a legtöbb modulra alkalmazható sablonkódot.
Ezután példányosítjuk a két teljesen összefüggő rétegünket,
és hozzárendeljük őket a `self.hidden` és `self.out` változókhoz.
Vegyük észre, hogy hacsak nem implementálunk új réteget,
nem kell aggódnunk a visszaterjedési metódus
vagy a paraméter inicializálása miatt.
A rendszer automatikusan generálja ezeket a metódusokat.
Próbáljuk ki.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
net = MLP()
if tab.selected('mxnet'):
    net.initialize()
net(X).shape
```

```{.python .input}
%%tab jax
net = MLP()
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

A modulabsztrakció fő erénye a sokoldalúsága.
Lehet alosztályozni egy modult rétegek létrehozásához
(mint a teljesen összefüggő réteg osztálya),
teljes modellek (mint a fenti `MLP` osztály),
vagy közbülső bonyolultságú különféle komponensek.
Ezt a sokoldalúságot kihasználjuk
az elkövetkező fejezetekben,
például konvolúciós neurális hálózatok tárgyalásakor.


## [**A Sequential modul**]
:label:`subsec_model-construction-sequential`

Most közelebbről megvizsgálhatjuk,
hogyan működik a `Sequential` osztály.
Emlékezzünk arra, hogy a `Sequential`-t arra tervezték,
hogy más modulokat láncoljon össze.
A saját egyszerűsített `MySequential`-unk felépítéséhez
csupán két kulcsfontosságú metódust kell definiálnunk:

1. Egy metódust a modulok egyenkénti listához való hozzáfűzéséhez.
1. Egy előreterjedési metódust a bemenet átadásához a modulok láncolatán, ugyanabban a sorrendben, ahogy hozzá lettek fűzve.

A következő `MySequential` osztály ugyanazt a
funkcionalitást nyújtja, mint az alapértelmezett `Sequential` osztály.

```{.python .input}
%%tab mxnet
class MySequential(nn.Block):
    def add(self, block):
        # A block itt a Block egy alosztályának példánya, amelynek egyedi neve
        # van. A Block osztály _children tagváltozójában tároljuk el, amely
        # OrderedDict típusú. A MySequential példány initialize híváskor
        # a rendszer automatikusan inicializálja a _children összes tagját
        self._children[block.name] = block

    def forward(self, X):
        # Az OrderedDict garantálja, hogy a tagok a hozzáadás sorrendjében
        # lesznek bejárva
        for block in self._children.values():
            X = block(X)
        return X
```

```{.python .input}
%%tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, X):
        for module in self.children():            
            X = module(X)
        return X
```

```{.python .input}
%%tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = args

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

```{.python .input}
%%tab jax
class MySequential(nn.Module):
    modules: List

    def __call__(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

:begin_tab:`mxnet`
Az `add` metódus egyetlen blokkot fűz hozzá
a rendezett `_children` szótárhoz.
Felmerülhet a kérdés, hogy minden Gluon `Block`
miért rendelkezik `_children` attribútummal,
és miért használtuk ezt ahelyett, hogy
egyszerűen magunk definiáltunk volna egy Python listát.
Röviden, a `_children` fő előnye,
hogy blokkunk paraméter inicializálásakor
a Gluon tudja, hogy a `_children`
szótárban kell keresnie azokat az al-blokkokat, amelyek
paramétereinek szintén inicializálódnia kell.
:end_tab:

:begin_tab:`pytorch`
Az `__init__` metódusban minden modult hozzáadunk
az `add_modules` metódus meghívásával. Ezek a modulok később a `children` metódussal érhetők el.
Ily módon a rendszer tudja a hozzáadott modulokat,
és megfelelően inicializálja minden modul paramétereit.
:end_tab:

Amikor a `MySequential` előreterjedési metódusa meghívódik,
minden hozzáadott modul végrehajtásra kerül
abban a sorrendben, amelyben hozzá lettek adva.
Most újraimplementálhatunk egy MLP-t
a `MySequential` osztályunk segítségével.

```{.python .input}
%%tab mxnet
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X).shape
```

```{.python .input}
%%tab pytorch
net = MySequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
net(X).shape
```

```{.python .input}
%%tab jax
net = MySequential([nn.Dense(256), nn.relu, nn.Dense(10)])
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

Vegyük észre, hogy a `MySequential` ezen használata
azonos az általunk korábban írt kóddal
a `Sequential` osztályhoz
(amint azt :numref:`sec_mlp` részben leírtuk).


## [**Kód végrehajtása az előreterjedési metódusban**]

A `Sequential` osztály megkönnyíti a modellépítést,
lehetővé téve, hogy új architektúrákat állítsunk össze
anélkül, hogy saját osztályt kellene definiálnunk.
Azonban nem minden architektúra egyszerű lánc.
Ha nagyobb rugalmasságra van szükség,
saját blokkokat kell definiálnunk.
Például érdemes lehet
Python vezérlési folyamatot végrehajtani az előreterjedési metóduson belül.
Sőt, tetszőleges matematikai műveleteket is érdemes elvégezni,
nem csupán előre meghatározott neurális hálózati rétegekre hagyatkozni.

Talán észrevettük, hogy mostanáig
hálózataink összes művelete
a hálózat aktivációira
és paraméterire hatott.
Néha azonban érdemes lehet
olyan tagokat beépíteni,
amelyek sem a korábbi rétegek eredményei,
sem frissíthető paraméterek.
Ezeket *konstans paramétereknek* nevezzük.
Tegyük fel például, hogy olyan réteget szeretnénk,
amely az $f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$ függvényt számítja ki,
ahol $\mathbf{x}$ a bemenet, $\mathbf{w}$ a paraméterünk,
és $c$ egy megadott konstans,
amely az optimalizálás során nem frissül.
Így implementáljuk a `FixedHiddenMLP` osztályt.

```{.python .input}
%%tab mxnet
class FixedHiddenMLP(nn.Block):
    def __init__(self):
        super().__init__()
        # A get_constant metódussal létrehozott véletlenszerű súlyparaméterek
        # tanítás során nem frissülnek (azaz konstans paraméterek)
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, X):
        X = self.dense(X)
        # A létrehozott konstans paraméterek, valamint a relu és dot
        # függvények használata
        X = npx.relu(np.dot(X, self.rand_weight.data()) + 1)
        # A teljesen összefüggő réteg újrafelhasználása. Ez egyenértékű
        # két teljesen összefüggő réteg paramétereinek megosztásával
        X = self.dense(X)
        # Vezérlési folyamat
        while np.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
%%tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Véletlenszerű súlyparaméterek, amelyek nem számítanak gradienseket,
        # így tanítás közben konstansak maradnak
        self.rand_weight = torch.rand((20, 20))
        self.linear = nn.LazyLinear(20)

    def forward(self, X):
        X = self.linear(X)        
        X = F.relu(X @ self.rand_weight + 1)
        # A teljesen összefüggő réteg újrafelhasználása. Ez egyenértékű
        # két teljesen összefüggő réteg paramétereinek megosztásával
        X = self.linear(X)
        # Vezérlési folyamat
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
%%tab tensorflow
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # A tf.constant segítségével létrehozott véletlenszerű súlyparaméterek
        # tanítás során nem frissülnek (azaz konstans paraméterek)
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        X = self.flatten(inputs)
        # A létrehozott konstans paraméterek, valamint a relu és
        # matmul függvények használata
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        # A teljesen összefüggő réteg újrafelhasználása. Ez egyenértékű
        # két teljesen összefüggő réteg paramétereinek megosztásával
        X = self.dense(X)
        # Vezérlési folyamat
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)
```

```{.python .input}
%%tab jax
class FixedHiddenMLP(nn.Module):
    # Véletlenszerű súlyparaméterek, amelyek nem számítanak gradienseket,
    # így tanítás közben konstansak maradnak
    rand_weight: jnp.array = jax.random.uniform(d2l.get_key(), (20, 20))

    def setup(self):
        self.dense = nn.Dense(20)

    def __call__(self, X):
        X = self.dense(X)
        X = nn.relu(X @ self.rand_weight + 1)
        # A teljesen összefüggő réteg újrafelhasználása. Ez egyenértékű
        # két teljesen összefüggő réteg paramétereinek megosztásával
        X = self.dense(X)
        # Vezérlési folyamat
        while jnp.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

Ebben a modellben
implementálunk egy rejtett réteget, amelynek súlyait
(`self.rand_weight`) véletlenszerűen inicializáljuk
a példányosításkor, és ezt követően konstansak maradnak.
Ez a súly nem modellparaméter,
így a visszaterjedés soha nem frissíti.
A hálózat ezután ennek a "rögzített" rétegnek a kimenetét adja át
egy teljesen összefüggő rétegen keresztül.

Vegyük észre, hogy a kimenet visszaadása előtt
a modellünk valami szokatlan dolgot tett.
Egy while-ciklust futtattunk, tesztelve
azt a feltételt, hogy az $\ell_1$ norma nagyobb-e $1$-nél,
és a kimeneti vektorunkat $2$-vel osztottuk el,
amíg a feltétel teljesül.
Végül az `X` bejegyzéseinek összegét adtuk vissza.
Tudomásunk szerint egyetlen szabványos neurális hálózat
sem végez ilyen műveletet.
Vegyük észre, hogy ez a konkrét művelet
valószínűleg nem hasznos egyetlen valós feladatban sem.
Csupán azt szeretnénk megmutatni, hogyan lehet
tetszőleges kódot integrálni
a neurális hálózat számításainak menetébe.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
net = FixedHiddenMLP()
if tab.selected('mxnet'):
    net.initialize()
net(X)
```

```{.python .input}
%%tab jax
net = FixedHiddenMLP()
params = net.init(d2l.get_key(), X)
net.apply(params, X)
```

[**Különféle módokon kombinálhatjuk és párosíthatjuk
a modulok összerakásának lehetőségeit.**]
A következő példában kreatív módon ágyazunk be modulokat
egymásba.

```{.python .input}
%%tab mxnet
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, X):
        return self.dense(self.net(X))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FixedHiddenMLP())
chimera.initialize()
chimera(X)
```

```{.python .input}
%%tab pytorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(64), nn.ReLU(),
                                 nn.LazyLinear(32), nn.ReLU())
        self.linear = nn.LazyLinear(16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.LazyLinear(20), FixedHiddenMLP())
chimera(X)
```

```{.python .input}
%%tab tensorflow
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))

chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
chimera(X)
```

```{.python .input}
%%tab jax
class NestMLP(nn.Module):
    def setup(self):
        self.net = nn.Sequential([nn.Dense(64), nn.relu,
                                  nn.Dense(32), nn.relu])
        self.dense = nn.Dense(16)

    def __call__(self, X):
        return self.dense(self.net(X))


chimera = nn.Sequential([NestMLP(), nn.Dense(20), FixedHiddenMLP()])
params = chimera.init(d2l.get_key(), X)
chimera.apply(params, X)
```

## Összefoglalás

Az egyes rétegek lehetnek modulok.
Sok réteg alkothat egy modult.
Sok modul alkothat egy modult.

Egy modul tartalmazhat kódot.
A modulok sok köztevékenységről gondoskodnak, beleértve a paraméterek inicializálását és a visszaterjedést.
A rétegek és modulok szekvenciális összefűzését a `Sequential` modul kezeli.


## Feladatok

1. Milyen problémák léphetnek fel, ha megváltoztatjuk a `MySequential`-t úgy, hogy a modulokat egy Python listában tárolja?
1. Implementálj egy modult, amely két modult vesz argumentumként, mondjuk `net1`-et és `net2`-t, és az előreterjedés során mindkét hálózat összefűzött kimenetét adja vissza. Ezt *párhuzamos modulnak* is nevezik.
1. Tegyük fel, hogy ugyanannak a hálózatnak több példányát szeretnénk összefűzni. Implementálj egy gyártó függvényt, amely több példányt generál ugyanabból a modulból, és ezekből épít fel egy nagyobb hálózatot.

:begin_tab:`mxnet`
[Megbeszélések](https://discuss.d2l.ai/t/54)
:end_tab:

:begin_tab:`pytorch`
[Megbeszélések](https://discuss.d2l.ai/t/55)
:end_tab:

:begin_tab:`tensorflow`
[Megbeszélések](https://discuss.d2l.ai/t/264)
:end_tab:

:begin_tab:`jax`
[Megbeszélések](https://discuss.d2l.ai/t/17989)
:end_tab:
