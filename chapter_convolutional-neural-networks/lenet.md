```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Konvolúciós neurális hálózatok (LeNet)
:label:`sec_lenet`

Most már rendelkezünk az összes összetevővel, amely szükséges egy teljesen funkcionális CNN összerakásához. A képadatokkal való korábbi találkozásunk során softmax regresszióval rendelkező lineáris modellt (:numref:`sec_softmax_scratch`) és MLP-t (:numref:`sec_mlp-implementation`) alkalmaztunk a Fashion-MNIST adathalmaz ruhadarab-képeire. Ahhoz, hogy az ilyen adatokat kezelhetővé tegyük, először minden képet egy $28\times28$-as mátrixból egy rögzített hosszúságú $784$-dimenziós vektorrá lapítottunk ki, majd ezután teljesen összekötött rétegeken dolgozzuk fel. Most, hogy jól értjük a konvolúciós rétegeket, megőrizhetjük a képeink térbeli struktúráját. A teljesen összekötött rétegek konvolúciós rétegekkel való felváltásának további előnyeként takarékosabb modelleket élvezhetünk, amelyek sokkal kevesebb paramétert igényelnek.

Ebben a részben bemutatjuk a *LeNetet*, az első publikált CNN-ek egyikét, amely széles körű figyelmet vívott ki a számítógépes látási feladatokban nyújtott teljesítményével. A modellt Yann LeCun vezette be (és nevezett el), aki akkor az AT&T Bell Labs kutatója volt, a képekben lévő kézzel írt számjegyek felismerésének céljából :cite:`LeCun.Bottou.Bengio.ea.1998`. Ez a munka egy évtizedes, a technológia fejlesztésére irányuló kutatás betetőzését jelentette; LeCun csapata tette közzé az első tanulmányt, amely sikeresen tanított CNN-eket visszaterjesztésen keresztül :cite:`LeCun.Boser.Denker.ea.1989`.

A LeNet abban az időben kiemelkedő eredményeket ért el, megfelelve a szupport vektor gépek teljesítményének, amely akkor az ellenőrzött tanulás domináns megközelítése volt, számjegyenként 1%-nál kisebb hibaarányt elérve. A LeNet végül az ATM gépekben lévő betétek feldolgozásához szükséges számjegyek felismerésére lett adaptálva. A mai napig néhány ATM még mindig azt a kódot futtatja, amelyet Yann LeCun és kollégája, Leon Bottou az 1990-es években írt!

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
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
from types import FunctionType
```

## LeNet

Magas szinten (**a LeNet (LeNet-5) két részből áll: (i) egy két konvolúciós rétegből álló konvolúciós kódolóból; és (ii) egy három teljesen összekötött rétegből álló sűrű blokkból**). Az architektúra a :numref:`img_lenet`-ben foglalható össze.

![Adatfolyam a LeNetben. A bemenet egy kézzel írt számjegy, a kimenet egy valószínűség 10 lehetséges eredmény felett.](../img/lenet.svg)
:label:`img_lenet`

Minden konvolúciós blokkban az alapegységek egy konvolúciós réteg, egy szigmoid aktivációs függvény és egy azt követő átlagos pooling művelet. Megjegyezzük, hogy bár a ReLU-k és a max-pooling jobban működnek, ezeket akkor még nem fedezték fel. Minden konvolúciós réteg $5\times 5$-ös kernelt és szigmoid aktivációs függvényt használ. Ezek a rétegek a térben elrendezett bemeneteket számos kétdimenziós jellemzőtérképre képezik le, általában növelve a csatornák számát. Az első konvolúciós rétegnek 6 kimeneti csatornája van, míg a másodiknak 16. Minden $2\times2$-es pooling művelet (2-es lépésköz) négyszeres mértékben csökkenti a dimenzionalitást térbeli lecsökkentéssel. A konvolúciós blokk egy kimenetet bocsát ki, amelynek alakja (batch méret, csatornák száma, magasság, szélesség).

A konvolúciós blokkból a sűrű blokkba való kimenet átadásához minden egyes példát ki kell lapítanunk a mini-batchből. Más szóval, ezt a négydimenziós bemenetet a teljesen összekötött rétegek által elvárt kétdimenziós bemenetre alakítjuk: emlékeztetőként az általunk kívánt kétdimenziós reprezentáció az első dimenziót a mini-batch példáinak indexelésére, a másodikat pedig minden példa lapos vektor-reprezentációjának megadására használja. A LeNet sűrű blokkjának három teljesen összekötött rétege van, rendre 120, 84 és 10 kimenettel. Mivel még mindig osztályozást végzünk, a 10-dimenziós kimeneti réteg megfelel a lehetséges kimeneti osztályok számának.

Bár egy kis munkát igényelhetett eljutni arra a pontra, hogy valóban megértse, mi történik a LeNeten belül, reméljük, hogy a következő kódrészlet meggyőz téged arról, hogy az ilyen modellek modern deep learning keretrendszerekkel való implementálása rendkívül egyszerű. Csak egy `Sequential` blokkot kell példányosítanunk, és a megfelelő rétegeket kell összeláncolnunk, Xavier inicializálást alkalmazva, ahogyan azt a :numref:`subsec_xavier` bevezette.

```{.python .input}
%%tab pytorch
def init_cnn(module):  #@save
    """CNN-ek súlyainak inicializálása."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
```

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LeNet(d2l.Classifier):  #@save
    """A LeNet-5 modell."""
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(channels=6, kernel_size=5, padding=2,
                          activation='sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Dense(120, activation='sigmoid'),
                nn.Dense(84, activation='sigmoid'),
                nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.LazyLinear(120), nn.Sigmoid(),
                nn.LazyLinear(84), nn.Sigmoid(),
                nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       activation='sigmoid', padding='same'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                                       activation='sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(120, activation='sigmoid'),
                tf.keras.layers.Dense(84, activation='sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class LeNet(d2l.Classifier):  #@save
    """A LeNet-5 modell."""
    lr: float = 0.1
    num_classes: int = 10
    kernel_init: FunctionType = nn.initializers.xavier_uniform

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(features=6, kernel_size=(5, 5), padding='SAME',
                    kernel_init=self.kernel_init()),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(features=16, kernel_size=(5, 5), padding='VALID',
                    kernel_init=self.kernel_init()),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),  # lapítás
            nn.Dense(features=120, kernel_init=self.kernel_init()),
            nn.sigmoid,
            nn.Dense(features=84, kernel_init=self.kernel_init()),
            nn.sigmoid,
            nn.Dense(features=self.num_classes, kernel_init=self.kernel_init())
        ])
```

Bizonyos szabadságot vettünk a LeNet reprodukálásában annyiban, hogy a Gauss-aktivációs réteget egy softmax réteggel váltottuk fel. Ez nagymértékben egyszerűsíti az implementációt, nem is utolsósorban azért, mert a Gauss-dekódolót manapság ritkán használják. Ettől eltekintve a hálózat megegyezik az eredeti LeNet-5 architektúrával.

:begin_tab:`pytorch, mxnet, tensorflow`
Lássuk, mi történik a hálózaton belül. Egycsatornás (fekete-fehér) $28 \times 28$-as képet átadva a hálózaton, és minden rétegben kinyomtatva a kimeneti alakot, [**megvizsgálhatjuk a modellt**], hogy megbizonyosodjunk arról, hogy műveletei megegyeznek a :numref:`img_lenet_vert`-ben várással.
:end_tab:

:begin_tab:`jax`
Lássuk, mi történik a hálózaton belül. Egycsatornás (fekete-fehér) $28 \times 28$-as képet átadva a hálózaton, és minden rétegben kinyomtatva a kimeneti alakot, [**megvizsgálhatjuk a modellt**], hogy megbizonyosodjunk arról, hogy műveletei megegyeznek a :numref:`img_lenet_vert`-ben várással. A Flax biztosítja az `nn.tabulate` elemet, egy ügyes módszert hálózatunk rétegeinek és paramétereinek összefoglalásához. Itt a `bind` módszert használjuk egy kötött modell létrehozásához. A változók most a `d2l.Module` osztályhoz vannak kötve, azaz ez a kötött modell egy állapotteljes objektummá válik, amelyet ezután a `Sequential` objektum `net` attribútumának és a benne lévő `layers`-nek elérésére lehet használni. Megjegyezzük, hogy a `bind` módszert csak interaktív kísérletezésre szabad használni, és nem közvetlen helyettesítője az `apply` módszernek.
:end_tab:

![A LeNet-5 tömörített jelölése.](../img/lenet-vert.svg)
:label:`img_lenet_vert`

```{.python .input}
%%tab mxnet, pytorch
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape):
    X = d2l.randn(*X_shape)
    for layer in self.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
        
model = LeNet()
model.layer_summary((1, 1, 28, 28))
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape):
    X = d2l.normal(X_shape)
    for layer in self.net.layers:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

model = LeNet()
model.layer_summary((1, 28, 28, 1))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape, key=d2l.get_key()):
    X = jnp.zeros(X_shape)
    params = self.init(key, X)
    bound_model = self.clone().bind(params, mutable=['batch_stats'])
    _ = bound_model(X)
    for layer in bound_model.net.layers:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

model = LeNet()
model.layer_summary((1, 28, 28, 1))
```

Vegyük figyelembe, hogy a reprezentáció magassága és szélessége a konvolúciós blokk minden rétegénél csökken (az előző réteghez képest). Az első konvolúciós réteg két pixel párnázást használ a magasság és szélesség csökkentésének kompenzálására, amely egyébként $5 \times 5$-ös kernel használatából adódna. Mellékesen megjegyezzük, hogy az eredeti MNIST OCR adathalmazban a $28 \times 28$ pixeles képméret az eredeti $32 \times 32$ pixeles szkennelésekből két pixel sor (és oszlop) *levágásának* eredménye. Ezt elsősorban helytakarékossági okokból tették (30%-os csökkentés) abban az időben, amikor a megabájtok számítottak.

Ezzel szemben a második konvolúciós réteg nem alkalmaz párnázást, ezért magasság és szélesség egyaránt négy pixellel csökken. Ahogy haladunk a rétegek sorában, a csatornák száma rétegenként növekszik: a bemenetben 1-ről az első konvolúciós réteg után 6-ra, a második konvolúciós réteg után 16-ra. Azonban minden pooling réteg felezi a magasságot és szélességet. Végül minden teljesen összekötött réteg csökkenti a dimenzionalitást, végül egy olyan kimenetet bocsátva ki, amelynek dimenziója megegyezik az osztályok számával.


## Tanítás

Most, hogy implementáltuk a modellt, [**futtassunk egy kísérletet, hogy lássuk, hogyan teljesít a LeNet-5 modell a Fashion-MNIST-en**].

Bár a CNN-eknek kevesebb paramétere van, számítási szempontból mégis drágábbak lehetnek, mint a hasonlóan mély MLP-k, mivel minden paraméter sokkal több szorzásban vesz részt. Ha van hozzáférésünk GPU-hoz, lehet, hogy most jó alkalom azt akcióba hozni a tanítás felgyorsítására. Vegyük figyelembe, hogy a `d2l.Trainer` osztály minden részletről gondoskodik. Alapértelmezés szerint inicializálja a modell paramétereit az elérhető eszközökön. Ahogyan az MLP-knél, veszteségfüggvényünk keresztentrópia, és mini-batch sztochasztikus gradienscsökkenés segítségével minimalizáljuk.

```{.python .input}
%%tab pytorch, mxnet, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = LeNet(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = LeNet(lr=0.1)
    trainer.fit(model, data)
```

## Összefoglalás

Ebben a fejezetben jelentős előrelépést tettünk. Az 1980-as évek MLP-iről az 1990-es évek és a 2000-es évek eleji CNN-eire léptünk. A javasolt architektúrák, például a LeNet-5 formájában, még ma is értékesek. Érdemes összehasonlítani a Fashion-MNIST-en LeNet-5-tel elérhető hibaarányokat mind az MLP-kkel elérhető legjobb eredménnyel (:numref:`sec_mlp-implementation`), mind a jóval fejlettebb architektúrákéval, mint például a ResNet (:numref:`sec_resnet`). A LeNet sokkal inkább hasonlít az utóbbihoz, mint az előbbihez. Az egyik elsődleges különbség, ahogyan majd látni fogjuk, az, hogy a nagyobb számítási kapacitás lényegesen összetettebb architektúrákat tett lehetővé.

A második különbség az a relatív egyszerűség, amellyel a LeNetet implementálni tudtuk. Ami korábban hónapnyi C++ és assembly kódot, az SN, egy korai Lisp-alapú deep learning eszköz :cite:`Bottou.Le-Cun.1988` fejlesztésének mérnöki kihívása volt, és végül a modellekkel való kísérletezés, az most percek alatt elvégezhető. Ez a hihetetlen produktivitás-növekedés az, ami a deep learning modell fejlesztést rendkívüli mértékben demokratizálta. A következő fejezetben belemegyünk ebbe a nyúlüregbe, hogy lássuk, hová vezet minket.

## Feladatok

1. Modernizáljuk a LeNetet. Implementáljuk és teszteljük a következő változtatásokat:
    1. Cseréljük le az átlagos poolingot max-poolingra.
    1. Cseréljük le a softmax réteget ReLU-ra.
1. Próbáljuk megváltoztatni a LeNet stílusú hálózat méretét a pontosság javítása érdekében a max-pooling és ReLU használata mellett.
    1. Állítsuk be a konvolúciós ablak méretét.
    1. Állítsuk be a kimeneti csatornák számát.
    1. Állítsuk be a konvolúciós rétegek számát.
    1. Állítsuk be a teljesen összekötött rétegek számát.
    1. Állítsuk be a tanulási rátákat és egyéb tanítási részleteket (pl. inicializálás és epochok száma).
1. Próbáljuk ki a javított hálózatot az eredeti MNIST adathalmazon.
1. Jelenítsük meg a LeNet első és második rétegének aktivációit különböző bemenetekhez (pl. pulóverek és kabátok).
1. Mi történik az aktivációkkal, ha lényegesen különböző képeket táplálunk a hálózatba (pl. macskák, autók vagy akár véletlenszerű zaj)?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/73)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/74)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/275)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18000)
:end_tab:
