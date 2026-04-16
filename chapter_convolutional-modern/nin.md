```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Network in Network (NiN)
:label:`sec_nin`

A LeNet, az AlexNet és a VGG mind ugyanazt az általános tervezési mintát osztja: térbeli struktúrát kiaknázó jellemzőkinyerés konvolúciók és pooling rétegek sorozatával, majd a reprezentációk utófeldolgozása teljesen összekötött rétegekkel.
Az AlexNet és a VGG által a LeNet-en végrehajtott fejlesztések főként abban rejlenek, ahogyan ezek a későbbi hálózatok kiszélesítik és elmélyítik ezt a két modult.

Ez a tervezés két fő kihívást vet fel.
Először is, az architektúra végén lévő teljesen összekötött rétegek óriási számú paramétert fogyasztanak. Például még egy egyszerű modell, mint a VGG-11 is egy szörnyű méretű mátrixot igényel, amely egyszeres pontosságban (FP32) közel 400 MB RAM-ot foglal el. Ez jelentős akadály a számítás szempontjából, különösen mobil- és beágyazott eszközökön. Elvégre még a csúcskategóriás mobiltelefonoknak is legfeljebb 8 GB RAM-juk van. Amikor a VGG-t feltalálták, ez egy nagyságrenddel kevesebb volt (az iPhone 4S 512 MB-tal rendelkezett). Így nehéz lett volna indokolni, hogy a memória nagy részét képosztályozásra fordítsák.

Másodszor, ugyanolyan lehetetlen teljesen összekötött rétegeket hozzáadni a hálózat korábbi részein a nemlinearitás fokának növelésére: ez megsemmisítené a térbeli struktúrát, és potenciálisan még több memóriát igényelne.

A *network in network* (*NiN*) blokkok :cite:`Lin.Chen.Yan.2013` alternatívát kínálnak, amelyek egyszerű stratégiával mindkét problémát képesek megoldani.
Egy nagyon egyszerű meglátáson alapulnak: (i) $1 \times 1$-es konvolúciókkal helyi nemlinearitásokat adnak hozzá a csatorna-aktivációkon keresztül, és (ii) globális átlagos poolingot alkalmaznak az utolsó reprezentációs réteg összes helyzetén való összesítéshez. Megjegyezzük, hogy a globális átlagos pooling nem lenne hatékony a hozzáadott nemlinearitások nélkül. Merüljünk el a részletekben.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
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
```

## (**NiN Blokkok**)

Idézzük fel a :numref:`subsec_1x1` részt. Ott azt mondtuk, hogy a konvolúciós rétegek bemenete és kimenete négydimenziós tenzorokból áll, amelyek tengelyei a mintának, a csatornának, a magasságnak és a szélességnek felelnek meg.
Idézzük fel azt is, hogy a teljesen összekötött rétegek bemenetei és kimenetei jellemzően kétdimenziós tenzorok, amelyek a mintának és a jellemzőnek felelnek meg.
A NiN mögötti ötlet az, hogy minden pixelhelyen (minden magasságra és szélességre) teljesen összekötött réteget alkalmazunk.
Az eredményül kapott $1 \times 1$-es konvolúció tekinthető egy teljesen összekötött rétegnek, amely minden pixelhelyen önállóan hat.

A :numref:`fig_nin` a VGG és a NiN architektúrájának, valamint blokkjainak főbb strukturális különbségeit szemlélteti.
Figyeljük meg a NiN blokkok különbségét (a kezdeti konvolúciót $1 \times 1$-es konvolúciók követik, míg a VGG megtartja a $3 \times 3$-as konvolúciókat), és a végét, ahol már nincs szükség óriási teljesen összekötött rétegre.

![A VGG és a NiN architektúrájának és blokkjainak összehasonlítása.](../img/nin.svg)
:width:`600px`
:label:`fig_nin`

```{.python .input}
%%tab mxnet
def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding,
                      activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk
```

```{.python .input}
%%tab pytorch
def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU())
```

```{.python .input}
%%tab tensorflow
def nin_block(out_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(out_channels, kernel_size, strides=strides,
                           padding=padding),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(out_channels, 1),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(out_channels, 1),
    tf.keras.layers.Activation('relu')])
```

```{.python .input}
%%tab jax
def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential([
        nn.Conv(out_channels, kernel_size, strides, padding),
        nn.relu,
        nn.Conv(out_channels, kernel_size=(1, 1)), nn.relu,
        nn.Conv(out_channels, kernel_size=(1, 1)), nn.relu])
```

## [**NiN Modell**]

A NiN ugyanolyan kezdeti konvolúciós méreteket alkalmaz, mint az AlexNet (nem sokkal azután javasolták).
A kernelméret sorrendben $11\times 11$, $5\times 5$ és $3\times 3$, a kimeneti csatornák száma megfelel az AlexNet-nek. Minden NiN blokkot egy max-pooling réteg követ 2-es lépésközzel és $3\times 3$-as ablakmérettel.

A NiN és az AlexNet, illetve a VGG közötti második lényeges különbség az, hogy a NiN teljesen mellőzi a teljesen összekötött rétegeket.
Ehelyett a NiN egy NiN blokkot alkalmaz, amelynek kimeneti csatornaszáma megegyezik a címkeosztályok számával, majd egy *globális* átlagos pooling réteg következik, amely logitok vektorát állítja elő.
Ez a tervezés jelentősen csökkenti a szükséges modellparaméterek számát, bár a tanítási idő esetleges növekedése árán.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class NiN(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nin_block(96, kernel_size=11, strides=4, padding=0),
                nn.MaxPool2D(pool_size=3, strides=2),
                nin_block(256, kernel_size=5, strides=1, padding=2),
                nn.MaxPool2D(pool_size=3, strides=2),
                nin_block(384, kernel_size=3, strides=1, padding=1),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding=1),
                nn.GlobalAvgPool2D(),
                nn.Flatten())
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nin_block(96, kernel_size=11, strides=4, padding=0),
                nn.MaxPool2d(3, stride=2),
                nin_block(256, kernel_size=5, strides=1, padding=2),
                nn.MaxPool2d(3, stride=2),
                nin_block(384, kernel_size=3, strides=1, padding=1),
                nn.MaxPool2d(3, stride=2),
                nn.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten())
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                nin_block(96, kernel_size=11, strides=4, padding='valid'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                nin_block(256, kernel_size=5, strides=1, padding='same'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                nin_block(384, kernel_size=3, strides=1, padding='same'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding='same'),
                tf.keras.layers.GlobalAvgPool2D(),
                tf.keras.layers.Flatten()])
```

```{.python .input}
%%tab jax
class NiN(d2l.Classifier):
    lr: float = 0.1
    num_classes = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nin_block(96, kernel_size=(11, 11), strides=(4, 4), padding=(0, 0)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nin_block(256, kernel_size=(5, 5), strides=(1, 1), padding=(2, 2)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nin_block(384, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nn.Dropout(0.5, deterministic=not self.training),
            nin_block(self.num_classes, kernel_size=(3, 3), strides=1, padding=(1, 1)),
            lambda x: nn.avg_pool(x, (5, 5)),  # globális átlagos pooling
            lambda x: x.reshape((x.shape[0], -1))  # lapítás
        ])
```

Létrehozunk egy adatpéldát, hogy lássuk [**az egyes blokkok kimeneti alakját**].

```{.python .input}
%%tab mxnet, pytorch
NiN().layer_summary((1, 1, 224, 224))
```

```{.python .input}
%%tab tensorflow
NiN().layer_summary((1, 224, 224, 1))
```

```{.python .input}
%%tab jax
NiN(training=False).layer_summary((1, 224, 224, 1))
```

## [**Tanítás**]

Mint korábban, a Fashion-MNIST-et használjuk a modell tanításához, ugyanazzal az optimalizálóval, amelyet az AlexNet és a VGG esetén is alkalmaztunk.

```{.python .input}
%%tab mxnet, pytorch, jax
model = NiN(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = NiN(lr=0.05)
    trainer.fit(model, data)
```

## Összefoglalás

A NiN-nek drámaian kevesebb paramétere van, mint az AlexNet-nek és a VGG-nek. Ez elsősorban abból ered, hogy nincs szüksége óriási teljesen összekötött rétegekre. Ehelyett globális átlagos poolingot alkalmaz a hálózat törzsének utolsó szakasza után az összes képhelyen való összesítéshez. Ez szükségtelenné teszi a drága (tanult) dimenziójú csökkentési műveleteket, amelyeket egy egyszerű átlaggal helyettesít. Ami meglepte a kutatókat, az volt, hogy ez az átlagoló művelet nem rontotta a pontosságot. Megjegyezzük, hogy az alacsony felbontású reprezentáción (sok csatornával) való átlagolás szintén növeli a hálózat eltolási invarianciáját.

Kevesebb konvolúció választása széles kernelekkel, és helyettük $1 \times 1$-es konvolúciók alkalmazása tovább segíti a kevesebb paraméter elérését. Ez lehetővé teszi a csatornák közötti jelentős nemlinearitás kezelését bármely adott helyen. Mind az $1 \times 1$-es konvolúciók, mind a globális átlagos pooling jelentősen befolyásolta a későbbi CNN tervezéseket.

## Feladatok

1. Miért van blokkokként két $1\times 1$-es konvolúciós réteg a NiN-ben? Növeljük számukat háromra. Csökkentsük számukat egyre. Mi változik?
1. Mi változik, ha az $1 \times 1$-es konvolúciókat $3 \times 3$-as konvolúciókkal helyettesítjük?
1. Mi történik, ha a globális átlagos poolingot teljesen összekötött réteggel helyettesítjük (sebesség, pontosság, paraméterek száma)?
1. Számítsd ki a NiN erőforrásfelhasználását.
    1. Mennyi a paraméterek száma?
    1. Mekkora a számítási igény?
    1. Mekkora memóriára van szükség tanítás közben?
    1. Mekkora memóriára van szükség inferencia során?
1. Mik lehetnek a lehetséges problémák azzal, ha a $384 \times 5 \times 5$-ös reprezentációt egy lépésben $10 \times 5 \times 5$-ösre csökkentjük?
1. Használd a VGG strukturális tervezési döntéseit, amelyek a VGG-11, VGG-16 és VGG-19-hez vezettek, egy NiN-szerű hálózatok famíliájának tervezéséhez.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/79)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/80)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18003)
:end_tab:
