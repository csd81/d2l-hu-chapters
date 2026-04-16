```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Többágú Hálózatok (GoogLeNet)
:label:`sec_googlenet`

2014-ben a *GoogLeNet* megnyerte az ImageNet Challenge-t :cite:`Szegedy.Liu.Jia.ea.2015`, egy olyan struktúrát alkalmazva, amely ötvözte a NiN :cite:`Lin.Chen.Yan.2013`, az ismétlődő blokkok :cite:`Simonyan.Zisserman.2014` és a konvolúciós kernelek koktéljának erősségeit. Ez volt valószínűleg az első olyan hálózat is, amelyben egy CNN-ben egyértelműen elkülönült a törzs (adatbevitel), a test (adatfeldolgozás) és a fej (predikció). Ez a tervezési minta azóta is megmaradt a mély hálózatok tervezésében: a *törzset* az első két vagy három konvolúció adja, amelyek a képen dolgoznak. Alacsony szintű jellemzőket kinyernek az alapjául szolgáló képekből. Ezt a konvolúciós blokkok *teste* követi. Végül a *fej* képezi le az eddig kapott jellemzőket a szükséges osztályozási, szegmentálási, detektálási vagy követési feladatra.

A GoogLeNet fő hozzájárulása a hálózat testének tervezése volt. Zseniális módon oldotta meg a konvolúciós kernelek kiválasztásának problémáját. Míg más munkák arra törekedtek, hogy meghatározzák, melyik konvolúció lenne a legjobb az $1 \times 1$-estől a $11 \times 11$-esig, ez egyszerűen *összefűzte* a többágú konvolúciókat.
Az alábbiakban a GoogLeNet kissé egyszerűsített változatát mutatjuk be: az eredeti tervezés számos trükköt tartalmazott a tanítás stabilizálására közbenső veszteségfüggvényeken keresztül, amelyeket a hálózat több rétegére alkalmaztak.
Ezekre már nincs szükség a jobb tanítási algoritmusok elérhetősége miatt.

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
from torch.nn import functional as F
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
from jax import numpy as jnp
import jax
```

## (**Inception Blokkok**)

A GoogLeNet alapvető konvolúciós blokkját *Inception blokknak* nevezzük, ami az *Inception* film "mélyebbre kell mennünk" memójából ered.

![Az Inception blokk szerkezete.](../img/inception.svg)
:label:`fig_inception`

Ahogy a :numref:`fig_inception` ábra mutatja, az Inception blokk négy párhuzamos ágból áll.
Az első három ág $1\times 1$, $3\times 3$ és $5\times 5$-ös ablakméretű konvolúciós rétegeket alkalmaz a különböző térbeli méretű információk kinyeréséhez.
A középső két ág egy $1\times 1$-es konvolúciót is hozzáad a bemenethez a csatornák számának csökkentése érdekében, így csökkentve a modell komplexitását.
A negyedik ág egy $3\times 3$-as max-pooling réteget alkalmaz, amelyet egy $1\times 1$-es konvolúciós réteg követ a csatornák számának megváltoztatásához.
Mind a négy ág megfelelő kitöltést alkalmaz, hogy a bemenetnek és a kimenetnek azonos magassága és szélessége legyen.
Végül az egyes ágak kimenetei a csatorna dimenzió mentén összefűzve alkotják a blokk kimenetét.
Az Inception blokk leggyakrabban hangolt hiperparaméterei a rétegenkénti kimeneti csatornák száma, azaz hogy hogyan osszuk el a kapacitást a különböző méretű konvolúciók között.

```{.python .input}
%%tab mxnet
class Inception(nn.Block):
    # c1--c4 az egyes ágak kimeneti csatornáinak száma
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 1. ág
        self.b1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # 2. ág
        self.b2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.b2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,
                              activation='relu')
        # 3. ág
        self.b3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.b3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2,
                              activation='relu')
        # 4. ág
        self.b4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.b4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

    def forward(self, x):
        b1 = self.b1_1(x)
        b2 = self.b2_2(self.b2_1(x))
        b3 = self.b3_2(self.b3_1(x))
        b4 = self.b4_2(self.b4_1(x))
        return np.concatenate((b1, b2, b3, b4), axis=1)
```

```{.python .input}
%%tab pytorch
class Inception(nn.Module):
    # c1--c4 az egyes ágak kimeneti csatornáinak száma
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 1. ág
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # 2. ág
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # 3. ág
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # 4. ág
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)
```

```{.python .input}
%%tab tensorflow
class Inception(tf.keras.Model):
    # c1--c4 az egyes ágak kimeneti csatornáinak száma
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        self.b1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')
        self.b2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.b2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same',
                                           activation='relu')
        self.b3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.b3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same',
                                           activation='relu')
        self.b4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.b4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')

    def call(self, x):
        b1 = self.b1_1(x)
        b2 = self.b2_2(self.b2_1(x))
        b3 = self.b3_2(self.b3_1(x))
        b4 = self.b4_2(self.b4_1(x))
        return tf.keras.layers.Concatenate()([b1, b2, b3, b4])
```

```{.python .input}
%%tab jax
class Inception(nn.Module):
    # `c1`--`c4` az egyes ágak kimeneti csatornáinak száma
    c1: int
    c2: tuple
    c3: tuple
    c4: int

    def setup(self):
        # 1. ág
        self.b1_1 = nn.Conv(self.c1, kernel_size=(1, 1))
        # 2. ág
        self.b2_1 = nn.Conv(self.c2[0], kernel_size=(1, 1))
        self.b2_2 = nn.Conv(self.c2[1], kernel_size=(3, 3), padding='same')
        # 3. ág
        self.b3_1 = nn.Conv(self.c3[0], kernel_size=(1, 1))
        self.b3_2 = nn.Conv(self.c3[1], kernel_size=(5, 5), padding='same')
        # 4. ág
        self.b4_1 = lambda x: nn.max_pool(x, window_shape=(3, 3),
                                          strides=(1, 1), padding='same')
        self.b4_2 = nn.Conv(self.c4, kernel_size=(1, 1))

    def __call__(self, x):
        b1 = nn.relu(self.b1_1(x))
        b2 = nn.relu(self.b2_2(nn.relu(self.b2_1(x))))
        b3 = nn.relu(self.b3_2(nn.relu(self.b3_1(x))))
        b4 = nn.relu(self.b4_2(self.b4_1(x)))
        return jnp.concatenate((b1, b2, b3, b4), axis=-1)
```

Hogy megértsük, miért működik ez a hálózat annyira jól, gondoljuk át a szűrők kombinációját.
Különböző méretű szűrőkkel vizsgálják a képet, ami azt jelenti, hogy különböző kiterjedésű részletek különböző méretű szűrőkkel hatékonyan felismerhetők.
Ugyanakkor különböző mennyiségű paramétert oszthatunk el a különböző szűrőknek.


## [**GoogLeNet Modell**]

Ahogy a :numref:`fig_inception_full` ábra mutatja, a GoogLeNet összesen 9 Inception blokkot használ, amelyek három csoportba rendezve max-poolinggal vannak elválasztva egymástól, és a fejben globális átlagos poolinggal generálja becsléseit.
Az Inception blokkok közötti max-pooling csökkenti a dimenzionalitást.
A törzsénél az első modul hasonló az AlexNet-hez és a LeNet-hez.

![A GoogLeNet architektúrája.](../img/inception-full-90.svg)
:label:`fig_inception_full`

Most darabonként implementálhatjuk a GoogLeNet-et. Kezdjük a törzzsel.
Az első modul egy 64 csatornás $7\times 7$-es konvolúciós réteget alkalmaz.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class GoogleNet(d2l.Classifier):
    def b1(self):
        if tab.selected('mxnet'):
            net = nn.Sequential()
            net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3,
                              activation='relu'),
                    nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            return net
        if tab.selected('pytorch'):
            return nn.Sequential(
                nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        if tab.selected('tensorflow'):
            return tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2,
                                          padding='same')])
```

```{.python .input}
%%tab jax
class GoogleNet(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10

    def setup(self):
        self.net = nn.Sequential([self.b1(), self.b2(), self.b3(), self.b4(),
                                  self.b5(), nn.Dense(self.num_classes)])

    def b1(self):
        return nn.Sequential([
                nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='same'),
                nn.relu,
                lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2),
                                      padding='same')])
```

A második modul két konvolúciós réteget alkalmaz: először egy 64 csatornás $1\times 1$-es konvolúciós réteget, amelyet egy $3\times 3$-as konvolúciós réteg követ, amely megháromszorozza a csatornák számát. Ez megfelel az Inception blokk második ágának, és lezárja a törzs tervezését. Ezen a ponton 192 csatornánk van.

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b2(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(nn.Conv2D(64, kernel_size=1, activation='relu'),
               nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
               nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if tab.selected('tensorflow'):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 1, activation='relu'),
            tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
    if tab.selected('jax'):
        return nn.Sequential([nn.Conv(64, kernel_size=(1, 1)),
                              nn.relu,
                              nn.Conv(192, kernel_size=(3, 3), padding='same'),
                              nn.relu,
                              lambda x: nn.max_pool(x, window_shape=(3, 3),
                                                    strides=(2, 2),
                                                    padding='same')])
```

A harmadik modul egymás után két teljes Inception blokkot kapcsol össze.
Az első Inception blokk kimeneti csatornáinak száma $64+128+32+32=256$. Ez a négy ág kimeneti csatornáinak $2:4:1:1$ arányát jelenti. Ennek eléréséhez először a bemeneti dimenziókat $\frac{1}{2}$-del és $\frac{1}{12}$-vel csökkentjük a második és harmadik ágban, hogy $96 = 192/2$, illetve $16 = 192/12$ csatornát kapjunk.

A második Inception blokk kimeneti csatornáinak száma $128+192+96+64=480$-ra nő, ami $128:192:96:64 = 4:6:3:2$ arányt jelent. Mint korábban, csökkenteni kell a közbenső dimenziók számát a második és harmadik csatornában. Elegendő $\frac{1}{2}$-es és $\frac{1}{8}$-os skálázás, ami $128$, illetve $32$ csatornát eredményez. Ezt rögzítik a következő `Inception` blokk konstruktorainak argumentumai.

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b3(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(Inception(64, (96, 128), (16, 32), 32),
               Inception(128, (128, 192), (32, 96), 64),
               nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(Inception(64, (96, 128), (16, 32), 32),
                             Inception(128, (128, 192), (32, 96), 64),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if tab.selected('tensorflow'):
        return tf.keras.models.Sequential([
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
    if tab.selected('jax'):
        return nn.Sequential([Inception(64, (96, 128), (16, 32), 32),
                              Inception(128, (128, 192), (32, 96), 64),
                              lambda x: nn.max_pool(x, window_shape=(3, 3),
                                                    strides=(2, 2),
                                                    padding='same')])
```

A negyedik modul bonyolultabb.
Öt Inception blokkot kapcsol egymás után, amelyek rendre $192+208+48+64=512$, $160+224+64+64=512$, $128+256+64+64=512$, $112+288+64+64=528$ és $256+320+128+128=832$ kimeneti csatornával rendelkeznek.
Az ezekhez az ágakhoz rendelt csatornák száma hasonló a harmadik moduléhoz: a $3\times 3$-as konvolúciós rétegű második ág adja a legtöbb kimeneti csatornát, ezt követi az csak $1\times 1$-es konvolúciós rétegű első ág, az $5\times 5$-ös konvolúciós rétegű harmadik ág és a $3\times 3$-as max-pooling rétegű negyedik ág.
A második és harmadik ág először csökkenti a csatornák számát a megadott arányban. Ezek az arányok kissé eltérnek a különböző Inception blokkokban.

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b4(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(Inception(192, (96, 208), (16, 48), 64),
                Inception(160, (112, 224), (24, 64), 64),
                Inception(128, (128, 256), (24, 64), 64),
                Inception(112, (144, 288), (32, 64), 64),
                Inception(256, (160, 320), (32, 128), 128),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(Inception(192, (96, 208), (16, 48), 64),
                             Inception(160, (112, 224), (24, 64), 64),
                             Inception(128, (128, 256), (24, 64), 64),
                             Inception(112, (144, 288), (32, 64), 64),
                             Inception(256, (160, 320), (32, 128), 128),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    if tab.selected('tensorflow'):
        return tf.keras.Sequential([
            Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
    if tab.selected('jax'):
        return nn.Sequential([Inception(192, (96, 208), (16, 48), 64),
                              Inception(160, (112, 224), (24, 64), 64),
                              Inception(128, (128, 256), (24, 64), 64),
                              Inception(112, (144, 288), (32, 64), 64),
                              Inception(256, (160, 320), (32, 128), 128),
                              lambda x: nn.max_pool(x, window_shape=(3, 3),
                                                    strides=(2, 2),
                                                    padding='same')])
```

Az ötödik modul két Inception blokkot tartalmaz $256+320+128+128=832$ és $384+384+128+128=1024$ kimeneti csatornával.
Az egyes ágakhoz rendelt csatornák száma megegyezik a harmadik és negyedik moduléval, de az adott értékek különböznek.
Megjegyezzük, hogy az ötödik blokkot követi a kimeneti réteg.
Ez a blokk globális átlagos pooling réteget alkalmaz, hogy minden csatorna magasságát és szélességét 1-re változtassa, ahogy a NiN-ben is.
Végül a kimenetet kétdimenziós tömbbé alakítjuk, amelyet egy teljesen összekötött réteg követ, amelynek kimenetei száma megegyezik a címkeosztályok számával.

```{.python .input}
%%tab all
@d2l.add_to_class(GoogleNet)
def b5(self):
    if tab.selected('mxnet'):
        net = nn.Sequential()
        net.add(Inception(256, (160, 320), (32, 128), 128),
                Inception(384, (192, 384), (48, 128), 128),
                nn.GlobalAvgPool2D())
        return net
    if tab.selected('pytorch'):
        return nn.Sequential(Inception(256, (160, 320), (32, 128), 128),
                             Inception(384, (192, 384), (48, 128), 128),
                             nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
    if tab.selected('tensorflow'):
        return tf.keras.Sequential([
            Inception(256, (160, 320), (32, 128), 128),
            Inception(384, (192, 384), (48, 128), 128),
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Flatten()])
    if tab.selected('jax'):
        return nn.Sequential([Inception(256, (160, 320), (32, 128), 128),
                              Inception(384, (192, 384), (48, 128), 128),
                              # A Flax nem biztosít GlobalAvgPool2D réteget
                              lambda x: nn.avg_pool(x,
                                                    window_shape=x.shape[1:3],
                                                    strides=x.shape[1:3],
                                                    padding='valid'),
                              lambda x: x.reshape((x.shape[0], -1))])
```

Most, hogy definiáltuk az összes `b1`-től `b5`-ig terjedő blokkot, már csak össze kell rakni őket egy teljes hálózattá.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(GoogleNet)
def __init__(self, lr=0.1, num_classes=10):
    super(GoogleNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.b1(), self.b2(), self.b3(), self.b4(), self.b5(),
                     nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.b1(), self.b2(), self.b3(), self.b4(),
                                 self.b5(), nn.LazyLinear(num_classes))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.Sequential([
            self.b1(), self.b2(), self.b3(), self.b4(), self.b5(),
            tf.keras.layers.Dense(num_classes)])
```

A GoogLeNet modell számítási szempontból összetett. Figyelemre méltó a viszonylag önkényes hiperparaméterek nagy száma a kiválasztott csatornák számát, a dimenzionalitáscsökkentés előtti blokkok számát, a kapacitás csatornák közötti relatív elosztását stb. illetően. Ennek nagy része abból fakad, hogy a GoogLeNet bevezetésekor még nem álltak rendelkezésre automatikus eszközök a hálózatok definiálásához vagy tervezési feltárásához. Például ma már magától értetődőnek tekintjük, hogy egy kompetens deep learning keretrendszer képes automatikusan kikövetkeztetni a bemeneti tenzorok dimenzióit. Akkoriban sok ilyen konfigurációt a kísérletezőnek kellett explicit módon megadnia, ami sokszor lassította az aktív kísérletezést. Ráadásul az automatikus feltáráshoz szükséges eszközök még fejlesztés alatt álltak, és a kezdeti kísérletek nagyrészt költséges nyers erő feltárásból, genetikus algoritmusokból és hasonló stratégiákból álltak.

Egyelőre az egyetlen módosítás, amelyet elvégzünk, az
[**a bemeneti magasság és szélesség 224-ről 96-ra csökkentése, hogy ésszerű tanítási időt kapjunk Fashion-MNIST-en.**]
Ez egyszerűsíti a számítást. Nézzük meg a kimeneti alak változásait a különböző modulok között.

```{.python .input}
%%tab mxnet, pytorch
model = GoogleNet().layer_summary((1, 1, 96, 96))
```

```{.python .input}
%%tab tensorflow, jax
model = GoogleNet().layer_summary((1, 96, 96, 1))
```

## [**Tanítás**]

Mint korábban, a Fashion-MNIST adathalmazt használjuk a modell tanításához. A tanítási eljárás meghívása előtt $96 \times 96$ pixeles felbontásra alakítjuk át.

```{.python .input}
%%tab mxnet, pytorch, jax
model = GoogleNet(lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = GoogleNet(lr=0.01)
    trainer.fit(model, data)
```

## Vita

A GoogLeNet egyik kulcstulajdonsága, hogy valójában *olcsóbban* számítható ki elődeinél, miközben jobb pontosságot is nyújt. Ez jelöli egy sokkal tudatosabb hálózattervezés kezdetét, amely egy hálózat kiértékelési költségét a hibák csökkentésével váltja fel. Ez jelöli a hálózattervezési hiperparaméterekkel való blokk szintű kísérletezés kezdetét is, bár ez akkoriban teljesen kézzel történt. Ezt a témát a :numref:`sec_cnn-design` részben fogjuk újra felvetni, amikor a hálózatstruktúra feltárási stratégiáit tárgyaljuk.

A következő szakaszokban számos tervezési választással találkozunk majd (pl. batchnormalizáció, reziduális kapcsolatok és csatorna-csoportosítás), amelyek lehetővé teszik a hálózatok jelentős javítását. Egyelőre büszkék lehetünk arra, hogy implementáltuk azt, ami minden bizonnyal az első igazán modern CNN.

## Feladatok

1. A GoogLeNet olyan sikeres volt, hogy több iteráción ment keresztül, fokozatosan javítva a sebességet és a pontosságot. Próbálj meg implementálni és futtatni néhányat ezek közül. Ide tartoznak a következők:
    1. Adj hozzá egy batchnormalizációs réteget :cite:`Ioffe.Szegedy.2015`, amint az a :numref:`sec_batch_norm` részben leírásra kerül.
    1. Végezz módosításokat az Inception blokkon (szélesség, konvolúciók választása és sorrendje), amint az :citet:`Szegedy.Vanhoucke.Ioffe.ea.2016` leírja.
    1. Alkalmazz label smoothingot a modell regularizálásához, amint az :citet:`Szegedy.Vanhoucke.Ioffe.ea.2016` leírja.
    1. Végezz további módosításokat az Inception blokkon reziduális kapcsolat hozzáadásával :cite:`Szegedy.Ioffe.Vanhoucke.ea.2017`, amint az a :numref:`sec_resnet` részben leírásra kerül.
1. Mi a minimális képméret, amelyre a GoogLeNet működéséhez szükség van?
1. Tervezhetsz-e olyan GoogLeNet-változatot, amely a Fashion-MNIST natív $28 \times 28$ pixeles felbontásán működik? Hogyan kellene megváltoztatnod a hálózat törzsét, testét és fejét, ha egyáltalán szükséges?
1. Hasonlítsd össze az AlexNet, a VGG, a NiN és a GoogLeNet modellparaméter-méreteit. Hogyan csökkenti az utóbbi két hálózati architektúra jelentősen a modellparaméter-méretet?
1. Hasonlítsd össze a GoogLeNet-ben és az AlexNet-ben szükséges számítási mennyiséget. Hogyan befolyásolja ez egy gyorsítóchip tervezését, pl. a memóriaméret, a memória sávszélesség, a gyorsítótár mérete, a számítási mennyiség és a specializált műveletek haszna szempontjából?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/81)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/82)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/316)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18004)
:end_tab:
