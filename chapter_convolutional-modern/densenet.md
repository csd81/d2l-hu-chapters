```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Sűrűn Összekötött Hálózatok (DenseNet)
:label:`sec_densenet`

A ResNet jelentősen megváltoztatta azt a szemléletet, hogyan parametrizáljuk a függvényeket mély hálózatokban. A *DenseNet* (sűrű konvolúciós hálózat) ennek valamiféle logikus kiterjesztése :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`.
A DenseNet-et jellemzi mind az összekapcsolási minta, amelynél minden réteg csatlakozik az összes megelőző réteghez, mind az összefűzési művelet (a ResNet-ben alkalmazott összeadás helyett) a korábbi rétegek jellemzőinek megőrzéséhez és újrafelhasználásához.
Hogy megértsük, hogyan jutottunk el idáig, tegyünk egy kis matematikai kitérőt.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import init, np, npx
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
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
```

## A ResNet-től a DenseNet-ig

Idézzük fel a függvények Taylor-sorfejtését. Az $x = 0$ pontban felírható:

$$f(x) = f(0) + x \cdot \left[f'(0) + x \cdot \left[\frac{f''(0)}{2!}  + x \cdot \left[\frac{f'''(0)}{3!}  + \cdots \right]\right]\right].$$


A kulcspont az, hogy egy függvényt egyre magasabb rendű tagokra bont. Hasonlóképpen a ResNet a függvényeket a következőre bontja:

$$f(\mathbf{x}) = \mathbf{x} + g(\mathbf{x}).$$

Vagyis a ResNet $f$-et egy egyszerű lineáris tagra és egy bonyolultabb nemlineáris tagra bontja.
Mi lenne, ha (nem feltétlenül összeadva) két tagnál több információt szeretnénk megragadni?
Egy ilyen megoldás a DenseNet :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`.

![A ResNet (bal) és a DenseNet (jobb) fő különbsége a rétegek közötti kapcsolatokban: összeadás és összefűzés alkalmazása.](../img/densenet-block.svg)
:label:`fig_densenet_block`

Ahogy a :numref:`fig_densenet_block` ábra mutatja, a ResNet és a DenseNet fő különbsége az, hogy az utóbbi esetben a kimenetek *összefűzésre* kerülnek (jelölve $[,]$-vel), nem összeadásra.
Ennek eredményeként egy egyre összetettebb függvénysorozatot alkalmazunk $\mathbf{x}$-ről az értékeire:

$$\mathbf{x} \to \left[
\mathbf{x},
f_1(\mathbf{x}),
f_2\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right)\right]\right), f_3\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right), f_2\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right)\right]\right)\right]\right), \ldots\right].$$

A végén ezeket a függvényeket egy MLP kombinálja, hogy ismét csökkentsék a jellemzők számát. Implementálás szempontjából ez meglehetősen egyszerű: tagok összeadása helyett összefűzzük őket. A DenseNet neve abból fakad, hogy a változók közötti függőségi gráf meglehetősen sűrűvé válik. Egy ilyen lánc utolsó rétege sűrűn kapcsolódik az összes korábbi réteghez. A sűrű kapcsolatokat a :numref:`fig_densenet` ábra mutatja.

![Sűrű kapcsolatok a DenseNet-ben. Figyeljük meg, hogyan nő a dimenzionalitás a mélységgel.](../img/densenet.svg)
:label:`fig_densenet`

A DenseNet-et alkotó fő komponensek a *sűrű blokkok* és az *átmeneti rétegek*. Az előbbiek meghatározzák, hogyan fűzik össze a bemeneteket és a kimeneteket, míg az utóbbiak szabályozzák a csatornák számát, hogy ne legyen túl nagy, mivel a $\mathbf{x} \to \left[\mathbf{x}, f_1(\mathbf{x}), f_2\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right)\right]\right), \ldots \right]$ kiterjesztés meglehetősen nagy dimenziójú lehet.


## [**Sűrű Blokkok**]

A DenseNet a ResNet módosított "batchnormalizáció, aktiváció és konvolúció" struktúráját alkalmazza (lásd a :numref:`sec_resnet` feladatát).
Először ezt a konvolúciós blokk struktúrát implementáljuk.

```{.python .input}
%%tab mxnet
def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk
```

```{.python .input}
%%tab pytorch
def conv_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=3, padding=1))
```

```{.python .input}
%%tab tensorflow
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3), padding='same')

        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x,y], axis=-1)
        return y
```

```{.python .input}
%%tab jax
class ConvBlock(nn.Module):
    num_channels: int
    training: bool = True

    @nn.compact
    def __call__(self, X):
        Y = nn.relu(nn.BatchNorm(not self.training)(X))
        Y = nn.Conv(self.num_channels, kernel_size=(3, 3), padding=(1, 1))(Y)
        Y = jnp.concatenate((X, Y), axis=-1)
        return Y
```

Egy *sűrű blokk* több konvolúciós blokkból áll, amelyek mindegyike azonos számú kimeneti csatornát alkalmaz. Az előrepasszolásban azonban minden konvolúciós blokk bemenetét és kimenetét összefűzzük a csatorna dimenzión. A lusta kiértékelés lehetővé teszi a dimenzionalitás automatikus beállítását.

```{.python .input}
%%tab mxnet
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels):
        super().__init__()
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Összefűzzük az egyes blokkok bemenetét és kimenetét a csatornák mentén
            X = np.concatenate((X, Y), axis=1)
        return X
```

```{.python .input}
%%tab pytorch
class DenseBlock(nn.Module):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Összefűzzük az egyes blokkok bemenetét és kimenetét a csatornák mentén
            X = torch.cat((X, Y), dim=1)
        return X
```

```{.python .input}
%%tab tensorflow
class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x
```

```{.python .input}
%%tab jax
class DenseBlock(nn.Module):
    num_convs: int
    num_channels: int
    training: bool = True

    def setup(self):
        layer = []
        for i in range(self.num_convs):
            layer.append(ConvBlock(self.num_channels, self.training))
        self.net = nn.Sequential(layer)

    def __call__(self, X):
        return self.net(X)
```

A következő példában [**egy `DenseBlock` példányt definiálunk**] 10 kimeneti csatornájú két konvolúciós blokkal.
Három csatornás bemenet esetén $3 + 10 + 10=23$ csatornás kimenetet kapunk. A konvolúciós blokk csatornáinak száma szabályozza a kimeneti csatornák számának növekedését a bemeneti csatornák számához képest. Ezt *növekedési rátának* (*growth rate*) is nevezik.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
blk = DenseBlock(2, 10)
if tab.selected('mxnet'):
    X = np.random.uniform(size=(4, 3, 8, 8))
    blk.initialize()
if tab.selected('pytorch'):
    X = torch.randn(4, 3, 8, 8)
if tab.selected('tensorflow'):
    X = tf.random.uniform((4, 8, 8, 3))
Y = blk(X)
Y.shape
```

```{.python .input}
%%tab jax
blk = DenseBlock(2, 10)
X = jnp.zeros((4, 8, 8, 3))
Y = blk.init_with_output(d2l.get_key(), X)[0]
Y.shape
```

## [**Átmeneti Rétegek**]

Mivel minden sűrű blokk növeli a csatornák számát, túl sok hozzáadásuk túlzottan összetett modellt eredményezne. Egy *átmeneti réteg* a modell összetettségének szabályozására szolgál. Csökkenti a csatornák számát egy $1\times 1$-es konvolúció segítségével. Ezenfelül 2-es lépésközű átlagos poolinggal felezi a magasságot és a szélességet.

```{.python .input}
%%tab mxnet
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
%%tab pytorch
def transition_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
```

```{.python .input}
%%tab tensorflow
class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)
```

```{.python .input}
%%tab jax
class TransitionBlock(nn.Module):
    num_channels: int
    training: bool = True

    @nn.compact
    def __call__(self, X):
        X = nn.BatchNorm(not self.training)(X)
        X = nn.relu(X)
        X = nn.Conv(self.num_channels, kernel_size=(1, 1))(X)
        X = nn.avg_pool(X, window_shape=(2, 2), strides=(2, 2))
        return X
```

[**Alkalmazzunk egy 10 csatornás átmeneti réteget**] az előző példában szereplő sűrű blokk kimenetére. Ez 10-re csökkenti a kimeneti csatornák számát, és felezi a magasságot és szélességet.

```{.python .input}
%%tab mxnet
blk = transition_block(10)
blk.initialize()
blk(Y).shape
```

```{.python .input}
%%tab pytorch
blk = transition_block(10)
blk(Y).shape
```

```{.python .input}
%%tab tensorflow
blk = TransitionBlock(10)
blk(Y).shape
```

```{.python .input}
%%tab jax
blk = TransitionBlock(10)
blk.init_with_output(d2l.get_key(), Y)[0].shape
```

## [**DenseNet Modell**]

Következőként megépítünk egy DenseNet modellt. A DenseNet először ugyanazt az egyetlen konvolúciós réteget és max-pooling réteget alkalmazza, mint a ResNet.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class DenseNet(d2l.Classifier):
    def b1(self):
        if tab.selected('mxnet'):
            net = nn.Sequential()
            net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
                nn.BatchNorm(), nn.Activation('relu'),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            return net
        if tab.selected('pytorch'):
            return nn.Sequential(
                nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                nn.LazyBatchNorm2d(), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        if tab.selected('tensorflow'):
            return tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(
                    64, kernel_size=7, strides=2, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(
                    pool_size=3, strides=2, padding='same')])
```

```{.python .input}
%%tab jax
class DenseNet(d2l.Classifier):
    num_channels: int = 64
    growth_rate: int = 32
    arch: tuple = (4, 4, 4, 4)
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = self.create_net()

    def b1(self):
        return nn.Sequential([
            nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='same'),
            nn.BatchNorm(not self.training),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3),
                                  strides=(2, 2), padding='same')
        ])
```

Ezután, hasonlóan a ResNet által használt reziduális blokkokból álló négy modulhoz, a DenseNet négy sűrű blokkot alkalmaz.
A ResNet-hez hasonlóan beállíthatjuk az egyes sűrű blokkokban alkalmazott konvolúciós rétegek számát. Itt 4-re állítjuk, összhangban a :numref:`sec_resnet` részben szereplő ResNet-18 modellel. Ezenkívül a sűrű blokk konvolúciós rétegeinek csatornaszámát (azaz a növekedési rátát) 32-re állítjuk, így minden sűrű blokkhoz 128 csatorna adódik.

A ResNet-ben a magasság és szélesség minden modul között 2-es lépésközű reziduális blokk által csökken. Itt az átmeneti réteget alkalmazzuk a magasság és szélesség felezéséhez, és a csatornák számának felezéséhez. A ResNet-hez hasonlóan globális pooling réteget és teljesen összekötött réteget kapcsolunk a végén a kimenet előállításához.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(DenseNet)
def __init__(self, num_channels=64, growth_rate=32, arch=(4, 4, 4, 4),
             lr=0.1, num_classes=10):
    super(DenseNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add(DenseBlock(num_convs, growth_rate))
            # Az előző sűrű blokk kimeneti csatornáinak száma
            num_channels += num_convs * growth_rate
            # Az átmeneti réteg, amely felezi a csatornák számát,
            # a sűrű blokkok közé kerül
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add(transition_block(num_channels))
        self.net.add(nn.BatchNorm(), nn.Activation('relu'),
                     nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add_module(f'dense_blk{i+1}', DenseBlock(num_convs,
                                                              growth_rate))
            # Az előző sűrű blokk kimeneti csatornáinak száma
            num_channels += num_convs * growth_rate
            # Az átmeneti réteg, amely felezi a csatornák számát,
            # a sűrű blokkok közé kerül
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add_module(f'tran_blk{i+1}', transition_block(
                    num_channels))
        self.net.add_module('last', nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add(DenseBlock(num_convs, growth_rate))
            # Az előző sűrű blokk kimeneti csatornáinak száma
            num_channels += num_convs * growth_rate
            # Az átmeneti réteg, amely felezi a csatornák számát,
            # a sűrű blokkok közé kerül
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add(TransitionBlock(num_channels))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes)]))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(DenseNet)
def create_net(self):
    net = self.b1()
    for i, num_convs in enumerate(self.arch):
        net.layers.extend([DenseBlock(num_convs, self.growth_rate,
                                      training=self.training)])
        # Az előző sűrű blokk kimeneti csatornáinak száma
        num_channels = self.num_channels + (num_convs * self.growth_rate)
        # Az átmeneti réteg, amely felezi a csatornák számát,
        # a sűrű blokkok közé kerül
        if i != len(self.arch) - 1:
            num_channels //= 2
            net.layers.extend([TransitionBlock(num_channels,
                                               training=self.training)])
    net.layers.extend([
        nn.BatchNorm(not self.training),
        nn.relu,
        lambda x: nn.avg_pool(x, window_shape=x.shape[1:3],
                              strides=x.shape[1:3], padding='valid'),
        lambda x: x.reshape((x.shape[0], -1)),
        nn.Dense(self.num_classes)
    ])
    return net
```

## [**Tanítás**]

Mivel itt mélyebb hálózatot alkalmazunk, ebben a szakaszban a bemeneti magasságot és szélességet 224-ről 96-ra csökkentjük a számítás egyszerűsítése érdekében.

```{.python .input}
%%tab mxnet, pytorch, jax
model = DenseNet(lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = DenseNet(lr=0.01)
    trainer.fit(model, data)
```

## Összefoglalás és Vita

A DenseNet-et alkotó fő komponensek a sűrű blokkok és az átmeneti rétegek. Az utóbbiak esetén a hálózat összeállításakor a dimenzionalitást kézben kell tartani, átmeneti rétegek hozzáadásával, amelyek ismét csökkentik a csatornák számát.
A rétegek közötti kapcsolatok szempontjából, ellentétben a ResNet-tel, ahol a bemeneteket és a kimeneteket összeadják, a DenseNet összefűzi a bemeneteket és a kimeneteket a csatorna dimenzióban.
Bár ezek az összefűzési műveletek a jellemzők újrafelhasználásával számítási hatékonyságot érnek el, sajnos nagy GPU memóriafelhasználáshoz vezet.
Ennek következtében a DenseNet alkalmazása memóriahatékonyabb implementációkat igényelhet, amelyek növelhetik a tanítási időt :cite:`pleiss2017memory`.


## Feladatok

1. Miért átlagos poolingot alkalmazunk, és nem max-poolingot az átmeneti rétegben?
1. A DenseNet cikkben említett egyik előny az, hogy a modellparaméterek kisebbek, mint a ResNet-é. Miért van ez így?
1. Az egyik probléma, amelyért a DenseNet-et kritizálják, a magas memóriafelhasználása.
    1. Valóban ez a helyzet? Próbáld meg $224\times 224$-re változtatni a bemeneti alakot, hogy empirikusan összehasonlítsd a tényleges GPU memóriafelhasználást.
    1. Gondolsz valamilyen alternatív módszerre a memóriafelhasználás csökkentésére? Hogyan kellene módosítani a keretrendszert?
1. Implementáld a DenseNet cikk 1. táblázatában bemutatott különböző DenseNet verziókat :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`.
1. Tervezz MLP-alapú modellt a DenseNet ötletének alkalmazásával. Alkalmazd a :numref:`sec_kaggle_house` lakásárak előrejelzési feladatára.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/87)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/88)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/331)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18008)
:end_tab:
