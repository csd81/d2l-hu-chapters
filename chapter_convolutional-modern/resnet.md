```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Reziduális Hálózatok (ResNet) és ResNeXt
:label:`sec_resnet`

Ahogy egyre mélyebb hálózatokat tervezünk, elengedhetetlenné válik megérteni, hogy a rétegek hozzáadása hogyan növeli a hálózat összetettségét és kifejező erejét.
Még fontosabb az a képesség, hogy olyan hálózatokat tervezzünk, ahol a rétegek hozzáadása szigorúan kifejezőbbé teszi a hálózatot, nem csupán másféle funkcionalitást hoz.
A haladáshoz egy kis matematikára van szükségünk.

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

## Függvényosztályok

Tekintsük $\mathcal{F}$-et, azt a függvényosztályt, amelyet egy adott hálózati architektúra (a tanulási rátákkal és más hiperparaméter-beállításokkal együtt) elérhet.
Vagyis minden $f \in \mathcal{F}$ esetén létezik egy paraméterállomány (pl. súlyok és torzítások), amelyek egy megfelelő adathalmazon való tanítással megszerezhetők.
Tegyük fel, hogy $f^*$ az az "igaz" függvény, amelyet valóban meg szeretnénk találni.
Ha $\mathcal{F}$-ben van, akkor jó helyzetben vagyunk, de jellemzően nem vagyunk ilyen szerencsések.
Ehelyett megpróbálunk találni egy $f^*_\mathcal{F}$-et, amely a legjobb megközelítésünk $\mathcal{F}$-en belül.
Például, adott egy $\mathbf{X}$ jellemzőjű és $\mathbf{y}$ címkéjű adathalmaz esetén megpróbálhatjuk megtalálni a következő optimalizálási probléma megoldásával:

$$f^*_\mathcal{F} \stackrel{\textrm{def}}{=} \mathop{\mathrm{argmin}}_f L(\mathbf{X}, \mathbf{y}, f) \textrm{ subject to } f \in \mathcal{F}.$$

Tudjuk, hogy a regularizáció :cite:`tikhonov1977solutions,morozov2012methods` szabályozhatja $\mathcal{F}$ összetettségét és konzisztenciát érhet el, így egy nagyobb tanítóadathalmaz általában jobb $f^*_\mathcal{F}$-hez vezet.
Ésszerű feltételezni, hogy ha egy másik és erőteljesebb $\mathcal{F}'$ architektúrát tervezünk, jobb eredményt kapunk. Más szóval, elvárjuk, hogy $f^*_{\mathcal{F}'}$ "jobb" legyen, mint $f^*_{\mathcal{F}}$. Ha azonban $\mathcal{F} \not\subseteq \mathcal{F}'$, nincs garancia arra, hogy ez be is következik. Sőt, $f^*_{\mathcal{F}'}$ akár rosszabb is lehet.
Ahogy a :numref:`fig_functionclasses` ábra szemlélteti, nem egymásba ágyazott függvényosztályok esetén egy nagyobb függvényosztály nem mindig kerül közelebb az "igaz" $f^*$ függvényhez. Például a :numref:`fig_functionclasses` bal oldalán, bár $\mathcal{F}_3$ közelebb van $f^*$-hoz, mint $\mathcal{F}_1$, $\mathcal{F}_6$ távolabb kerül, és nincs garancia arra, hogy a komplexitás növelése csökkenti a távolságot $f^*$-tól.
A $\mathcal{F}_1 \subseteq \cdots \subseteq \mathcal{F}_6$ egymásba ágyazott függvényosztályokkal a :numref:`fig_functionclasses` jobb oldalán elkerülhetjük a nem egymásba ágyazott függvényosztályok előbb említett problémáját.


![Nem egymásba ágyazott függvényosztályoknál egy nagyobb (területtel jelzett) függvényosztály nem garantálja, hogy közelebb kerülünk az "igaz" $\mathit{f}^*$ függvényhez. Egymásba ágyazott függvényosztályoknál ez nem történik meg.](../img/functionclasses.svg)
:label:`fig_functionclasses`

Ezért csak akkor garantált, hogy a hálózat kifejező erejét szigorúan növeljük a nagyobb függvényosztályokkal, ha a nagyobb függvényosztályok tartalmazzák a kisebb osztályokat.
Mély neurális hálózatoknál, ha az újonnan hozzáadott réteget identitásfüggvénnyé ($f(\mathbf{x}) = \mathbf{x}$) tudjuk tanítani, az új modell ugyanolyan hatékony lesz, mint az eredeti. Mivel az új modell jobb megoldást kaphat a tanítóadathalmazra való illesztéshez, a hozzáadott réteg megkönnyítheti a tanítási hibák csökkentését.

Ezt a kérdést vizsgálta :citet:`He.Zhang.Ren.ea.2016`, amikor nagyon mély számítógépes látási modelleken dolgoztak.
Az általuk javasolt *reziduális hálózat* (*ResNet*) lényege az az gondolat, hogy minden egyes hozzáadott rétegnek könnyebben tartalmaznia kell az identitásfüggvényt elemei egyikeként.
Ezek a megfontolások meglehetősen mélyek, de meglepően egyszerű megoldáshoz vezettek: a *reziduális blokkhoz*.
Segítségével a ResNet megnyerte az ImageNet Large Scale Visual Recognition Challenge-t 2015-ben. A tervezés mélyen befolyásolta a mély neurális hálózatok építésének módját. Például reziduális blokkokat adtak a visszatérő hálózatokhoz :cite:`prakash2016neural,kim2017residual`. Hasonlóképpen, a Transformerek :cite:`Vaswani.Shazeer.Parmar.ea.2017` azokat alkalmazzák sok hálózatréteg hatékony egymásra rakásához. Gráf neurális hálózatokban :cite:`Kipf.Welling.2016` is alkalmazzák, és alapvető fogalomként széles körben alkalmazták a számítógépes látásban :cite:`Redmon.Farhadi.2018,Ren.He.Girshick.ea.2015`.
Megjegyezzük, hogy a reziduális hálózatokat megelőzik az autópálya-hálózatok :cite:`srivastava2015highway`, amelyek osztoznak egyes motivációkban, de az identitásfüggvény körüli elegáns parametrizáció nélkül.


## (**Reziduális Blokkok**)
:label:`subsec_residual-blks`

Koncentráljunk egy neurális hálózat helyi részére, ahogy a :numref:`fig_residual_block` ábra mutatja. Jelöljük a bemenetet $\mathbf{x}$-szel.
Feltételezzük, hogy $f(\mathbf{x})$, a tanulással elérni kívánt alapleképezés, bemenetként szolgál a tetején lévő aktivációs függvényhez.
A bal oldalon a szaggatott vonalú doboz belsejében lévő résznek közvetlenül kell megtanulnia $f(\mathbf{x})$-et.
A jobb oldalon a szaggatott vonalú doboz belsejében lévő résznek kell megtanulnia a *reziduális leképezést* $g(\mathbf{x}) = f(\mathbf{x}) - \mathbf{x}$, ebből fakad a reziduális blokk neve.
Ha az $f(\mathbf{x}) = \mathbf{x}$ identitásleképezés a kívánt alapleképezés, a reziduális leképezés $g(\mathbf{x}) = 0$, és így könnyebb megtanulni: csak a szaggatott vonalú dobozban lévő felső súlyréteg (pl. teljesen összekötött réteg és konvolúciós réteg) súlyait és torzításait kell nullára tolnunk.
A jobb oldali ábra szemlélteti a ResNet *reziduális blokkját*, ahol a rétegebemenetet $\mathbf{x}$-et az összeadó operátorhoz vezető folytonos vonal *reziduális kapcsolatnak* (vagy *átugrási kapcsolatnak*) nevezünk.
Reziduális blokkokkal a bemenetek gyorsabban terjeszkedhetnek előre a reziduális kapcsolatokon keresztül a rétegeken át.
Valójában a reziduális blokk a többágú Inception blokk speciális esetének tekinthető: két ággal rendelkezik, amelyek egyike az identitásleképezés.

![Egy szabályos blokkban (bal) a szaggatott vonalú dobozban lévő résznek közvetlenül kell megtanulnia az $\mathit{f}(\mathbf{x})$ leképezést. Egy reziduális blokkban (jobb) a szaggatott vonalú dobozban lévő résznek a reziduális leképezést $\mathit{g}(\mathbf{x}) = \mathit{f}(\mathbf{x}) - \mathbf{x}$ kell megtanulnia, ezáltal az identitásleképezés $\mathit{f}(\mathbf{x}) = \mathbf{x}$ tanulása könnyebbé válik.](../img/residual-block.svg)
:label:`fig_residual_block`


A ResNet a VGG teljes $3\times 3$-as konvolúciós réteg tervezését alkalmazza. A reziduális blokk két $3\times 3$-as konvolúciós rétegből áll, azonos számú kimeneti csatornával. Minden konvolúciós réteget batch normalizációs réteg és ReLU aktivációs függvény követ. Ezután kihagyjuk ezt a két konvolúciós műveletet, és a bemenetet közvetlenül adjuk hozzá a végső ReLU aktivációs függvény előtt.
Ez a fajta tervezés megköveteli, hogy a két konvolúciós réteg kimenete ugyanolyan alakú legyen, mint a bemenet, hogy össze lehessen őket adni. Ha módosítani szeretnénk a csatornák számát, be kell vezetnünk egy további $1\times 1$-es konvolúciós réteget a bemenet kívánt alakra való transzformálásához az összeadási művelethez. Nézzük meg az alábbi kódot.

```{.python .input}
%%tab mxnet
class Residual(nn.Block):  #@save
    """A ResNet modellek reziduális blokkja."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return npx.relu(Y + X)
```

```{.python .input}
%%tab pytorch
class Residual(nn.Module):  #@save
    """A ResNet modellek reziduális blokkja."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```

```{.python .input}
%%tab tensorflow
class Residual(tf.keras.Model):  #@save
    """A ResNet modellek reziduális blokkja."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(num_channels, padding='same',
                                            kernel_size=3, strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                            padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                                                strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)
```

```{.python .input}
%%tab jax
class Residual(nn.Module):  #@save
    """A ResNet modellek reziduális blokkja."""
    num_channels: int
    use_1x1conv: bool = False
    strides: tuple = (1, 1)
    training: bool = True

    def setup(self):
        self.conv1 = nn.Conv(self.num_channels, kernel_size=(3, 3),
                             padding='same', strides=self.strides)
        self.conv2 = nn.Conv(self.num_channels, kernel_size=(3, 3),
                             padding='same')
        if self.use_1x1conv:
            self.conv3 = nn.Conv(self.num_channels, kernel_size=(1, 1),
                                 strides=self.strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm(not self.training)
        self.bn2 = nn.BatchNorm(not self.training)

    def __call__(self, X):
        Y = nn.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return nn.relu(Y)
```

Ez a kód kétféle hálózatot generál: egyet, ahol a bemenetet hozzáadjuk a kimenethez a ReLU nemlinearitás alkalmazása előtt, ha `use_1x1conv=False`; és egyet, ahol $1 \times 1$-es konvolúció segítségével állítjuk be a csatornákat és a felbontást az összeadás előtt. Ezt szemlélteti a :numref:`fig_resnet_block` ábra.

![ResNet blokk $1 \times 1$-es konvolúcióval és anélkül, amely a bemenetet a kívánt alakra transzformálja az összeadási művelethez.](../img/resnet-block.svg)
:label:`fig_resnet_block`

Most nézzük meg [**azt az esetet, amikor a bemenet és a kimenet azonos alakú**], ahol $1 \times 1$-es konvolúcióra nincs szükség.

```{.python .input}
%%tab mxnet, pytorch
if tab.selected('mxnet'):
    blk = Residual(3)
    blk.initialize()
if tab.selected('pytorch'):
    blk = Residual(3)
X = d2l.randn(4, 3, 6, 6)
blk(X).shape
```

```{.python .input}
%%tab tensorflow
blk = Residual(3)
X = d2l.normal((4, 6, 6, 3))
Y = blk(X)
Y.shape
```

```{.python .input}
%%tab jax
blk = Residual(3)
X = jax.random.normal(d2l.get_key(), (4, 6, 6, 3))
blk.init_with_output(d2l.get_key(), X)[0].shape
```

Lehetőségünk van [**a kimeneti magasság és szélesség felezésére, miközben a kimeneti csatornák számát növeljük**].
Ebben az esetben $1 \times 1$-es konvolúciókat használunk `use_1x1conv=True` segítségével. Ez hasznos az egyes ResNet blokkok elején a térbeli dimenzionalitás csökkentéséhez `strides=2` alkalmazásával.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
blk = Residual(6, use_1x1conv=True, strides=2)
if tab.selected('mxnet'):
    blk.initialize()
blk(X).shape
```

```{.python .input}
%%tab jax
blk = Residual(6, use_1x1conv=True, strides=(2, 2))
blk.init_with_output(d2l.get_key(), X)[0].shape
```

## [**ResNet Modell**]

A ResNet első két rétege megegyezik a korábban leírt GoogLeNet rétegével: a 64 kimeneti csatornás és 2-es lépésközű $7\times 7$-es konvolúciós réteget a $3\times 3$-as max-pooling réteg követi 2-es lépésközzel. A különbség az, hogy a ResNet-ben minden konvolúciós réteg után batch normalizációs réteget adtak hozzá.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class ResNet(d2l.Classifier):
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
                tf.keras.layers.Conv2D(64, kernel_size=7, strides=2,
                                       padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2,
                                          padding='same')])
```

```{.python .input}
%%tab jax
class ResNet(d2l.Classifier):
    arch: tuple
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = self.create_net()

    def b1(self):
        return nn.Sequential([
            nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='same'),
            nn.BatchNorm(not self.training), nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2),
                                  padding='same')])
```

A GoogLeNet Inception blokkokból álló négy modult használ.
A ResNet viszont reziduális blokkokból álló négy modult alkalmaz, amelyek mindegyike több, azonos számú kimeneti csatornájú reziduális blokkot tartalmaz.
Az első modul csatornáinak száma megegyezik a bemeneti csatornák számával. Mivel egy 2-es lépésközű max-pooling réteget már alkalmaztunk, nem szükséges csökkenteni a magasságot és a szélességet. Az egyes következő modulok első reziduális blokkjában a csatornák száma megduplázódik az előző modulhoz képest, és a magasság és szélesség felezésre kerül.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels))
    return nn.Sequential(*blk)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = tf.keras.models.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
```

```{.python .input}
%%tab jax
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(num_channels, use_1x1conv=True,
                                strides=(2, 2), training=self.training))
        else:
            blk.append(Residual(num_channels, training=self.training))
    return nn.Sequential(blk)
```

Ezután hozzáadjuk az összes modult a ResNet-hez. Itt minden modulhoz két reziduális blokkot alkalmazunk. Végül, akárcsak a GoogLeNet-nél, globális átlagos pooling réteget adunk hozzá, amelyet a teljesen összekötött réteg kimenete követ.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(ResNet)
def __init__(self, arch, lr=0.1, num_classes=10):
    super(ResNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.b1())
        for i, b in enumerate(arch):
            self.net.add(self.block(*b, first_block=(i==0)))
        self.net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i+2}', self.block(*b, first_block=(i==0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add(self.block(*b, first_block=(i==0)))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(units=num_classes)]))
```

```{.python .input}
# %%tab jax
@d2l.add_to_class(ResNet)
def create_net(self):
    net = nn.Sequential([self.b1()])
    for i, b in enumerate(self.arch):
        net.layers.extend([self.block(*b, first_block=(i==0))])
    net.layers.extend([nn.Sequential([
        # A Flax nem biztosít GlobalAvg2D réteget
        lambda x: nn.avg_pool(x, window_shape=x.shape[1:3],
                              strides=x.shape[1:3], padding='valid'),
        lambda x: x.reshape((x.shape[0], -1)),
        nn.Dense(self.num_classes)])])
    return net
```

Minden modulban négy konvolúciós réteg van (kivéve a $1\times 1$-es konvolúciós réteget). Az első $7\times 7$-es konvolúciós réteggel és a végső teljesen összekötött réteggel együtt összesen 18 réteg van. Ezért ezt a modellt általában ResNet-18-nak nevezik.
A modulban lévő csatornák és reziduális blokkok számának különböző beállításával különböző ResNet-modelleket hozhatunk létre, például a mélyebb, 152 rétegű ResNet-152-t. Bár a ResNet fő architektúrája hasonló a GoogLeNet-hez, a ResNet struktúrája egyszerűbb és könnyebben módosítható. Mindezek a tényezők a ResNet gyors és széles körű elterjedéséhez vezettek. A :numref:`fig_resnet18` ábra mutatja a teljes ResNet-18-at.

![A ResNet-18 architektúrája.](../img/resnet18-90.svg)
:label:`fig_resnet18`

A ResNet tanítása előtt [**nézzük meg, hogyan változik a bemeneti alak a ResNet különböző moduljaiban**]. Mint az összes korábbi architektúrában, a felbontás csökken, míg a csatornák száma nő egészen addig, amíg egy globális átlagos pooling réteg aggregálja az összes jellemzőt.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class ResNet18(ResNet):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)),
                       lr, num_classes)
```

```{.python .input}
%%tab jax
class ResNet18(ResNet):
    arch: tuple = ((2, 64), (2, 128), (2, 256), (2, 512))
    lr: float = 0.1
    num_classes: int = 10
```

```{.python .input}
%%tab pytorch, mxnet
ResNet18().layer_summary((1, 1, 96, 96))
```

```{.python .input}
%%tab tensorflow
ResNet18().layer_summary((1, 96, 96, 1))
```

```{.python .input}
%%tab jax
ResNet18(training=False).layer_summary((1, 96, 96, 1))
```

## [**Tanítás**]

A ResNet-et a Fashion-MNIST adathalmazon tanítjuk, akárcsak korábban. A ResNet meglehetősen erős és rugalmas architektúra. A tanítási és validációs veszteséget rögzítő ábra jelentős különbséget mutat a két görbe között, a tanítási veszteség jóval alacsonyabb. Egy ilyen rugalmas hálózatnál több tanítóadat egyértelműen hasznos lenne a különbség csökkentésében és a pontosság javításában.

```{.python .input}
%%tab mxnet, pytorch, jax
model = ResNet18(lr=0.01)
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
    model = ResNet18(lr=0.01)
    trainer.fit(model, data)
```

## ResNeXt
:label:`subsec_resnext`

A ResNet tervezésénél felmerülő egyik kihívás a nemlinearitás és a dimenzionalitás kompromisszuma egy adott blokkon belül. Vagyis növelhetjük a nemlinearitást a rétegek számának növelésével vagy a konvolúciók szélességének növelésével. Egy alternatív stratégia a blokkok között információt hordozó csatornák számának növelése. Sajnos az utóbbi négyzetes büntetéssel jár, mivel a $c_\textrm{i}$ csatorna feldolgozásának és a $c_\textrm{o}$ csatorna kibocsátásának számítási költsége arányos $\mathcal{O}(c_\textrm{i} \cdot c_\textrm{o})$-val (lásd megbeszélésünket a :numref:`sec_channels` részben).

Inspirációt meríthetünk a :numref:`fig_inception` Inception blokkjából, amelyben az információ külön csoportokban áramlik a blokkon keresztül. A :numref:`fig_resnet_block` ResNet blokkjára alkalmazva a több független csoport ötletét a ResNeXt tervezéséhez vezetett :cite:`Xie.Girshick.Dollar.ea.2017`.
Az Inception transzformációinak kavalkádjától eltérően a ResNeXt *ugyanazt* a transzformációt alkalmazza az összes ágban, így minimalizálva az egyes ágak manuális hangolásának szükségességét.

![A ResNeXt blokk. A $\mathit{g}$ csoportos csoportos konvolúció alkalmazása $\mathit{g}$-szer gyorsabb, mint a sűrű konvolúció. Bottleneck reziduális blokk, ha a közbenső csatornák $\mathit{b}$ száma kisebb, mint $\mathit{c}$.](../img/resnext-block.svg)
:label:`fig_resnext_block`

Egy $c_\textrm{i}$-ről $c_\textrm{o}$ csatornára való konvolúció $g$ darab $c_\textrm{i}/g$ méretű csoportra való felosztása, amelyek $g$ darab $c_\textrm{o}/g$ méretű kimenetet állítanak elő, igen találóan *csoportos konvolúciónak* (*grouped convolution*) nevezik. A számítási költség (arányosan) $\mathcal{O}(c_\textrm{i} \cdot c_\textrm{o})$-ról $\mathcal{O}(g \cdot (c_\textrm{i}/g) \cdot (c_\textrm{o}/g)) = \mathcal{O}(c_\textrm{i} \cdot c_\textrm{o} / g)$-re csökken, azaz $g$-szer gyorsabb. Sőt, a kimenet generálásához szükséges paraméterek száma is csökken egy $c_\textrm{i} \times c_\textrm{o}$ méretű mátrixról $g$ darab $(c_\textrm{i}/g) \times (c_\textrm{o}/g)$ méretű kisebb mátrixra, ismét $g$-szeres csökkentéssel. A továbbiakban feltételezzük, hogy mind $c_\textrm{i}$, mind $c_\textrm{o}$ osztható $g$-vel.

Ebben a tervezésben az egyetlen kihívás az, hogy a $g$ csoport között nem történik információcsere. A :numref:`fig_resnext_block` ResNeXt blokkja ezt két módon korrigálja: a $3 \times 3$-as kernelű csoportos konvolúció két $1 \times 1$-es konvolúció közé kerül "szendvicsbe". A második a csatornák számának visszaállítására is szolgál. Az előny az, hogy csak $\mathcal{O}(c \cdot b)$ költséget fizetünk az $1 \times 1$-es kernelekért, és megelégedhetünk $\mathcal{O}(b^2 / g)$ költséggel a $3 \times 3$-as kernelekért. Hasonlóan a :numref:`subsec_residual-blks` reziduális blokk implementációjához, a reziduális kapcsolatot egy $1 \times 1$-es konvolúció váltja fel (és általánosítja).

A :numref:`fig_resnext_block` jobb oldali ábra sokkal tömörebb összefoglalót nyújt az eredményül kapott hálóblokkról. Döntő szerepet fog játszani az általános modern CNN-ek tervezésében is a :numref:`sec_cnn-design` részben. Megjegyezzük, hogy a csoportos konvolúciók ötlete visszanyúlik az AlexNet implementációjáig :cite:`Krizhevsky.Sutskever.Hinton.2012`. Amikor a hálózatot két korlátozott memóriájú GPU-ra osztotta el az implementáció, az egyes GPU-kat saját csatornájukként kezelte káros mellékhatások nélkül.

A `ResNeXtBlock` osztály következő implementációja argumentumként fogadja a `groups` ($g$) paramétert és `bot_channels` ($b$) közbenső (bottleneck) csatornákat. Végül, amikor a reprezentáció magasságát és szélességét kell csökkenteni, 2-es lépésközt adunk hozzá a `use_1x1conv=True, strides=2` beállítással.

```{.python .input}
%%tab mxnet
class ResNeXtBlock(nn.Block):  #@save
    """A ResNeXt blokk."""
    def __init__(self, num_channels, groups, bot_mul,
                 use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.Conv2D(bot_channels, kernel_size=1, padding=0,
                               strides=1)
        self.conv2 = nn.Conv2D(bot_channels, kernel_size=3, padding=1, 
                               strides=strides, groups=bot_channels//groups)
        self.conv3 = nn.Conv2D(num_channels, kernel_size=1, padding=0,
                               strides=1)
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()
        self.bn3 = nn.BatchNorm()
        if use_1x1conv:
            self.conv4 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
            self.bn4 = nn.BatchNorm()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = npx.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return npx.relu(Y + X)
```

```{.python .input}
%%tab pytorch
class ResNeXtBlock(nn.Module):  #@save
    """A ResNeXt blokk."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3,
                                   stride=strides, padding=1,
                                   groups=bot_channels//groups)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1, 
                                       stride=strides)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)
```

```{.python .input}
%%tab tensorflow
class ResNeXtBlock(tf.keras.Model):  #@save
    """A ResNeXt blokk."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = tf.keras.layers.Conv2D(bot_channels, 1, strides=1)
        self.conv2 = tf.keras.layers.Conv2D(bot_channels, 3, strides=strides,
                                            padding="same",
                                            groups=bot_channels//groups)
        self.conv3 = tf.keras.layers.Conv2D(num_channels, 1, strides=1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        if use_1x1conv:
            self.conv4 = tf.keras.layers.Conv2D(num_channels, 1,
                                                strides=strides)
            self.bn4 = tf.keras.layers.BatchNormalization()
        else:
            self.conv4 = None

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = tf.keras.activations.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return tf.keras.activations.relu(Y + X)
```

```{.python .input}
%%tab jax
class ResNeXtBlock(nn.Module):  #@save
    """A ResNeXt blokk."""
    num_channels: int
    groups: int
    bot_mul: int
    use_1x1conv: bool = False
    strides: tuple = (1, 1)
    training: bool = True

    def setup(self):
        bot_channels = int(round(self.num_channels * self.bot_mul))
        self.conv1 = nn.Conv(bot_channels, kernel_size=(1, 1),
                               strides=(1, 1))
        self.conv2 = nn.Conv(bot_channels, kernel_size=(3, 3),
                               strides=self.strides, padding='same',
                               feature_group_count=bot_channels//self.groups)
        self.conv3 = nn.Conv(self.num_channels, kernel_size=(1, 1),
                               strides=(1, 1))
        self.bn1 = nn.BatchNorm(not self.training)
        self.bn2 = nn.BatchNorm(not self.training)
        self.bn3 = nn.BatchNorm(not self.training)
        if self.use_1x1conv:
            self.conv4 = nn.Conv(self.num_channels, kernel_size=(1, 1),
                                       strides=self.strides)
            self.bn4 = nn.BatchNorm(not self.training)
        else:
            self.conv4 = None

    def __call__(self, X):
        Y = nn.relu(self.bn1(self.conv1(X)))
        Y = nn.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return nn.relu(Y + X)
```

Alkalmazása teljesen analóg a korábban tárgyalt `ResNetBlock`-éval. Például `use_1x1conv=False, strides=1` alkalmazásakor a bemenet és a kimenet azonos alakú. Alternatív módon a `use_1x1conv=True, strides=2` beállítás felezi a kimenet magasságát és szélességét.

```{.python .input}
%%tab mxnet, pytorch
blk = ResNeXtBlock(32, 16, 1)
if tab.selected('mxnet'):
    blk.initialize()
X = d2l.randn(4, 32, 96, 96)
blk(X).shape
```

```{.python .input}
%%tab tensorflow
blk = ResNeXtBlock(32, 16, 1)
X = d2l.normal((4, 96, 96, 32))
Y = blk(X)
Y.shape
```

```{.python .input}
%%tab jax
blk = ResNeXtBlock(32, 16, 1)
X = jnp.zeros((4, 96, 96, 32))
blk.init_with_output(d2l.get_key(), X)[0].shape
```

## Összefoglalás és Vita

Az egymásba ágyazott függvényosztályok kívánatosak, mivel lehetővé teszik, hogy a kapacitás növelésekor szigorúan *erőteljesebb*, ne csupán halkan *különböző* függvényosztályokat kapjunk. Ennek egyik módja az, hogy hagyjuk, hogy a kiegészítő rétegek egyszerűen átadják a bemenetet a kimenetnek. A reziduális kapcsolatok ezt teszik lehetővé. Ennek következtében ez megváltoztatja az induktív torzítást az egyszerű $f(\mathbf{x}) = 0$ alakú függvényekről az $f(\mathbf{x}) = \mathbf{x}$ kinézetű egyszerű függvényekre.

A reziduális leképezés könnyebben megtanulhatja az identitásfüggvényt, például a súlyrétegben lévő paraméterek nullára tolásával. Reziduális blokkok alkalmazásával hatékony *mély* neurális hálózatot taníthatunk. A bemenetek gyorsabban terjeszkedhetnek előre a reziduális kapcsolatokon keresztül a rétegeken át. Ennek következtében sokkal mélyebb hálózatokat taníthatunk. Például az eredeti ResNet cikk :cite:`He.Zhang.Ren.ea.2016` akár 152 réteg alkalmazását is lehetővé tette. A reziduális hálózatok másik előnye, hogy lehetővé teszi rétegek hozzáadását, identitásfüggvényként inicializálva, *a* tanítási folyamat *alatt*. Elvégre egy réteg alapértelmezett viselkedése az adatok változatlan átengedése. Ez egyes esetekben felgyorsíthatja a nagyon nagy hálózatok tanítását.

A reziduális kapcsolatok előtt kapuegységekkel rendelkező megkerülő útvonalakat vezettek be a 100 rétegnél mélyebb autópálya-hálózatok hatékony tanításához :cite:`srivastava2015highway`.
Identitásfüggvények megkerülő útvonalként való alkalmazásával a ResNet kiváló teljesítményt nyújtott több számítógépes látási feladatban.
A reziduális kapcsolatok komoly hatással voltak a későbbi mély neurális hálózatok tervezésére, akár konvolúciós, akár szekvenciális jellegű hálózatokra.
Ahogy a későbbiekben bemutatjuk, a Transformer architektúra :cite:`Vaswani.Shazeer.Parmar.ea.2017` reziduális kapcsolatokat (más tervezési döntésekkel együtt) alkalmaz, és elterjedt a nyelvészet, a látás, a hangfeldolgozás és a megerősítéses tanulás területein.

A ResNeXt egy példa arra, hogyan fejlődött idővel a konvolúciós neurális hálózatok tervezése: a számítással való takarékossággal és azt az aktivációk méretével (csatornák száma) való felcserélésével gyorsabb és pontosabb hálózatokat tesz lehetővé alacsonyabb költséggel. A csoportos konvolúciók egy alternatív értelmezési módja a konvolúciós súlyokra vonatkozó blokk-átlós mátrix. Megjegyezzük, hogy elég sok ilyen "trükk" létezik, amelyek hatékonyabb hálózatokhoz vezetnek. Például a ShiftNet :cite:`wu2018shift` a $3 \times 3$-as konvolúció hatásait utánozza pusztán eltolt aktivációk csatornákhoz való hozzáadásával, megnövelt függvénykomplexitást kínálva, ez alkalommal bármiféle számítási költség nélkül.

Az eddig tárgyalt tervek közös jellemzője, hogy a hálózattervezés meglehetősen kézi jellegű, elsősorban a tervező leleményességére támaszkodva a "megfelelő" hálózati hiperparaméterek megtalálásában. Bár ez egyértelműen megvalósítható, emberi időben is nagyon költséges, és nincs garancia arra, hogy az eredmény bármilyen értelemben optimális. A :numref:`sec_cnn-design` részben számos stratégiát tárgyalunk a jó minőségű hálózatok automatikusabb módon való megszerzésére. Különösen felülvizsgáljuk a *hálózattervezési terek* fogalmát, amely a RegNetX/Y modellekhez :cite:`Radosavovic.Kosaraju.Girshick.ea.2020` vezetett.

## Feladatok

1. Melyek a fő különbségek a :numref:`fig_inception` Inception blokkja és a reziduális blokk között? Hogyan hasonlítanak egymáshoz a számítás, a pontosság és az általuk leírható függvényosztályok szempontjából?
1. A ResNet cikk 1. táblázatát :cite:`He.Zhang.Ren.ea.2016` felhasználva implementáld a hálózat különböző változatait.
1. Mélyebb hálózatoknál a ResNet "bottleneck" architektúrát vezet be a modell összetettségének csökkentése érdekében. Próbáld meg implementálni.
1. A ResNet későbbi verzióiban a szerzők a "konvolúció, batch normalizáció és aktiváció" struktúrát a "batch normalizáció, aktiváció és konvolúció" struktúrára változtatták. Hajtsd végre ezt a fejlesztést saját magad. A részletekért lásd az 1. ábrát :citet:`He.Zhang.Ren.ea.2016*1`-ben.
1. Miért nem növelhetjük a függvények összetettségét korlátlanul, még akkor sem, ha a függvényosztályok egymásba ágyazottak?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/85)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/86)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/8737)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18006)
:end_tab:
