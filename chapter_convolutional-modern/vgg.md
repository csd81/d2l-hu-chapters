```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Blokkokat Alkalmazó Hálózatok (VGG)
:label:`sec_vgg`

Bár az AlexNet empirikus bizonyítékot nyújtott arra, hogy a mély CNN-ek jó eredményeket érhetnek el, nem biztosított általános sablont az ezt követő kutatók számára az új hálózatok tervezéséhez.
A következő szakaszokban bemutatunk néhány heurisztikus koncepciót, amelyeket általánosan alkalmaznak mély hálózatok tervezésénél.

Az ezen a területen elért haladás tükrözi a VLSI (nagyon nagyméretű integráció) fejlődését a chiptervezésben, ahol a mérnökök a tranzisztorok elhelyezésétől az logikai elemekig, majd a logikai blokkokig jutottak :cite:`Mead.1980`.
Hasonlóképpen a neurális hálózati architektúrák tervezése egyre elvontabbá vált: a kutatók az egyes neuronoktól az egész rétegekig, majd a blokkokig, a rétegek ismétlődő mintáiig jutottak. Egy évtizeddel később ez odáig fejlődött, hogy a kutatók egész betanított modelleket használnak fel különböző, bár rokon feladatokra. Az ilyen nagy előre betanított modelleket általában *alapmodelleknek* (*foundation models*) :cite:`bommasani2021opportunities` nevezzük.

Vissza a hálózattervezéshez. A blokkok használatának ötlete először az Oxfordi Egyetem Visual Geometry Group (VGG) csoportjától származott, az általuk névadóan elnevezett *VGG* hálózatban :cite:`Simonyan.Zisserman.2014`.
Ezeket az ismétlődő struktúrákat könnyű megvalósítani kódban bármely modern deep learning keretrendszerrel hurkok és szubrutinok segítségével.

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
```

## (**VGG Blokkok**)
:label:`subsec_vgg-blocks`

A CNN-ek alapvető építőeleme a következő sorozat:
(i) konvolúciós réteg kitöltéssel a felbontás megőrzéséhez,
(ii) nemlinearitás, például ReLU,
(iii) pooling réteg, például max-pooling a felbontás csökkentéséhez. Az egyik probléma ezzel a megközelítéssel az, hogy a térbeli felbontás meglehetősen gyorsan csökken. Ez különösen kemény korlátot szab: legfeljebb $\log_2 d$ konvolúciós réteg használható, mielőtt az összes dimenzió ($d$) elfogy. Például ImageNet esetén így legfeljebb 8 konvolúciós réteget lehetne alkalmazni.

:citet:`Simonyan.Zisserman.2014` kulcsgondolata az volt, hogy *több* konvolúciót alkalmazzanak a max-pooling általi leskálázások között, egy blokk formájában. Elsősorban azt vizsgálták, hogy a mély vagy a széles hálózatok teljesítenek-e jobban. Például két egymás utáni $3 \times 3$-as konvolúció ugyanazokat a pixeleket érinti, mint egyetlen $5 \times 5$-ös konvolúció. Ugyanakkor az utóbbi körülbelül annyi paramétert használ ($25 \cdot c^2$), mint három $3 \times 3$-as konvolúció ($3 \cdot 9 \cdot c^2$).
Egy meglehetősen részletes elemzésben megmutatták, hogy a mély és keskeny hálózatok jelentősen felülmúlják sekély társaikat. Ez a deep learninget az egyre mélyebb hálózatok keresésére indította, tipikus alkalmazásoknál 100 réteg feletti mélységgel.
A $3 \times 3$-as konvolúciók egymásra halmozása a későbbi mély hálózatok aranystandardjává vált (ezt a tervezési döntést :citet:`liu2022convnet` csak a közelmúltban vizsgálta felül). Ennek következtében a kis konvolúciók gyors implementációi GPU-kon is elterjedtek :cite:`lavin2016fast`.

Visszatérve a VGG-hez: egy VGG blokk $3\times3$-as kernelű konvolúciók *sorozatából* áll, 1-es kitöltéssel (megőrizve a magasságot és a szélességet), amelyet egy $2 \times 2$-es max-pooling réteg követ 2-es lépésközzel (minden blokkban felezve a magasságot és a szélességet).
Az alábbi kódban egy `vgg_block` nevű függvényt definiálunk egy VGG blokk implementálásához.

A következő függvény két argumentumot fogad: a konvolúciós rétegek számát (`num_convs`) és a kimeneti csatornák számát (`num_channels`).

```{.python .input  n=2}
%%tab mxnet
def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input  n=3}
%%tab pytorch
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

```{.python .input  n=4}
%%tab tensorflow
def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(
            tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                   padding='same', activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
%%tab jax
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv(out_channels, kernel_size=(3, 3), padding=(1, 1)))
        layers.append(nn.relu)
    layers.append(lambda x: nn.max_pool(x, window_shape=(2, 2), strides=(2, 2)))
    return nn.Sequential(layers)
```

## [**VGG Hálózat**]
:label:`subsec_vgg-network`

Az AlexNet-hez és a LeNet-hez hasonlóan a VGG hálózat két részre osztható: az első főként konvolúciós és pooling rétegekből áll, a második pedig teljesen összekötött rétegekből, amelyek azonosak az AlexNet-ben lévőkkel.
A fő különbség az, hogy a konvolúciós rétegek nemlineáris transzformációkba vannak csoportosítva, amelyek változatlanul hagyják a dimenziókat, majd egy felbontáscsökkentő lépés következik, ahogy a :numref:`fig_vgg` ábra mutatja.

![Az AlexNet-től a VGG-ig. A fő különbség az, hogy a VGG réteges blokkokból áll, míg az AlexNet rétegei mind egyénileg tervezett.](../img/vgg.svg)
:width:`400px`
:label:`fig_vgg`

A hálózat konvolúciós részén egymás után több VGG blokk kapcsolódik össze a :numref:`fig_vgg` ábrából (a `vgg_block` függvényben is definiálva). A konvolúciók ilyen csoportosítása egy olyan minta, amely az elmúlt évtizedben szinte változatlan maradt, bár a konkrét műveletek megválasztása jelentősen módosult.
Az `arch` változó tuple-ok listájából áll (blokkokként egy-egy), ahol mindegyik két értéket tartalmaz: a konvolúciós rétegek számát és a kimeneti csatornák számát, amelyek pontosan a `vgg_block` függvény meghívásához szükséges argumentumok. Így a VGG hálózatok egy *családját* definiálja, nem csupán egyetlen konkrét megvalósítást. Egy konkrét hálózat felépítéséhez egyszerűen végigiteráljuk az `arch`-ot, és összerakjuk a blokkokat.

```{.python .input  n=5}
%%tab pytorch, mxnet, tensorflow
class VGG(d2l.Classifier):
    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            for (num_convs, num_channels) in arch:
                self.net.add(vgg_block(num_convs, num_channels))
            self.net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                         nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                         nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            conv_blks = []
            for (num_convs, out_channels) in arch:
                conv_blks.append(vgg_block(num_convs, out_channels))
            self.net = nn.Sequential(
                *conv_blks, nn.Flatten(),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                nn.LazyLinear(num_classes))
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            for (num_convs, num_channels) in arch:
                self.net.add(vgg_block(num_convs, num_channels))
            self.net.add(
                tf.keras.models.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes)]))
```

```{.python .input  n=5}
%%tab jax
class VGG(d2l.Classifier):
    arch: list
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        conv_blks = []
        for (num_convs, out_channels) in self.arch:
            conv_blks.append(vgg_block(num_convs, out_channels))

        self.net = nn.Sequential([
            *conv_blks,
            lambda x: x.reshape((x.shape[0], -1)),  # lapítás
            nn.Dense(4096), nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(4096), nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(self.num_classes)])
```

Az eredeti VGG hálózatnak öt konvolúciós blokkja volt: az első kettő egyenként egy konvolúciós réteget tartalmaz, a maradék három mindegyike két konvolúciós réteget.
Az első blokk 64 kimeneti csatornával rendelkezik, és minden egymást követő blokk megduplázza a kimeneti csatornák számát, amíg az 512-t el nem éri.
Mivel ez a hálózat nyolc konvolúciós réteget és három teljesen összekötött réteget alkalmaz, gyakran VGG-11-nek nevezik.

```{.python .input  n=6}
%%tab pytorch, mxnet
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary(
    (1, 1, 224, 224))
```

```{.python .input  n=7}
%%tab tensorflow
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary(
    (1, 224, 224, 1))
```

```{.python .input}
%%tab jax
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)),
    training=False).layer_summary((1, 224, 224, 1))
```

Amint látható, minden blokknál felezzük a magasságot és a szélességet, végül 7-es magasságot és szélességet érve el, mielőtt kiterítjük a reprezentációkat a hálózat teljesen összekötött részének feldolgozásához.
:citet:`Simonyan.Zisserman.2014` a VGG számos más változatát is leírta.
Sőt, manapság az is normává vált, hogy egy új architektúra bevezetésekor hálózatok *família­ját* ajánlják különböző sebesség–pontosság kompromisszumokkal.

## Tanítás

[**Mivel a VGG-11 számítási szempontból igényesebb az AlexNet-nél, kisebb csatornaszámú hálózatot építünk.**]
Ez Fashion-MNIST tanítására több mint elegendő.
A [**modelltanítás**] folyamata hasonló az AlexNet-éhez a :numref:`sec_alexnet`-ben.
Ismét megfigyelhető a validációs és tanítási veszteség szoros egyezése, ami csak kis mértékű túlillesztésre utal.

```{.python .input  n=8}
%%tab mxnet, pytorch, jax
model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input  n=9}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.01)
    trainer.fit(model, data)
```

## Összefoglalás

Talán a VGG tekinthető az első igazán modern konvolúciós neurális hálózatnak. Bár az AlexNet bevezette a deep learning hatékonyságához szükséges összetevők nagy részét, a VGG volt az, amely bevezette az olyan kulcstulajdonságokat, mint a több konvolúcióból álló blokkok és a mély, keskeny hálózatok előnyben részesítése. Ez egyben az első hálózat, amely valójában hasonló paraméterezésű modellek egész famíliája, és a szakember számára bőséges kompromisszumot kínál a komplexitás és a sebesség között. Ez az a hely is, ahol a modern deep learning keretrendszerek igazán ragyognak. Már nem szükséges XML konfigurációs fájlokat generálni a hálózat megadásához, hanem egyszerű Python kódon keresztül lehet összerakni a hálózatokat.

A közelmúltban a ParNet :cite:`Goyal.Bochkovskiy.Deng.ea.2021` megmutatta, hogy jóval sekélyebb architektúrával, nagyszámú párhuzamos számítás révén is versenyképes teljesítmény érhető el. Ez egy izgalmas fejlemény, és van remény arra, hogy a jövőben befolyásolni fogja az architektúra tervezését. A fejezet hátralévő részében azonban a tudományos haladás elmúlt évtizedes útját követjük.

## Feladatok

1. Az AlexNet-hez képest a VGG sokkal lassabb számítási szempontból, és több GPU memóriát is igényel.
    1. Hasonlítsd össze az AlexNet és a VGG által igényelt paraméterek számát.
    1. Hasonlítsd össze a konvolúciós rétegekben és a teljesen összekötött rétegekben felhasznált lebegőpontos műveletek számát.
    1. Hogyan lehetne csökkenteni a teljesen összekötött rétegek által okozott számítási költséget?
1. A hálózat különböző rétegeihez kapcsolódó dimenziók megjelenítésekor csak nyolc blokk (plusz néhány segédtranszformáció) információit látjuk, bár a hálózatnak 11 rétege van. Hová kerültek a maradék három réteg?
1. Használd a VGG cikk 1. táblázatát :cite:`Simonyan.Zisserman.2014` más elterjedt modellek, például a VGG-16 vagy a VGG-19 felépítéséhez.
1. A Fashion-MNIST felbontásának nyolcszorosra ($28 \times 28$-ról $224 \times 224$ dimenzióra) való felskálázása rendkívül pazarló. Próbáld módosítani a hálózati architektúrát és a felbontás-konverziót, pl. 56 vagy 84 dimenzióra a bemenethez. Meg tudod-e tenni a hálózat pontosságának csökkentése nélkül? Ötletekért a leskálázás előtti további nemlinearitások hozzáadásához nézd meg a VGG cikket :cite:`Simonyan.Zisserman.2014`.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/77)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/78)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/277)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18002)
:end_tab:
