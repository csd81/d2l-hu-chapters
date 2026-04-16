```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# A képosztályozási adathalmaz
:label:`sec_fashion_mnist`

(~~Az MNIST adathalmaz az egyik legelterjedtebb adathalmaz képosztályozáshoz, azonban referenciaként túl egyszerűnek számít. Ehelyett a hasonló, de bonyolultabb Fashion-MNIST adathalmazt fogjuk használni. ~~)

A képosztályozásban széles körben alkalmazott adathalmaz a kézzel írt számjegyeket tartalmazó [MNIST adathalmaz](https://en.wikipedia.org/wiki/MNIST_database) :cite:`LeCun.Bottou.Bengio.ea.1998`. Az 1990-es években való megjelenésekor komoly kihívást jelentett a legtöbb gépi tanulási algoritmus számára: 60 000 darab $28 \times 28$ pixeles képből áll (plusz egy 10 000 képes tesztadathalmaz). Az összefüggések érzékeltetéséhez: 1995-ben az AT&T Bell Laboratories-ban a gépi tanulásban csúcstechnológiának számított egy Sun SPARCStation 5, amelynek óriási 64 MB RAM-ja és lenyűgöző 5 MFLOPs teljesítménye volt. A számjegyfelismerésben elért magas pontosság kulcsfontosságú volt a levélosztályozás automatizálásában az USPS-nél az 1990-es években. A mély hálózatok, mint a LeNet-5 :cite:`LeCun.Jackel.Bottou.ea.1995`, invarianciákkal rendelkező szupportvektor-gépek :cite:`Scholkopf.Burges.Vapnik.1996` és az érintőtávolság-alapú osztályozók :cite:`Simard.LeCun.Denker.ea.1998` mind 1% alatti hibaarányt értek el.

Több mint egy évtizeden át az MNIST volt *a* referenciaadathalmaz a gépi tanulási algoritmusok összehasonlításában.
Bár jól betöltötte ezt a szerepet, a mai egyszerű modellek is 95%-os pontosságot érnek el,
így alkalmatlanná vált az erős és gyengébb modellek megkülönböztetésére. Sőt, az adathalmaz lehetővé teszi *rendkívül* magas pontosság elérését, ami az osztályozási feladatoknál nem tipikus. Ez algoritmikus fejlesztések terén bizonyos algoritmuscsaládok — mint az aktív halmaz módszerek és a határkeresési aktív halmaz algoritmusok — felé tolta a fejlesztőket.
Ma az MNIST inkább helyesség-ellenőrzésre (sanity check) szolgál, mintsem referenciaadathalmazként. Az ImageNet :cite:`Deng.Dong.Socher.ea.2009` sokkal relevánsabb kihívást jelent. Sajnos az ImageNet mérete miatt nem megfelelő a könyv példáinak és szemléltetéseinek nagy részéhez, mivel a tanítás túl sokáig tartana az interaktív példákhoz. Helyette a következő szakaszokban egy minőségileg hasonló, de sokkal kisebb adathalmazra fókuszálunk: a 2017-ben megjelent Fashion-MNIST adathalmazra :cite:`Xiao.Rasul.Vollgraf.2017`, amely 10 ruházati kategória $28 \times 28$ pixeles képeit tartalmazza.

```{.python .input}
%%tab mxnet
%matplotlib inline
import time
from d2l import mxnet as d2l
from mxnet import gluon, npx
from mxnet.gluon.data.vision import transforms
npx.set_np()

d2l.use_svg_display()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
import time
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms

d2l.use_svg_display()
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
import time
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import time
import tensorflow as tf
import tensorflow_datasets as tfds

d2l.use_svg_display()
```

## Az adathalmaz betöltése

Mivel a Fashion-MNIST adathalmaz annyira hasznos, a legtöbb major keretrendszer előre feldolgozott verziókat biztosít belőle. [**A beépített keretrendszer-segédeszközök segítségével letölthetjük és betölthetjük a memóriába.**]

```{.python .input}
%%tab mxnet
class FashionMNIST(d2l.DataModule):  #@save
    """A Fashion-MNIST adathalmaz."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = gluon.data.vision.FashionMNIST(
            train=True).transform_first(trans)
        self.val = gluon.data.vision.FashionMNIST(
            train=False).transform_first(trans)
```

```{.python .input}
%%tab pytorch
class FashionMNIST(d2l.DataModule):  #@save
    """A Fashion-MNIST adathalmaz."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
```

```{.python .input}
%%tab tensorflow, jax
class FashionMNIST(d2l.DataModule):  #@save
    """A Fashion-MNIST adathalmaz."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        self.train, self.val = tf.keras.datasets.fashion_mnist.load_data()
```

A Fashion-MNIST 10 kategóriából álló képeket tartalmaz; minden kategóriát
6000 kép képvisel a tanítási adathalmazban és 1000 a tesztadathalmazban.
A *tesztadathalmazt* a modell teljesítményének kiértékelésére használjuk (tanításhoz nem szabad felhasználni).
Ebből következően a tanítóhalmaz és a teszthalmaz rendre 60 000 és 10 000 képet tartalmaz.

```{.python .input}
%%tab mxnet, pytorch
data = FashionMNIST(resize=(32, 32))
len(data.train), len(data.val)
```

```{.python .input}
%%tab tensorflow, jax
data = FashionMNIST(resize=(32, 32))
len(data.train[0]), len(data.val[0])
```

A képek szürkeárnyalatosak és fentebb $32 \times 32$ pixelre nagyítottak. Ez hasonló az eredeti MNIST adathalmazhoz, amely (bináris) fekete-fehér képekből állt. Megjegyezzük azonban, hogy a legtöbb modern képadat három csatornával rendelkezik (piros, zöld, kék), és a hiperspektrális képek több mint 100 csatornát is tartalmazhatnak (a HyMap szenzornak 126 csatornája van).
Megállapodásból egy képet $c \times h \times w$ tenzoralakban tárolunk, ahol $c$ a színcsatornák száma, $h$ a magasság és $w$ a szélesség.

```{.python .input}
%%tab all
data.train[0][0].shape
```

[~~Két segédfüggvény az adathalmaz megjelenítéséhez~~]

A Fashion-MNIST kategóriái emberi értelmű nevekkel bírnak.
Az alábbi segédmetódus a numerikus és szöveges címkék között konvertál.

```{.python .input}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def text_labels(self, indices):
    """Szöveges címkéket ad vissza."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]
```

## mini-batch olvasása

Hogy megkönnyítsük az életünket a tanítási és teszthalmazból való olvasáskor,
a beépített adatiterátort használjuk ahelyett, hogy nulláról hoznánk létre egyet.
Emlékeztetőül: minden iterációban az adatiterátor
[**`batch_size` méretű adat mini-batch-et olvas.**]
A tanítási adatiterátor esetén a példákat véletlenszerűen is megkeverjük.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return gluon.data.DataLoader(data, self.batch_size, shuffle=train,
                                 num_workers=self.num_workers)
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)
```

```{.python .input}
%%tab tensorflow, jax
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (tf.image.resize_with_pad(X, *self.resize), y)
    shuffle_buf = len(data[0]) if train else 1
    if tab.selected('tensorflow'):
        return tf.data.Dataset.from_tensor_slices(process(*data)).batch(
            self.batch_size).map(resize_fn).shuffle(shuffle_buf)
    if tab.selected('jax'):
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(process(*data)).batch(
                self.batch_size).map(resize_fn).shuffle(shuffle_buf))
```

Hogy lássuk, hogyan működik, töltsük be képek egy mini-batch-ét a `train_dataloader` metódus meghívásával. 64 képet tartalmaz.

```{.python .input}
%%tab all
X, y = next(iter(data.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)
```

Nézzük meg, mennyi idő szükséges a képek beolvasásához. Bár beépített betöltőről van szó, nem villámgyors. Ennek ellenére ez elegendő, mivel a képek mély hálózattal való feldolgozása jóval tovább tart. Tehát a hálózat tanítása nem lesz I/O-korlátozott.

```{.python .input}
%%tab all
tic = time.time()
for X, y in data.train_dataloader():
    continue
f'{time.time() - tic:.2f} sec'
```

## Megjelenítés

A Fashion-MNIST adathalmazt sokszor fogjuk majd használni. A `show_images` segédfüggvénnyel megjeleníthetjük a képeket és a hozzájuk tartozó címkéket.
Az implementáció részleteit kihagyva csak az interfészt mutatjuk be: ilyen segédfüggvényeknél csak azt kell tudnunk, hogyan kell meghívni a `d2l.show_images` függvényt, nem azt, hogyan működik.

```{.python .input}
%%tab all
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Képek listáját ábrázolja."""
    raise NotImplementedError
```

Tegyük hasznossá. Általában jó ötlet megjeleníteni és ellenőrizni az adatokat, amelyeken tanítunk.
Az emberek nagyon jók a furcsaságok észlelésében, és emiatt a vizualizáció kiegészítő védelmet nyújt a kísérletek tervezésekor elkövetett hibák ellen. Íme [**a tanítási adathalmaz első néhány példájának képei és a hozzájuk tartozó (szöveges) címkék**].

```{.python .input}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    if tab.selected('mxnet', 'pytorch'):
        d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
    if tab.selected('tensorflow'):
        d2l.show_images(tf.squeeze(X), nrows, ncols, titles=labels)
    if tab.selected('jax'):
        d2l.show_images(jnp.squeeze(X), nrows, ncols, titles=labels)

batch = next(iter(data.val_dataloader()))
data.visualize(batch)
```

Most már készen állunk arra, hogy a következő szakaszokban dolgozzunk a Fashion-MNIST adathalmazzal.

## Összefoglalás

Most egy kicsit realistikusabb adathalmazunk van az osztályozáshoz. A Fashion-MNIST egy ruházati osztályozási adathalmaz, amely 10 kategóriát ábrázoló képekből áll. Ezt az adathalmazt a következő szakaszokban és fejezetekben fogjuk felhasználni különböző hálózati tervek kiértékelésére, az egyszerű lineáris modelltől a fejlett maradékos hálózatokig. Ahogy általában a képekkel szoktuk, (batch méret, csatornák száma, magasság, szélesség) alakú tenzorként olvassuk be őket. Egyelőre csak egy csatornánk van, mivel a képek szürkeárnyalatosak (a fenti megjelenítés hamis színpalettát használ a jobb láthatóság érdekében).

Végül az adatiterálók kulcsfontosságú komponensek a hatékony működéshez. Például GPU-kat használhatunk a hatékony képtömörítés-visszafejtéshez, videó-átkódoláshoz vagy egyéb előfeldolgozáshoz. Amikor csak lehetséges, jól megvalósított adatiterálókra kell hagyatkozni, amelyek kihasználják a nagy teljesítményű számítástechnikát, hogy elkerüljük a tanítási ciklus lassítását.


## Feladatok

1. Befolyásolja-e a `batch_size` csökkentése (például 1-re) az olvasási teljesítményt?
1. Az adatiterátor teljesítménye fontos. Gondolod, hogy a jelenlegi implementáció elég gyors? Vizsgálj meg különböző lehetőségeket a javítására. Rendszer-profilozóval derítsd ki, hol vannak a szűk keresztmetszetek!
1. Nézd meg a keretrendszer online API dokumentációját! Milyen más adathalmazok elérhetők?

:begin_tab:`mxnet`
[Megbeszélések](https://discuss.d2l.ai/t/48)
:end_tab:

:begin_tab:`pytorch`
[Megbeszélések](https://discuss.d2l.ai/t/49)
:end_tab:

:begin_tab:`tensorflow`
[Megbeszélések](https://discuss.d2l.ai/t/224)
:end_tab:

:begin_tab:`jax`
[Megbeszélések](https://discuss.d2l.ai/t/17980)
:end_tab:
