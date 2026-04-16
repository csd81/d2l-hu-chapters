# Az objektumdetektálási adathalmaz
:label:`sec_object-detection-dataset`

Az objektumdetektálás területén nincsenek olyan kis adathalmazok, mint az MNIST vagy a Fashion-MNIST.
Az objektumdetektálási modellek gyors bemutatásához
[**összegyűjtöttünk és felcímkéztünk egy kis adathalmazt**].
Először az irodánkban lévő szabad banánokról készítettünk fényképeket, és
különböző forgatásokkal és méretekkel 1000 banánképet generáltunk.
Majd minden banánképet véletlenszerű pozícióban helyeztünk el egy háttérképen.
Végül befoglaló téglalapokat jelöltünk meg a képeken lévő banánokhoz.


## [**Az adathalmaz letöltése**]

A banánfelismerési adathalmaz az összes képpel és csv-címkefájllal közvetlenül letölthető az internetről.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os
import pandas as pd

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
import pandas as pd
```

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
```

## Az adathalmaz beolvasása

A [**banánfelismerési adathalmaz beolvasásához**] az alábbi `read_data_bananas` függvényt fogjuk használni.
Az adathalmaz tartalmaz egy csv-fájlt az objektumok osztálycímkéihez és a valódi befoglaló téglalapok koordinátáihoz a bal felső és a jobb alsó sarokban.

```{.python .input}
#@tab mxnet
#@save
def read_data_bananas(is_train=True):
    """A banánfelismerési adathalmaz képeinek és címkéinek beolvasása."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(image.imread(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # A `target` tartalmazza: (osztály, bal felső x, bal felső y,
        # jobb alsó x, jobb alsó y), ahol az összes kép ugyanahhoz
        # a banán osztályhoz tartozik (index 0)
        targets.append(list(target))
    return images, np.expand_dims(np.array(targets), 1) / 256
```

```{.python .input}
#@tab pytorch
#@save
def read_data_bananas(is_train=True):
    """A banánfelismerési adathalmaz képeinek és címkéinek beolvasása."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # A `target` tartalmazza: (osztály, bal felső x, bal felső y,
        # jobb alsó x, jobb alsó y), ahol az összes kép ugyanahhoz
        # a banán osztályhoz tartozik (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

A `read_data_bananas` függvény segítségével képek és címkék beolvasásával
a következő `BananasDataset` osztály lehetővé teszi számunkra egy [**testreszabott `Dataset` példány létrehozását**]
a banánfelismerési adathalmaz betöltéséhez.

```{.python .input}
#@tab mxnet
#@save
class BananasDataset(gluon.data.Dataset):
    """Testreszabott adathalmaz a banánfelismerési adathalmaz betöltéséhez."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].astype('float32').transpose(2, 0, 1),
                self.labels[idx])

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class BananasDataset(torch.utils.data.Dataset):
    """Testreszabott adathalmaz a banánfelismerési adathalmaz betöltéséhez."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

Végül definiáljuk a `load_data_bananas` függvényt, amely [**visszaad két adatiterátor-példányt mind a tanítási, mind a teszthalmazhoz.**]
A tesztelési adathalmaznál nincs szükség véletlenszerű sorrendben való olvasásra.

```{.python .input}
#@tab mxnet
#@save
def load_data_bananas(batch_size):
    """A banánfelismerési adathalmaz betöltése."""
    train_iter = gluon.data.DataLoader(BananasDataset(is_train=True),
                                       batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(BananasDataset(is_train=False),
                                     batch_size)
    return train_iter, val_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_bananas(batch_size):
    """A banánfelismerési adathalmaz betöltése."""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
```

[**Olvassunk be egy mini-batch-et, és nyomtassuk ki a képek és a címkék alakját**] ebben a mini-batch-ben.
A képek mini-batch-ének alakja
(batch méret, csatornák száma, magasság, szélesség)
ismerős:
ugyanaz, mint a korábbi képosztályozási feladatoknál.
A címkék mini-batch-ének alakja
(batch méret, $m$, 5),
ahol $m$ az adathalmazban bármely képen lévő befoglaló téglalapok maximálisan lehetséges száma.

Bár a mini-batch-ekben való számítás hatékonyabb,
megköveteli, hogy az összes képpélda azonos számú befoglaló téglalapot tartalmazzon az összefűzés útján való mini-batch-formáláshoz.
Általában a képeknek eltérő számú befoglaló téglalapjuk lehet;
ezért
az $m$-nél kevesebb befoglaló téglalapot tartalmazó képeket érvénytelen befoglaló téglalapokkal egészítik ki,
amíg el nem érik az $m$ értéket.
Ezután minden befoglaló téglalap címkéjét egy 5 hosszúságú tömbként ábrázolják.
A tömb első eleme a befoglaló téglalapban lévő objektum osztálya,
ahol a -1 kitöltéshez szükséges érvénytelen befoglaló téglalapot jelöl.
A tömb maradék négy eleme a befoglaló téglalap bal felső sarkának és jobb alsó sarkának ($x$, $y$)-koordináta értékei (értékkészlet 0 és 1 között).
A banán adathalmaz esetén,
mivel minden képen csak egy befoglaló téglalap van,
$m=1$.

```{.python .input}
#@tab all
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
```

## [**Demonstráció**]

Mutassunk be tíz képet a felcímkézett valódi befoglaló téglalapjaikkal.
Láthatjuk, hogy a banánok forgatása, mérete és pozíciói eltérnek az összes képen.
Természetesen ez csupán egy egyszerű mesterséges adathalmaz.
A gyakorlatban a valós adathalmazok általában sokkal bonyolultabbak.

```{.python .input}
#@tab mxnet
imgs = (batch[0][:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

```{.python .input}
#@tab pytorch
imgs = (batch[0][:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

## Összefoglalás

* Az összegyűjtött banánfelismerési adathalmaz objektumdetektálási modellek bemutatására használható.
* Az objektumdetektáláshoz szükséges adatbetöltés hasonló a képosztályozáséhoz. Azonban az objektumdetektálásban a címkék a valódi befoglaló téglalapok információit is tartalmazzák, ami a képosztályozásban hiányzik.


## Feladatok

1. Mutass be más képeket valódi befoglaló téglalapjaikkal a banánfelismerési adathalmazból. Miben különböznek a befoglaló téglalapok és az objektumok szempontjából?
1. Tegyük fel, hogy adataugmentációt, például véletlen kivágást szeretnénk alkalmazni az objektumdetektáláshoz. Miben különbözhet ez a képosztályozásnál alkalmazottól? Tipp: mi van akkor, ha a kivágott kép csak egy kis részét tartalmazza az objektumnak?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/372)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1608)
:end_tab:
