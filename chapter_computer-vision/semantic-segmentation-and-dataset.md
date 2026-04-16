# Szemantikai szegmentáció és az adathalmaz
:label:`sec_semantic_segmentation`

Az objektumdetektálási feladatok tárgyalásakor a :numref:`sec_bbox`--:numref:`sec_rcnn` fejezetekben befoglaló téglalapokat alkalmazunk a képeken lévő objektumok felcímkézésére és jóslására.
Ez a fejezet a *szemantikai szegmentáció* problémáját tárgyalja, amely arra összpontosít, hogyan osztjuk fel a képet a különböző szemantikai osztályokba tartozó régiókra.
Eltérően az objektumdetektálástól, a szemantikai szegmentáció pixelszinten ismeri fel és érti a képeken lévőket: a szemantikai régiók felcímkézése és jóslása pixelszintű.
A :numref:`fig_segmentation` ábra bemutatja a kép kutyájának, macskájának és hátterének szemantikai szegmentációs felcímkézését.
Az objektumdetektáláshoz képest a szemantikai szegmentációban felcímkézett pixelszintű határok nyilvánvalóan finomabbak.


![A kép kutyájának, macskájának és hátterének felcímkézése szemantikai szegmentációban.](../img/segmentation.svg)
:label:`fig_segmentation`


## Képszegmentáció és példányszegmentáció

A számítógépes látás területén van még két fontos feladat, amely hasonló a szemantikai szegmentációhoz, nevezetesen a képszegmentáció és a példányszegmentáció.
Ezeket röviden megkülönböztetjük a szemantikai szegmentációtól az alábbiak szerint.

* A *képszegmentáció* egy képet több alkotórégióra oszt fel. Az ilyen típusú problémák megoldásához szükséges módszerek általában a képen lévő pixelek közötti összefüggést használják ki. A tanítás során nem igényel képpixelekre vonatkozó felcímkézési információt, és nem garantálhatja, hogy a szegmentált régiók olyan szemantikával rendelkeznek majd, amelyet a jóslás során szeretnénk kapni. A :numref:`fig_segmentation` ábrán látható képet bemenetként véve a képszegmentáció a kutyát két régióra oszthatja: az egyik a főként fekete szájat és szemeket, a másik a főként sárga testmaradékot fedi.
* A *példányszegmentáció* más nevén *egyidejű felismerés és szegmentáció*. Azt tanulmányozza, hogyan ismerjük fel egy kép minden objektumpéldányának pixelszintű régióit. A szemantikai szegmentációtól eltérően a példányszegmentációnak nemcsak a szemantikát kell megkülönböztetnie, hanem a különböző objektumpéldányokat is. Például ha a képen két kutya van, a példányszegmentációnak meg kell különböztetnie, hogy a két kutya közül melyikhez tartozik egy pixel.


## A Pascal VOC2012 szemantikai szegmentációs adathalmaz

[**Az egyik legfontosabb szemantikai szegmentációs adathalmaz a [Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).**]
A következőkben megvizsgáljuk ezt az adathalmazt.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
```

Az adathalmaz tar fájlja körülbelül 2 GB, így a fájl letöltése eltarthat egy ideig.
A kicsomagolt adathalmaz az `../data/VOCdevkit/VOC2012` könyvtárban található.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
```

A `../data/VOCdevkit/VOC2012` elérési út megadása után láthatjuk az adathalmaz különböző összetevőit.
Az `ImageSets/Segmentation` elérési út tanítási és tesztelési mintákat meghatározó szöveges fájlokat tartalmaz, míg a `JPEGImages` és `SegmentationClass` elérési utak rendre minden példány bemeneti képét és felcímkézett képét tárolják.
A felcímkézett kép itt is képformátumban van, azonos méretű a felcímkézett bemeneti képpel.
Emellett bármely felcímkézett képen az azonos színű pixelek ugyanahhoz a szemantikai osztályhoz tartoznak.
A következő definiálja a `read_voc_images` függvényt [**az összes bemeneti kép és felcímkézett kép memóriába való beolvasásához**].

```{.python .input}
#@tab mxnet
#@save
def read_voc_images(voc_dir, is_train=True):
    """Az összes VOC jellemző- és címkekép beolvasása."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(image.imread(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(image.imread(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png')))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

```{.python .input}
#@tab pytorch
#@save
def read_voc_images(voc_dir, is_train=True):
    """Az összes VOC jellemző- és címkekép beolvasása."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

[**Megrajzoljuk az első öt bemeneti képet és felcímkézett képüket**].
A felcímkézett képeken a fehér és fekete rendre határokat és hátteret jelölnek, míg a többi szín különböző osztályoknak felel meg.

```{.python .input}
#@tab mxnet
n = 5
imgs = train_features[:n] + train_labels[:n]
d2l.show_images(imgs, 2, n);
```

```{.python .input}
#@tab pytorch
n = 5
imgs = train_features[:n] + train_labels[:n]
imgs = [img.permute(1,2,0) for img in imgs]
d2l.show_images(imgs, 2, n);
```

Ezután [**felsoroljuk az RGB színértékeket és az osztályneveket**] az adathalmaz összes felcímkézett képéhez.

```{.python .input}
#@tab all
#@save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
```

A fent definiált két állandóval kényelmesen [**megtalálhatjuk minden pixel osztályindexét egy felcímkézett képen**].
Definiáljuk a `voc_colormap2label` függvényt a fenti RGB színértékekből az osztályindexekbe való leképezés felépítéséhez, és a `voc_label_indices` függvényt, amely bármely RGB értéket a Pascal VOC2012 adathalmaz osztályindexeire képez le.

```{.python .input}
#@tab mxnet
#@save
def voc_colormap2label():
    """Leképezés felépítése RGB-ről osztályindexekre VOC címkékhez."""
    colormap2label = np.zeros(256 ** 3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """Bármely RGB érték leképezése osztályindexekre VOC címkékben."""
    colormap = colormap.astype(np.int32)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

```{.python .input}
#@tab pytorch
#@save
def voc_colormap2label():
    """Leképezés felépítése RGB-ről osztályindexekre VOC címkékhez."""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """Bármely RGB érték leképezése osztályindexekre VOC címkékben."""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

[**Például**] az első példaképen a repülő elejének osztályindexe 1, a háttér indexe pedig 0.

```{.python .input}
#@tab all
y = voc_label_indices(train_labels[0], voc_colormap2label())
y[105:115, 130:140], VOC_CLASSES[1]
```

### Adatok előfeldolgozása

A korábbi kísérletekben, például a :numref:`sec_alexnet`--:numref:`sec_googlenet` fejezetekben a képeket átméretezik a modell szükséges bemeneti alakjára.
A szemantikai szegmentációban azonban ez megköveteli a jósolt pixelosztályok visszaméretezését a bemeneti kép eredeti alakjára.
Az ilyen átméretezés pontatlan lehet, különösen a különböző osztályokba tartozó szegmentált régiók esetén. Ennek elkerülése érdekében a képet *rögzített* alakra vágjuk ki átméretezés helyett. Konkrétabban, [**képaugmentációból véletlen kivágást alkalmazva egyaránt kivágjuk a bemeneti kép és a felcímkézett kép azonos területét**].

```{.python .input}
#@tab mxnet
#@save
def voc_rand_crop(feature, label, height, width):
    """Jellemző- és címkeképek véletlen kivágása."""
    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    return feature, label
```

```{.python .input}
#@tab pytorch
#@save
def voc_rand_crop(feature, label, height, width):
    """Jellemző- és címkeképek véletlen kivágása."""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label
```

```{.python .input}
#@tab mxnet
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

```{.python .input}
#@tab pytorch
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

imgs = [img.permute(1, 2, 0) for img in imgs]
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

### [**Egyéni szemantikai szegmentációs adathalmaz osztály**]

Egyéni szemantikai szegmentációs adathalmaz osztályt, a `VOCSegDataset`-et definiálunk, örökölve a magas szintű API-k által biztosított `Dataset` osztályt.
A `__getitem__` függvény implementálásával tetszőlegesen hozzáférhetünk az adathalmazban `idx` indexű bemeneti képhez és a képen lévő minden pixel osztályindexéhez.
Mivel az adathalmaz egyes képeinek mérete kisebb, mint a véletlen kivágás kimeneti mérete, ezeket a példányokat egy egyéni `filter` függvény kiszűri.
Emellett definiáljuk a `normalize_image` függvényt a bemeneti képek három RGB csatornájának értékeinek normalizálásához.

```{.python .input}
#@tab mxnet
#@save
class VOCSegDataset(gluon.data.Dataset):
    """Egyéni adathalmaz a VOC adathalmaz betöltéséhez."""
    def __init__(self, is_train, crop_size, voc_dir):
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature.transpose(2, 0, 1),
                voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class VOCSegDataset(torch.utils.data.Dataset):
    """Egyéni adathalmaz a VOC adathalmaz betöltéséhez."""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

### [**Az adathalmaz beolvasása**]

Az egyéni `VOCSegDataset` osztályt használjuk a tanítóhalmaz és a teszthalmaz példányainak létrehozásához.
Tegyük fel, hogy megadjuk, hogy a véletlenszerűen kivágott képek kimeneti alakja $320\times 480$.
Az alábbiakban megtekinthetjük a tanítóhalmazban és a teszthalmazban megmaradó példányok számát.

```{.python .input}
#@tab all
crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)
```

A batch méret 64-re állításával definiáljuk a tanítóhalmaz adatiterátorát.
Nyomtassuk ki az első minibatch alakját.
A képosztályozástól vagy az objektumdetektálástól eltérően itt a felcímkézett képek háromdimenziós tenzorok.

```{.python .input}
#@tab mxnet
batch_size = 64
train_iter = gluon.data.DataLoader(voc_train, batch_size, shuffle=True,
                                   last_batch='discard',
                                   num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

```{.python .input}
#@tab pytorch
batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

### [**Összerakva**]

Végül definiáljuk a következő `load_data_voc` függvényt a Pascal VOC2012 szemantikai szegmentációs adathalmaz letöltésére és beolvasására.
Ez adatiterátorokat ad vissza mind a tanítási, mind a tesztelési adathalmazhoz.

```{.python .input}
#@tab mxnet
#@save
def load_data_voc(batch_size, crop_size):
    """A VOC szemantikai szegmentációs adathalmaz betöltése."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = gluon.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, last_batch='discard', num_workers=num_workers)
    test_iter = gluon.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        last_batch='discard', num_workers=num_workers)
    return train_iter, test_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_voc(batch_size, crop_size):
    """A VOC szemantikai szegmentációs adathalmaz betöltése."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter
```

## Összefoglalás

* A szemantikai szegmentáció pixelszinten ismeri fel és érti a képen lévőket, a képet a különböző szemantikai osztályokhoz tartozó régiókra osztva.
* Az egyik legfontosabb szemantikai szegmentációs adathalmaz a Pascal VOC2012.
* A szemantikai szegmentációban, mivel a bemeneti kép és a felcímkézett kép pixelszinten egy-az-egyhez felel meg, a bemeneti képet rögzített alakra vágjuk ki átméretezés helyett.


## Feladatok

1. Hogyan alkalmazható a szemantikai szegmentáció az önvezető járművekben és az orvosi képdiagnosztikában? Tudsz más alkalmazásokra gondolni?
1. Emlékezz a képaugmentáció leírásaira a :numref:`sec_image_augmentation` fejezetben. A képosztályozásban alkalmazott képaugmentációs módszerek közül melyiket lenne kivitelezhetetlen alkalmazni a szemantikai szegmentációban?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/375)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1480)
:end_tab:
