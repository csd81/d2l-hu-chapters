# Képosztályozás (CIFAR-10) Kaggle-ön
:label:`sec_kaggle_cifar10`

Eddig a mélytanulás keretrendszerek magas szintű API-jait használtuk, hogy közvetlenül tensor formátumban kapjuk meg a képadathalmazokat.
Az egyedi képadathalmazok
azonban általában képfájlok formájában állnak rendelkezésre.
Ebben a szakaszban
nyers képfájlokból indulunk ki,
majd lépésről lépésre rendszerezzük, beolvassuk,
és tensor formátumba alakítjuk őket.

A CIFAR-10 adathalmazzal már kísérleteztünk a :numref:`sec_image_augmentation` szakaszban –
ez egy fontos adathalmaz a számítógépes látás területén.
Ebben a szakaszban
az előző részekben tanultakat felhasználva
a CIFAR-10 képosztályozási Kaggle-verseny
feladatain fogunk gyakorolni.
(**A verseny webcíme: https://www.kaggle.com/c/cifar-10**)

A :numref:`fig_kaggle_cifar10` ábra a verseny weboldalán található információkat mutatja.
Az eredmények beküldéséhez
Kaggle-fiókot kell regisztrálni.

![A CIFAR-10 képosztályozási verseny weboldalának információi. A verseny adathalmazát az „Adatok" fülre kattintva lehet letölteni.](../img/kaggle-cifar10.png)
:width:`600px`
:label:`fig_kaggle_cifar10`

```{.python .input}
#@tab mxnet
import collections
from d2l import mxnet as d2l
import math
from mxnet import gluon, init, npx
from mxnet.gluon import nn
import os
import pandas as pd
import shutil

npx.set_np()
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
import torchvision
from torch import nn
import os
import pandas as pd
import shutil
```

## Az adathalmaz letöltése és rendszerezése

A verseny adathalmazát tanítóhalmazra és teszthalmazra osztják fel,
amelyek rendre 50 000, illetve 300 000 képet tartalmaznak.
A teszthalmaz
10 000 képét kiértékelésre használják,
a fennmaradó 290 000 kép nem kerül kiértékelésre:
ezeket csak azért szerepeltetik,
hogy nehezebb legyen
a teszthalmaz *kézzel* megcímkézett eredményeivel
csalni.
Az adathalmaz képei
mind png formátumú, színes (RGB csatornás) képfájlok,
amelyek magassága és szélessége egyaránt 32 képpont.
A képek összesen 10 kategóriát fednek le: repülők, autók, madarak, macskák, őzek, kutyák, békák, lovak, hajók és teherautók.
A :numref:`fig_kaggle_cifar10` bal felső sarkában az adathalmazból látható néhány repülős, autós és madár kategóriájú kép.


### Az adathalmaz letöltése

Bejelentkezés után a :numref:`fig_kaggle_cifar10` ábrán látható CIFAR-10 képosztályozási verseny weboldalán az „Adatok" fülre kattintva, majd az „Összes letöltése" gombra kattintva tölthetjük le az adathalmazt.
A letöltött fájl `../data` mappába való kicsomagolása, majd a benne lévő `train.7z` és `test.7z` fájlok kibontása után az adathalmaz a következő elérési utakon érhető el:

* `../data/cifar-10/train/[1-50000].png`
* `../data/cifar-10/test/[1-300000].png`
* `../data/cifar-10/trainLabels.csv`
* `../data/cifar-10/sampleSubmission.csv`

ahol a `train` és `test` könyvtárak rendre a tanító- és tesztképeket tartalmazzák, a `trainLabels.csv` a tanítóképek címkéit tartalmazza, a `sample_submission.csv` pedig egy minta beküldési fájl.

A könnyebb indulás érdekében [**egy kis méretű adathalmazmutatót biztosítunk, amely
az első 1000 tanítóképet és 5 véletlenszerű tesztképet tartalmazza.**]
A Kaggle-verseny teljes adathalmazának használatához az alábbi `demo` változót `False` értékre kell állítani.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')

# Ha a Kaggle-versenyhez letoltott teljes adathalmazt hasznalod,
# allitsd a `demo` erteket False-ra
demo = True

if demo:
    data_dir = d2l.download_extract('cifar10_tiny')
else:
    data_dir = '../data/cifar-10/'
```

### [**Az adathalmaz rendszerezése**]

Az adathalmazt rendszerezni kell, hogy megkönnyítsük a modell tanítását és tesztelését.
Először olvassuk be a címkéket a csv fájlból.
Az alábbi függvény egy szótárat ad vissza, amely
a fájlnév kiterjesztés nélküli részét a hozzá tartozó címkére képezi le.

```{.python .input}
#@tab all
#@save
def read_csv_labels(fname):
    """Beolvassa az `fname` fajlt, es fajlnev-cimke szotarat ad vissza."""
    with open(fname, 'r') as f:
        # A fejlecsor (oszlopnev) kihagyasa
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
print('# training examples:', len(labels))
print('# classes:', len(set(labels.values())))
```

Következő lépésként definiáljuk a `reorg_train_valid` függvényt, amely [**kiválasztja a validációs halmazt az eredeti tanítóhalmazból.**]
A függvény `valid_ratio` argumentuma a validációs halmazban lévő példányok számának és az eredeti tanítóhalmazban lévő példányok számának aránya.
Pontosabban:
legyen $n$ a legkevesebb példánnyal rendelkező osztály képeinek száma, $r$ pedig az arány.
Az validációs halmazba minden osztályból
$\max(\lfloor nr\rfloor,1)$ kép kerül ki.
Vegyük például a `valid_ratio=0.1` értéket. Mivel az eredeti tanítóhalmaz 50 000 képet tartalmaz,
45 000 képet fognak tanításra használni a `train_valid_test/train` útvonal alatt,
míg a fennmaradó 5000 kép
a validációs halmazba kerül a `train_valid_test/valid` útvonalon. Az adathalmaz rendszerezése után az azonos osztályba tartozó képek azonos mappába kerülnek.

```{.python .input}
#@tab all
#@save
def copyfile(filename, target_dir):
    """Fajl masolasa a celkonyvtarba."""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

#@save
def reorg_train_valid(data_dir, labels, valid_ratio):
    """Az ervenyesitesi halmaz kivalasztasa az eredeti tanitohalmaz-bol."""
    # A legkevesebb peldannyal rendelkezo osztaly peldanyainak szama
    # a tanitoadathalmazban
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # Az ervenyesitesi halmazban osztalyonkent hasznalt peldanyszam
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label
```

Az alábbi `reorg_test` függvény [**a teszthalmazt rendszerezi az adatok betöltéséhez az előrejelzés során.**]

```{.python .input}
#@tab all
#@save
def reorg_test(data_dir):
    """A teszthalmaz rendszerezese az elorejeles kozbeni adatbetolteshez."""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))
```

Végül egy függvény segítségével [**meghívjuk**]
a fent definiált `read_csv_labels`, `reorg_train_valid` és `reorg_test` (**függvényeket.**)

```{.python .input}
#@tab all
def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)
```

Az adathalmaz kis méretű mintájához itt a batch méretet 32-re állítjuk.
A Kaggle-verseny teljes adathalmazának tanításakor és tesztelésekor
a `batch_size` értékét nagyobb egész számra, például 128-ra kell állítani.
A tanítási példák 10%-át validációs halmazként választjuk ki a hiperparaméterek hangolásához.

```{.python .input}
#@tab all
batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)
```

## [**Képaugmentáció**]

Képaugmentációt alkalmazunk a túlilleszkedés kezelésére.
Például tanítás közben a képeket véletlenszerűen vízszintesen tükrözhetjük.
A színes képek három RGB csatornáján standardizálást is végezhetünk. Az alábbiakban felsoroltunk néhány ilyen műveletet, amelyeket hangolhatsz.

```{.python .input}
#@tab mxnet
transform_train = gluon.data.vision.transforms.Compose([
    # A kep felmeretezese 40x40 pixeles negyzetre
    gluon.data.vision.transforms.Resize(40),
    # Veletlen vagassal egy 40x40 pixeles negyzetes kepbol az eredeti
    # kepmeret 0,64-1-szerese teruletu kisebb negyzetet vagjuk ki,
    # majd 32x32 pixelesre meretezzuk
    gluon.data.vision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    # A kep minden csatornajanak standardizalasa
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])
```

```{.python .input}
#@tab pytorch
transform_train = torchvision.transforms.Compose([
    # A kep felmeretezese 40x40 pixeles negyzetre
    torchvision.transforms.Resize(40),
    # Veletlen vagassal egy 40x40 pixeles negyzetes kepbol az eredeti
    # kepmeret 0,64-1-szerese teruletu kisebb negyzetet vagjuk ki,
    # majd 32x32 pixelesre meretezzuk
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # A kep minden csatornajanak standardizalasa
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

Tesztelés során
csak standardizálást végzünk a képeken,
hogy a kiértékelési eredményekből
kizárjuk a véletlenszerűséget.

```{.python .input}
#@tab mxnet
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                           [0.2023, 0.1994, 0.2010])])
```

```{.python .input}
#@tab pytorch
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

## Az adathalmaz beolvasása

Ezután [**a rendszerezett, nyers képfájlokból álló adathalmazt olvassuk be**]. Minden példány egy képet és egy címkét tartalmaz.

```{.python .input}
#@tab mxnet
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ['train', 'valid', 'train_valid', 'test']]
```

```{.python .input}
#@tab pytorch
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]
```

Tanítás során
[**meg kell adnunk a fent definiált összes képaugmentációs műveletet**].
Ha a validációs halmazt a hiperparaméter-hangolás során a modell kiértékelésére használjuk,
nem szabad véletlenszerűséget bevezetni a képaugmentáció révén.
A végső előrejelzés előtt
a modellt a tanítóhalmaz és a validációs halmaz kombinációján tanítjuk be, hogy minden rendelkezésre álló felcímkézett adatot hasznosítsunk.

```{.python .input}
#@tab mxnet
train_iter, train_valid_iter = [gluon.data.DataLoader(
    dataset.transform_first(transform_train), batch_size, shuffle=True,
    last_batch='discard') for dataset in (train_ds, train_valid_ds)]

valid_iter = gluon.data.DataLoader(
    valid_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='discard')

test_iter = gluon.data.DataLoader(
    test_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='keep')
```

```{.python .input}
#@tab pytorch
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)
```

## A [**modell**] definiálása

:begin_tab:`mxnet`
Itt a `HybridBlock` osztályra alapozva építjük fel a reziduális blokkokat, ami kissé
eltér a :numref:`sec_resnet` szakaszban leírt implementációtól.
Ez a számítási hatékonyság javítása érdekében történik.
:end_tab:

```{.python .input}
#@tab mxnet
class Residual(nn.HybridBlock):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
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

    def hybrid_forward(self, F, X):
        Y = F.npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.npx.relu(Y + X)
```

:begin_tab:`mxnet`
Következő lépésként definiáljuk a ResNet-18 modellt.
:end_tab:

```{.python .input}
#@tab mxnet
def resnet18(num_classes):
    net = nn.HybridSequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))

    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
        return blk

    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

:begin_tab:`mxnet`
A tanítás megkezdése előtt a :numref:`subsec_xavier` szakaszban leírt Xavier-inicializálást alkalmazzuk.
:end_tab:

:begin_tab:`pytorch`
A :numref:`sec_resnet` szakaszban leírt ResNet-18 modellt definiáljuk.
:end_tab:

```{.python .input}
#@tab mxnet
def get_net(devices):
    num_classes = 10
    net = resnet18(num_classes)
    net.initialize(ctx=devices, init=init.Xavier())
    return net

loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net

loss = nn.CrossEntropyLoss(reduction="none")
```

## A [**tanítási függvény**] definiálása

A modelleket a validációs halmazon mutatott teljesítményük alapján választjuk ki és hangoljuk a hiperparamétereket.
Az alábbiakban definiáljuk a `train` modell-tanítási függvényt.

```{.python .input}
#@tab mxnet
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(
                net, features, labels.astype('float32'), loss, trainer,
                devices, d2l.split_batch)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpus(net, valid_iter,
                                                   d2l.split_batch)
            animator.add(epoch + 1, (None, None, valid_acc))
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
        scheduler.step()
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

## [**A modell tanítása és validálása**]

Most már taníthatjuk és validálhatjuk a modellt.
Az összes alábbi hiperparaméter hangolható.
Például növelhetjük az epochok számát.
Ha a `lr_period` és `lr_decay` értékét rendre 4-re és 0,9-re állítjuk, akkor az optimalizáló algoritmus tanulási rátáját minden 4 epochban 0,9-del szorozza meg. A szemléltetés egyszerűsége érdekében
itt csak 20 epochon tanítunk.

```{.python .input}
#@tab mxnet
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 0.02, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input}
#@tab pytorch
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4, 5e-4
lr_period, lr_decay, net = 4, 0.9, get_net()
net(next(iter(train_iter))[0])
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## [**A teszthalmaz osztályozása**] és az eredmények beküldése Kaggle-re

Miután egy megfelelő modellt kaptunk a hiperparaméterekkel,
az összes felcímkézett adatot (beleértve a validációs halmazt is) felhasználva újra tanítjuk a modellt, és osztályozzuk a teszthalmazt.

```{.python .input}
#@tab mxnet
net, preds = get_net(devices), []
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X.as_in_ctx(devices[0]))
    preds.extend(y_hat.argmax(axis=1).astype(int).asnumpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission.csv', index=False)
```

```{.python .input}
#@tab pytorch
net, preds = get_net(), []
net(next(iter(train_valid_iter))[0])
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)
```

A fenti kód
egy `submission.csv` fájlt fog létrehozni,
amelynek formátuma
megfelel a Kaggle-verseny követelményeinek.
Az eredmények Kaggle-re való beküldésének
módja hasonló a :numref:`sec_kaggle_house` szakaszban leírtakhoz.

## Összefoglalás

* A nyers képfájlokat tartalmazó adathalmazokat beolvashatjuk, ha előbb a szükséges formátumba rendszerezzük őket.

:begin_tab:`mxnet`
* Egy képosztályozási versenyen konvolúciós neurális hálózatokat, képaugmentációt és hibrid programozást alkalmazhatunk.
:end_tab:

:begin_tab:`pytorch`
* Egy képosztályozási versenyen konvolúciós neurális hálózatokat és képaugmentációt alkalmazhatunk.
:end_tab:

## Feladatok

1. Használd a teljes CIFAR-10 adathalmazt ehhez a Kaggle-versenyhez. Állítsd be a hiperparamétereket a következőre: `batch_size = 128`, `num_epochs = 100`, `lr = 0.1`, `lr_period = 50`, `lr_decay = 0.1`. Milyen pontosságot és helyezést tudsz elérni ezen a versenyen? Tovább lehet-e javítani?
1. Milyen pontosságot kapsz képaugmentáció nélkül?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/379)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1479)
:end_tab:
