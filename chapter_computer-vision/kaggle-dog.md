# Kutyafajta-azonosítás (ImageNet Dogs) Kaggle-ön

Ebben a részben a Kaggle-ön szereplő kutyafajta-azonosítási feladatot gyakoroljuk.
(**A verseny webcíme: https://www.kaggle.com/c/dog-breed-identification**)

Ebben a versenyben
120 különböző kutyafajtát kell felismerni.
Az adathalmaz valójában
az ImageNet adathalmaz egy részhalmazát alkotja.
A :numref:`sec_kaggle_cifar10` szakaszban szereplő CIFAR-10 adathalmaz képeivel ellentétben
az ImageNet adathalmaz képei változó méretűek, és mind magasságban, mind szélességben nagyobbak.
A :numref:`fig_kaggle_dog` ábra a verseny weboldalán található információkat mutatja. Az eredmények beküldéséhez Kaggle-fiókra van szükség.


![A kutyafajta-azonosítási verseny weboldala. A verseny adathalmazát az „Adatok" fülre kattintva lehet letölteni.](../img/kaggle-dog.jpg)
:width:`400px`
:label:`fig_kaggle_dog`

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import os
```

## Az adathalmaz beszerzése és rendszerezése

A verseny adathalmazát egy tanítóhalmazra és egy teszthalmazra osztják fel, amelyek rendre 10222 és 10357 JPEG képet tartalmaznak három RGB (színes) csatornával.
A tanítóadatok között
120 kutyafajta szerepel,
például labrador retriever, uszkár, tacskó, szamojéd, husky, csivava és yorkshire terrier.


### Az adathalmaz letöltése

Miután bejelentkeztél a Kaggle-re,
kattints az „Adatok" fülre
a :numref:`fig_kaggle_dog` ábrán látható versenyoldalon, majd kattints az „Összes letöltése" gombra az adathalmaz letöltéséhez.
A letöltött fájl `../data` mappába való kicsomagolása után a teljes adathalmazt a következő elérési utakon találod:

* ../data/dog-breed-identification/labels.csv
* ../data/dog-breed-identification/sample_submission.csv
* ../data/dog-breed-identification/train
* ../data/dog-breed-identification/test

Észreveheted, hogy a fenti struktúra hasonló
a :numref:`sec_kaggle_cifar10` szakaszban szereplő CIFAR-10 verseny struktúrájához: a `train/` és `test/` mappák rendre a tanítási és tesztelési kutyaképeket tartalmazzák, a `labels.csv` pedig a tanítóképek címkéit tárolja.
Hasonlóan, a könnyebb kezdés érdekében [**az adathalmaz egy kis mintáját is elérhetővé tesszük**]: `train_valid_test_tiny.zip`.
Ha a teljes adathalmazt szeretnéd használni a Kaggle-versenyen, az alábbi `demo` változót állítsd `False` értékre.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

# Ha a Kaggle-versenyhez letoltott teljes adathalmazt hasznalod,
# allitsd az alabbi valtozot `False`-ra
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')
```

### [**Az adathalmaz rendszerezése**]

Az adathalmazt a :numref:`sec_kaggle_cifar10` szakaszhoz hasonló módon rendszerezhetjük: az eredeti tanítóhalmazból kiválasztunk egy validációs halmazt, és a képeket a címkék szerint csoportosított almappákba helyezzük.

Az alábbi `reorg_dog_data` függvény beolvassa
a tanítóadatok címkéit, kialakítja a validációs halmazt, és rendszerezi a tanítóhalmazt.

```{.python .input}
#@tab all
def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)


batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)
```

## [**Képaugmentálás**]

Ne feledjük, hogy ez a kutyafajta-adathalmaz
az ImageNet adathalmaz egy részhalmazát alkotja,
amelynek képei
nagyobbak, mint a :numref:`sec_kaggle_cifar10` szakaszban szereplő
CIFAR-10 adathalmaz képei.
Az alábbiakban felsorolunk néhány képaugmentálási műveletet,
amelyek hasznosak lehetnek a viszonylag nagyobb képek esetén.

```{.python .input}
#@tab mxnet
transform_train = gluon.data.vision.transforms.Compose([
    # A kep veletlen vagasa, hogy az eredeti terulet 0,08-1-szeresenek
    # megfelelo teruletu, 3/4 es 4/3 kozotti magassag-szelesseg aranyu
    # kepet kapjunk. Majd atmeretezzuk 224 x 224 pixelesre
    gluon.data.vision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                   ratio=(3.0/4.0, 4.0/3.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    # A fenyero, kontraszt es telitettseg veletlen valtoztatasa
    gluon.data.vision.transforms.RandomColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4),
    # Veletlen zaj hozzaadasa
    gluon.data.vision.transforms.RandomLighting(0.1),
    gluon.data.vision.transforms.ToTensor(),
    # A kep minden csatornajanak standardizalasa
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_train = torchvision.transforms.Compose([
    # A kep veletlen vagasa, hogy az eredeti terulet 0,08-1-szeresenek
    # megfelelo teruletu, 3/4 es 4/3 kozotti magassag-szelesseg aranyu
    # kepet kapjunk. Majd atmeretezzuk 224 x 224 pixelesre
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # A fenyero, kontraszt es telitettseg veletlen valtoztatasa
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # Veletlen zaj hozzaadasa
    torchvision.transforms.ToTensor(),
    # A kep minden csatornajanak standardizalasa
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

Az előrejelzés során
csak véletlenszerűség nélküli képelőfeldolgozási műveleteket alkalmazunk.

```{.python .input}
#@tab mxnet
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    # 224 x 224 pixeles negyzet kivagasa a kep kozeppontjabol
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # 224 x 224 pixeles negyzet kivagasa a kep kozeppontjabol
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

## [**Az adathalmaz beolvasása**]

A :numref:`sec_kaggle_cifar10` szakaszhoz hasonlóan
a rendszerezett adathalmazt nyers képfájlokból olvashatjuk be.

```{.python .input}
#@tab mxnet
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ('train', 'valid', 'train_valid', 'test')]
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

Az alábbiakban adatiterátorokat hozunk létre
ugyanúgy, ahogyan a :numref:`sec_kaggle_cifar10` szakaszban tettük.

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

## [**Előtanított modell finomhangolása**]

Mivel ez a verseny adata az ImageNet adathalmaz egy részhalmaza,
alkalmazhatjuk a :numref:`sec_fine_tuning` szakaszban tárgyalt megközelítést:
választunk egy, a teljes ImageNet adathalmazon előtanított modellt, amellyel képjellemzőket nyerünk ki, majd ezeket egy egyedi, kisebb kimeneti hálózatba vezetjük.
A deep learning keretrendszerek magas szintű API-jai
számos, az ImageNet adathalmazon előtanított modellt kínálnak.
Ebben az esetben
egy előtanított ResNet-34 modellt választunk,
amelynek kimeneti rétegéhez vezető bemenetet
(azaz a kinyert jellemzőket) közvetlenül újra felhasználjuk.
Ezután az eredeti kimeneti réteget lecseréljük egy kis egyedi kimeneti hálózatra, amely tanítható,
például két teljesen összekötött réteget egymásra helyezve.
A :numref:`sec_fine_tuning` szakasz kísérletétől eltérően
az alábbiakban nem tanítjuk újra a jellemzőkinyerésre használt előtanított modellt. Ez csökkenti a tanítási időt és
a gradiensek tárolásához szükséges memóriát.

Ne felejtsük el, hogy a képeket a teljes ImageNet adathalmaz három RGB-csatornájának átlagai és szórásai alapján standardizáltuk.
Ez valójában összhangban van
az ImageNet-en előtanított modell által alkalmazott standardizálási eljárással.

```{.python .input}
#@tab mxnet
def get_net(devices):
    finetune_net = gluon.model_zoo.vision.resnet34_v2(pretrained=True)
    # Uj kimeneti halozat definialasa
    finetune_net.output_new = nn.HybridSequential(prefix='')
    finetune_net.output_new.add(nn.Dense(256, activation='relu'))
    # 120 kimeneti kategoria van
    finetune_net.output_new.add(nn.Dense(120))
    # A kimeneti halozat inicializalasa
    finetune_net.output_new.initialize(init.Xavier(), ctx=devices)
    # A modellparameterek elosztasa a szamitashoz hasznalt CPU-kra/GPU-kra
    finetune_net.collect_params().reset_ctx(devices)
    return finetune_net
```

```{.python .input}
#@tab pytorch
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # Uj kimeneti halozat definialasa (120 kimeneti kategoria van)
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # A modell athelyezese az eszkozokre
    finetune_net = finetune_net.to(devices[0])
    # A jellemzoretegek parametereinek befagyasztasa
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
```

Mielőtt [**a veszteséget kiszámítanánk**],
először megkapjuk az előtanított modell kimeneti rétegének bemenetét, vagyis a kinyert jellemzőket.
Ezután ezt a jellemzőt adjuk be az egyedi kisméretű kimeneti hálózatunkba a veszteség kiszámításához.

```{.python .input}
#@tab mxnet
loss = gluon.loss.SoftmaxCrossEntropyLoss()

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        X_shards, y_shards = d2l.split_batch(features, labels, devices)
        output_features = [net.features(X_shard) for X_shard in X_shards]
        outputs = [net.output_new(feature) for feature in output_features]
        ls = [loss(output, y_shard).sum() for output, y_shard
              in zip(outputs, y_shards)]
        l_sum += sum([float(l.sum()) for l in ls])
        n += labels.size
    return l_sum / n
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return l_sum / n
```

## [**A tanítási függvény**] meghatározása

A modellt és a hiperparamétereket a validációs halmazon elért teljesítmény alapján választjuk ki és hangoljuk. A `train` tanítási függvény csak
az egyedi kisméretű kimeneti hálózat paramétereit iterálja.

```{.python .input}
#@tab mxnet
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Csak a kis egyedi kimeneti halozatot tanitjuk
    trainer = gluon.Trainer(net.output_new.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            X_shards, y_shards = d2l.split_batch(features, labels, devices)
            output_features = [net.features(X_shard) for X_shard in X_shards]
            with autograd.record():
                outputs = [net.output_new(feature)
                           for feature in output_features]
                ls = [loss(output, y_shard).sum() for output, y_shard
                      in zip(outputs, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            metric.add(sum([float(l.sum()) for l in ls]), labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss))
    measures = f'train loss {metric[0] / metric[1]:.3f}'
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Csak a kis egyedi kimeneti halozatot tanitjuk
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach().cpu()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

## [**A modell tanítása és érvényesítése**]

Most taníthatjuk és érvényesíthetjük a modellt.
Az alábbi hiperparaméterek mind hangolhatók.
Például az epochok száma növelhető. Mivel `lr_period` és `lr_decay` értéke rendre 2 és 0,9, az optimalizáló algoritmus tanulási rátáját minden 2. epoch után megszorozzuk 0,9-cel.

```{.python .input}
#@tab mxnet
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 5e-3, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input}
#@tab pytorch
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## [**A teszthalmaz osztályozása**] és az eredmények beküldése Kaggle-re


A :numref:`sec_kaggle_cifar10` szakasz utolsó lépéséhez hasonlóan
végül az összes felcímkézett adatot (beleértve a validációs halmazt is) a modell tanítására és a teszthalmaz osztályozására használjuk.
Az osztályozáshoz a tanított egyedi kimeneti hálózatot alkalmazzuk.

```{.python .input}
#@tab mxnet
net = get_net(devices)
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output_features = net.features(data.as_in_ctx(devices[0]))
    output = npx.softmax(net.output_new(output_features))
    preds.extend(output.asnumpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

```{.python .input}
#@tab pytorch
net = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=1)
    preds.extend(output.cpu().detach().numpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

A fenti kód
létrehoz egy `submission.csv` fájlt,
amelyet a :numref:`sec_kaggle_house` szakaszban leírt módon
lehet beküldeni a Kaggle-re.


## Összefoglalás


* Az ImageNet adathalmaz képei nagyobbak (változó méretűek), mint a CIFAR-10 képei. Más adathalmazra vonatkozó feladatoknál módosíthatjuk a képaugmentálási műveleteket.
* Az ImageNet adathalmaz egy részhalmazának osztályozásához kihasználhatjuk a teljes ImageNet adathalmazon előtanított modelleket a jellemzők kinyerésére, és csak egy egyedi kisméretű kimeneti hálózatot tanítunk. Ez kevesebb számítási időt és memóriaköltséget eredményez.


## Feladatok

1. A teljes Kaggle-verseny adathalmazát használva milyen eredményeket érhetsz el, ha növeled a `batch_size` (batch méretét) és a `num_epochs` (epochok számát), miközben a többi hiperparamétert a következőképpen állítod be: `lr = 0.01`, `lr_period = 10` és `lr_decay = 0.1`?
1. Jobb eredményeket érsz el, ha mélyebb előtanított modellt használsz? Hogyan hangolod a hiperparamétereket? Tovább javíthatod az eredményeket?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/380)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1481)
:end_tab:
