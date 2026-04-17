# Képaugmentáció
:label:`sec_image_augmentation`

A :numref:`sec_alexnet` fejezetben már szó volt arról, hogy a nagy adathalmazok elengedhetetlen feltételei a mély neurális hálózatok sikerének különböző alkalmazásokban.
A *képaugmentáció* a tanítóképek véletlen módosításainak sorozatával hasonló, de különböző tanítópéldákat generál, ezáltal bővíti a tanítóhalmaz méretét.
Másrészt a képaugmentáció indokolható azzal is, hogy a tanítópéldák véletlen módosítása lehetővé teszi, hogy a modellek kevésbé támaszkódjanak bizonyos tulajdonságokra, ezáltal javítva általánosítóképességüket.
Például különböző módokon vághatjuk ki a képet, hogy az érdeklődési objektum különböző pozíciókban jelenjen meg, csökkentve ezzel a modell érzékenységét az objektum helyzetére.
A szín és a fényerő módosításával csökkenthetjük a modell érzékenységét a képek színére.
Valószínűleg igaz, hogy a képaugmentáció elengedhetetlen volt az AlexNet akkori sikeréhez.
Ebben a fejezetben ezt a számítógépes látásban széles körben alkalmazott technikát tárgyaljuk.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
```

## Általános képaugmentációs módszerek

A gyakori képaugmentációs módszerek vizsgálatához a következő $400\times 500$ pixeles képet használjuk példaként.

```{.python .input}
#@tab mxnet
d2l.set_figsize()
img = image.imread('../img/cat1.jpg')
d2l.plt.imshow(img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img);
```

A legtöbb képaugmentációs módszernek van bizonyos fokú véletlenszerűsége. Hogy könnyebben megfigyelhessük a képaugmentáció hatását, definiálunk egy segédfüggvényt: `apply`. Ez a függvény az `aug` képaugmentációs módszert többször alkalmazza a bemeneti `img` képre, és megmutatja az összes eredményt.

```{.python .input}
#@tab all
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
```

### Tükrözés és vágás

:begin_tab:`mxnet`
[**A kép vízszintes tükrözése**] általában nem változtatja meg az objektum kategóriáját.
Ez az egyik legkorábbi és legszélesebb körben alkalmazott képaugmentációs módszer.
A következőkben a `transforms` modullal létrehozunk egy `RandomFlipLeftRight` példányt, amely 50%-os valószínűséggel vízszintesen tükrözi a képet.
:end_tab:

:begin_tab:`pytorch`
[**A kép vízszintes tükrözése**] általában nem változtatja meg az objektum kategóriáját.
Ez az egyik legkorábbi és legszélesebb körben alkalmazott képaugmentációs módszer.
A következőkben a `transforms` modullal létrehozunk egy `RandomHorizontalFlip` példányt, amely 50%-os valószínűséggel vízszintesen tükrözi a képet.
:end_tab:

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomFlipLeftRight())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomHorizontalFlip())
```

:begin_tab:`mxnet`
[**A kép függőleges tükrözése**] kevésbé elterjedt, mint a vízszintes tükrözés. Legalábbis ennél a példaképnél a függőleges tükrözés sem akadályozza a felismerést.
Ezután létrehozunk egy `RandomFlipTopBottom` példányt, amely 50%-os valószínűséggel függőlegesen tükrözi a képet.
:end_tab:

:begin_tab:`pytorch`
[**A kép függőleges tükrözése**] kevésbé elterjedt, mint a vízszintes tükrözés. Legalábbis ennél a példaképnél a függőleges tükrözés sem akadályozza a felismerést.
Ezután létrehozunk egy `RandomVerticalFlip` példányt, amely 50%-os valószínűséggel függőlegesen tükrözi a képet.
:end_tab:

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomFlipTopBottom())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomVerticalFlip())
```

A példaképünkön a macska a kép közepén van, de általában ez nem feltétlenül így van.
A :numref:`sec_pooling` fejezetben elmagyaráztuk, hogy a pooling réteg csökkenti a konvolúciós réteg érzékenységét a célpozícióra.
Emellett véletlenszerűen is kivághatjuk a képet, hogy az objektumok különböző pozíciókban és léptékekben jelenjenek meg, ami szintén csökkenti a modell érzékenységét a célpozícióra.

Az alábbi kódban [**véletlenszerűen kivágunk**] egy területet, amelynek mérete az eredeti terület $10\% \sim 100\%$-a, és e terület szélességének és magasságának aránya véletlenszerűen kerül kiválasztásra a $0.5 \sim 2$ tartományból. Ezután a kivágott terület szélességét és magasságát egyaránt 200 pixelre skálázzuk.
Ha másképpen nem jelezzük, ebben a fejezetben az $a$ és $b$ közötti véletlen szám az $[a, b]$ intervallumból egyenletesen véletlenszerűen vett folytonos értékre utal.

```{.python .input}
#@tab mxnet
shape_aug = gluon.data.vision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

```{.python .input}
#@tab pytorch
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

### Színek módosítása

Egy másik augmentációs módszer a színek módosítása. A képszín négy aspektusát változtathatjuk meg: fényerő, kontraszt, telítettség és árnyalat. Az alábbi példában [**véletlenszerűen módosítjuk a kép fényerejét**] az eredeti kép 50%-a ($1-0.5$) és 150%-a ($1+0.5$) közé.

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomBrightness(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
```

Hasonlóképpen [**véletlenszerűen módosíthatjuk a kép árnyalatát**] is.

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomHue(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
```

Létrehozhatunk egy `RandomColorJitter` példányt is, és megadhatjuk, hogyan [**módosítsuk egyszerre véletlenszerűen a kép `brightness` (fényerő), `contrast` (kontraszt), `saturation` (telítettség) és `hue` (árnyalat) értékeit**].

```{.python .input}
#@tab mxnet
color_aug = gluon.data.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

```{.python .input}
#@tab pytorch
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

### Több képaugmentációs módszer kombinálása

A gyakorlatban [**több képaugmentációs módszert kombinálunk**].
Például kombinálhatjuk a fent definiált különböző képaugmentációs módszereket, és minden képre alkalmazhatjuk őket egy `Compose` példányon keresztül.

```{.python .input}
#@tab mxnet
augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])
apply(img, augs)
```

```{.python .input}
#@tab pytorch
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
```

## [**Tanítás képaugmentációval**]

Tanítsunk egy modellt képaugmentációval.
Itt a korábban használt Fashion-MNIST adathalmaz helyett a CIFAR-10 adathalmazt használjuk.
Ez azért van, mert a Fashion-MNIST adathalmazban az objektumok pozíciója és mérete normalizált, míg a CIFAR-10 adathalmazban az objektumok színe és mérete jelentősebb különbségeket mutat.
A CIFAR-10 adathalmaz első 32 tanítóképe az alábbi ábrán látható.

```{.python .input}
#@tab mxnet
d2l.show_images(gluon.data.vision.CIFAR10(
    train=True)[:32][0], 4, 8, scale=0.8);
```

```{.python .input}
#@tab pytorch
all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);
```

A jóslás során határozott eredmények elérése érdekében általában csak a tanítópéldákra alkalmazzuk a képaugmentációt, és nem használunk véletlen műveleteket tartalmazó képaugmentációt a jósláskor.
[**Itt csak a legegyszerűbb véletlen vízszintes tükrözési módszert használjuk**]. Emellett egy `ToTensor` példányt használunk a képek mini-batch-ének a mélytanulás keretrendszer által igényelt formátumba konvertálásához, azaz 0 és 1 közötti 32 bites lebegőpontos számokká, amelyek alakja (batch méret, csatornák száma, magasság, szélesség).

```{.python .input}
#@tab mxnet
train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor()])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor()])
```

```{.python .input}
#@tab pytorch
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])
```

:begin_tab:`mxnet`
Ezután definiálunk egy segédfüggvényt, amely megkönnyíti a képek olvasását és a képaugmentáció alkalmazását.
A Gluon adathalmazai által biztosított `transform_first` függvény a képaugmentációt minden tanítópélda (kép és címke) első elemére, azaz a képre alkalmazza.
A `DataLoader` részletes bemutatásáért lásd: :numref:`sec_fashion_mnist`.
:end_tab:

:begin_tab:`pytorch`
Ezután [**definiálunk egy segédfüggvényt, amely megkönnyíti a képek olvasását és a képaugmentáció alkalmazását**].
A PyTorch adathalmazai által biztosított `transform` argumentum augmentációt alkalmaz a képek transzformálásához.
A `DataLoader` részletes bemutatásáért lásd: :numref:`sec_fashion_mnist`.
:end_tab:

```{.python .input}
#@tab mxnet
def load_cifar10(is_train, augs, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train,
        num_workers=d2l.get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader
```

### Többprocesszoros GPU-s tanítás

A ResNet-18 modellt a :numref:`sec_resnet` fejezetből a CIFAR-10 adathalmazon tanítjuk.
Emlékezz a több GPU-val való tanítás bemutatójára a :numref:`sec_multi_gpu_concise` fejezetből.
A következőkben [**definiálunk egy függvényt a modell több GPU-val való tanítására és kiértékelésére**].

```{.python .input}
#@tab mxnet
#@save
def train_batch_ch13(net, features, labels, loss, trainer, devices,
                     split_f=d2l.split_batch):
    """Tanítás egy mini-batch-re több GPU-val (a 13. fejezetben definiálva)."""
    X_shards, y_shards = split_f(features, labels, devices)
    with autograd.record():
        pred_shards = [net(X_shard) for X_shard in X_shards]
        ls = [loss(pred_shard, y_shard) for pred_shard, y_shard
              in zip(pred_shards, y_shards)]
    for l in ls:
        l.backward()
    # A `True` jelző lehetővé teszi az elavult gradiensű paramétereket,
    # ami később hasznos (pl. BERT finomhangolásánál)
    trainer.step(labels.shape[0], ignore_stale_grad=True)
    train_loss_sum = sum([float(l.sum()) for l in ls])
    train_acc_sum = sum(d2l.accuracy(pred_shard, y_shard)
                        for pred_shard, y_shard in zip(pred_shards, y_shards))
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab pytorch
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Tanítás egy mini-batch-re több GPU-val (a 13. fejezetben definiálva)."""
    if isinstance(X, list):
        # A BERT finomhangoláshoz szükséges (később tárgyaljuk)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab mxnet
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus(), split_f=d2l.split_batch):
    """Modell tanítása több GPU-val (a 13. fejezetben definiálva)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # Tanítási veszteség összege, tanítási pontosság összege,
        # példák száma, predikciók száma
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices, split_f)
            metric.add(l, acc, labels.shape[0], labels.size)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpus(net, test_iter, split_f)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Modell tanítása több GPU-val (a 13. fejezetben definiálva)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Tanítási veszteség összege, tanítási pontosság összege,
        # példák száma, predikciók száma
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

Most [**definiálhatjuk a `train_with_data_aug` függvényt, amely képaugmentációval tanítja a modellt**].
Ez a függvény megszerzi az összes elérhető GPU-t, az Adam optimalizálási algoritmust használja, képaugmentációt alkalmaz a tanítóadathalmazra, és végül meghívja az imént definiált `train_ch13` függvényt a modell tanítására és kiértékelésére.

```{.python .input}
#@tab mxnet
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10)
net.initialize(init=init.Xavier(), ctx=devices)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

```{.python .input}
#@tab pytorch
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)
net.apply(d2l.init_cnn)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    net(next(iter(train_iter))[0])
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

[**Tanítsuk a modellt**] véletlen vízszintes tükrözésen alapuló képaugmentációval.

```{.python .input}
#@tab all
train_with_data_aug(train_augs, test_augs, net)
```

## Összefoglalás

* A képaugmentáció a meglévő tanítóadatok alapján véletlenszerű képeket generál, javítva ezzel a modellek általánosítóképességét.
* A jóslás során határozott eredmények elérése érdekében általában csak a tanítópéldákra alkalmazzuk a képaugmentációt, és nem használunk véletlen műveleteket tartalmazó képaugmentációt a jósláskor.
* A mélytanulás keretrendszerek számos különböző képaugmentációs módszert biztosítanak, amelyek egyszerre alkalmazhatók.


## Feladatok

1. Tanítsd a modellt képaugmentáció nélkül: `train_with_data_aug(test_augs, test_augs)`. Hasonlítsd össze a tanítási és tesztelési pontosságot képaugmentáció alkalmazásával és anélkül. Alátámasztja-e ez az összehasonlító kísérlet azt az érvet, hogy a képaugmentáció csökkentheti a túlillesztést? Miért?
1. Kombinálj több különböző képaugmentációs módszert a CIFAR-10 adathalmazon való modellezés során. Javítja-e a teszt pontosságát?
1. Tekintsd meg a mélytanulás keretrendszer online dokumentációját. Milyen más képaugmentációs módszereket biztosít?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/367)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1404)
:end_tab:
