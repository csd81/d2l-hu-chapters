# Több GPU tömör implementációja
:label:`sec_multi_gpu_concise`

A párhuzamosság minden új modellhez való nulláról való implementálása nem kellemes feladat. Ráadásul a szinkronizációs eszközök magas teljesítményre való optimalizálásából komoly előnyök származnak. A következőkben bemutatjuk, hogyan valósítsuk ezt meg a deep learning keretrendszerek magas szintű API-jainak segítségével.
A matematika és az algoritmusok ugyanazok, mint a :numref:`sec_multi_gpu`-ban.
Nem meglepő módon ehhez a szakaszhoz legalább két GPU szükséges a kód futtatásához.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## [**Egy egyszerű hálózat**]

Használjunk egy valamivel értelmesebb hálózatot, mint a :numref:`sec_multi_gpu`-ban szereplő LeNet, amelyet még mindig elég könnyű és gyors tanítani.
Egy ResNet-18 variánst választunk :cite:`He.Zhang.Ren.ea.2016`. Mivel a bemeneti képek kicsik, kissé módosítjuk. Különösen a :numref:`sec_resnet`-től való eltérés az, hogy kisebb konvolúciós kernelt, lépésközt és párnázást használunk a kezdetben.
Emellett eltávolítjuk a max-pooling réteget.

```{.python .input}
#@tab mxnet
#@save
def resnet18(num_classes):
    """Egy kissé módosított ResNet-18 modell."""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # Ez a modell kisebb konvolúciós kernelt, lépésközt és párnázást használ,
    # és eltávolítja a max-pooling réteget
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

```{.python .input}
#@tab pytorch
#@save
def resnet18(num_classes, in_channels=1):
    """Egy kissé módosított ResNet-18 modell."""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(out_channels, use_1x1conv=True, 
                                        strides=2))
            else:
                blk.append(d2l.Residual(out_channels))
        return nn.Sequential(*blk)

    # Ez a modell kisebb konvolúciós kernelt, lépésközt és párnázást használ,
    # és eltávolítja a max-pooling réteget
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

## Hálózat inicializálása

:begin_tab:`mxnet`
Az `initialize` függvény lehetővé teszi a paraméterek inicializálását tetszőleges eszközön.
Az inicializálási módszerek felfrissítéséhez lásd :numref:`sec_numerical_stability`. Különösen kényelmes, hogy egyszerre *több* eszközön is inicializálhatjuk a hálózatot. Nézzük meg, hogyan működik ez a gyakorlatban.
:end_tab:

:begin_tab:`pytorch`
A hálózatot a tanítási cikluson belül inicializáljuk.
Az inicializálási módszerek felfrissítéséhez lásd :numref:`sec_numerical_stability`.
:end_tab:

```{.python .input}
#@tab mxnet
net = resnet18(10)
# GPU-k listájának lekérése
devices = d2l.try_all_gpus()
# A hálózat összes paraméterének inicializálása
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

```{.python .input}
#@tab pytorch
net = resnet18(10)
# GPU-k listájának lekérése
devices = d2l.try_all_gpus()
# A hálózatot a tanítási cikluson belül inicializáljuk
```

:begin_tab:`mxnet`
A :numref:`sec_multi_gpu`-ban bemutatott `split_and_load` függvénnyel egy minibatch adatot eloszthatunk, és az egyes részeket átmásolhatjuk a `devices` változó által megadott eszközök listájára. A hálózatpéldány *automatikusan* a megfelelő GPU-t használja az előterjesztés értékének kiszámításához. Itt 4 megfigyelést generálunk, és elosztjuk őket a GPU-k között.
:end_tab:

```{.python .input}
#@tab mxnet
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

:begin_tab:`mxnet`
Miután az adatok átmentek a hálózaton, a megfelelő paraméterek *azon az eszközön* inicializálódnak, amelyen az adatok áthaladtak.
Ez azt jelenti, hogy az inicializálás eszközönként történik. Mivel a 0-s és az 1-es GPU-t választottuk inicializáláshoz, a hálózat csak ott inicializálódik, a CPU-n nem. Valójában a paraméterek a CPU-n nem is léteznek. Ezt ellenőrizhetjük a paraméterek kiírásával, és figyelhetjük az esetlegesen felmerülő hibákat.
:end_tab:

```{.python .input}
#@tab mxnet
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(devices[0])[0], weight.data(devices[1])[0]
```

:begin_tab:`mxnet`
Ezután cseréljük le a [**pontosság kiértékelésére**] szolgáló kódot olyanra, amely (**párhuzamosan működik több eszközön**). Ez a :numref:`sec_lenet`-ben szereplő `evaluate_accuracy_gpu` függvény felváltásaként szolgál. A fő különbség az, hogy a hálózat meghívása előtt felosztjuk a minibatch-et. Minden más lényegében azonos.
:end_tab:

```{.python .input}
#@tab mxnet
#@save
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    """Egy modell pontosságának kiszámítása adathalmazon, több GPU-val."""
    # Az eszközök listájának lekérdezése
    devices = list(net.collect_params().values())[0].list_ctx()
    # Helyes előrejelzések száma, összes előrejelzés száma
    metric = d2l.Accumulator(2)
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        # Párhuzamos futtatás
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```

## [**Tanítás**]

Mint korábban, a tanítási kódnak több alapvető funkciót kell ellátnia a hatékony párhuzamossághoz:

* A hálózat paramétereit minden eszközön inicializálni kell.
* Az adathalmazon való iterálás során a minibatch-eket el kell osztani az összes eszköz között.
* A veszteséget és a gradiensét párhuzamosan számítjuk ki az eszközökön.
* A gradienseket összegyűjtjük, és ennek megfelelően frissítjük a paramétereket.

Végül kiszámítjuk a pontosságot (ismét párhuzamosan), hogy jelenthessük a hálózat végső teljesítményét. A tanítási rutin igen hasonló az előző fejezetekben szereplő implementációkhoz, azzal a különbséggel, hogy az adatokat fel kell osztani és össze kell gyűjteni.

```{.python .input}
#@tab mxnet
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [d2l.try_gpu(i) for i in range(num_gpus)]
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for features, labels in train_iter:
            X_shards, y_shards = d2l.split_batch(features, labels, ctx)
            with autograd.record():
                ls = [loss(net(X_shard), y_shard) for X_shard, y_shard
                      in zip(X_shards, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        npx.waitall()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpus(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(ctx)}')
```

```{.python .input}
#@tab pytorch
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(module):
        if type(module) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)
    # A modell beállítása több GPU-ra
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

Nézzük meg, hogyan működik ez a gyakorlatban. Bemelegítésképpen [**egyetlen GPU-n tanítjuk a hálózatot.**]

```{.python .input}
#@tab mxnet
train(num_gpus=1, batch_size=256, lr=0.1)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

Ezután [**2 GPU-t használunk a tanításhoz**]. A :numref:`sec_multi_gpu`-ban értékelt LeNet-hez képest
a ResNet-18 modell lényegesen összetettebb. Ez az a pont, ahol a párhuzamosítás megmutatja előnyét. A számítás ideje érdemben nagyobb, mint a paraméterek szinkronizálásának ideje. Ez javítja a skálázhatóságot, mivel a párhuzamosítás többletterhe kevésbé releváns.

```{.python .input}
#@tab mxnet
train(num_gpus=2, batch_size=512, lr=0.2)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## Összefoglalás

:begin_tab:`mxnet`
* A Gluon primitíveket biztosít a modell inicializálásához több eszközön, egy kontextuslista megadásával.
:end_tab:
* Az adatok automatikusan azon az eszközön kerülnek kiértékelésre, ahol megtalálhatók.
* Ügyelj arra, hogy az egyes eszközökön inicializáld a hálózatot, mielőtt megpróbálod elérni az adott eszközön lévő paramétereket. Ellenkező esetben hibával találkozol.
* Az optimalizálási algoritmusok automatikusan összegyűjtik a gradienseket több GPU-ról.



## Gyakorlatok

:begin_tab:`mxnet`
1. Ez a szakasz a ResNet-18-at használja. Próbálj ki különböző epoch-számokat, batch-méreteket és tanulási rátákat. Használj több GPU-t a számításhoz. Mi történik, ha 16 GPU-val próbálod (pl. egy AWS p2.16xlarge példányon)?
1. Néha különböző eszközök különböző számítási teljesítményt nyújtanak. Egyszerre használhatjuk a GPU-kat és a CPU-t is. Hogyan osszuk el a munkát? Megéri a fáradságot? Miért igen? Miért nem?
1. Mi történik, ha elhagyjuk az `npx.waitall()` hívást? Hogyan módosítanád a tanítást úgy, hogy legfeljebb két lépés átfedése legyen a párhuzamosítás érdekében?
:end_tab:

:begin_tab:`pytorch`
1. Ez a szakasz a ResNet-18-at használja. Próbálj ki különböző epoch-számokat, batch-méreteket és tanulási rátákat. Használj több GPU-t a számításhoz. Mi történik, ha 16 GPU-val próbálod (pl. egy AWS p2.16xlarge példányon)?
1. Néha különböző eszközök különböző számítási teljesítményt nyújtanak. Egyszerre használhatjuk a GPU-kat és a CPU-t is. Hogyan osszuk el a munkát? Megéri a fáradságot? Miért igen? Miért nem?
:end_tab:



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/365)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1403)
:end_tab:
