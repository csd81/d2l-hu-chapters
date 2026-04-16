# tanulási ráta ütemezése
:label:`sec_scheduler`

Eddig elsősorban az optimalizálási *algoritmusokra* összpontosítottunk, arra, hogyan frissítsük a súlyvektorokat, nem pedig arra a *sebességre*, amellyel a frissítések történnek. Ennek ellenére a tanulási ráta módosítása éppoly fontos, mint maga az algoritmus. Számos szempontot kell figyelembe venni:

* A legkézenfekvőbben a tanulási ráta *nagysága* számít. Ha túl nagy, az optimalizálás divergál; ha túl kicsi, a tanítás túl sokáig tart, vagy szuboptimális eredményt kapunk. Korábban láttuk, hogy a probléma kondíciószáma számít (részletekért lásd pl. a :numref:`sec_momentum` szakaszt). Intuitívan ez a legkevésbé érzékeny irányban bekövetkező változás mértékének és a legérzékenyebb irányban bekövetkezőnek az aránya.
* Másodszor, az üteme is éppoly fontos. Ha a tanulási ráta nagy marad, elképzelhető, hogy egyszerűen a minimum körül pattogunk, és nem érjük el az optimalitást. A :numref:`sec_mini-batch_sgd` szakasz részletesen tárgyalta ezt, és a :numref:`sec_sgd` szakaszban teljesítménygaranciákat is elemeztük. Röviden: azt szeretnénk, hogy az ütem csökkenjön, de valószínűleg lassabban, mint $\mathcal{O}(t^{-\frac{1}{2}})$, ami konvex problémáknál jó választás lenne.
* Egy másik, éppoly fontos szempont az *inicializálás*. Ez vonatkozik arra is, hogyan állítjuk be kezdetben a paramétereket (részletekért lásd a :numref:`sec_numerical_stability` szakaszt), és hogyan fejlődnek kezdetben. Ezt *bemelegítésnek* (warmup) is nevezik: milyen gyorsan kezdünk el a megoldás felé haladni. A kezdeti nagy lépések nem feltétlenül hasznosak, különösen, mivel a paraméterek kezdeti készlete véletlenszerű. A kezdeti frissítési irányok is meglehetősen értelmetlenek lehetnek.
* Végül vannak olyan optimalizálási változatok, amelyek ciklikus tanulási ráta módosítást végeznek. Ez meghaladja az aktuális fejezet kereteit. Javasoljuk az olvasónak, hogy tekintse meg :citet:`Izmailov.Podoprikhin.Garipov.ea.2018` részleteit, pl. hogyan kaphatunk jobb megoldásokat a paraméterek teljes *útjának* átlagolásával.

Tekintettel arra, hogy a tanulási ráták kezeléséhez sok részletre van szükség, a legtöbb mélytanulási keretrendszer rendelkezik eszközökkel ennek automatikus kezeléséhez. Az aktuális fejezetben áttekintjük, milyen hatással vannak a különböző ütemezők a pontosságra, és azt is bemutatjuk, hogyan kezelhető ez hatékonyan egy *tanulási ráta ütemező* segítségével.

## Játékprobléma

Egy olyan játékproblémával kezdünk, amely elég olcsón számítható, mégis kellőképpen nem-triviális ahhoz, hogy szemléltesse a kulcsaspektusokat. Ehhez a LeNet egy kissé modernizált változatát választjuk (`relu` aktiváció `sigmoid` helyett, MaxPooling AveragePooling helyett), a Fashion-MNIST adathalmazra alkalmazva. Ráadásul hibridizáljuk a hálózatot a teljesítmény érdekében. Mivel a kód nagy része szabványos, csupán az alapokat mutatjuk be részletes tárgyalás nélkül. Szükség esetén lásd a :numref:`chap_cnn` fejezetet áttekintésként.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, lr_scheduler, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.HybridSequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# A kód szinte azonos a `d2l.train_ch6` függvénnyel, amelyet a
# konvolúciós neurális hálózatok fejezet lenet szakaszában definiáltunk
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device):
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.optim import lr_scheduler

def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# A kód szinte azonos a `d2l.train_ch6` függvénnyel, amelyet a
# konvolúciós neurális hálózatok fejezet lenet szakaszában definiáltunk
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
          scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Beépített PyTorch ütemező használata
                scheduler.step()
            else:
                # Egyénileg definiált ütemező használata
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# A kód szinte azonos a `d2l.train_ch6` függvénnyel, amelyet a
# konvolúciós neurális hálózatok fejezet lenet szakaszában definiáltunk
def train(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu(), custom_callback = False):
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    if custom_callback is False:
        net.fit(train_iter, epochs=num_epochs, verbose=0,
                callbacks=[callback])
    else:
         net.fit(train_iter, epochs=num_epochs, verbose=0,
                 callbacks=[callback, custom_callback])
    return net
```

Nézzük meg, mi történik, ha az alapértelmezett beállításokkal hívjuk meg az algoritmust, például $0.3$-as tanulási rátával, $30$ iteráción át tanítva. Figyeljük meg, hogy a tanítási pontosság folyamatosan növekszik, miközben a teszt pontosságában mért előrehaladás egy ponton megáll. A két görbe közötti rés túlillesztést jelzi.

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.3, 30
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab tensorflow
lr, num_epochs = 0.3, 30
train(net, train_iter, test_iter, num_epochs, lr)
```

## Ütemezők

A tanulási ráta módosításának egyik módja az explicit megadás minden lépésnél. Ez kényelmesen megvalósítható a `set_learning_rate` módszerrel. Minden epoc után (vagy akár minden mini-batch után) csökkenthetjük, pl. dinamikus módon, az optimalizálás előrehaladásának megfelelően.

```{.python .input}
#@tab mxnet
trainer.set_learning_rate(0.1)
print(f'learning rate is now {trainer.learning_rate:.2f}')
```

```{.python .input}
#@tab pytorch
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')
```

```{.python .input}
#@tab tensorflow
lr = 0.1
dummy_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
dummy_model.compile(tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
print(f'learning rate is now ,', dummy_model.optimizer.lr.numpy())
```

Általánosabban ütemezőt szeretnénk definiálni. Ha a frissítések számával hívjuk meg, visszaadja a tanulási ráta megfelelő értékét. Definiáljunk egy egyszerűt, amely a tanulási rátát $\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$-re állítja.

```{.python .input}
#@tab all
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
```

Jelenítsük meg a viselkedését értéktartomány felett.

```{.python .input}
#@tab all
scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Nézzük meg, hogyan érvényesül ez a Fashion-MNIST tanításakor. Egyszerűen az ütemezőt kiegészítő argumentumként adjuk át a tanítási algoritmusnak.

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

Ez lényegesen jobban működött, mint korábban. Két dolog tűnik ki: a görbe simább volt, mint korábban. Másodszor, kevesebb volt a túlillesztés. Sajnos nem teljesen megoldott kérdés, hogy *elméletileg* miért vezet egyes stratégiák kevesebb túlillesztéshez. Van némi érvelés amellett, hogy a kisebb lépésméretek nullához közelibb paramétereket eredményeznek, és ezáltal egyszerűbbeket. Ez azonban nem magyarázza teljes mértékben a jelenséget, mivel nem állítjuk le korán a tanítást, csupán óvatosan csökkentjük a tanulási rátát.

## Stratégiák

Bár nem tudjuk az összes tanulási ráta ütemező változatát áttekinteni, az alábbiakban rövid összefoglalót nyújtunk a népszerű stratégiákról. Általánosan alkalmazott választások a polinomiális csökkentés és a lépésenkénti konstans ütemek. Ezen túl a koszinusz tanulási ráta ütemezők bizonyos problémáknál empirikusan jól teljesítenek. Végül egyes problémáknál előnyös lehet az optimalizálót nagy tanulási ráták alkalmazása előtt bemelegíteni.

### Faktoros ütemező

A polinomiális csökkentés alternatívája a szorzó jellegű, vagyis $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ valamely $\alpha \in (0, 1)$ esetén. Annak megakadályozása érdekében, hogy a tanulási ráta egy ésszerű alsó határ alá csökkenjen, a frissítési egyenletet gyakran módosítják: $\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$.

```{.python .input}
#@tab all
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(d2l.arange(50), [scheduler(t) for t in range(50)])
```

Ez az MXNet-ben is elérhető beépített ütemezővel, a `lr_scheduler.FactorScheduler` objektumon keresztül. Néhány további paramétert vesz fel, például bemelegítési periódust, bemelegítési módot (lineáris vagy konstans), a kívánt frissítések maximális számát stb. A továbbiakban szükség esetén a beépített ütemezőket alkalmazzuk, és csak a funkcionalitásukat magyarázzuk el. Ahogy látható, saját ütemező létrehozása is meglehetősen egyszerű szükség esetén.

### Többlépéses faktoros ütemező

Mély hálózatok tanításának egyik elterjedt stratégiája a tanulási ráta lépésenkénti konstanson tartása és meghatározott időközönként egy adott mértékű csökkentés. Vagyis egy $s = \{5, 10, 20\}$ időpontok halmazát megadva $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ csökkentés következik be, ha $t \in s$. Feltéve, hogy az értékek minden lépésnél felezők, az alábbiak szerint implementálhatjuk.

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5,
                                              base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(d2l.arange(num_epochs), [get_lr(trainer, scheduler)
                                  for t in range(num_epochs)])
```

```{.python .input}
#@tab tensorflow
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr

    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

A lépésenkénti konstans tanulási ráta ütemező mögötti intuíció az, hogy az optimalizálást addig hagyjuk haladni, amíg a súlyvektorok eloszlása tekintetében stacionárius pontra nem jutunk. Csak ekkor (és nem korábban) csökkentjük a sebességet, hogy egy jó lokális minimum jobb minőségű közelítőjét kapjuk. Az alábbi példa bemutatja, hogyan eredményez ez minden esetben enyhén jobb megoldásokat.

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### Koszinusz ütemező

Meglehetősen meglepő heurisztikát javasolt :citet:`Loshchilov.Hutter.2016`. Arra az megfigyelésre épül, hogy elképzelhető, hogy nem szeretnénk a tanulási rátát túlságosan drasztikusan csökkenteni az elején, és ráadásul a végén nagyon kis tanulási rátával szeretnénk „finomítani" a megoldást. Ez koszinuszszerű ütemezést eredményez, amelynek funkcionális alakja a $t \in [0, T]$ tartományon lévő tanulási rátákre:

$$\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2} \left(1 + \cos(\pi t/T)\right)$$


ahol $\eta_0$ a kezdeti tanulási ráta, $\eta_T$ a $T$ időpontbeli célsebesség. Ráadásul $t > T$ esetén egyszerűen $\eta_T$-n tartjuk az értéket, növelés nélkül. Az alábbi példában a maximális frissítési lépést $T = 20$-ra állítjuk.

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(max_update=20, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

A számítógépes látás kontextusában ez az ütemezés *javíthat* az eredményeken. Fontos azonban megjegyezni, hogy az ilyen javulások nem garantáltak (amint az alábbiakban látható).

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### Bemelegítés

Egyes esetekben a paraméterek inicializálása önmagában nem elégséges a jó megoldás garantálásához. Ez különösen problémás egyes fejlett hálózati architektúráknál, amelyek instabil optimalizálási problémákhoz vezethetnek. Ezt kezelhetjük kellően kis tanulási ráta megválasztásával a divergencia megelőzéséhez az elején. Sajnos ez lassú előrehaladást jelent. Ezzel szemben a kezdeti nagy tanulási ráta divergenciához vezet.

Ennek a dilemmának egy meglehetősen egyszerű megoldása egy bemelegítési periódus alkalmazása, amely alatt a tanulási ráta *növekszik* a kezdeti maximumig, majd az optimalizálási folyamat végéig csökken. Az egyszerűség kedvéért erre általában lineáris növelést alkalmaznak. Ez az alábbi alakú ütemezéshez vezet.

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(20, warmup_steps=5, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Megjegyezzük, hogy a hálózat kezdetben jobban konvergál (különösen figyeljük meg a teljesítményt az első 5 epochban).

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

A bemelegítés bármely ütemezőre alkalmazható (nem csak a koszinuszra). A tanulási ráta ütemezők és számos további kísérlet részletesebb tárgyalásához lásd :cite:`Gotmare.Keskar.Xiong.ea.2018`. Különösen azt találják, hogy egy bemelegítési szakasz korlátozza a paraméterek eltérésének mértékét nagyon mély hálózatokban. Ez intuitívan is érthető, mivel a véletlenszerű inicializáció miatt várható jelentős eltérés a hálózat azon részeinél, amelyeknek a legtöbb időre van szükségük az előrehaladáshoz az elején.

## Összefoglalás

* A tanulási ráta csökkentése a tanítás során javíthatja a pontosságot és (talán meglepő módon) csökkentheti a modell túlillesztését.
* A tanulási ráta lépésenkénti csökkentése, ha a haladás megállt, hatékony a gyakorlatban. Ez lényegében biztosítja, hogy hatékonyan konvergálunk egy megfelelő megoldáshoz, és csak ezután csökkentjük a paraméterek inherens varianciáját a tanulási ráta csökkentésével.
* A koszinusz ütemezők népszerűek egyes számítógépes látási problémáknál. Részletekért lásd pl. a [GluonCV](http://gluon-cv.mxnet.io) oldalt.
* Az optimalizálás előtti bemelegítési periódus megakadályozhatja a divergenciát.
* Az optimalizálás a mélytanulásban több célt szolgál. A tanítási célfüggvény minimalizálásán kívül az optimalizálási algoritmusok és a tanulási ráta ütemezők különböző megválasztása meglehetősen eltérő általánosítási és túlillesztési mértéket eredményezhet a teszthalmazon (azonos tanítási hiba mellett).

## Gyakorló feladatok

1. Kísérletezz az optimalizálási viselkedéssel adott rögzített tanulási ráta esetén. Milyen a legjobb modell, amit így lehet elérni?
1. Hogyan változik a konvergencia, ha megváltoztatod a tanulási ráta csökkentésének kitevőjét? Kényelmed érdekében a kísérletekhez használd a `PolyScheduler`-t.
1. Alkalmazd a koszinusz ütemezőt nagy számítógépes látási problémákra, pl. az ImageNet tanításakor. Hogyan befolyásolja a teljesítményt más ütemezőkhöz képest?
1. Mennyi ideig tartson a bemelegítés?
1. Összekapcsolható-e az optimalizálás és a mintavételezés? Kezdd :citet:`Welling.Teh.2011` sztochasztikus gradiens Langevin-dinamikára vonatkozó eredményeivel.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/359)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1080)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1081)
:end_tab:
