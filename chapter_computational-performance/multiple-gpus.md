# Tanítás több GPU-n
:label:`sec_multi_gpu`

Eddig azt tárgyaltuk, hogyan lehet modelleket hatékonyan tanítani CPU-kon és GPU-kon. Megmutattuk azt is, hogy a deep learning keretrendszerek hogyan teszik lehetővé a számítás és a kommunikáció automatikus párhuzamosítását közöttük a :numref:`sec_auto_para` részben. Szintén bemutattuk a :numref:`sec_use_gpu` részben, hogyan lehet felsorolni egy számítógép összes elérhető GPU-ját az `nvidia-smi` paranccsal.
Amit *nem* tárgyaltunk, az az, hogy a deep learning tanítást valójában hogyan lehet párhuzamosítani.
Ehelyett csak mellékesen jeleztük, hogy az adatokat valahogy el kell osztani több eszköz között, és működésre bírni a rendszert. A jelen rész kitölti ezeket a részleteket, és megmutatja, hogyan lehet egy hálózatot párhuzamosan tanítani a semmiből indulva. A magas szintű API-k funkcionalitásának kihasználásáról részletesebben a :numref:`sec_multi_gpu_concise` részben lesz szó.
Feltételezzük, hogy ismered a mini-batch sztochasztikus gradienscsökkenés algoritmusokat, mint amilyenek a :numref:`sec_mini-batch_sgd` részben szerepelnek.


## A feladat felosztása

Kezdjük egy egyszerű számítógépes látási feladattal és egy kissé elavult hálózattal, például olyannal, amelynek több konvolúciós és pooling rétege van, és esetleg néhány teljesen összekötött réteg is van a végén.
Vagyis kezdjük egy olyan hálózattal, amely meglehetősen hasonlít a LeNet :cite:`LeCun.Bottou.Bengio.ea.1998` vagy az AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012` hálózathoz.
Több GPU esetén (2, ha asztali szerver, 4 egy AWS g4dn.12xlarge példányon, 8 egy p3.16xlarge-on, vagy 16 egy p2.16xlarge-on) a tanítást úgy szeretnénk felosztani, hogy jó gyorsulást érjünk el, miközben egyszerű és reprodukálható tervezési döntésekből profitálunk. A több GPU végső soron mind a *memória*, mind a *számítási* kapacitást növeli. Röviden szólva a következő lehetőségeink vannak, ha adott egy tanítási adatok mini-batchje, amelyet osztályozni szeretnénk.

Először feloszthatjuk a hálózatot több GPU között. Vagyis minden GPU bemenetként kapja az egy adott rétegbe beáramló adatokat, feldolgozza azokat néhány egymást követő rétegen át, majd elküldi az adatokat a következő GPU-nak.
Ez lehetővé teszi, hogy nagyobb hálózatokkal dolgozzunk, mint amelyeket egyetlen GPU képes kezelni.
Emellett
a GPU-nkénti memóriaigény jól szabályozható (ez a teljes hálózat memóriaigényének csak egy töredéke).

Az egyes rétegek (és így a GPU-k) közötti interfész azonban szoros szinkronizálást igényel. Ez trükkös lehet, különösen akkor, ha a számítási terhelés nincs megfelelően elosztva a rétegek között. A probléma sok GPU esetén tovább súlyosbodik.
A rétegek közötti interfész
nagy mennyiségű adatátvitelt is igényel,
például aktivációk és gradiensek formájában.
Ez túlterhelheti a GPU-buszok sávszélességét.
Továbbá, a számításigényes, de szekvenciális műveletek felosztása nem triviális. Lásd például :citet:`Mirhoseini.Pham.Le.ea.2017` a legjobb megközelítésért e téren. Ez mindmáig nehéz probléma, és nem világos, hogy nem triviális feladatoknál lehetséges-e jó (lineáris) skálázódást elérni. Nem ajánljuk ezt a megközelítést, hacsak nem áll rendelkezésre kiváló keretrendszer vagy operációs rendszer támogatás a több GPU láncolásához.


Másodszor, feloszthatjuk a munkát rétegenként. Például ahelyett, hogy 64 csatornát számítanánk egyetlen GPU-n, feloszthatjuk a feladatot 4 GPU között, amelyek mindegyike 16 csatorna adatait állítja elő.
Hasonlóképpen, egy teljesen összekötött réteg esetén feloszthatjuk a kimeneti egységek számát. A :numref:`fig_alexnet_original` ábra (forrása: :citet:`Krizhevsky.Sutskever.Hinton.2012`)
szemlélteti ezt a tervezést, ahol ezt a stratégiát alkalmazták, hogy megbirkózzanak a nagyon kis memóriájú GPU-kkal (akkoriban 2 GB volt).
Ez jó skálázódást tesz lehetővé a számítás szempontjából, feltéve, hogy a csatornák (vagy egységek) száma nem túl kicsi.
Emellett
több GPU egyre nagyobb hálózatokat képes feldolgozni, mivel a rendelkezésre álló memória lineárisan skálázódik.

![Modellpárhuzamosság az eredeti AlexNet tervezésben a korlátozott GPU-memória miatt.](../img/alexnet-original.svg)
:label:`fig_alexnet_original`

Azonban
*nagyon nagy* számú szinkronizálási vagy barrier műveletre van szükségünk, mivel minden egyes réteg az összes többi réteg eredményeitől függ.
Ráadásul az átviendő adatok mennyisége potenciálisan még nagyobb, mint amikor a rétegeket osztjuk el a GPU-k között. Ezért ezt a megközelítést sem ajánljuk a sávszélesség-igénye és összetettsége miatt.

Végül feloszthatjuk az adatokat több GPU között. Így minden GPU ugyanolyan típusú munkát végez, bár különböző megfigyeléseken. A gradienseket minden tanítási adat-mini-batch után összesítik a GPU-k között.
Ez a legegyszerűbb megközelítés, és bármilyen helyzetben alkalmazható.
Csupán minden mini-batch után kell szinkronizálni. Ugyanakkor nagyon kívánatos, hogy a gradiensparamétereket már csere közben is kiszámítsuk, amíg a többiek még folyamatban vannak.
Ráadásul a több GPU nagyobb mini-batch méretekhez vezet, ezáltal növelve a tanítás hatékonyságát.
Azonban több GPU hozzáadása nem teszi lehetővé, hogy nagyobb modelleket tanítsunk.


![Párhuzamosítás több GPU-n. Balról jobbra: eredeti feladat, hálózatpartícionálás, réteges partícionálás, adatpárhuzamosság.](../img/splitting.svg)
:label:`fig_splitting`


A több GPU-n való párhuzamosítás különböző módjainak összehasonlítása a :numref:`fig_splitting` ábrán látható.
Általánosságban az adatpárhuzamosság a legkényelmesebb megközelítés, feltéve, hogy elegendően nagy memóriával rendelkező GPU-khoz van hozzáférésünk. Lásd még :cite:`Li.Andersen.Park.ea.2014` az elosztott tanításhoz való partícionálás részletes leírásáért. A GPU-memória a deep learning korai napjaiban volt gondot okozó tényező. Mára ez a probléma megoldódott a legkülönlegesebb esetek kivételével. A továbbiakban az adatpárhuzamosságra összpontosítunk.

## Adatpárhuzamosság

Tegyük fel, hogy egy gépen $k$ GPU található. Az adott tanítandó modell esetén minden GPU önállóan fenntartja a modellparaméterek teljes készletét, bár a GPU-k közötti paraméterértékek azonosak és szinkronizáltak.
Például
a :numref:`fig_data_parallel` ábra szemlélteti
az adatpárhuzamossággal végzett tanítást,
amikor $k=2$.


![Minibatch sztochasztikus gradienscsökkenés kiszámítása adatpárhuzamossággal két GPU-n.](../img/data-parallel.svg)
:label:`fig_data_parallel`

Általánosságban a tanítás a következőképpen zajlik:

* A tanítás bármely iterációjában, adott egy véletlenszerű mini-batch, a batch példáit $k$ részre osztjuk, és egyenletesen elosztjuk a GPU-k között.
* Minden GPU kiszámítja a veszteséget és a modellparaméterek gradiensét a hozzá rendelt mini-batch-részlet alapján.
* A $k$ GPU mindegyikének helyi gradienseit összesítik, hogy megkapják az aktuális mini-batch sztochasztikus gradienst.
* Az összesített gradienst visszaterjesztik minden GPU-ra.
* Minden GPU ezt a mini-batch sztochasztikus gradienst használja a fenntartott modellparaméterek teljes készletének frissítéséhez.




Megjegyzendő, hogy a gyakorlatban $k$-szorosára *növeljük* a mini-batch méretét $k$ GPU-n való tanítás esetén, hogy minden GPU ugyanannyi munkát végezzen, mintha egyetlen GPU-n tanítanánk. Egy 16 GPU-s szerveren ez jelentősen megnövelheti a mini-batch méretet, és előfordulhat, hogy a tanulási rátát is ennek megfelelően kell növelni.
Megjegyzendő az is, hogy a :numref:`sec_batch_norm` részben tárgyalt batch normalizálást módosítani kell, például GPU-nkénti külön batch normalizálási együttható fenntartásával.
A következőkben egy egyszerű hálózatot fogunk használni a több GPU-s tanítás szemléltetéséhez.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

## [**Egy egyszerű hálózat**]

A :numref:`sec_lenet` részben bemutatott LeNet-et használjuk (némi módosítással). A semmiből definiáljuk, hogy részletesen szemléltessük a paramétercsere és szinkronizálás menetét.

```{.python .input}
#@tab mxnet
# A modellparaméterek inicializálása
scale = 0.01
W1 = np.random.normal(scale=scale, size=(20, 1, 3, 3))
b1 = np.zeros(20)
W2 = np.random.normal(scale=scale, size=(50, 20, 5, 5))
b2 = np.zeros(50)
W3 = np.random.normal(scale=scale, size=(800, 128))
b3 = np.zeros(128)
W4 = np.random.normal(scale=scale, size=(128, 10))
b4 = np.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# A modell definiálása
def lenet(X, params):
    h1_conv = npx.convolution(data=X, weight=params[0], bias=params[1],
                              kernel=(3, 3), num_filter=20)
    h1_activation = npx.relu(h1_conv)
    h1 = npx.pooling(data=h1_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2_conv = npx.convolution(data=h1, weight=params[2], bias=params[3],
                              kernel=(5, 5), num_filter=50)
    h2_activation = npx.relu(h2_conv)
    h2 = npx.pooling(data=h2_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = np.dot(h2, params[4]) + params[5]
    h3 = npx.relu(h3_linear)
    y_hat = np.dot(h3, params[6]) + params[7]
    return y_hat

# Keresztentrópia veszteségfüggvény
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
# A modellparaméterek inicializálása
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# A modell definiálása
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# Keresztentrópia veszteségfüggvény
loss = nn.CrossEntropyLoss(reduction='none')
```

## Adatszinkronizálás

A hatékony több GPU-s tanításhoz két alapvető műveletre van szükségünk.
Először szükségünk van arra, hogy [**paraméterek listáját több eszközre tudjuk elosztani**], és gradienseket csatolhassunk hozzájuk (`get_params`). Paraméterek nélkül lehetetlen a hálózatot egy GPU-n kiértékelni.
Másodszor szükségünk van arra, hogy paramétereket tudjunk összesíteni több eszközről, vagyis szükségünk van egy `allreduce` függvényre.

```{.python .input}
#@tab mxnet
def get_params(params, device):
    new_params = [p.copyto(device) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params
```

```{.python .input}
#@tab pytorch
def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params
```

Próbáljuk ki egy GPU-ra másolva a modellparamétereket.

```{.python .input}
#@tab all
new_params = get_params(params, d2l.try_gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)
```

Mivel még nem végeztünk semmilyen számítást, az elfogultsági paraméterhez tartozó gradiens még mindig nulla.
Tegyük fel most, hogy van egy több GPU-ra elosztott vektorunk. A következő [**`allreduce` függvény összeadja az összes vektort, és az eredményt visszasugározza minden GPU-ra**]. Megjegyzendő, hogy ehhez az eredményeket összegyűjtő eszközre kell másolni az adatokat.

```{.python .input}
#@tab mxnet
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].ctx)
    for i in range(1, len(data)):
        data[0].copyto(data[i])
```

```{.python .input}
#@tab pytorch
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)
```

Teszteljük ezt különböző értékekkel rendelkező vektorok létrehozásával különböző eszközökön, majd összesítsük őket.

```{.python .input}
#@tab mxnet
data = [np.ones((1, 2), ctx=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])
```

```{.python .input}
#@tab pytorch
data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])
```

## Adatok elosztása

Szükségünk van egy egyszerű segédfüggvényre, amely [**egy mini-batch-et egyenletesen oszt el több GPU között**]. Például két GPU esetén az adatok felét szeretnénk az egyik, másik felét a másik GPU-ra másolni.
Mivel kényelmesebb és tömörebb, a deep learning keretrendszer beépített függvényét használjuk, amelyet egy $4 \times 5$-ös mátrixon próbálunk ki.

```{.python .input}
#@tab mxnet
data = np.arange(20).reshape(4, 5)
devices = [npx.gpu(0), npx.gpu(1)]
split = gluon.utils.split_and_load(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

```{.python .input}
#@tab pytorch
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```

A későbbi felhasználás érdekében definiálunk egy `split_batch` függvényt, amely az adatokat és a címkéket is felosztja.

```{.python .input}
#@tab mxnet
#@save
def split_batch(X, y, devices):
    """Az `X` és `y` felosztása több eszközre."""
    assert X.shape[0] == y.shape[0]
    return (gluon.utils.split_and_load(X, devices),
            gluon.utils.split_and_load(y, devices))
```

```{.python .input}
#@tab pytorch
#@save
def split_batch(X, y, devices):
    """Az `X` és `y` felosztása több eszközre."""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))
```

## Tanítás

Most megvalósíthatjuk a [**több GPU-s tanítást egyetlen mini-batch-re**]. Megvalósítása elsősorban az ebben a részben leírt adatpárhuzamossági megközelítésen alapul. Az imént tárgyalt segédfüggvényeket, az `allreduce`-t és a `split_and_load`-ot fogjuk használni az adatok szinkronizálásához több GPU között. Megjegyezzük, hogy nem kell semmiféle speciális kódot írnunk a párhuzamosítás eléréséhez. Mivel a számítási gráfnak nincsenek eszközök közötti függőségei egy mini-batchon belül, az végrehajtása *automatikusan* párhuzamos lesz.

```{.python .input}
#@tab mxnet
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    with autograd.record():  # A veszteség minden GPU-n külön kerül kiszámításra
        ls = [loss(lenet(X_shard, device_W), y_shard)
              for X_shard, y_shard, device_W in zip(
                  X_shards, y_shards, device_params)]
    for l in ls:  # A visszaterjesztés minden GPU-n külön történik
        l.backward()
    # Az összes GPU gradienseinek összesítése és visszasugárzása minden GPU-ra
    for i in range(len(device_params[0])):
        allreduce([device_params[c][i].grad for c in range(len(devices))])
    # A modellparaméterek minden GPU-n külön frissülnek
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0])  # Itt teljes méretű batch-et használunk
```

```{.python .input}
#@tab pytorch
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # A veszteség minden GPU-n külön kerül kiszámításra
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
              X_shards, y_shards, device_params)]
    for l in ls:  # A visszaterjesztés minden GPU-n külön történik
        l.backward()
    # Az összes GPU gradienseinek összesítése és visszasugárzása minden GPU-ra
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    # A modellparaméterek minden GPU-n külön frissülnek
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # Itt teljes méretű batch-et használunk
```

Most definiálhatjuk [**a tanítófüggvényt**]. Ez kissé eltér az előző fejezetekben használtaktól: el kell osztanunk a GPU-kat, és az összes modellparamétert minden eszközre másolni kell.
Nyilvánvalóan minden batch feldolgozása a `train_batch` függvénnyel történik a több GPU kezeléséhez. A kényelmesség (és a kód tömörségének) kedvéért a pontosságot egyetlen GPU-n számítjuk ki, bár ez *nem hatékony*, mivel a többi GPU tétlen.

```{.python .input}
#@tab mxnet
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # Modellparaméterek másolása `num_gpus` GPU-ra
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # Több GPU-s tanítás egyetlen mini-batch-re
            train_batch(X, y, device_params, devices, lr)
            npx.waitall()
        timer.stop()
        # A modell kiértékelése a 0-s GPU-n
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # Modellparaméterek másolása `num_gpus` GPU-ra
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # Több GPU-s tanítás egyetlen mini-batch-re
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
        # A modell kiértékelése a 0-s GPU-n
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

Nézzük meg, hogyan működik ez [**egyetlen GPU-n**].
Először 256-os batch méretet és 0,2-es tanulási rátát használunk.

```{.python .input}
#@tab all
train(num_gpus=1, batch_size=256, lr=0.2)
```

A batch méretet és a tanulási rátát változatlanul hagyva, [**a GPU-k számát 2-re növelve**] láthatjuk, hogy a tesztelési pontosság nagyjából azonos marad az előző kísérlethez képest.
Az optimalizálási algoritmusok szempontjából azonosak. Sajnos itt nem érhető el érdemi gyorsulás: a modell egyszerűen túl kicsi; ráadásul csak kis adathalmazzal rendelkezünk, ahol a több GPU-s tanítás megvalósításának kissé kifinomultatlan megközelítése jelentős Python-overheadet szenvedett el. A továbbiakban összetettebb modellekkel és kifinomultabb párhuzamosítási módszerekkel fogunk találkozni.
Lássuk mindenesetre, mi történik a Fashion-MNIST esetén.

```{.python .input}
#@tab all
train(num_gpus=2, batch_size=256, lr=0.2)
```

## Összefoglalás

* A deep hálózatok tanításának több GPU-ra való elosztásának több módja is van. Feloszthatjuk rétegek között, rétegeken belül vagy adatok szerint. Az első két módszer szorosan koordinált adatátvitelt igényel. Az adatpárhuzamosság a legegyszerűbb stratégia.
* Az adatpárhuzamos tanítás egyszerűen megvalósítható. Ugyanakkor a hatékonyság érdekében megnöveli a tényleges mini-batch méretet.
* Az adatpárhuzamosságban az adatok több GPU között oszlanak el, ahol minden GPU elvégzi a saját előre- és visszaszámítási műveletét, majd a gradienseket összesítik, és az eredményeket visszasugározzák a GPU-kra.
* Nagyobb mini-batch-ekhez esetleg kissé megnövelt tanulási rátát érdemes alkalmazni.

## Feladatok

1. Amikor $k$ GPU-n tanítunk, változtassuk meg a mini-batch méretét $b$-ről $k \cdot b$-re, vagyis skálázzuk fel a GPU-k számával.
1. Hasonlítsuk össze a pontosságot különböző tanulási ráták esetén. Hogyan skálázódik a GPU-k számával?
1. Valósítsunk meg egy hatékonyabb `allreduce` függvényt, amely különböző paramétereket aggregál különböző GPU-kon! Miért hatékonyabb ez?
1. Valósítsuk meg a több GPU-s tesztelési pontosság kiszámítását!

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/364)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1669)
:end_tab:
