# Finomhangolás
:label:`sec_fine_tuning`

A korábbi fejezetekben megvitattuk, hogyan tanítsunk modelleket a Fashion-MNIST tanítóadathalmazon, amely mindössze 60 000 képet tartalmaz. Az ImageNet adathalmazt is ismertettük, amely az akadémiai kutatásban legelterjedtebb nagy méretű képadathalmaz: több mint 10 millió képet és 1000 objektumkategóriát tartalmaz. Az általunk általában használt adathalmazok mérete e két adathalmaz között van.

Tegyük fel, hogy képek alapján különböző széktípusokat szeretnénk felismerni, majd vásárlási linkeket ajánlani a felhasználóknak.
Egyik lehetséges módszer: először azonosítunk 100 általános széket, minden székről különböző szögekből 1000 képet készítünk, majd osztályozási modellt tanítunk az összegyűjtött képadathalmazon.
Bár ez a székadathalmaz nagyobb lehet a Fashion-MNIST adathalmaznál, a példák száma még mindig kevesebb, mint az ImageNet egy tizede.
Ez az ImageNet számára megfelelő összetett modellek túlillesztéséhez vezethet ezen a székadathalmazon.
Emellett, a korlátozott számú tanítópélda miatt, a betanított modell pontossága esetleg nem felel meg a gyakorlati követelményeknek.

Az előbbi problémák megoldására egy nyilvánvaló megoldás az adatok bővítése.
Az adatok gyűjtése és felcímkézése azonban rengeteg időt és pénzt igényelhet.
Például az ImageNet adathalmaz összegyűjtéséhez a kutatók millió dollárokat fordítottak kutatási forrásokból.
Bár a jelenlegi adatgyűjtési költségek jelentősen csökkentek, ezek a költségek még mindig nem hagyhatók figyelmen kívül.

Egy másik megoldás a *transzfertanulás* alkalmazása, amely a *forrásadathalmazból* tanult tudást átviszi a *célhalmaz* felé.
Például, bár az ImageNet adathalmaz képeinek többsége nincs kapcsolatban a székekkel, az ezen az adathalmazon betanított modell általánosabb képjellemzőket vonhat ki, amelyek segíthetnek az élek, textúrák, formák és tárgyösszetételek azonosításában.
Ezek a hasonló jellemzők hatékonyak lehetnek a székek felismeréséhez is.

## Lépések

Ebben a fejezetben a transzfertanulás egyik általános technikáját mutatjuk be: a *finomhangolást*. Ahogy a :numref:`fig_finetune` ábrán látható, a finomhangolás a következő négy lépésből áll:

1. Neurális hálózati modell, azaz a *forrásmodell* előtanítása egy forrásadathalmazon (pl. az ImageNet adathalmazon).
1. Új neurális hálózati modell, azaz a *célmodell* létrehozása. Ez másolja a forrásmodell összes modelltervét és paraméterét, kivéve a kimeneti réteget. Feltételezzük, hogy ezek a modellparaméterek tartalmazzák a forrásadathalmazból tanult tudást, és ez a tudás a célhalmazra is alkalmazható. Feltételezzük azt is, hogy a forrásmodell kimeneti rétege szorosan kapcsolódik a forrásadathalmaz címkéihez; ezért a célmodellben nem alkalmazzák.
1. Kimeneti réteg hozzáadása a célmodellhez, amelynek kimenetei száma a célhalmaz kategóriáinak száma. Ezután véletlenszerűen inicializálják e réteg modellparamétereit.
1. A célmodell tanítása a célhalmazon, például egy székadathalmazon. A kimeneti réteg teljesen nulláról kerül betanításra, míg az összes többi réteg paramétereit finomhangolják a forrásmodell paraméterei alapján.

![Finomhangolás.](../img/finetune.svg)
:label:`fig_finetune`

Ha a célhalmazok sokkal kisebbek a forrásadathalmazoknál, a finomhangolás segít javítani a modellek általánosítóképességét.

## Hot dog felismerés

Mutassuk be a finomhangolást egy konkrét esettel: hot dog felismeréssel.
Egy kis adathalmazon finomhangolunk egy ResNet modellt, amelyet az ImageNet adathalmazon tanítottunk elő.
Ez a kis adathalmaz néhány ezer hot dogot tartalmazó és nem tartalmazó képből áll.
A finomhangolt modellt hot dogok képekből való felismerésére fogjuk használni.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from torch import nn
import torch
import torchvision
import os
```

### Az adathalmaz beolvasása

[**Az általunk használt hot dog adathalmaz online képekből készült**].
Ez az adathalmaz 1400 pozitív osztályú, hot dogot tartalmazó képből, és ugyanennyi negatív osztályú, más ételeket tartalmazó képből áll.
Mindkét osztályból 1000 kép kerül tanításra, a többi tesztelésre.

A letöltött adathalmaz kicsomagolása után két mappát kapunk: `hotdog/train` és `hotdog/test`. Mindkét mappa tartalmaz `hotdog` és `not-hotdog` almappákat, amelyek mindegyike a megfelelő osztály képeit tartalmazza.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')
```

Két példányt hozunk létre a tanítási és tesztelési adathalmazok összes képfájljának beolvasásához.

```{.python .input}
#@tab mxnet
train_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'train'))
test_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'test'))
```

```{.python .input}
#@tab pytorch
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
```

Az első 8 pozitív példa és az utolsó 8 negatív kép az alábbiakban látható. Ahogy látható, [**a képek mérete és képaránya eltérő**].

```{.python .input}
#@tab all
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);
```

Tanítás során először véletlenszerű méretű és képarányú területet vágunk ki véletlenszerűen a képből, majd ezt a területet $224 \times 224$ pixeles bemeneti képpé méretezzük.
Tesztelés során a kép szélességét és magasságát egyaránt 256 pixelre méretezzük, majd középső $224 \times 224$ pixeles területet vágunk ki bemenetként.
Emellett a három RGB (vörös, zöld és kék) színcsatorna értékeit csatornánként *normalizáljuk*.
Konkrétan: minden értékből kivonjuk az adott csatorna átlagát, majd az eredményt elosztjuk az adott csatorna szórásával.

[~~Data augmentations~~]

```{.python .input}
#@tab mxnet
# A három RGB csatorna átlagainak és szórásainak megadása
# az egyes csatornák standardizálásához
normalize = gluon.data.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomResizedCrop(224),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    normalize])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    normalize])
```

```{.python .input}
#@tab pytorch
# A három RGB csatorna átlagainak és szórásainak megadása
# az egyes csatornák standardizálásához
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])
```

### [**A modell definiálása és inicializálása**]

A forrásmodellként az ImageNet adathalmazon előtanított ResNet-18-at használjuk. Itt a `pretrained=True` megadásával automatikusan letöltjük az előtanított modellparamétereket.
Ha ezt a modellt először használjuk, internetkapcsolat szükséges a letöltéshez.

```{.python .input}
#@tab mxnet
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
```

:begin_tab:`mxnet`
Az előtanított forrásmodell példány két tagváltozót tartalmaz: `features` és `output`. Az előbbi tartalmazza a modell összes rétegét a kimeneti réteg kivételével, az utóbbi a modell kimeneti rétege.
Ennek az elkülönítésnek az a fő célja, hogy megkönnyítse a modellparaméterek finomhangolását az összes rétegben a kimeneti réteg kivételével. A forrásmodell `output` tagváltozója az alábbiakban látható.
:end_tab:

:begin_tab:`pytorch`
Az előtanított forrásmodell példány számos jellemzőréteget és egy `fc` kimeneti réteget tartalmaz.
Ennek az elkülönítésnek az a fő célja, hogy megkönnyítse a modellparaméterek finomhangolását az összes rétegben a kimeneti réteg kivételével. A forrásmodell `fc` tagváltozója az alábbiakban látható.
:end_tab:

```{.python .input}
#@tab mxnet
pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net.fc
```

Teljesen összekötött rétegként ez a ResNet végső globális átlag-pooling kimeneteit az ImageNet adathalmaz 1000 osztályú kimeneteivé alakítja.
Ezután egy új neurális hálózatot építünk célmodellként. Ez ugyanolyan módon van definiálva, mint az előtanított forrásmodell, azzal a különbséggel, hogy az utolsó réteg kimenetének száma a célhalmaz osztályainak számára van állítva (nem 1000-re).

Az alábbi kódban a `finetune_net` célmodell példány kimeneti réteg előtti modellparamétereit a forrásmodell megfelelő rétegeinek modellparamétereivel inicializálják.
Mivel ezeket a modellparamétereket ImageNet előtanítással szerezték, hatékonyak.
Ezért csak kis tanulási rátát kell használni az ilyen előtanított paraméterek *finomhangolásához*.
Ezzel szemben a kimeneti réteg modellparamétereit véletlenszerűen inicializálják, és általában nagyobb tanulási rátát igényelnek a nulláról való tanuláshoz.
Legyen az alaptanulási ráta $\eta$; a kimeneti réteg modellparamétereinek iterálásához $10\eta$ tanulási rátát alkalmazunk.

```{.python .input}
#@tab mxnet
finetune_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
# A kimeneti réteg modellparamétereit tízszer nagyobb tanulási rátával
# iteráljuk
finetune_net.output.collect_params().setattr('lr_mult', 10)
```

```{.python .input}
#@tab pytorch
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight);
```

### [**A modell finomhangolása**]

Először definiálunk egy `train_fine_tuning` nevű tanítási függvényt, amely finomhangolást alkalmaz, és többször meghívható.

```{.python .input}
#@tab mxnet
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    train_iter = gluon.data.DataLoader(
        train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(
        test_imgs.transform_first(test_augs), batch_size)
    devices = d2l.try_all_gpus()
    net.collect_params().reset_ctx(devices)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': 0.001})
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

```{.python .input}
#@tab pytorch
# Ha `param_group=True`, a kimeneti réteg modellparamétereit tízszer nagyobb
# tanulási rátával frissítjük
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)    
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

[**Az alaptanulási rátát kis értékre állítjuk**], hogy finomhangoljuk az előtanítással nyert modellparamétereket. Az előző beállítások alapján a célmodell kimeneti rétegének paramétereit tízszer nagyobb tanulási rátával tanítjuk nulláról.

```{.python .input}
#@tab mxnet
train_fine_tuning(finetune_net, 0.01)
```

```{.python .input}
#@tab pytorch
train_fine_tuning(finetune_net, 5e-5)
```

[**Összehasonlításképpen**] definiálunk egy azonos modellt, de (**az összes modellparamétert véletlenszerű értékekre inicializáljuk**). Mivel az egész modellt nulláról kell betanítani, nagyobb tanulási rátát használhatunk.

```{.python .input}
#@tab mxnet
scratch_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
train_fine_tuning(scratch_net, 0.1)
```

```{.python .input}
#@tab pytorch
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
```

Ahogy látható, a finomhangolt modell általában jobban teljesít ugyanannyi epoch alatt, mivel a kezdeti paraméterértékei hatékonyabbak.

## Összefoglalás

* A transzfertanulás a forrásadathalmazból tanult tudást átviszi a célhalmazra. A finomhangolás a transzfertanulás egyik általános technikája.
* A célmodell másolja a forrásmodell összes modelltervét és paramétereit, kivéve a kimeneti réteget, majd finomhangolja ezeket a paramétereket a célhalmaz alapján. Ezzel szemben a célmodell kimeneti rétegét nulláról kell betanítani.
* Általában a finomhangolt paraméterek kisebb tanulási rátát igényelnek, míg a kimeneti réteg nulláról való tanítása nagyobb tanulási rátát alkalmazhat.

## Feladatok

1. Növeld folyamatosan a `finetune_net` tanulási rátáját. Hogyan változik a modell pontossága?
2. Módosítsd tovább a `finetune_net` és `scratch_net` hiperparamétereit az összehasonlító kísérletben. Még mindig különböznek-e a pontosságban?
3. Állítsd a `finetune_net` kimeneti réteg előtti paramétereit a forrásmodell paramétereire, és *ne* frissítsd ezeket tanítás közben. Hogyan változik a modell pontossága? Az alábbi kódot használhatod.

```{.python .input}
#@tab mxnet
finetune_net.features.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
for param in finetune_net.parameters():
    param.requires_grad = False
```

4. Valójában van egy "hotdog" osztály az `ImageNet` adathalmazban. A megfelelő súlyparaméter a kimeneti rétegben az alábbi kóddal kapható meg. Hogyan lehet kihasználni ezt a súlyparamétert?

```{.python .input}
#@tab mxnet
weight = pretrained_net.output.weight
hotdog_w = np.split(weight.data(), 1000, axis=0)[713]
hotdog_w.shape
```

```{.python .input}
#@tab pytorch
weight = pretrained_net.fc.weight
hotdog_w = torch.split(weight.data, 1, dim=0)[934]
hotdog_w.shape
```

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/368)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1439)
:end_tab:
