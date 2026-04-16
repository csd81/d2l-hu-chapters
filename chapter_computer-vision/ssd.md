# Egylépéses többdobozos felismerés (SSD)
:label:`sec_ssd`

A :numref:`sec_bbox`--:numref:`sec_object-detection-dataset` fejezetekben bemutattuk a befoglaló téglalapokat, a horgonydobozokat, a többléptékű objektumdetektálást és az objektumdetektálási adathalmazt.
Most készen állunk arra, hogy ezeket az előzetes ismereteket felhasználva tervezzünk egy objektumdetektálási modellt: az egylépéses többdobozos felismerést (SSD) :cite:`Liu.Anguelov.Erhan.ea.2016`.
Ez a modell egyszerű, gyors és széles körben használt.
Bár ez csupán egy a rengeteg objektumdetektálási modell közül, ebben a fejezetben egyes tervezési elvek és implementációs részletek más modellekre is alkalmazhatók.


## Modell

A :numref:`fig_ssd` ábra áttekintést nyújt az egylépéses többdobozos felismerés tervéről.
Ez a modell főleg egy alaphálózatból és azt követő számos többléptékű jellemzőtérkép-blokkból áll.
Az alaphálózat a bemeneti képből vonja ki a jellemzőket, így mély konvolúciós neurális hálózatot alkalmazhat.
Például az eredeti egylépéses többdobozos felismerési cikk egy, az osztályozási réteg előtt csonkított VGG hálózatot alkalmaz :cite:`Liu.Anguelov.Erhan.ea.2016`, míg a ResNet-et is széles körben alkalmazzák.
A tervünkön keresztül az alaphálózatot úgy alakíthatjuk, hogy nagyobb jellemzőtérképeket adjon ki, így több horgonydobozt generálunk a kisebb objektumok felismeréséhez.
Ezután minden többléptékű jellemzőtérkép-blokk csökkenti (pl. felére) az előző blokk jellemzőtérképeinek magasságát és szélességét, és lehetővé teszi a jellemzőtérképek minden egységének, hogy növelje receptív mezőjét a bemeneti képen.


Emlékeztetünk a mély neurális hálózatok rétegenkénti képreprezentációin keresztül megvalósuló többléptékű objektumdetektálás tervére a :numref:`sec_multiscale-object-detection` fejezetből.
Mivel a :numref:`fig_ssd` ábra tetejéhez közelebb lévő többléptékű jellemzőtérképek kisebbek, de nagyobb receptív mezőkkel rendelkeznek, alkalmasabbak kevesebb, de nagyobb objektum felismerésére.

Röviden összefoglalva, alaphálózatán és számos többléptékű jellemzőtérkép-blokkján keresztül az egylépéses többdobozos felismerés változó számú, különböző méretű horgonydobozt generál, és változó méretű objektumokat ismer fel ezen horgonydobozok osztályainak és eltolásainak (és így a befoglaló téglalapoknak) jóslásával; így ez egy többléptékű objektumdetektálási modell.


![Többléptékű objektumdetektálási modellként az egylépéses többdobozos felismerés főleg egy alaphálózatból és azt követő számos többléptékű jellemzőtérkép-blokkból áll.](../img/ssd.svg)
:label:`fig_ssd`


A következőkben leírjuk a :numref:`fig_ssd` ábra különböző blokkjainak implementációs részleteit. Először megvizsgáljuk, hogyan implementálhatjuk az osztály- és befoglaló téglalap-jóslást.


### [**Osztályjóslási réteg**]

Legyen az objektumok osztályainak száma $q$.
Ekkor a horgonydobozoknak $q+1$ osztályuk van, ahol a 0. osztály a háttér.
Bizonyos léptéken tegyük fel, hogy a jellemzőtérképek magassága és szélessége rendre $h$ és $w$.
Amikor $a$ horgonydobozt generálnak, minden térbeli pozíciót középpontként véve,
összesen $hwa$ horgonydobozt kell osztályozni.
Ez gyakran teszi a teljesen összekötött rétegekkel való osztályozást kivitelezhetetlenné a valószínűleg magas paraméterezési költségek miatt.
Emlékeztetünk arra, hogyan használtuk a konvolúciós rétegek csatornáit osztályok jóslásához a :numref:`sec_nin` fejezetben.
Az egylépéses többdobozos felismerés ugyanezt a technikát alkalmazza a modell komplexitásának csökkentésére.

Konkrétan az osztályjóslási réteg konvolúciós réteget alkalmaz a jellemzőtérképek szélességét vagy magasságát nem módosítva.
Ily módon a jellemzőtérképek azonos térbeli dimenzióin (szélességén és magasságán) a kimenetek és a bemenetek között egy-az-egyhez megfeleltetés lehetséges.
Konkrétabban, a bemeneti jellemzőtérképek ($x$, $y$) bármely térbeli pozíciójában lévő kimeneti jellemzőtérkép csatornái az ($x$, $y$)-t középpontként vevő összes horgonydoboz osztályjóslásait ábrázolják.
Érvényes jóslások előállításához $a(q+1)$ kimeneti csatornának kell lennie, ahol az azonos térbeli pozícióban lévő $i(q+1) + j$ indexű kimeneti csatorna az $i$ ($0 \leq i < a$) horgonydoboz $j$ ($0 \leq j \leq q$) osztályának jóslását ábrázolja.

Az alábbiakban definiálunk egy ilyen osztályjóslási réteget, ahol $a$-t és $q$-t rendre a `num_anchors` és `num_classes` argumentumokon keresztül adjuk meg.
Ez a réteg 1 párnázású $3\times3$-as konvolúciós réteget alkalmaz.
Ennek a konvolúciós rétegnek a bemenetének és kimenetének szélessége és magassága változatlan marad.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,
                     padding=1)
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)
```

### (**Befoglaló téglalap-jóslási réteg**)

A befoglaló téglalap-jóslási réteg kialakítása hasonló az osztályjóslási rétegéhez.
Az egyetlen különbség minden horgonydoboz kimeneteinek számában rejlik: itt négy eltolást kell jósolni $q+1$ osztály helyett.

```{.python .input}
#@tab mxnet
def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)
```

```{.python .input}
#@tab pytorch
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
```

### [**Jóslások összefűzése több léptékre**]

Ahogy említettük, az egylépéses többdobozos felismerés többléptékű jellemzőtérképeket használ horgonydobozok generálásához, és osztályaik és eltolásaik jóslásához.
Különböző léptékeken a jellemzőtérképek alakjai, vagy az azonos egységet középpontként vevő horgonydobozok száma eltérhet.
Ezért a jóslási kimenetek alakjai különböző léptékeken eltérhetnek.

Az alábbi példában azonos mini-batch-re két különböző léptéken építünk jellemzőtérképeket: `Y1` és `Y2`, ahol `Y2` magassága és szélessége `Y1`-nek a fele.
Példaként vegyük az osztályjóslást.
Tegyük fel, hogy `Y1` és `Y2` minden egységéhez rendre 5 és 3 horgonydoboz generálódik.
Tegyük fel továbbá, hogy az objektumok osztályainak száma 10.
A `Y1` és `Y2` jellemzőtérképekhez az osztályjóslási kimenetek csatornáinak száma rendre $5\times(10+1)=55$ és $3\times(10+1)=33$, ahol mindkét kimenet alakja (batch méret, csatornák száma, magasság, szélesség).

```{.python .input}
#@tab mxnet
def forward(x, block):
    block.initialize()
    return block(x)

Y1 = forward(np.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
Y2 = forward(np.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
Y1.shape, Y2.shape
```

```{.python .input}
#@tab pytorch
def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
Y1.shape, Y2.shape
```

Amint látható, a batch méret dimenzióján kívül a többi három dimenzió mind különböző méretű.
E két jóslási kimenet összefűzéséhez a hatékonyabb számítás érdekében ezeket a tenzorokat egységesebb formátumra alakítjuk.

Vegyük figyelembe, hogy a csatorna dimenzió tartalmazza az azonos középpontú horgonydobozok jóslásait.
Először ezt a dimenziót a legbelsőbb helyre helyezzük.
Mivel a batch méret különböző léptékeken azonos marad, a jóslási kimenetet kétdimenziós tenzorrá alakíthatjuk (batch méret, magasság $\times$ szélesség $\times$ csatornák száma) alakban.
Ezután az ilyen kimeneteket különböző léptékeken az 1. dimenzió mentén összefűzhetjük.

```{.python .input}
#@tab mxnet
def flatten_pred(pred):
    return npx.batch_flatten(pred.transpose(0, 2, 3, 1))

def concat_preds(preds):
    return np.concatenate([flatten_pred(p) for p in preds], axis=1)
```

```{.python .input}
#@tab pytorch
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)
```

Ily módon, bár `Y1` és `Y2` eltérő méretű csatornákban, magasságban és szélességben, az azonos mini-batch-re vonatkozó két különböző léptékű jóslási kimenetet még mindig összefűzhetjük.

```{.python .input}
#@tab all
concat_preds([Y1, Y2]).shape
```

### [**Lefele mintavételezési blokk**]

Több léptéken való objektumdetektálás céljából definiáljuk a következő `down_sample_blk` lefele mintavételezési blokkot, amely a bemeneti jellemzőtérképek magasságát és szélességét felezi.
Valójában ez a blokk alkalmazza a VGG blokkok tervét a :numref:`subsec_vgg-blocks` fejezetből.
Konkrétabban minden lefele mintavételezési blokk két, 1-es párnázású $3\times3$-as konvolúciós rétegből áll, amelyeket egy 2-es lépésközt alkalmazó $2\times2$-es max-pooling réteg követ.
Ahogy tudjuk, az 1-es párnázású $3\times3$-as konvolúciós rétegek nem változtatják meg a jellemzőtérképek alakját.
Azonban a következő $2\times2$-es max-pooling felezi a bemeneti jellemzőtérképek magasságát és szélességét.
A lefele mintavételezési blokk bemeneti és kimeneti jellemzőtérképeire egyaránt, mivel $1\times 2+(3-1)+(3-1)=6$, a kimenet minden egységének $6\times6$-os receptív mezeje van a bemeneten.
Ezért a lefele mintavételezési blokk megnöveli a kimeneti jellemzőtérképek minden egységének receptív mezőjét.

```{.python .input}
#@tab mxnet
def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    return blk
```

```{.python .input}
#@tab pytorch
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
```

Az alábbi példában az általunk épített lefele mintavételezési blokk megváltoztatja a bemeneti csatornák számát, és felezi a bemeneti jellemzőtérképek magasságát és szélességét.

```{.python .input}
#@tab mxnet
forward(np.zeros((2, 3, 20, 20)), down_sample_blk(10)).shape
```

```{.python .input}
#@tab pytorch
forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape
```

### [**Alaphálózati blokk**]

Az alaphálózati blokk a bemeneti képekből vonja ki a jellemzőket.
Az egyszerűség kedvéért egy kis alaphálózatot építünk, amely három lefele mintavételezési blokkból áll, amelyek minden blokkon megkétszerezik a csatornák számát.
Adott egy $256\times256$ pixeles bemeneti képen ez az alaphálózati blokk $32 \times 32$-es jellemzőtérképeket ad ki ($256/2^3=32$).

```{.python .input}
#@tab mxnet
def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk

forward(np.zeros((2, 3, 256, 256)), base_net()).shape
```

```{.python .input}
#@tab pytorch
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

forward(torch.zeros((2, 3, 256, 256)), base_net()).shape
```

### A teljes modell


[**A teljes egylépéses többdobozos felismerési modell öt blokkból áll.**]
Minden blokk által előállított jellemzőtérképeket egyrészt horgonydobozok generálására, másrészt ezen horgonydobozok osztályainak és eltolásainak jóslására használják.
E öt blokk közül az első az alaphálózati blokk, a második-negyedik lefele mintavételezési blokkok, az utolsó blokk pedig globális max-poolingot alkalmaz a magasság és a szélesség egyaránt 1-re csökkentéséhez.
Technikailag a második-ötödik blokkok mind :numref:`fig_ssd` ábrán látható többléptékű jellemzőtérkép-blokkok.

```{.python .input}
#@tab mxnet
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk
```

```{.python .input}
#@tab pytorch
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk
```

Most [**definiáljuk az előreterjesztést**] minden blokkhoz.
Eltérően a képosztályozási feladatoktól, a kimenetek itt a következőket tartalmazzák: (i) `Y` konvolúciós neurális hálózati jellemzőtérképek, (ii) az aktuális léptéken `Y` felhasználásával generált horgonydobozok, és (iii) ezen horgonydobozokra vonatkozó jósolt (a `Y` alapján) osztályok és eltolások.

```{.python .input}
#@tab mxnet
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

```{.python .input}
#@tab pytorch
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

Emlékeztetünk arra, hogy a :numref:`fig_ssd` ábrán a tetejéhez közelebb lévő többléptékű jellemzőtérkép-blokk nagyobb objektumok felismerésére szolgál; ezért nagyobb horgonydobozokat kell generálnia.
A fenti előreterjesztésben minden többléptékű jellemzőtérkép-blokknál két méretértékből álló listát adunk meg a `multibox_prior` függvény (a :numref:`sec_anchor` fejezetben leírva) `sizes` argumentumán keresztül.
A következőkben a 0.2 és 1.05 közötti intervallumot öt egyenlő részre osztják fel, hogy meghatározzák a kisebb méretértékeket az öt blokknál: 0.2, 0.37, 0.54, 0.71 és 0.88.
Ezután a nagyobb méretértékek $\sqrt{0.2 \times 0.37} = 0.272$, $\sqrt{0.37 \times 0.54} = 0.447$ stb.

[~~Hyperparameters for each block~~]

```{.python .input}
#@tab all
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
```

Most [**definiálhatjuk a teljes modellt**] `TinySSD` az alábbiak szerint.

```{.python .input}
#@tab mxnet
class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # Ekvivalens a `self.blk_i = get_blk(i)` értékadással
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Itt a `getattr(self, 'blk_%d' % i)` a `self.blk_i`-t éri el
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = np.concatenate(anchors, axis=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

```{.python .input}
#@tab pytorch
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # Ekvivalens a `self.blk_i = get_blk(i)` értékadással
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Itt a `getattr(self, 'blk_%d' % i)` a `self.blk_i`-t éri el
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

[**Létrehozunk egy modellpéldányt és előreterjesztést végzünk**] $256 \times 256$ pixeles képek `X` mini-batch-én.

Ahogy a szakasz elején bemutattuk, az első blokk $32 \times 32$-es jellemzőtérképeket ad ki.
Emlékeztetünk arra, hogy a második-negyedik lefele mintavételezési blokkok felezik a magasságot és a szélességet, az ötödik blokk pedig globális poolingot alkalmaz.
Mivel 4 horgonydoboz generálódik minden egységhez a jellemzőtérképek térbeli dimenzióiban, mind az öt léptéken összesen $(32^2 + 16^2 + 8^2 + 4^2 + 1)\times 4 = 5444$ horgonydoboz generálódik minden képhez.

```{.python .input}
#@tab mxnet
net = TinySSD(num_classes=1)
net.initialize()
X = np.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

```{.python .input}
#@tab pytorch
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

## Tanítás

Most elmagyarázzuk, hogyan tanítható az egylépéses többdobozos felismerési modell objektumdetektáláshoz.


### Az adathalmaz beolvasása és a modell inicializálása

Először is [**olvassuk be a banánfelismerési adathalmazt**] a :numref:`sec_object-detection-dataset` fejezetből.

```{.python .input}
#@tab all
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)
```

A banánfelismerési adathalmazban csak egy osztály van. A modell definiálása után **inicializálnunk kell a paramétereit és meg kell határoznunk az optimalizálási algoritmust**.

```{.python .input}
#@tab mxnet
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
net.initialize(init=init.Xavier(), ctx=device)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'wd': 5e-4})
```

```{.python .input}
#@tab pytorch
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
```

### [**Veszteség- és kiértékelési függvények definiálása**]

Az objektumdetektálásnak két típusú veszteségfüggvénye van.
Az első a horgonydobozok osztályaival foglalkozik: kiszámítása egyszerűen felhasználhatja a képosztályozásnál alkalmazott kereszt-entrópia veszteségfüggvényt.
A második a pozitív (nem háttér) horgonydobozok eltolásaival foglalkozik: ez egy regressziós probléma.
Ehhez a regressziós problémához azonban nem alkalmazzuk a :numref:`subsec_normal_distribution_and_squared_loss` fejezetben leírt négyzetes veszteséget.
Ehelyett a $\ell_1$ normaveszteséget alkalmazzuk, a jóslás és a valódi érték különbségének abszolút értékét.
A `bbox_masks` maszkváltozó kiszűri a negatív horgonydobozokat és az érvénytelen (kitöltött) horgonydobozokat a veszteség kiszámításakor.
Végül összeadjuk a horgonydoboz osztályveszteségét és a horgonydoboz eltolási veszteségét, hogy megkapjuk a modell veszteségfüggvényét.

```{.python .input}
#@tab mxnet
cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
bbox_loss = gluon.loss.L1Loss()

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox
```

```{.python .input}
#@tab pytorch
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox
```

Az osztályozási eredmények kiértékeléséhez pontosságot alkalmazhatunk.
Az eltolásokhoz alkalmazott $\ell_1$ normaveszteség miatt a jósolt befoglaló téglalapok kiértékeléséhez a *közepes abszolút hibát* alkalmazzuk.
Ezek a jóslási eredmények a generált horgonydobozokból és a rájuk vonatkozó jósolt eltolásokból kaphatók.

```{.python .input}
#@tab mxnet
def cls_eval(cls_preds, cls_labels):
    # Mivel az osztályjóslási eredmények az utolsó dimenzión vannak,
    # az `argmax`-nak meg kell adni ezt a dimenziót
    return float((cls_preds.argmax(axis=-1).astype(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((np.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

```{.python .input}
#@tab pytorch
def cls_eval(cls_preds, cls_labels):
    # Mivel az osztályjóslási eredmények az utolsó dimenzión vannak,
    # az `argmax`-nak meg kell adni ezt a dimenziót
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

### [**A modell tanítása**]

A modell tanításakor az előreterjesztésben többléptékű horgonydobozokat (`anchors`) kell generálnunk, és jósolnunk kell osztályaikat (`cls_preds`) és eltolásaikat (`bbox_preds`).
Ezután felcímkézzük a generált horgonydobozok osztályait (`cls_labels`) és eltolásait (`bbox_labels`) az `Y` címkeinformáció alapján.
Végül a veszteségfüggvényt az osztályok és eltolások jósolt és felcímkézett értékei alapján számítjuk ki.
A tömör implementáció érdekében a teszthalmaz kiértékelése itt kimaradt.

```{.python .input}
#@tab mxnet
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
for epoch in range(num_epochs):
    # Tanítási pontosság összege, példák száma a pontosság összegében,
    # Abszolút hiba összege, példák száma az abszolút hiba összegében
    metric = d2l.Accumulator(4)
    for features, target in train_iter:
        timer.start()
        X = features.as_in_ctx(device)
        Y = target.as_in_ctx(device)
        with autograd.record():
            # Többléptékű horgonydobozok generálása és osztályaik
            # és eltolásaik jóslása
            anchors, cls_preds, bbox_preds = net(X)
            # Ezen horgonydobozok osztályainak és eltolásainak felcímkézése
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors,
                                                                      Y)
            # A veszteségfüggvény kiszámítása az osztályok és eltolások
            # jósolt és felcímkézett értékei alapján
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
        l.backward()
        trainer.step(batch_size)
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.size,
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.size)
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter._dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

```{.python .input}
#@tab pytorch
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # Tanítási pontosság összege, példák száma a pontosság összegében,
    # Abszolút hiba összege, példák száma az abszolút hiba összegében
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # Többléptékű horgonydobozok generálása és osztályaik
        # és eltolásaik jóslása
        anchors, cls_preds, bbox_preds = net(X)
        # Ezen horgonydobozok osztályainak és eltolásainak felcímkézése
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # A veszteségfüggvény kiszámítása az osztályok és eltolások
        # jósolt és felcímkézett értékei alapján
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

## [**Jóslás**]

A jóslás során a cél a képen lévő összes érdeklődési objektum felismerése.
Az alábbiakban beolvasunk és átméretezünk egy teszt képet, és átalakítjuk a konvolúciós rétegek által igényelt négydimensziós tenzorrá.

```{.python .input}
#@tab mxnet
img = image.imread('../img/banana.jpg')
feature = image.imresize(img, 256, 256).astype('float32')
X = np.expand_dims(feature.transpose(2, 0, 1), axis=0)
```

```{.python .input}
#@tab pytorch
X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
```

Az alábbi `multibox_detection` függvény használatával a jósolt befoglaló téglalapokat a horgonydobozokból és jósolt eltolásaikból kapjuk meg.
Majd nem-maximum szuppressziót alkalmazunk a hasonló jósolt befoglaló téglalapok eltávolítására.

```{.python .input}
#@tab mxnet
def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_ctx(device))
    cls_probs = npx.softmax(cls_preds).transpose(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

```{.python .input}
#@tab pytorch
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

Végül [**megjelenítjük az összes legalább 0.9-es megbízhatóságú jósolt befoglaló téglalapot**] kimenetként.

```{.python .input}
#@tab mxnet
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img.asnumpy())
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[:2]
        bbox = [row[2:6] * np.array((w, h, w, h), ctx=row.ctx)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output, threshold=0.9)
```

```{.python .input}
#@tab pytorch
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)
```

## Összefoglalás

* Az egylépéses többdobozos felismerés egy többléptékű objektumdetektálási modell. Alaphálózatán és számos többléptékű jellemzőtérkép-blokkján keresztül az egylépéses többdobozos felismerés változó számú, különböző méretű horgonydobozt generál, és változó méretű objektumokat ismer fel ezen horgonydobozok osztályainak és eltolásainak jóslásával (és így a befoglaló téglalapokkal).
* Az egylépéses többdobozos felismerési modell tanítása során a veszteségfüggvényt a horgonydoboz osztályok és eltolások jósolt és felcímkézett értékei alapján számítják ki.



## Feladatok

1. Javítható-e az egylépéses többdobozos felismerés a veszteségfüggvény fejlesztésével? Például cseréld a $\ell_1$ normaveszteséget simított $\ell_1$ normaveszteségre a jósolt eltolásokhoz. Ez a veszteségfüggvény a nulla körüli simasághoz egy négyzetes függvényt alkalmaz, amelyet a $\sigma$ hiperparaméter szabályoz:

$$
f(x) =
    \begin{cases}
    (\sigma x)^2/2,& \textrm{ha }|x| < 1/\sigma^2\\
    |x|-0.5/\sigma^2,& \textrm{különben}
    \end{cases}
$$

Ha $\sigma$ nagyon nagy, ez a veszteség hasonló a $\ell_1$ normaveszteséghez. Kisebb értéknél a veszteségfüggvény simább.

```{.python .input}
#@tab mxnet
sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = np.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = npx.smooth_l1(x, scalar=s)
    d2l.plt.plot(x.asnumpy(), y.asnumpy(), l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def smooth_l1(data, scalar):
    out = []
    for i in data:
        if abs(i) < 1 / (scalar ** 2):
            out.append(((scalar * i) ** 2) / 2)
        else:
            out.append(abs(i) - 0.5 / (scalar ** 2))
    return torch.tensor(out)

sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = torch.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = smooth_l1(x, scalar=s)
    d2l.plt.plot(x, y, l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

Emellett a kísérletben az osztályjósláshoz kereszt-entrópia veszteséget alkalmaztunk: jelölve $p_j$-vel a $j$ valódi osztály jósolt valószínűségét, a kereszt-entrópia veszteség $-\log p_j$. Alkalmazhatjuk a fokális veszteséget is :cite:`Lin.Goyal.Girshick.ea.2017`: adott $\gamma > 0$ és $\alpha > 0$ hiperparaméterek esetén ez a veszteség a következőképpen definiálható:

$$ - \alpha (1-p_j)^{\gamma} \log p_j.$$

Amint látható, a $\gamma$ növelése hatékonyan csökkenti a jól osztályozott példák relatív veszteségét (pl. $p_j > 0.5$), így a tanítás jobban koncentrálhat azokra a nehéz példákra, amelyeket tévesen osztályoztak.

```{.python .input}
#@tab mxnet
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * np.log(x)

x = np.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x.asnumpy(), focal_loss(gamma, x).asnumpy(), l,
                     label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * torch.log(x)

x = torch.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x, focal_loss(gamma, x), l, label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

2. Helyhiány miatt elhagytunk néhány implementációs részletet az egylépéses többdobozos felismerési modellből ebben a fejezetben. Tovább javítható-e a modell a következő szempontokból:
    1. Ha egy objektum sokkal kisebb a képhez képest, a modell nagyobbra méretezheti a bemeneti képet.
    1. Általában nagyszámú negatív horgonydoboz van. Az osztályeloszlás kiegyensúlyozottabbá tételéhez le lehetne mintavételezni a negatív horgonydobozokat.
    1. A veszteségfüggvényben különböző súlyhiperparamétereket rendelünk az osztályveszteséghez és az eltolási veszteséghez.
    1. Más módszereket is alkalmazhatunk az objektumdetektálási modell kiértékeléséhez, például az egylépéses többdobozos felismerési cikkben :cite:`Liu.Anguelov.Erhan.ea.2016` leírtakat.



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/373)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1604)
:end_tab:
