# Faktorizációs gépek

A faktorizációs gépek (FM), amelyeket :citet:`Rendle.2010` javasolt, felügyelt algoritmus, amely osztályozási, regressziós és rangsorolási feladatokhoz használható. Gyorsan népszerű és hatásos módszerré vált előrejelzések és ajánlások készítésében. Különösen a lineáris regressziós modell és a mátrixfaktorizációs modell általánosításának tekinthető. Emellett a polinomiális kerneles támogatóvektor-gépekhez is hasonlít. A faktorizációs gépek előnyei a lineáris regresszióval és a mátrixfaktorizációval szemben a következők: (1) képesek $\chi$-utas változóinterakciók modellezésére, ahol $\chi$ a polinomiális rendet jelöli, és általában kettőre állítjuk. (2) A faktorizációs gépekhez társított gyors optimalizálási algoritmus a polinomiális számítási időt lineáris bonyolultságra csökkenti, így különösen nagy dimenziós, ritka bemeneteknél rendkívül hatékony. Ezen okok miatt a faktorizációs gépeket széles körben használják a modern hirdetési és termékajánlási rendszerekben. A technikai részleteket és a megvalósítást az alábbiakban ismertetjük.


## 2-es rendű faktorizációs gépek

Formálisan legyen $x \in \mathbb{R}^d$ egy minta jellemzővektora, és legyen $y$ a megfelelő címke, amely lehet valós értékű címke vagy osztálycímke, például a bináris "kattint/nem kattint" osztály. A másodrendű faktorizációs gép modellje a következő:

$$
\hat{y}(x) = \mathbf{w}_0 + \sum_{i=1}^d \mathbf{w}_i x_i + \sum_{i=1}^d\sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j
$$

ahol $\mathbf{w}_0 \in \mathbb{R}$ a globális bias; a $\mathbf{w} \in \mathbb{R}^d$ az $i$-edik változó súlyait jelöli; a $\mathbf{V} \in \mathbb{R}^{d\times k}$ a jellemzőbeágyazásokat reprezentálja; a $\mathbf{v}_i$ a $\mathbf{V}$ $i^\textrm{edik}$ sorát jelenti; $k$ a látens faktorok dimenziója; az $\langle\cdot, \cdot \rangle$ két vektor skalárszorzata. Az $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ az $i^\textrm{edik}$ és $j^\textrm{edik}$ jellemző közötti kölcsönhatást modellezi. Egyes jellemzőinterakciók könnyen értelmezhetők, ezért szakértők által is megtervezhetők. A legtöbb más jellemzőinterakció azonban rejtve marad az adatokban, és nehéz azonosítani. Ezért a jellemzők közötti kölcsönhatások automatikus modellezése jelentősen csökkentheti a jellemzőmérnökségre fordított erőfeszítéseket. Nyilvánvaló, hogy az első két tag a lineáris regressziós modellnek felel meg, míg az utolsó tag a mátrixfaktorizációs modell kiterjesztése. Ha az $i$ jellemző egy elemet, a $j$ jellemző pedig egy felhasználót reprezentál, akkor a harmadik tag pontosan a felhasználó- és elembeágyazások skalárszorzata. Érdemes megjegyezni, hogy az FM magasabb rendekre is általánosítható (rend > 2). Ennek ellenére a numerikus stabilitás ronthatja az általánosítást.


## Hatékony optimalizálási kritérium

A faktorizációs gépek közvetlen optimalizálása $\mathcal{O}(kd^2)$ bonyolultságú, mivel minden páronkénti kölcsönhatást ki kell számítani. Ennek a hatékonysági problémának a megoldására az FM harmadik tagját újraírhatjuk, ami jelentősen csökkenti a számítási költséget, és lineáris időbonyolultságot eredményez ($\mathcal{O}(kd)$). A páronkénti interakciós tag átalakítása a következő:

$$
\begin{aligned}
&\sum_{i=1}^d \sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j \\
 &= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d\langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j - \frac{1}{2}\sum_{i=1}^d \langle\mathbf{v}_i, \mathbf{v}_i\rangle x_i x_i \\
 &= \frac{1}{2} \big (\sum_{i=1}^d \sum_{j=1}^d \sum_{l=1}^k\mathbf{v}_{i, l} \mathbf{v}_{j, l} x_i x_j - \sum_{i=1}^d \sum_{l=1}^k \mathbf{v}_{i, l} \mathbf{v}_{i, l} x_i x_i \big)\\
 &=  \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i) (\sum_{j=1}^d \mathbf{v}_{j, l}x_j) - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2 \big ) \\
 &= \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i)^2 - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2)
 \end{aligned}
$$

Ezzel az átalakítással a modell bonyolultsága jelentősen csökken. Ráadásul ritka jellemzők esetén csak a nemnulla elemeket kell kiszámítani, így az összesített bonyolultság lineáris marad a nemnulla jellemzők számával.

Az FM modell tanításához regressziós feladathoz az MSE-veszteség, osztályozási feladatokhoz a keresztentrópia-veszteség, rangsorolási feladathoz pedig a BPR-veszteség használható. Az optimalizáláshoz olyan szabványos optimalizálók is megfelelnek, mint a sztochasztikus gradient descent vagy az Adam.

```{.python .input  n=2}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## A modell megvalósítása
A következő kód valósítja meg a faktorizációs gépeket. Jól látható, hogy az FM egy lineáris regressziós blokkból és egy hatékony jellemzőinterakciós blokkból áll. A végső pontszámra szigmoid függvényt alkalmazunk, mivel a CTR-előrejelzést osztályozási feladatként kezeljük.

```{.python .input  n=2}
#@tab mxnet
class FM(nn.Block):
    def __init__(self, field_dims, num_factors):
        super(FM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)

    def forward(self, x):
        square_of_sum = np.sum(self.embedding(x), axis=1) ** 2
        sum_of_square = np.sum(self.embedding(x) ** 2, axis=1)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
        x = npx.sigmoid(x)
        return x
```

## A hirdetési adathalmaz betöltése
Az előző szakasz CTR-adatburkolóját használjuk az online hirdetési adathalmaz betöltéséhez.

```{.python .input  n=3}
#@tab mxnet
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
```

## A modell tanítása
Ezután megtanítjuk a modellt. A tanulási ráta alapértelmezés szerint 0.02, a beágyazás mérete pedig 20. A modell tanításához az `Adam` optimalizálót és a `SigmoidBinaryCrossEntropyLoss` veszteséget használjuk.

```{.python .input  n=5}
#@tab mxnet
devices = d2l.try_all_gpus()
net = FM(train_data.field_dims, num_factors=20)
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.02, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## Összefoglalás

* Az FM egy általános keretrendszer, amely számos feladatra alkalmazható, például regresszióra, osztályozásra és rangsorolásra.
* A jellemzők közötti kölcsönhatás/kombináció fontos az előrejelzési feladatokban, és a 2-es rendű kölcsönhatás az FM-mel hatékonyan modellezhető.

## Gyakorlatok

* Próbáld ki az FM-et más adathalmazokon, például az Avazu, MovieLens és Criteo adathalmazokon!
* Változtasd meg a beágyazás méretét, és nézd meg a teljesítményre gyakorolt hatását. Megfigyelsz hasonló mintázatot, mint a mátrixfaktorizációnál?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/406)
:end_tab:
