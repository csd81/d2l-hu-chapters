# Mély faktorizációs gépek

A hatékony jellemzőkombinációk megtanulása kulcsfontosságú a kattintási arány előrejelzésének sikerében. A faktorizációs gépek a jellemzőinterakciókat lineáris paradigmában modellezik (például bilineáris kölcsönhatásokkal). Ez a valós adatoknál gyakran nem elegendő, mivel a jellemzők közötti belső keresztezési struktúrák rendszerint nagyon összetettek és nemlineárisak. Ráadásul a gyakorlatban a faktorizációs gépek többnyire csak másodrendű jellemzőinterakciókat használnak. Elméletileg magasabb rendű jellemzőkombinációk is modellezhetők faktorizációs gépekkel, de ezt általában a numerikus instabilitás és a magas számítási komplexitás miatt nem alkalmazzák.

Egy hatékony megoldás a mély neurális hálózatok használata. A mély neurális hálózatok erősek a jellemzőreprezentációk tanulásában, és képesek összetett jellemzőinterakciókat is megtanulni. Ezért természetes a mély neurális hálózatokat a faktorizációs gépekkel integrálni. A nemlineáris transzformációs rétegek hozzáadása lehetővé teszi, hogy a faktorizációs gépek alacsony és magas rendű jellemzőkombinációkat is modellezzenek. Emellett a bemenetek nemlineáris belső szerkezetei is megragadhatók mély neurális hálózatokkal. Ebben a szakaszban a mély faktorizációs gépek (DeepFM) :cite:`Guo.Tang.Ye.ea.2017` nevű reprezentatív modellt mutatjuk be, amely az FM-et és a mély neurális hálózatokat kombinálja.


## Modellarchitektúrák

A DeepFM egy FM-komponensből és egy mély komponensből áll, amelyek párhuzamos szerkezetben kapcsolódnak össze. Az FM-komponens megegyezik a 2-es rendű faktorizációs gépekkel, és az alacsony rendű jellemzőinterakciók modellezésére szolgál. A mély komponens egy MLP, amely a magas rendű jellemzőinterakciókat és a nemlinearitásokat ragadja meg. A két komponens ugyanazokat a bemeneteket/beágyazásokat osztja meg, és kimeneteiket összeadva kapjuk a végső előrejelzést. Érdemes megjegyezni, hogy a DeepFM szellemisége hasonlít a Wide \& Deep architektúráéra, amely egyszerre képes a memorizálásra és az általánosításra. A DeepFM előnye a Wide \& Deep modellel szemben, hogy automatikusan azonosítja a jellemzőkombinációkat, így csökkenti a kézzel készített jellemzőmérnökség szükségességét.

Rövidség kedvéért az FM-komponens leírását elhagyjuk, és a kimenetet $\hat{y}^{(FM)}$-mel jelöljük. További részletekért az olvasó az előző szakaszra hivatkozhat. Legyen $\mathbf{e}_i \in \mathbb{R}^{k}$ az $i^\textrm{edik}$ mező látens jellemzővektora. A mély komponens bemenete az összes mező sűrű beágyazásának összefűzése, amelyeket a ritka kategorikus jellemzőbemenet alapján keresünk le, és ezt a következőképpen jelöljük:

$$
\mathbf{z}^{(0)}  = [\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_f],
$$

ahol $f$ a mezők száma. Ezután a következő neurális hálózatba kerül:

$$
\mathbf{z}^{(l)}  = \alpha(\mathbf{W}^{(l)}\mathbf{z}^{(l-1)} + \mathbf{b}^{(l)}),
$$

ahol $\alpha$ az aktivációs függvény. A $\mathbf{W}_{l}$ és $\mathbf{b}_{l}$ az $l^\textrm{edik}$ réteg súlya és bias-a. Jelölje $y_{DNN}$ az előrejelzés kimenetét. A DeepFM végső előrejelzése az FM és a DNN kimeneteinek összege. Így:

$$
\hat{y} = \sigma(\hat{y}^{(FM)} + \hat{y}^{(DNN)}),
$$

ahol $\sigma$ a szigmoid függvény. A DeepFM architektúráját az alábbi ábra szemlélteti.
![Illustration of the DeepFM model](../img/rec-deepfm.svg)

Érdemes megjegyezni, hogy a DeepFM nem az egyetlen módja a mély neurális hálózatok és az FM kombinálásának. A jellemzőinterakciók fölé nemlineáris rétegeket is építhetünk :cite:`He.Chua.2017`.

```{.python .input  n=2}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## A DeepFM megvalósítása
A DeepFM megvalósítása hasonló az FM-hez. Az FM-részt változatlanul hagyjuk, és egy `relu` aktivációjú MLP blokkot használunk. A modell regularizálására dropoutot is alkalmazunk. Az MLP neuronjainak száma az `mlp_dims` hiperparaméterrel állítható.

```{.python .input  n=2}
#@tab mxnet
class DeepFM(nn.Block):
    def __init__(self, field_dims, num_factors, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)
        input_dim = self.embed_output_dim = len(field_dims) * num_factors
        self.mlp = nn.Sequential()
        for dim in mlp_dims:
            self.mlp.add(nn.Dense(dim, 'relu', True, in_units=input_dim))
            self.mlp.add(nn.Dropout(rate=drop_rate))
            input_dim = dim
        self.mlp.add(nn.Dense(in_units=input_dim, units=1))

    def forward(self, x):
        embed_x = self.embedding(x)
        square_of_sum = np.sum(embed_x, axis=1) ** 2
        sum_of_square = np.sum(embed_x ** 2, axis=1)
        inputs = np.reshape(embed_x, (-1, self.embed_output_dim))
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True) \
            + self.mlp(inputs)
        x = npx.sigmoid(x)
        return x
```

## A modell tanítása és értékelése
Az adatbetöltési folyamat megegyezik az FM esetével. A DeepFM MLP komponensét egy háromrétegű, piramisszerkezetű (30-20-10) sűrű hálózatra állítjuk. Az összes többi hiperparaméter megegyezik az FM-ével.

```{.python .input  n=4}
#@tab mxnet
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
field_dims = train_data.field_dims
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
devices = d2l.try_all_gpus()
net = DeepFM(field_dims, num_factors=10, mlp_dims=[30, 20, 10])
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.01, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

Az FM-hez képest a DeepFM gyorsabban konvergál és jobb teljesítményt ér el.

## Összefoglalás

* A neurális hálózatok FM-mel való integrálása lehetővé teszi összetett és magas rendű interakciók modellezését.
* A DeepFM a hirdetési adathalmazon jobb teljesítményt nyújt, mint az eredeti FM.

## Gyakorlatok

* Változtasd meg az MLP szerkezetét, és vizsgáld meg a modell teljesítményére gyakorolt hatását.
* Cseréld le az adathalmazt Criteo-ra, és hasonlítsd össze az eredeti FM modellel.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/407)
:end_tab:
