# AutoRec: Értékbecslés autokódolókkal

Bár a mátrixfaktorizációs modell elfogadható teljesítményt ér el az értékbecslési feladaton, lényegében lineáris modell. Ezért az ilyen modellek nem képesek megragadni az összetett, nemlineáris és finom kapcsolódásokat, amelyek a felhasználói preferenciák előrejelzésében hasznosak lehetnek. Ebben a szakaszban egy nemlineáris neurális hálózati kollaboratív szűrési modellt, az AutoRec-et :cite:`Sedhain.Menon.Sanner.ea.2015` mutatjuk be. Ez a kollaboratív szűrést (CF) autokódoló-architektúrával azonosítja, és arra törekszik, hogy az explicit visszajelzés alapján nemlineáris transzformációkat építsen be a CF-be. A neurális hálózatokról bebizonyosodott, hogy bármely folytonos függvényt képesek közelíteni, ezért jól alkalmasak a mátrixfaktorizáció korlátainak kezelésére és kifejezőképességének növelésére.

Egyrészt az AutoRec ugyanazzal a szerkezettel rendelkezik, mint egy autokódoló, vagyis bemeneti rétegből, rejtett rétegből és rekonstrukciós (kimeneti) rétegből áll. Az autokódoló olyan neurális hálózat, amely megtanulja a bemenetét a kimenetére másolni, hogy a bemeneteket a rejtett, általában alacsony dimenziós reprezentációkba kódolja. Az AutoRec esetében a felhasználókat/elemeket nem explicit módon ágyazza be alacsony dimenziós térbe, hanem az interakciós mátrix adott oszlopát/sorát használja bemenetként, majd a kimeneti rétegben rekonstruálja az interakciós mátrixot.

Másrészt az AutoRec eltér a hagyományos autokódolótől: ahelyett, hogy a rejtett reprezentációk megtanulására összpontosítana, az AutoRec a kimeneti réteg tanulására/rekonstrukciójára koncentrál. Részben megfigyelt interakciós mátrixot használ bemenetként, és egy teljes értékbecslési mátrix rekonstruálására törekszik. Eközben a bemenet hiányzó elemei a kimeneti rétegben, rekonstrukció útján kerülnek kitöltésre az ajánlás céljából.

Az AutoRec-nek két változata van: felhasználó-alapú és elem-alapú. Rövidség kedvéért itt csak az elem-alapú AutoRec-et mutatjuk be. A felhasználó-alapú AutoRec ebből hasonlóan levezethető.


## Modell

Legyen $\mathbf{R}_{*i}$ az értékbecslési mátrix $i^\textrm{edik}$ oszlopa, ahol az ismeretlen értékeléseket alapértelmezés szerint nullákra állítjuk. A neurális architektúra a következő:

$$
h(\mathbf{R}_{*i}) = f(\mathbf{W} \cdot g(\mathbf{V} \mathbf{R}_{*i} + \mu) + b)
$$

ahol az $f(\cdot)$ és $g(\cdot)$ aktivációs függvényeket jelöl, a $\mathbf{W}$ és $\mathbf{V}$ súlymátrixok, a $\mu$ és $b$ pedig biasok. Jelölje $h(\cdot)$ az AutoRec teljes hálózatát. Az $h(\mathbf{R}_{*i})$ kimenet az értékbecslési mátrix $i^\textrm{edik}$ oszlopának rekonstrukciója.

A következő célfüggvény a rekonstrukciós hibát minimalizálja:

$$
\underset{\mathbf{W},\mathbf{V},\mu, b}{\mathrm{argmin}} \sum_{i=1}^M{\parallel \mathbf{R}_{*i} - h(\mathbf{R}_{*i})\parallel_{\mathcal{O}}^2} +\lambda(\| \mathbf{W} \|_F^2 + \| \mathbf{V}\|_F^2)
$$

ahol a $\| \cdot \|_{\mathcal{O}}$ azt jelenti, hogy csak a megfigyelt értékelések hozzájárulását vesszük figyelembe, vagyis visszaterjesztés során csak a megfigyelt bemenetekhez tartozó súlyok frissülnek.

```{.python .input  n=3}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx

npx.set_np()
```

## A modell megvalósítása

Egy tipikus autokódoló egy kódolóből és egy dekódolóből áll. Az kódoló a bemenetet rejtett reprezentációkba vetíti, a dekódoló pedig a rejtett réteget a rekonstrukciós rétegbe képezi le. Ezt a gyakorlatot követjük, és a kódolót, illetve a dekódolót teljesen összekötött rétegekkel hozzuk létre. Az kódoló aktivációja alapértelmezés szerint `sigmoid`, a dekódolóhez pedig nem alkalmazunk aktivációt. A túlillesztés csökkentésére a kódolási transzformáció után dropoutot használunk. A nem megfigyelt bemenetek gradiensét maszkoljuk, hogy csak a megfigyelt értékelések járuljanak hozzá a modell tanulási folyamatához.

```{.python .input  n=2}
#@tab mxnet
class AutoRec(nn.Block):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Dense(num_hidden, activation='sigmoid',
                                use_bias=True)
        self.decoder = nn.Dense(num_users, use_bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(self.encoder(input))
        pred = self.decoder(hidden)
        if autograd.is_training():  # Gradiens maszkolása tanítás közben
            return pred * np.sign(input)
        else:
            return pred
```

## Az értékelő újraimplementálása

Mivel a bemenet és a kimenet megváltozott, az értékelő függvényt is újra kell implementálnunk, miközben továbbra is az RMSE-t használjuk pontossági mérőszámként.

```{.python .input  n=3}
#@tab mxnet
def evaluator(network, inter_matrix, test_data, devices):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, devices, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # A teszt RMSE kiszámítása
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                   / np.sum(np.sign(test_data)))
    return float(rmse)
```

## A modell tanítása és értékelése

Most tanítsuk és értékeljük az AutoRec-et a MovieLens adathalmazon. Jól látható, hogy a teszt RMSE alacsonyabb, mint a mátrixfaktorizációs modellé, ami megerősíti a neurális hálózatok hatékonyságát az értékbecslési feladatban.

```{.python .input  n=4}
#@tab mxnet
devices = d2l.try_all_gpus()
# A MovieLens 100K adathalmaz betöltése
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items)
_, _, _, train_inter_mat = d2l.load_data_ml100k(train_data, num_users,
                                                num_items)
_, _, _, test_inter_mat = d2l.load_data_ml100k(test_data, num_users,
                                               num_items)
train_iter = gluon.data.DataLoader(train_inter_mat, shuffle=True,
                                   last_batch="rollover", batch_size=256,
                                   num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(np.array(train_inter_mat), shuffle=False,
                                  last_batch="keep", batch_size=1024,
                                  num_workers=d2l.get_dataloader_workers())
# Modell inicializálása, tanítása és értékelése
net = AutoRec(500, num_users)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 25, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
d2l.train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices, evaluator, inter_mat=test_inter_mat)
```

## Összefoglalás

* Az autokódolókkal a mátrixfaktorizációs algoritmus nemlineáris rétegekkel és dropout regularizációval egészíthető ki.
* A MovieLens 100K adathalmazon végzett kísérletek azt mutatják, hogy az AutoRec jobb teljesítményt ér el, mint a mátrixfaktorizáció.



## Gyakorlatok

* Változtasd meg az AutoRec rejtett dimenzióját, és figyeld meg a modell teljesítményére gyakorolt hatását.
* Próbálj meg több rejtett réteget hozzáadni. Segít ez javítani a modell teljesítményét?
* Tudsz jobb kombinációt találni a dekódoló és az kódoló aktivációs függvényeiből?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/401)
:end_tab:
