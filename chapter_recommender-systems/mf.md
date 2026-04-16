# Mátrixfaktorizáció

A mátrixfaktorizáció :cite:`Koren.Bell.Volinsky.2009` egy régóta bevált algoritmus az ajánlórendszerek szakirodalmában. A mátrixfaktorizációs modell első változatát Simon Funk javasolta egy híres [blogbejegyzésben](https://sifter.org/%7Esimon/journal/20061211.html), amelyben leírta az interakciós mátrix faktorizálásának ötletét. Ezután széles körben ismertté vált a 2006-ban megrendezett Netflix-verseny révén. Abban az időben a Netflix, egy médiaszolgáltató és filmkölcsönző vállalat, versenyt hirdetett ajánlórendszerének javítására. Az a csapat, amely a Netflix alaptervéhez (azaz a Cinematch-hez) képest 10 százalékos javulást ér el, egymillió dollár díjat nyerhet. Ez a verseny rengeteg figyelmet vonzott az ajánlórendszerek kutatásának területére. Végül a nagydíjat a BellKor's Pragmatic Chaos csapata nyerte – ez a BellKor, a Pragmatic Theory és a BigChaos közös csapata volt (ezekkel az algoritmusokkal most nem kell foglalkoznunk). Bár a végső eredmény egy ensemble megoldás (azaz sok algoritmus kombinációja) volt, a mátrixfaktorizációs algoritmus kulcsszerepet játszott az összeállításban. A Netflix nagydíj megoldásának műszaki jelentése :cite:`Toscher.Jahrer.Bell.2009` részletes bevezetést nyújt az alkalmazott modellbe. Ebben a szakaszban a mátrixfaktorizációs modell részleteit és megvalósítását mutatjuk be.


## A mátrixfaktorizációs modell

A mátrixfaktorizáció az együttműködési szűrési modellek egy osztálya. Konkrétan a modell a felhasználó–elem interakciós mátrixot (például értékelési mátrixot) két alacsonyabb rangú mátrix szorzatára faktorizálja, megragadva a felhasználó–elem interakciók alacsony rangú struktúráját.

Jelölje $\mathbf{R} \in \mathbb{R}^{m \times n}$ az interakciós mátrixot $m$ felhasználóval és $n$ elemmel, ahol $\mathbf{R}$ értékei explicit értékelések. A felhasználó–elem interakció egy $\mathbf{P} \in \mathbb{R}^{m \times k}$ felhasználói látens mátrixra és egy $\mathbf{Q} \in \mathbb{R}^{n \times k}$ elem látens mátrixra faktorizálható, ahol $k \ll m, n$ a látens faktor mérete. Jelölje $\mathbf{p}_u$ a $\mathbf{P}$ mátrix $u^\textrm{-adik}$ sorát, és $\mathbf{q}_i$ a $\mathbf{Q}$ mátrix $i^\textrm{-edik}$ sorát. Egy adott $i$ elem esetén a $\mathbf{q}_i$ elemei azt mérik, hogy az elem milyen mértékben rendelkezik bizonyos jellemzőkkel, például egy film műfajával vagy nyelvével. Egy adott $u$ felhasználónál a $\mathbf{p}_u$ elemei azt mérik, hogy a felhasználó mennyire érdeklődik az elemek megfelelő jellemzői iránt. Ezek a látens faktorok mérhetnek nyilvánvaló dimenziókat, mint a fenti példákban, vagy teljesen értelmezhetetlenek lehetnek. A becsült értékelések a következőképpen számíthatók:

$$\hat{\mathbf{R}} = \mathbf{PQ}^\top$$

ahol $\hat{\mathbf{R}}\in \mathbb{R}^{m \times n}$ a becsült értékelési mátrix, amelynek alakja megegyezik $\mathbf{R}$-ével. Ennek az előrejelzési szabálynak az egyik fő problémája, hogy nem képes modellezni a felhasználók és az elemek torzításait. Például egyes felhasználók általában magasabb értékeléseket adnak, vagy egyes elemek gyengébb minőségük miatt mindig alacsonyabb értékeléseket kapnak. Ezek a torzítások mindennaposak a valós alkalmazásokban. Ezek megragadásához felhasználó-specifikus és elem-specifikus torzítási tagokat vezetünk be. Konkrétan az $u$ felhasználó $i$ elemre adott becsült értékelése a következőképpen számítható:

$$
\hat{\mathbf{R}}_{ui} = \mathbf{p}_u\mathbf{q}^\top_i + b_u + b_i
$$

Ezt követően a mátrixfaktorizációs modellt a becsült és a tényleges értékelési pontszámok közötti négyzetes középhiba minimalizálásával tanítjuk. A célfüggvény a következőképpen definiálható:

$$
\underset{\mathbf{P}, \mathbf{Q}, b}{\mathrm{argmin}} \sum_{(u, i) \in \mathcal{K}} \| \mathbf{R}_{ui} -
\hat{\mathbf{R}}_{ui} \|^2 + \lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}
\|^2_F + b_u^2 + b_i^2 )
$$

ahol $\lambda$ a regularizációs együttható. A $\lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}
\|^2_F + b_u^2 + b_i^2 )$ regularizációs tag a paraméterek nagyságát büntetve kerüli el a túltanulást. Azok az $(u, i)$ párok, amelyekre $\mathbf{R}_{ui}$ ismert, a $\mathcal{K}=\{(u, i) \mid \mathbf{R}_{ui} \textrm{ ismert}\}$ halmazban vannak tárolva. A modell paraméterei egy optimalizáló algoritmussal, például sztochasztikus gradienscsökkenéssel vagy Adam-mal taníthatók.

A mátrixfaktorizációs modell szemléletes ábrázolása az alábbiakban látható:

![Illustration of matrix factorization model](../img/rec-mf.svg)

A szakasz hátralévő részében elmagyarázzuk a mátrixfaktorizáció megvalósítását, és betanítjuk a modellt a MovieLens adathalmazon.

```{.python .input  n=2}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
npx.set_np()
```

## A modell megvalósítása

Először megvalósítjuk a fent leírt mátrixfaktorizációs modellt. A felhasználói és elem látens faktorok az `nn.Embedding` segítségével hozhatók létre. Az `input_dim` az elemek/felhasználók száma, az `output_dim` pedig a látens faktorok $k$ dimenziója. Az `nn.Embedding` felhasználható a felhasználói/elem torzítások létrehozására is, ha az `output_dim`-et egyre állítjuk. A `forward` függvényben a felhasználói és elem azonosítók az embeddingjük kikeresésére szolgálnak.

```{.python .input  n=4}
#@tab mxnet
class MF(nn.Block):
    def __init__(self, num_factors, num_users, num_items, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.P = nn.Embedding(input_dim=num_users, output_dim=num_factors)
        self.Q = nn.Embedding(input_dim=num_items, output_dim=num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id, item_id):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)
        outputs = (P_u * Q_i).sum(axis=1) + np.squeeze(b_u) + np.squeeze(b_i)
        return outputs.flatten()
```

## Kiértékelési mérőszámok

Ezután megvalósítjuk az RMSE (négyzetes középhiba gyöke) mérőszámot, amelyet általánosan használnak a modell által becsült értékelési pontszámok és a tényleges megfigyelt értékelések (valós értékek) közötti különbségek mérésére :cite:`Gunawardana.Shani.2015`. Az RMSE definíciója:

$$
\textrm{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|}\sum_{(u, i) \in \mathcal{T}}(\mathbf{R}_{ui} -\hat{\mathbf{R}}_{ui})^2}
$$

ahol $\mathcal{T}$ a kiértékelni kívánt felhasználó–elem párok halmaza, $|\mathcal{T}|$ ennek a halmaznak a mérete. Az `mx.metric` által biztosított RMSE függvényt használhatjuk.

```{.python .input  n=3}
#@tab mxnet
def evaluator(net, test_iter, devices):
    rmse = mx.metric.RMSE()  # RMSE lekérése
    rmse_list = []
    for idx, (users, items, ratings) in enumerate(test_iter):
        u = gluon.utils.split_and_load(users, devices, even_split=False)
        i = gluon.utils.split_and_load(items, devices, even_split=False)
        r_ui = gluon.utils.split_and_load(ratings, devices, even_split=False)
        r_hat = [net(u, i) for u, i in zip(u, i)]
        rmse.update(labels=r_ui, preds=r_hat)
        rmse_list.append(rmse.get()[1])
    return float(np.mean(np.array(rmse_list)))
```

## A modell tanítása és kiértékelése


A tanítási függvényben $\ell_2$ veszteséget alkalmazunk súlycsökkenéssel. A súlycsökkenési mechanizmus ugyanolyan hatású, mint az $\ell_2$ regularizáció.

```{.python .input  n=4}
#@tab mxnet
#@save
def train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices=d2l.try_all_gpus(), evaluator=None,
                        **kwargs):
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 2],
                            legend=['train loss', 'test RMSE'])
    for epoch in range(num_epochs):
        metric, l = d2l.Accumulator(3), 0.
        for i, values in enumerate(train_iter):
            timer.start()
            input_data = []
            values = values if isinstance(values, list) else [values]
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, devices))
            train_feat = input_data[:-1] if len(values) > 1 else input_data
            train_label = input_data[-1]
            with autograd.record():
                preds = [net(*t) for t in zip(*train_feat)]
                ls = [loss(p, s) for p, s in zip(preds, train_label)]
            [l.backward() for l in ls]
            l += sum([l.asnumpy() for l in ls]).mean() / len(devices)
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
            timer.stop()
        if len(kwargs) > 0:  # Az AutoRec szakaszban kerül felhasználásra
            test_rmse = evaluator(net, test_iter, kwargs['inter_mat'],
                                  devices)
        else:
            test_rmse = evaluator(net, test_iter, devices)
        train_l = l / (i + 1)
        animator.add(epoch + 1, (train_l, test_rmse))
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test RMSE {test_rmse:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')
```

Végül fűzzük össze az összes részt, és tanítsuk be a modellt. A látens faktor dimenziót 30-ra állítjuk.

```{.python .input  n=5}
#@tab mxnet
devices = d2l.try_all_gpus()
num_users, num_items, train_iter, test_iter = d2l.split_and_load_ml100k(
    test_ratio=0.1, batch_size=512)
net = MF(30, num_users, num_items)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 20, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                    devices, evaluator)
```

Az alábbiakban a betanított modellt használjuk annak előrejelzésére, hogy egy felhasználó (20-as azonosító) milyen értékelést adhat egy elemre (30-as azonosító).

```{.python .input  n=6}
#@tab mxnet
scores = net(np.array([20], dtype='int', ctx=devices[0]),
             np.array([30], dtype='int', ctx=devices[0]))
scores
```

## Összefoglalás

* A mátrixfaktorizációs modell széles körben használatos az ajánlórendszerekben. Alkalmazható annak előrejelzésére, hogy egy felhasználó milyen értékelést adhat egy elemre.
* Megvalósítható és betanítható a mátrixfaktorizáció ajánlórendszerek számára.


## Gyakorlatok

* Változtasd meg a látens faktorok méretét! Hogyan befolyásolja a látens faktorok mérete a modell teljesítményét?
* Próbálj ki különböző optimalizálókat, tanulási rátákat és súlycsökkenési rátákat!
* Ellenőrizd más felhasználók becsült értékelési pontszámait egy adott filmre!


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/400)
:end_tab:
