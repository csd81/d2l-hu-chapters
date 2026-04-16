# Neurális együttműködési szűrés személyre szabott rangsoroláshoz

Ez a szakasz túllép az explicit visszajelzésen, és bemutatja a neurális együttműködési szűrés (NCF) keretrendszerét az implicit visszajelzésen alapuló ajánláshoz. Az implicit visszajelzés mindenütt jelen van az ajánlórendszerekben. Az olyan műveletek, mint a kattintások, vásárlások és megtekintések, általános implicit visszajelzések, amelyek könnyen gyűjthetők és jelzik a felhasználók preferenciáit. A bemutatandó modell, amelynek neve NeuMF :cite:`He.Liao.Zhang.ea.2017` – a neurális mátrixfaktorizáció rövidítése –, a személyre szabott rangsorolási feladatot kívánja megoldani implicit visszajelzéssel. Ez a modell a neurális hálózatok rugalmasságát és nemlinearitását aknázza ki a mátrixfaktorizáció skaláris szorzatainak kiváltásához, azzal a céllal, hogy növelje a modell kifejezőképességét. Konkrétan a modell két alháló struktúrával rendelkezik: általánosított mátrixfaktorizációval (GMF) és MLP-vel, és az interakciókat két útvonalon, nem pedig egyszerű skaláris szorzatokkal modellezi. A két hálózat kimeneteit összefűzik a végső előrejelzési pontszámok kiszámításához. Az AutoRec értékelési feladatával ellentétben ez a modell rangsorolt ajánlási listát állít elő minden felhasználónak az implicit visszajelzés alapján. Az előző szakaszban bevezetett személyre szabott rangsorolási veszteséget fogjuk használni a modell tanításához.

## A NeuMF modell

Ahogy már szó volt róla, a NeuMF két alhálót ötvöz. A GMF a mátrixfaktorizáció általános neurális hálózati változata, amelynek bemenete a felhasználói és elem látens faktorok elemenként vett szorzata. Két neurális rétegből áll:

$$
\mathbf{x} = \mathbf{p}_u \odot \mathbf{q}_i \\
\hat{y}_{ui} = \alpha(\mathbf{h}^\top \mathbf{x}),
$$

ahol $\odot$ a vektorok Hadamard-szorzatát jelöli. $\mathbf{P} \in \mathbb{R}^{m \times k}$ és $\mathbf{Q} \in \mathbb{R}^{n \times k}$ rendre a felhasználói és elem látens mátrixnak felel meg. $\mathbf{p}_u \in \mathbb{R}^{ k}$ a $P$ mátrix $u^\textrm{-adik}$ sora, $\mathbf{q}_i \in \mathbb{R}^{ k}$ pedig a $Q$ mátrix $i^\textrm{-edik}$ sora. $\alpha$ és $h$ a kimeneti réteg aktivációs függvényét és súlyát jelöli. $\hat{y}_{ui}$ az az előrejelzési pontszám, amelyet az $u$ felhasználó az $i$ elemre adhat.

A modell másik komponense az MLP. A modell rugalmasságának növelése érdekében az MLP alháló nem osztja meg a felhasználói és elem embeddingeket a GMF-fel. A felhasználói és elem embeddingjek összefűzését használja bemenetként. Az összetett kapcsolatoknak és a nemlineáris transzformációknak köszönhetően képes becsülni a felhasználók és az elemek közötti bonyolult interakciókat. Pontosabban az MLP alháló a következőképpen definiálható:

$$
\begin{aligned}
z^{(1)} &= \phi_1(\mathbf{U}_u, \mathbf{V}_i) = \left[ \mathbf{U}_u, \mathbf{V}_i \right] \\
\phi^{(2)}(z^{(1)})  &= \alpha^1(\mathbf{W}^{(2)} z^{(1)} + b^{(2)}) \\
&... \\
\phi^{(L)}(z^{(L-1)}) &= \alpha^L(\mathbf{W}^{(L)} z^{(L-1)} + b^{(L)})) \\
\hat{y}_{ui} &= \alpha(\mathbf{h}^\top\phi^L(z^{(L-1)}))
\end{aligned}
$$

ahol $\mathbf{W}^*, \mathbf{b}^*$ és $\alpha^*$ a súlymátrixot, a torzításvektort és az aktivációs függvényt jelöli. $\phi^*$ a megfelelő réteg függvényét jelöli. $\mathbf{z}^*$ a megfelelő réteg kimenetét jelöli.

A GMF és az MLP eredményeinek összefűzéséhez – egyszerű összeadás helyett – a NeuMF a két alháló utolsó előtti rétegeit fűzi össze, hogy egy jellemzővektort hozzon létre, amely a további rétegekbe tovább adható. Ezt követően a kimeneteket a $\mathbf{h}$ mátrixszal és egy szigmoid aktivációs függvénnyel vetítik ki. Az előrejelzési réteg a következőképpen formalizálható:
$$
\hat{y}_{ui} = \sigma(\mathbf{h}^\top[\mathbf{x}, \phi^L(z^{(L-1)})]).
$$

A következő ábra a NeuMF modell architektúráját szemlélteti.

![Illustration of the NeuMF model](../img/rec-neumf.svg)

```{.python .input  n=1}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import random

npx.set_np()
```

## A modell megvalósítása
Az alábbi kód megvalósítja a NeuMF modellt. Egy általánosított mátrixfaktorizációs modellből és egy MLP-ből áll, amelyek különböző felhasználói és elem embeddingvektorokat használnak. Az MLP struktúráját a `nums_hiddens` paraméter szabályozza. Alapértelmezett aktivációs függvényként a ReLU-t használjuk.

```{.python .input  n=2}
#@tab mxnet
class NeuMF(nn.Block):
    def __init__(self, num_factors, num_users, num_items, nums_hiddens,
                 **kwargs):
        super(NeuMF, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.U = nn.Embedding(num_users, num_factors)
        self.V = nn.Embedding(num_items, num_factors)
        self.mlp = nn.Sequential()
        for num_hiddens in nums_hiddens:
            self.mlp.add(nn.Dense(num_hiddens, activation='relu',
                                  use_bias=True))
        self.prediction_layer = nn.Dense(1, activation='sigmoid', use_bias=False)

    def forward(self, user_id, item_id):
        p_mf = self.P(user_id)
        q_mf = self.Q(item_id)
        gmf = p_mf * q_mf
        p_mlp = self.U(user_id)
        q_mlp = self.V(item_id)
        mlp = self.mlp(np.concatenate([p_mlp, q_mlp], axis=1))
        con_res = np.concatenate([gmf, mlp], axis=1)
        return self.prediction_layer(con_res)
```

## Testreszabott adathalmaz negatív mintavételezéssel

A páronkénti rangsorolási veszteséghez fontos lépés a negatív mintavételezés. Minden felhasználónál azok az elemek, amelyekkel a felhasználó nem lépett kapcsolatba, jelöltelemeként (nem megfigyelt bejegyzésként) szolgálnak. Az alábbi függvény a felhasználók azonosítóját és a jelöltelemeket veszi bemenetként, és minden felhasználóhoz véletlenszerűen mintáz negatív elemeket a jelöltek halmazából. A tanítási szakasz során a modell biztosítja, hogy azok az elemek, amelyeket a felhasználó kedvel, magasabb rangot kapjanak azoknál, amelyeket nem kedvel vagy amelyekkel nem lépett kapcsolatba.

```{.python .input  n=3}
#@tab mxnet
class PRDataset(gluon.data.Dataset):
    def __init__(self, users, items, candidates, num_items):
        self.users = users
        self.items = items
        self.cand = candidates
        self.all = set([i for i in range(num_items)])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        neg_items = list(self.all - set(self.cand[int(self.users[idx])]))
        indices = random.randint(0, len(neg_items) - 1)
        return self.users[idx], self.items[idx], neg_items[indices]
```

## Kiértékelő
Ebben a szakaszban az időbeli felosztási stratégiát alkalmazzuk a tanító- és teszthalmaz létrehozásához. A modell hatékonyságának értékelésére két kiértékelési mérőszámot használunk: az adott $\ell$ levágásnál mért találati arányt ($\textrm{Hit}@\ell$) és a ROC-görbe alatti területet (AUC). Az adott $\ell$ pozíciónál mért találati arány minden felhasználóra azt jelzi, hogy a javasolt elem szerepel-e az első $\ell$ rangsorolt elem között. A formális definíció a következő:

$$
\textrm{Hit}@\ell = \frac{1}{m} \sum_{u \in \mathcal{U}} \textbf{1}(rank_{u, g_u} <= \ell),
$$

ahol $\textbf{1}$ egy indikátorfüggvényt jelöl, amely 1, ha a valós elem az első $\ell$ listán szerepel, egyébként 0. $rank_{u, g_u}$ az $u$ felhasználó $g_u$ valós elemének rangsorát jelöli az ajánlási listán (az ideális rangsor 1). $m$ a felhasználók száma. $\mathcal{U}$ a felhasználók halmaza.

Az AUC definíciója a következő:

$$
\textrm{AUC} = \frac{1}{m} \sum_{u \in \mathcal{U}} \frac{1}{|\mathcal{I} \backslash S_u|} \sum_{j \in I \backslash S_u} \textbf{1}(rank_{u, g_u} < rank_{u, j}),
$$

ahol $\mathcal{I}$ az elemek halmaza. $S_u$ az $u$ felhasználó jelöltelemeinek halmaza. Megjegyzendő, hogy számos más kiértékelési protokoll is alkalmazható, például a precízió, a visszahívás és a normalizált diszkontált kumulatív nyereség (NDCG).

Az alábbi függvény kiszámítja az egyes felhasználók találatszámát és AUC-értékét.

```{.python .input  n=4}
#@tab mxnet
#@save
def hit_and_auc(rankedlist, test_matrix, k):
    hits_k = [(idx, val) for idx, val in enumerate(rankedlist[:k])
              if val in set(test_matrix)]
    hits_all = [(idx, val) for idx, val in enumerate(rankedlist)
                if val in set(test_matrix)]
    max = len(rankedlist) - 1
    auc = 1.0 * (max - hits_all[0][0]) / max if len(hits_all) > 0 else 0
    return len(hits_k), auc
```

Ezután az összesített találati arány és az AUC a következőképpen számítható.

```{.python .input  n=5}
#@tab mxnet
#@save
def evaluate_ranking(net, test_input, seq, candidates, num_users, num_items,
                     devices):
    ranked_list, ranked_items, hit_rate, auc = {}, {}, [], []
    all_items = set([i for i in range(num_users)])
    for u in range(num_users):
        neg_items = list(all_items - set(candidates[int(u)]))
        user_ids, item_ids, x, scores = [], [], [], []
        [item_ids.append(i) for i in neg_items]
        [user_ids.append(u) for _ in neg_items]
        x.extend([np.array(user_ids)])
        if seq is not None:
            x.append(seq[user_ids, :])
        x.extend([np.array(item_ids)])
        test_data_iter = gluon.data.DataLoader(
            gluon.data.ArrayDataset(*x), shuffle=False, last_batch="keep",
            batch_size=1024)
        for index, values in enumerate(test_data_iter):
            x = [gluon.utils.split_and_load(v, devices, even_split=False)
                 for v in values]
            scores.extend([list(net(*t).asnumpy()) for t in zip(*x)])
        scores = [item for sublist in scores for item in sublist]
        item_scores = list(zip(item_ids, scores))
        ranked_list[u] = sorted(item_scores, key=lambda t: t[1], reverse=True)
        ranked_items[u] = [r[0] for r in ranked_list[u]]
        temp = hit_and_auc(ranked_items[u], test_input[u], 50)
        hit_rate.append(temp[0])
        auc.append(temp[1])
    return np.mean(np.array(hit_rate)), np.mean(np.array(auc))
```

## A modell tanítása és kiértékelése

A tanítási függvény az alábbiakban van definiálva. A modellt páronkénti módon tanítjuk.

```{.python .input  n=6}
#@tab mxnet
#@save
def train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter,
                  num_users, num_items, num_epochs, devices, evaluator,
                  candidates, eval_step=1):
    timer, hit_rate, auc = d2l.Timer(), 0, 0
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['test hit rate', 'test AUC'])
    for epoch in range(num_epochs):
        metric, l = d2l.Accumulator(3), 0.
        for i, values in enumerate(train_iter):
            input_data = []
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, devices))
            with autograd.record():
                p_pos = [net(*t) for t in zip(*input_data[:-1])]
                p_neg = [net(*t) for t in zip(*input_data[:-2],
                                              input_data[-1])]
                ls = [loss(p, n) for p, n in zip(p_pos, p_neg)]
            [l.backward(retain_graph=False) for l in ls]
            l += sum([l.asnumpy() for l in ls]).mean()/len(devices)
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
            timer.stop()
        with autograd.predict_mode():
            if (epoch + 1) % eval_step == 0:
                hit_rate, auc = evaluator(net, test_iter, test_seq_iter,
                                          candidates, num_users, num_items,
                                          devices)
                animator.add(epoch + 1, (hit_rate, auc))
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test hit rate {float(hit_rate):.3f}, test AUC {float(auc):.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')
```

Most betölthetjük a MovieLens 100k adathalmazt, és betaníthatjuk a modellt. Mivel a MovieLens adathalmazban csak értékelések szerepelnek, némi pontosságveszteséggel binarizáljuk ezeket az értékeléseket nullákra és egyesekre. Ha egy felhasználó értékelt egy elemet, az implicit visszajelzést egynek tekintjük, egyébként nullának. Egy elem értékelésének művelete implicit visszajelzés nyújtásának tekinthető. Az adathalmazt `seq-aware` módban osztjuk fel, ahol a felhasználók legutóbb interakcióba lépett elemeit tesztelésre tartjuk fenn.

```{.python .input  n=11}
#@tab mxnet
batch_size = 1024
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items,
                                              'seq-aware')
users_train, items_train, ratings_train, candidates = d2l.load_data_ml100k(
    train_data, num_users, num_items, feedback="implicit")
users_test, items_test, ratings_test, test_iter = d2l.load_data_ml100k(
    test_data, num_users, num_items, feedback="implicit")
train_iter = gluon.data.DataLoader(
    PRDataset(users_train, items_train, candidates, num_items ), batch_size,
    True, last_batch="rollover", num_workers=d2l.get_dataloader_workers())
```

Ezután létrehozzuk és inicializáljuk a modellt. Három rétegű, állandó 10-es rejtett méretű MLP-t alkalmazunk.

```{.python .input  n=8}
#@tab mxnet
devices = d2l.try_all_gpus()
net = NeuMF(10, num_users, num_items, nums_hiddens=[10, 10, 10])
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
```

Az alábbi kód tanítja be a modellt.

```{.python .input  n=12}
#@tab mxnet
lr, num_epochs, wd, optimizer = 0.01, 10, 1e-5, 'adam'
loss = d2l.BPRLoss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
train_ranking(net, train_iter, test_iter, loss, trainer, None, num_users,
              num_items, num_epochs, devices, evaluate_ranking, candidates)
```

## Összefoglalás

* Nemlinearitás hozzáadása a mátrixfaktorizációs modellhez előnyös a modell képességének és hatékonyságának javítása szempontjából.
* A NeuMF a mátrixfaktorizáció és egy MLP kombinációja. Az MLP a felhasználói és elem embeddingjek összefűzését veszi bemenetként.

## Gyakorlatok

* Változtasd meg a látens faktorok méretét! Hogyan befolyásolja a látens faktorok mérete a modell teljesítményét?
* Változtasd meg az MLP architektúráját (például a rétegek számát és az egyes rétegek neuronjainak számát), és vizsgáld meg a teljesítményre gyakorolt hatását!
* Próbálj ki különböző optimalizálókat, tanulási rátákat és súlycsökkenési rátákat!
* Próbáld meg az előző szakaszban definiált csuklóveszteséget (hinge loss) alkalmazni a modell optimalizálásához!

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/403)
:end_tab:
