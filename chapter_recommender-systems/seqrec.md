# Sorrendtudatos ajánlórendszerek

Az előző szakaszokban az ajánlási feladatot mátrixkitöltési problémaként kezeltük, a felhasználók rövid távú viselkedésének figyelembevétele nélkül. Ebben a szakaszban egy olyan ajánlási modellt mutatunk be, amely figyelembe veszi a felhasználók időrendbe rendezett interakciós naplóit. Ez egy sorrendtudatos ajánló :cite:`Quadrana.Cremonesi.Jannach.2018`, ahol a bemenet a múltbeli felhasználói tevékenységek rendezett, gyakran időbélyeggel ellátott listája. Számos friss szakirodalmi munka igazolta, hogy az ilyen információk beépítése hasznos a felhasználók időbeli viselkedési mintáinak modellezéséhez és érdeklődési irányuk változásának feltárásához.

A bemutatandó modell, a Caser :cite:`Tang.Wang.2018` — a konvolúciós szekvencia-beágyazásos ajánlómodell rövidítése —, konvolúciós neurális hálózatokat alkalmaz a felhasználók közelmúltbeli tevékenységeinek dinamikus mintáiból fakadó hatások megragadásához. A Caser fő összetevője egy vízszintes konvolúciós hálózat és egy függőleges konvolúciós hálózat, amelyek célja az unió-szintű, illetve pont-szintű szekvenciaminták feltárása. A pont-szintű minta az előzmények egyes elemeinek a célelemre gyakorolt hatását jelzi, míg az unió-szintű minta több korábbi tevékenység együttes hatását fejezi ki a következő célra. Például ha valaki egyszerre vesz tejet és vajat, nagyobb valószínűséggel vásárol lisztet is, mintha csak az egyiket vette volna. Emellett a felhasználók általános érdeklődése, azaz hosszú távú preferenciái szintén modellezésre kerülnek az utolsó teljesen összekötött rétegekben, így a felhasználói érdeklődés átfogóbb képe rajzolódik ki. A modell részleteit az alábbiakban ismertetjük.

## A modell architektúrája

Sorrendtudatos ajánlórendszerben minden felhasználóhoz az elemkészlet valamely elemeinek egy sorozata tartozik. Jelöljük $S^u = (S_1^u, ... S_{|S_u|}^u)$ a rendezett sorozatot. A Caser célja olyan elem ajánlása, amely figyelembe veszi a felhasználó általános ízlését és rövid távú szándékát. Tegyük fel, hogy az előző $L$ elemet vesszük figyelembe; ekkor a $t$ időlépéshez tartozó korábbi interakciókat reprezentáló beágyazási mátrix a következőképpen konstruálható:

$$
\mathbf{E}^{(u, t)} = [ \mathbf{q}_{S_{t-L}^u} , ..., \mathbf{q}_{S_{t-2}^u}, \mathbf{q}_{S_{t-1}^u} ]^\top,
$$

ahol $\mathbf{Q} \in \mathbb{R}^{n \times k}$ az elembeágyazásokat, $\mathbf{q}_i$ a $i^\textrm{th}$ sort jelöli. Az $\mathbf{E}^{(u, t)} \in \mathbb{R}^{L \times k}$ mátrix a $u$ felhasználó átmeneti érdeklődésének inferenciájára használható a $t$ időlépésben. A bemeneti $\mathbf{E}^{(u, t)}$ mátrixot képként értelmezhetjük, amely a két rákövetkező konvolúciós összetevő bemenete.

A vízszintes konvolúciós rétegnek $d$ vízszintes szűrője van: $\mathbf{F}^j \in \mathbb{R}^{h \times k}, 1 \leq j \leq d, h = \{1, ..., L\}$, a függőleges konvolúciós rétegnek pedig $d'$ függőleges szűrője: $\mathbf{G}^j \in \mathbb{R}^{ L \times 1}, 1 \leq j \leq d'$. Konvolúciós és összevonási műveletek sorozata után két kimenetet kapunk:

$$
\mathbf{o} = \textrm{HConv}(\mathbf{E}^{(u, t)}, \mathbf{F}) \\
\mathbf{o}'= \textrm{VConv}(\mathbf{E}^{(u, t)}, \mathbf{G}) ,
$$

ahol $\mathbf{o} \in \mathbb{R}^d$ a vízszintes konvolúciós hálózat kimenete, $\mathbf{o}' \in \mathbb{R}^{kd'}$ a függőleges konvolúciós hálózat kimenete. Az egyszerűség kedvéért a konvolúciós és összevonási műveletek részleteit elhagyjuk. Ezeket összefűzzük, majd egy teljesen összekötött neurális hálózati rétegbe táplálva magasabb szintű reprezentációkat kapunk.

$$
\mathbf{z} = \phi(\mathbf{W}[\mathbf{o}, \mathbf{o}']^\top + \mathbf{b}),
$$

ahol $\mathbf{W} \in \mathbb{R}^{k \times (d + kd')}$ a súlymátrix és $\mathbf{b} \in \mathbb{R}^k$ az eltolásvektor. A kapott $\mathbf{z} \in \mathbb{R}^k$ vektor a felhasználó rövid távú szándékának reprezentációja.

Végül az előrejelzési függvény a felhasználó rövid távú és általános ízlését egyesíti:

$$
\hat{y}_{uit} = \mathbf{v}_i \cdot [\mathbf{z}, \mathbf{p}_u]^\top + \mathbf{b}'_i,
$$

ahol $\mathbf{V} \in \mathbb{R}^{n \times 2k}$ egy másik elembeágyazási mátrix. $\mathbf{b}' \in \mathbb{R}^n$ az elemspecifikus eltolásvektor. $\mathbf{P} \in \mathbb{R}^{m \times k}$ a felhasználók általános ízlését reprezentáló felhasználói beágyazási mátrix. $\mathbf{p}_u \in \mathbb{R}^{ k}$ a $P$ mátrix $u^\textrm{th}$ sora, $\mathbf{v}_i \in \mathbb{R}^{2k}$ a $\mathbf{V}$ mátrix $i^\textrm{th}$ sora.

A modell BPR- vagy Hinge-veszteséggel tanítható. A Caser architektúráját az alábbi ábra szemlélteti:

![A Caser modell illusztrációja](../img/rec-caser.svg)

Először importáljuk a szükséges könyvtárakat.

```{.python .input  n=3}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import random

npx.set_np()
```

## A modell implementációja
Az alábbi kód implementálja a Caser modellt. Egy függőleges konvolúciós rétegből, egy vízszintes konvolúciós rétegből és egy teljesen összekötött rétegből áll.

```{.python .input  n=4}
#@tab mxnet
class Caser(nn.Block):
    def __init__(self, num_factors, num_users, num_items, L=5, d=16,
                 d_prime=4, drop_ratio=0.05, **kwargs):
        super(Caser, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.d_prime, self.d = d_prime, d
        # Függőleges konvolúciós réteg
        self.conv_v = nn.Conv2D(d_prime, (L, 1), in_channels=1)
        # Vízszintes konvolúciós réteg
        h = [i + 1 for i in range(L)]
        self.conv_h, self.max_pool = nn.Sequential(), nn.Sequential()
        for i in h:
            self.conv_h.add(nn.Conv2D(d, (i, num_factors), in_channels=1))
            self.max_pool.add(nn.MaxPool1D(L - i + 1))
        # Teljesen összekötött réteg
        self.fc1_dim_v, self.fc1_dim_h = d_prime * num_factors, d * len(h)
        self.fc = nn.Dense(in_units=d_prime * num_factors + d * L,
                           activation='relu', units=num_factors)
        self.Q_prime = nn.Embedding(num_items, num_factors * 2)
        self.b = nn.Embedding(num_items, 1)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, user_id, seq, item_id):
        item_embs = np.expand_dims(self.Q(seq), 1)
        user_emb = self.P(user_id)
        out, out_h, out_v, out_hs = None, None, None, []
        if self.d_prime:
            out_v = self.conv_v(item_embs)
            out_v = out_v.reshape(out_v.shape[0], self.fc1_dim_v)
        if self.d:
            for conv, maxp in zip(self.conv_h, self.max_pool):
                conv_out = np.squeeze(npx.relu(conv(item_embs)), axis=3)
                t = maxp(conv_out)
                pool_out = np.squeeze(t, axis=2)
                out_hs.append(pool_out)
            out_h = np.concatenate(out_hs, axis=1)
        out = np.concatenate([out_v, out_h], axis=1)
        z = self.fc(self.dropout(out))
        x = np.concatenate([z, user_emb], axis=1)
        q_prime_i = np.squeeze(self.Q_prime(item_id))
        b = np.squeeze(self.b(item_id))
        res = (x * q_prime_i).sum(1) + b
        return res
```

## Szekvenciális adathalmaz negatív mintavételezéssel
A szekvenciális interakciós adatok feldolgozásához újra kell implementálnunk a `Dataset` osztályt. Az alábbi kód egy `SeqDataset` nevű új adathalmaz-osztályt hoz létre. Minden mintában a modell a felhasználó azonosítóját, az előző $L$ interakcióban szereplő elemet mint sorozatot, és a következő interakcióban érintett elemet mint célelemet adja ki. Az alábbi ábra egy felhasználóra vonatkozó adatbetöltési folyamatot szemlélteti. Tegyük fel, hogy ez a felhasználó 9 filmet kedvelt; ezeket a filmeket időrendbe rendezzük. A legutóbbi filmet tesztelési elemként félretesszük. A fennmaradó nyolc filmből három tanítási mintát kaphatunk, ahol minden minta öt ($L=5$) filmet tartalmazó sorozatból és az azt követő célelemből áll. A testreszabott adathalmazba negatív minták is bekerülnek.

![Az adatgenerálási folyamat illusztrációja](../img/rec-seq-data.svg)

```{.python .input  n=5}
#@tab mxnet
class SeqDataset(gluon.data.Dataset):
    def __init__(self, user_ids, item_ids, L, num_users, num_items,
                 candidates):
        user_ids, item_ids = np.array(user_ids), np.array(item_ids)
        sort_idx = np.array(sorted(range(len(user_ids)),
                                   key=lambda k: user_ids[k]))
        u_ids, i_ids = user_ids[sort_idx], item_ids[sort_idx]
        temp, u_ids, self.cand = {}, u_ids.asnumpy(), candidates
        self.all_items = set([i for i in range(num_items)])
        [temp.setdefault(u_ids[i], []).append(i) for i, _ in enumerate(u_ids)]
        temp = sorted(temp.items(), key=lambda x: x[0])
        u_ids = np.array([i[0] for i in temp])
        idx = np.array([i[1][0] for i in temp])
        self.ns = ns = int(sum([c - L if c >= L + 1 else 1 for c
                                in np.array([len(i[1]) for i in temp])]))
        self.seq_items = np.zeros((ns, L))
        self.seq_users = np.zeros(ns, dtype='int32')
        self.seq_tgt = np.zeros((ns, 1))
        self.test_seq = np.zeros((num_users, L))
        test_users, _uid = np.empty(num_users), None
        for i, (uid, i_seq) in enumerate(self._seq(u_ids, i_ids, idx, L + 1)):
            if uid != _uid:
                self.test_seq[uid][:] = i_seq[-L:]
                test_users[uid], _uid = uid, uid
            self.seq_tgt[i][:] = i_seq[-1:]
            self.seq_items[i][:], self.seq_users[i] = i_seq[:L], uid

    def _win(self, tensor, window_size, step_size=1):
        if len(tensor) - window_size >= 0:
            for i in range(len(tensor), 0, - step_size):
                if i - window_size >= 0:
                    yield tensor[i - window_size:i]
                else:
                    break
        else:
            yield tensor

    def _seq(self, u_ids, i_ids, idx, max_len):
        for i in range(len(idx)):
            stop_idx = None if i >= len(idx) - 1 else int(idx[i + 1])
            for s in self._win(i_ids[int(idx[i]):stop_idx], max_len):
                yield (int(u_ids[i]), s)

    def __len__(self):
        return self.ns

    def __getitem__(self, idx):
        neg = list(self.all_items - set(self.cand[int(self.seq_users[idx])]))
        i = random.randint(0, len(neg) - 1)
        return (self.seq_users[idx], self.seq_items[idx], self.seq_tgt[idx],
                neg[i])
```

## A MovieLens 100K adathalmaz betöltése

Ezt követően beolvassuk és szekvenciatudatos módban szétbontjuk a MovieLens 100K adathalmazt, majd a fent implementált szekvenciális adatbetöltővel betöltjük a tanítási adatokat.

```{.python .input  n=6}
#@tab mxnet
TARGET_NUM, L, batch_size = 1, 5, 4096
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items,
                                              'seq-aware')
users_train, items_train, ratings_train, candidates = d2l.load_data_ml100k(
    train_data, num_users, num_items, feedback="implicit")
users_test, items_test, ratings_test, test_iter = d2l.load_data_ml100k(
    test_data, num_users, num_items, feedback="implicit")
train_seq_data = SeqDataset(users_train, items_train, L, num_users,
                            num_items, candidates)
train_iter = gluon.data.DataLoader(train_seq_data, batch_size, True,
                                   last_batch="rollover",
                                   num_workers=d2l.get_dataloader_workers())
test_seq_iter = train_seq_data.test_seq
train_seq_data[0]
```

A fent látható tanítási adatstruktúra első eleme a felhasználó azonosítója, a következő lista az adott felhasználó által utoljára kedvelt öt elemet tartalmazza, az utolsó elem pedig az, amelyet a felhasználó az öt elem után kedvelt meg.

## A modell tanítása
Most tanítsuk be a modellt. Az eredmények összehasonlíthatósága érdekében az előző szakaszban szereplő NeuMF-fel azonos beállításokat használunk, beleértve a tanulási rátát, az optimalizálót és a $k$ értékét.

```{.python .input  n=7}
#@tab mxnet
devices = d2l.try_all_gpus()
net = Caser(10, num_users, num_items, L)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.04, 8, 1e-5, 'adam'
loss = d2l.BPRLoss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})

# A futtatás > 1 órát vesz igénybe (függőben lévő MXNet-javítás)
# d2l.train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter, num_users, num_items, num_epochs, devices, d2l.evaluate_ranking, candidates, eval_step=1)
```

## Összefoglalás
* A felhasználó rövid és hosszú távú érdeklődésének modellezése hatékonyabbá teszi a következő preferált elem előrejelzését.
* A konvolúciós neurális hálózatok alkalmasak arra, hogy a szekvenciális interakciókból megragadják a felhasználók rövid távú érdeklődését.

## Gyakorlatok

* Végezz ablációs vizsgálatot a vízszintes, illetve a függőleges konvolúciós hálózat eltávolításával — melyik összetevő bizonyul fontosabbnak?
* Változtasd a $L$ hiperparamétert! Magasabb pontosságot eredményez-e a hosszabb előzményi interakció?
* A fent bemutatott sorrendtudatos ajánlási feladaton kívül létezik egy másik, munkamenet-alapú ajánlásnak nevezett sorrendtudatos ajánlási feladattípus is :cite:`Hidasi.Karatzoglou.Baltrunas.ea.2015`. Meg tudod magyarázni a két feladat közötti különbségeket?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/404)
:end_tab:
