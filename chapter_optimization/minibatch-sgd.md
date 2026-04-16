# Minibatch sztochasztikus gradienscsökkenés
:label:`sec_minibatch_sgd`

Eddig két szélsőséget tapasztaltunk a gradiens alapú tanulás megközelítésében: a :numref:`sec_gd` szakasz a teljes adathalmazt használja a gradiensek kiszámításához és a paraméterek frissítéséhez, egyszerre egy átmenetben. Ezzel szemben a :numref:`sec_sgd` szakasz egyszerre egy tanítási példát dolgoz fel a haladás érdekében.
Mindkettőnek megvannak a maga hátrányai.
A gradienscsökkenés nem különösebben *adathatékony*, ha az adatok nagyon hasonlóak.
A sztochasztikus gradienscsökkenés nem különösebben *számítási szempontból hatékony*, mivel a CPU-k és GPU-k nem tudják teljes mértékben kihasználni a vektorizálás erejét.
Ez arra utal, hogy valahol a kettő között kellene lennie egy megoldásnak –
és valójában ezt alkalmaztuk az eddig tárgyalt példákban.

## Vektorizálás és gyorsítótárak

A minibatchek alkalmazásának középpontjában a számítási hatékonyság áll. Ez legjobban akkor érthető meg, ha párhuzamos futtatást több GPU-n és több szerveren vizsgálunk. Ebben az esetben legalább egy képet kell küldeni minden egyes GPU-ra. Szerverenként 8 GPU-val és 16 szerverrel már legalább 128-as minibatch-mérethez jutunk.

A dolgok kissé árnyaltabbak egyetlen GPU vagy akár CPU esetén. Ezeknek az eszközöknek többféle típusú memóriájuk van, sokszor többféle számítási egységük, és különböző sávszélesség-korlátaik vannak köztük.
Például egy CPU-nak van egy kis regiszterkészlete, majd L1, L2, és néhány esetben még L3 gyorsítótára (amelyet a különböző processzormagok megosztanak).
Ezek a gyorsítótárak növekvő méretűek és késleltetésűek (és egyidejűleg csökkenő sávszélességűek).
Elegendő annyit mondani, hogy a processzor sokkal több műveletet tud végrehajtani, mint amennyit a főmemória-interfész képes biztosítani.

Először is, egy 16 maggal és AVX-512 vektorizálással rendelkező 2GHz-es CPU másodpercenként legfeljebb $2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$ bájtot tud feldolgozni. A GPU-k képessége ezt az értéket könnyen 100-szorosával haladja meg. Másrészt egy középkategóriás szerverprocesszornak nem lehet sokkal több mint 100 GB/s sávszélessége, azaz kevesebb mint egytized annyi, amennyire a processzor ellátásához szükség lenne. Ráadásul nem minden memória-hozzáférés egyforma: a memória-interfészek általában 64 bit vagy annál szélesebb sávszélességűek (pl. GPU-kon akár 384 bit), így egyetlen bájt olvasása sokkal szélesebb hozzáférés költségével jár.

Másodszor, az első hozzáférésnél jelentős az overhead, míg a szekvenciális hozzáférés viszonylag olcsó (ezt gyakran burst read-nek nevezik). Sok más szempontot is figyelembe kell venni, például a gyorsítótárazást több foglalattal, chipletekkel és egyéb struktúrákkal rendelkező rendszerekben.
A mélyebb tárgyalásért lásd ezt a [Wikipédia-cikket](https://en.wikipedia.org/wiki/Cache_hierarchy).

Ezeknek a korlátoknak az enyhítésének módja a CPU-gyorsítótárak hierarchiájának alkalmazása, amelyek valóban elég gyorsak ahhoz, hogy adatokat biztosítsanak a processzor számára. Ez a *legfőbb* mozgatórugója a kötegelésnek a mélytanulásban. Az egyszerűség kedvéért tekintsük a mátrix-mátrix szorzást, mondjuk $\mathbf{A} = \mathbf{B}\mathbf{C}$. Az $\mathbf{A}$ kiszámítására több lehetőségünk van. Például megpróbálhatjuk a következőket:

1. Kiszámíthatjuk $\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}$-t, azaz skaláris szorzatok segítségével elemenként kiszámíthatjuk.
1. Kiszámíthatjuk $\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}$-t, azaz oszloponként kiszámíthatjuk. Hasonlóképpen $\mathbf{A}$-t soronként, $\mathbf{A}_{i,:}$ formájában is kiszámíthatjuk.
1. Egyszerűen kiszámíthatjuk $\mathbf{A} = \mathbf{B} \mathbf{C}$-t.
1. Feldarabolhatjuk $\mathbf{B}$-t és $\mathbf{C}$-t kisebb blokk-mátrixokra, és $\mathbf{A}$-t blokkonként számíthatjuk ki.

Az első lehetőség követése esetén minden $\mathbf{A}_{ij}$ elem kiszámításakor egy sor- és egy oszlopvektort kell a CPU-ba másolni. Sőt, mivel a mátrix elemei szekvenciálisan vannak elrendezve, a memóriából való olvasásnál a két vektor egyikéhez sok nem összefüggő helyet kell elérni. A második lehetőség sokkal kedvezőbb. Ebben képesek vagyunk a $\mathbf{C}_{:,j}$ oszlopvektort a CPU gyorsítótárában tartani, miközben $\mathbf{B}$-n áthaladunk. Ez felezi a memória-sávszélességi követelményt, ennek megfelelően gyorsabb hozzáféréssel. Természetesen a 3. lehetőség a legkívánatosabb. Sajnos a legtöbb mátrix nem fér el teljesen a gyorsítótárban (elvégre erről van szó). A 4. lehetőség azonban egy gyakorlatilag hasznos alternatívát kínál: a mátrix blokkjait a gyorsítótárba mozgathatjuk, és helyben szorozhatjuk meg őket. Az optimalizált könyvtárak gondoskodnak erről helyettünk. Nézzük meg, milyen hatékonyak ezek a műveletek a gyakorlatban.

A számítási hatékonyságon túl a Python és maga a mélytanulási keretrendszer által bevezetett overhead is jelentős. Felidézve: minden alkalommal, amikor parancsot hajtunk végre, a Python interpreter parancsot küld az MXNet motornak, amelynek be kell illeszteni a számítási gráfba, és ütemezés közben kezelni kell. Ez az overhead meglehetősen káros lehet. Röviden: erősen ajánlott a vektorizálás (és a mátrixok) alkalmazása, valahányszor lehetséges.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
import time
npx.set_np()

A = np.zeros((256, 256))
B = np.random.normal(0, 1, (256, 256))
C = np.random.normal(0, 1, (256, 256))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import time
import torch
from torch import nn

A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
import time

A = tf.Variable(d2l.zeros((256, 256)))
B = tf.Variable(d2l.normal([256, 256], 0, 1))
C = tf.Variable(d2l.normal([256, 256], 0, 1))
```

Mivel a könyv hátralévő részében gyakran mérjük majd a futási időt, definiáljunk egy időzítőt.

```{.python .input}
#@tab all
class Timer:  #@save
    """Több futási idő rögzítése."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Az időzítő elindítása."""
        self.tik = time.time()

    def stop(self):
        """Az időzítő leállítása és az idő rögzítése egy listában."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Az átlagos idő visszaadása."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Az idők összegének visszaadása."""
        return sum(self.times)

    def cumsum(self):
        """A kumulált idő visszaadása."""
        return np.array(self.times).cumsum().tolist()

timer = Timer()
```

Az elemenként történő értékadás egyszerűen végigiterál $\mathbf{B}$ összes során és $\mathbf{C}$ összes oszlopán, hogy értéket rendeljen $\mathbf{A}$-hoz.

```{.python .input}
#@tab mxnet
# Az A = BC kiszámítása elemenként
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = np.dot(B[i, :], C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Az A = BC kiszámítása elemenként
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
# Az A = BC kiszámítása elemenként
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j].assign(tf.tensordot(B[i, :], C[:, j], axes=1))
timer.stop()
```

Gyorsabb stratégia az oszlopokra bontott értékadás.

```{.python .input}
#@tab mxnet
# Az A = BC kiszámítása oszloponként
timer.start()
for j in range(256):
    A[:, j] = np.dot(B, C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Az A = BC kiszámítása oszloponként
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(256):
    A[:, j].assign(tf.tensordot(B, C[:, j], axes=1))
timer.stop()
```

Végül a leghatékonyabb módszer az egész művelet egyetlen blokkban való végrehajtása. 
Megjegyezzük, hogy bármely két $\mathbf{B} \in \mathbb{R}^{m \times n}$ és $\mathbf{C} \in \mathbb{R}^{n \times p}$ mátrix szorzása körülbelül $2mnp$ lebegőpontos műveletet igényel,
ha a skaláris szorzást és az összeadást külön műveletnek számítjuk (a gyakorlatban fúzálva vannak).
Ezért két $256 \times 256$-os mátrix szorzása
$0.03$ milliárd lebegőpontos műveletet igényel.
Nézzük meg, milyen a műveletek megfelelő sebessége.

```{.python .input}
#@tab mxnet
# Az A = BC kiszámítása egyszerre
timer.start()
A = np.dot(B, C)
A.wait_to_read()
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab pytorch
# Az A = BC kiszámítása egyszerre
timer.start()
A = torch.mm(B, C)
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
A.assign(tf.tensordot(B, C, axes=1))
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

## Minibatchek

:label:`sec_minibatches`

Eddig természetesnek vettük, hogy az adatokat *minibatchenként* olvassuk be, nem egyenként, a paraméterek frissítéséhez. Most rövid magyarázatot adunk erre. Az egyes megfigyelések feldolgozása sok egyszeres mátrix-vektor (vagy akár vektor-vektor) szorzást igényel, ami meglehetősen drága, és jelentős overheadet okoz a mögöttes mélytanulási keretrendszer részéről. Ez egyaránt vonatkozik a hálózat kiértékelésére adatokon (amelyet gyakran következtetésnek neveznek) és a gradiensek kiszámítására a paraméterek frissítéséhez. Vagyis ez érvényes minden alkalommal, amikor végrehajtjuk a $\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$ frissítést, ahol

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w})$$

Ennek a műveletnek a *számítási* hatékonyságát növelhetjük azzal, hogy egyszerre megfigyelések minibatchére alkalmazzuk. Vagyis egyetlen megfigyelés $\mathbf{g}_t$ gradienssét egy kis batch feletti gradiensre cseréljük:

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w})$$

Vizsgáljuk meg, hogy ez mit tesz $\mathbf{g}_t$ statisztikai tulajdonságaival: mivel $\mathbf{x}_t$ és a $\mathcal{B}_t$ minibatch összes eleme egyenletesen véletlenszerűen kerül mintavételezésre a tanítóhalmazból, a gradiens várható értéke változatlan marad. A variancia viszont jelentősen csökken. Mivel a minibatch gradiens $b \stackrel{\textrm{def}}{=} |\mathcal{B}_t|$ független gradiens átlagából áll, szórása $b^{-\frac{1}{2}}$ faktorral csökken. Ez önmagában is jó dolog, mivel azt jelenti, hogy a frissítések megbízhatóbban igazodnak a teljes gradienshez.

Naivan ez arra utalna, hogy nagy $\mathcal{B}_t$ minibatch választása általánosan kívánatos lenne. Sajnos egy bizonyos ponton túl a szórás további csökkentése elhanyagolható a számítási költség lineáris növekedéséhez képest. A gyakorlatban olyan minibatch-méretet választunk, amely elég nagy a jó számítási hatékonysághoz, miközben elfér a GPU memóriájában. A megtakarítások szemléltetéséhez nézzünk meg egy kódpéldát. Ebben ugyanazt a mátrix-mátrix szorzást végezzük el, de ezúttal egyszerre 64 oszlopból álló „minibatchekre" bontva.

```{.python .input}
#@tab mxnet
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = np.dot(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab pytorch
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64].assign(tf.tensordot(B, C[:, j:j+64], axes=1))
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

Ahogy látható, a minibatch-en végzett számítás lényegében ugyanolyan hatékony, mint a teljes mátrixon. Egy megjegyzés azonban szükséges. A :numref:`sec_batch_norm` szakaszban egy olyan regularizálási típust alkalmaztunk, amely erősen függött a minibatch-en belüli varianciától. Ahogy növeljük az utóbbit, a variancia csökken, és ezzel együtt a batch normalizálásból eredő zaj-injekció előnye is. A megfelelő tagok átskálázásáról és kiszámításáról részletekért lásd például :citet:`Ioffe.2017`.

## Az adathalmaz beolvasása

Nézzük meg, hogyan generálhatók hatékonyan minibatchek adatokból. A következőkben egy NASA által kifejlesztett adathalmazt alkalmazunk, amelyet különböző repülőgépek [szárnyzajának](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) tesztelésére hoztak létre, az optimalizálási algoritmusok összehasonlítása céljából. Kényelmesség szempontjából csak az első $1500$ példát használjuk. Az adatokat fehérítik az előfeldolgozás során, vagyis eltávolítják az átlagot, és a varianciát $1$-re skálázzák koordinátánként.

```{.python .input}
#@tab mxnet
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab pytorch
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab tensorflow
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

## Implementálás alapoktól

Idézzük fel a minibatch sztochasztikus gradienscsökkenés implementációját a :numref:`sec_linear_scratch` szakaszból. Az alábbiakban egy kissé általánosabb implementációt nyújtunk. Kényelmi szempontból ugyanolyan hívási aláírása van, mint a fejezet későbbi részében bemutatott többi optimalizálási algoritmusnak. Konkrétan hozzáadjuk az `states` állapot bemenetet, és a hiperparamétert egy `hyperparams` szótárba helyezzük. Emellett a tanítási függvényben minden minibatch példa veszteségének átlagát vesszük, így az optimalizálási algoritmusban lévő gradienst nem kell elosztani a batch méretével.

```{.python .input}
#@tab mxnet
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```

```{.python .input}
#@tab pytorch
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, states, hyperparams):
    for param, grad in zip(params, grads):
        param.assign_sub(hyperparams['lr']*grad)
```

Ezután implementálunk egy általános tanítási függvényt a fejezet későbbi részében bemutatott többi optimalizálási algoritmus egyszerűsített alkalmazásához. Egy lineáris regressziós modellt inicializál, amelyet minibatch sztochasztikus gradienscsökkenéssel és a később bemutatott algoritmusokkal lehet betanítani.

```{.python .input}
#@tab mxnet
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Inicializálás
    w = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Tanítás
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab pytorch
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Inicializálás
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Tanítás
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Inicializálás
    w = tf.Variable(tf.random.normal(shape=(feature_dim, 1),
                                   mean=0, stddev=0.01),trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    # Tanítás
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
          with tf.GradientTape() as g:
            l = tf.math.reduce_mean(loss(net(X), y))

          dw, db = g.gradient(l, [w, b])
          trainer_fn([w, b], [dw, db], states, hyperparams)
          n += X.shape[0]
          if n % 200 == 0:
              timer.stop()
              p = n/X.shape[0]
              q = p/tf.data.experimental.cardinality(data_iter).numpy()
              r = (d2l.evaluate_loss(net, data_iter, loss),)
              animator.add(q, r)
              timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

Nézzük meg, hogyan halad az optimalizálás a batch gradienscsökkenés esetén. Ez úgy érhető el, hogy a minibatch-méretet 1500-ra állítjuk (azaz a példák teljes számára). Ennek eredményeként a modell paraméterei epocsonként csak egyszer frissülnek. Alig tapasztalható haladás. Valójában 6 lépés után a haladás megáll.

```{.python .input}
#@tab all
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

Ha a batch-méret egyenlő 1-gyel, sztochasztikus gradienscsökkenést alkalmazunk az optimalizáláshoz. Az implementáció egyszerűsége érdekében állandó (bár kis) tanulási sebességet választottunk. A sztochasztikus gradienscsökkenésben a modell paraméterei minden egyes feldolgozott példa után frissülnek. Esetünkben ez epocsonként 1500 frissítést jelent. Ahogy látható, a célfüggvény értékének csökkenése lassul az első epoc után. Bár mindkét eljárás 1500 példát dolgozott fel egy epoc alatt, a sztochasztikus gradienscsökkenés kísérletünkben több időt vesz igénybe, mint a gradienscsökkenés. Ennek oka, hogy a sztochasztikus gradienscsökkenés gyakrabban frissítette a paramétereket, és az egyes megfigyelések egyenkénti feldolgozása kevésbé hatékony.

```{.python .input}
#@tab all
sgd_res = train_sgd(0.005, 1)
```

Végül, ha a batch-méret egyenlő 100-zal, minibatch sztochasztikus gradienscsökkenést alkalmazunk az optimalizáláshoz. Az epocsonkénti szükséges idő rövidebb, mint a sztochasztikus gradienscsökkenés esetén, és rövidebb, mint a batch gradienscsökkenés esetén is.

```{.python .input}
#@tab all
mini1_res = train_sgd(.4, 100)
```

A batch-méret 10-re csökkentése esetén minden epoc ideje növekszik, mivel az egyes batchek munkaterhelése kevésbé hatékonyan hajtható végre.

```{.python .input}
#@tab all
mini2_res = train_sgd(.05, 10)
```

Most összehasonlíthatjuk az idő és a veszteség viszonyát az előző négy kísérletben. Ahogy látható, bár a sztochasztikus gradienscsökkenés gyorsabban konvergál, mint a GD a feldolgozott példák számát tekintve, több időt vesz igénybe ugyanolyan veszteség eléréséhez, mint a GD, mivel a gradiens példánkénti kiszámítása nem olyan hatékony. A minibatch sztochasztikus gradienscsökkenés egyensúlyt teremt a konvergenciasebesség és a számítási hatékonyság között. A 10-es minibatch-méret hatékonyabb, mint a sztochasztikus gradienscsökkenés; a 100-as minibatch-méret még a GD-t is felülmúlja futási idő tekintetében.

```{.python .input}
#@tab all
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
```

## Tömör implementáció

A Gluon-ban a `Trainer` osztályt használhatjuk optimalizálási algoritmusok meghívásához. Ezt egy általános tanítási függvény implementálásához alkalmazzuk. Ezt az aktuális fejezet egészében alkalmazni fogjuk.

```{.python .input}
#@tab mxnet
#@save
def train_concise_ch11(tr_name, hyperparams, data_iter, num_epochs=2):
    # Inicializálás
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), tr_name, hyperparams)
    loss = gluon.loss.L2Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(X.shape[0])
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

```{.python .input}
#@tab pytorch
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # Inicializálás
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(module):
        if type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # Az `MSELoss` az 1/2 szorzó nélkül számítja a négyzetes hibát
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

```{.python .input}
#@tab tensorflow
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2):
    # Inicializálás
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    optimizer = trainer_fn(**hyperparams)
    loss = tf.keras.losses.MeanSquaredError()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)
                params = net.trainable_variables
                grads = g.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                p = n/X.shape[0]
                q = p/tf.data.experimental.cardinality(data_iter).numpy()
                # A `MeanSquaredError` az 1/2 szorzó nélkül számítja
                # a négyzetes hibát
                r = (d2l.evaluate_loss(net, data_iter, loss) / 2,)
                animator.add(q, r)
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

A Gluon alkalmazásával az utolsó kísérlet megismétlése azonos viselkedést mutat.

```{.python .input}
#@tab mxnet
data_iter, _ = get_data_ch11(10)
train_concise_ch11('sgd', {'learning_rate': 0.05}, data_iter)
```

```{.python .input}
#@tab pytorch
data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
data_iter, _ = get_data_ch11(10)
trainer = tf.keras.optimizers.SGD
train_concise_ch11(trainer, {'learning_rate': 0.05}, data_iter)
```

## Összefoglalás

* A vektorizálás hatékonyabbá teszi a kódot a mélytanulási keretrendszerből eredő overhead csökkentésével, valamint a CPU-kon és GPU-kon jobb memória-lokalitás és gyorsítótárazás révén.
* Kompromisszum áll fenn a sztochasztikus gradienscsökkenésből eredő statisztikai hatékonyság és a nagy adatbatchek egyszeri feldolgozásából eredő számítási hatékonyság között.
* A minibatch sztochasztikus gradienscsökkenés mindkét világ előnyeit kínálja: számítási és statisztikai hatékonyságot.
* A minibatch sztochasztikus gradienscsökkenésben a tanítóadatok véletlenszerű permutációjával előállított adatbatcheket dolgozunk fel (azaz minden megfigyelést epocsonként csak egyszer dolgozunk fel, bár véletlenszerű sorrendben).
* Tanítás során célszerű csökkenteni a tanulási sebességet.
* Általánosságban a minibatch sztochasztikus gradienscsökkenés gyorsabb, mint a sztochasztikus gradienscsökkenés és a gradienscsökkenés a kisebb kockázathoz való konvergenciában, ha falióra-időben mérjük.

## Gyakorló feladatok

1. Módosítsd a batch-méretet és a tanulási sebességet, és figyeld meg a célfüggvény értékének csökkenési ütemét és az egyes epochokban eltelt időt.
1. Olvasd el az MXNet dokumentációját, és a `Trainer` osztály `set_learning_rate` függvényét alkalmazva csökkentsd a minibatch sztochasztikus gradienscsökkenés tanulási sebességét az előző értékének 1/10-ére minden epoc után.
1. Hasonlítsd össze a minibatch sztochasztikus gradienscsökkenést egy olyan változattal, amely *visszatevéssel mintavételez* a tanítóhalmazból. Mi történik?
1. Egy gonosz szellem megkettőzi az adathalmazát anélkül, hogy szólna (azaz minden megfigyelés kétszer szerepel, és az adathalmaz kétszer akkora lesz, de senki sem mondta el). Hogyan változik a sztochasztikus gradienscsökkenés, a minibatch sztochasztikus gradienscsökkenés és a gradienscsökkenés viselkedése?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/353)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1068)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1069)
:end_tab:
