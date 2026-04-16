# Momentum
:label:`sec_momentum`

A :numref:`sec_sgd` szakaszban áttekintettük, mi történik a sztochasztikus gradienscsökkenés végrehajtásakor, vagyis amikor az optimalizálás során csak a gradiens zajos változata áll rendelkezésre. Különösen azt vettük észre, hogy zajos gradiensek esetén rendkívül körültekintőnek kell lennünk a tanulási ráta megválasztásakor a zaj figyelembevételével. Ha túl gyorsan csökkentjük, a konvergencia megáll. Ha túl engedékenyek vagyunk, nem konvergálunk elég jó megoldáshoz, mivel a zaj folyamatosan az optimalitástól visz el bennünket.

## Alapok

Ebben a szakaszban hatékonyabb optimalizálási algoritmusokat vizsgálunk, különösen bizonyos, a gyakorlatban elterjedt optimalizálási problémák esetén.


### Kiszivárgó átlagok

Az előző szakaszban a minibatch SGD-t tárgyaltuk a számítás gyorsításának eszközeként. Ennek kellemesebb mellékhatása is volt: a gradiensek átlagolása csökkentette a variancia mértékét. A minibatch sztochasztikus gradienscsökkenés a következőképpen számítható:

$$\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

A jelölés egyszerűsítése érdekében $\mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1})$-t az $i$ mintához tartozó sztochasztikus gradienscsökkenésként használtuk a $t-1$ időpontban frissített súlyokkal.
Jó lenne, ha a varianciaredukció hatásából a minibatch-en belüli gradienátlagoláson túl is profitálhatnánk. Ennek egyik módja a gradiens számítás helyettesítése egy „kiszivárgó átlaggal":

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$

valamely $\beta \in (0, 1)$ esetén. Ez lényegében a pillanatnyi gradienst helyettesíti egy olyannal, amelyet több *korábbi* gradiens átlagából számítanak. $\mathbf{v}$-t *sebességnek* nevezzük. Felhalmozza a korábbi gradieneket, hasonlóan ahhoz, ahogy egy nehéz golyó a célfüggvény felületén lefelé gurulva integrálja a korábbi erőket. A részletek jobb megértéséhez fejtsük ki $\mathbf{v}_t$-t rekurzívan:

$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$

A nagy $\beta$ hosszú távú átlagolást jelent, míg a kis $\beta$ csak enyhe korrekciót a gradiensmódszerhez képest. Az új gradiens-helyettesítő többé nem az adott példán mutat a legmeredekebb ereszkedés irányába, hanem a korábbi gradiensek súlyozott átlagának irányába. Ez lehetővé teszi számunkra, hogy a batch feletti átlagolás legtöbb előnyét elérjük anélkül, hogy ténylegesen ki kellene számítanunk a gradieneket rajta. Ezt az átlagolási eljárást egy későbbi részben részletesebben tárgyaljuk.

A fenti érvelés alapozta meg azt, amit ma *gyorsított* gradiensmódszerekként ismerünk, mint például a momentummal rendelkező gradiensek. Ezeknek az az előnyük, hogy sokkal hatékonyabbak rosszul kondicionált optimalizálási problémák esetén (vagyis ahol egyes irányokban a haladás sokkal lassabb, mint másokban, ami egy szűk kanyonhoz hasonlít). Ráadásul lehetővé teszik az egymást követő gradiensek átlagolását, hogy stabilabb ereszkedési irányokat kapjunk. Valóban, a gyorsítás szempontja még zaj-mentes konvex problémák esetén is az egyik kulcsoka annak, hogy a momentum miért működik, és miért olyan jól.

Ahogy várható, hatékonysága miatt a momentum jól kutatott téma a mélytanulás és az azon túlmutató optimalizálásban. Lásd például :citet:`Goh.2017` szép [ismertető cikkét](https://distill.pub/2017/momentum/) a mélyreható elemzésért és interaktív animációért. :citet:`Polyak.1964` javasolta. :citet:`Nesterov.2018` részletes elméleti tárgyalást végez a konvex optimalizálás kontextusában. A momentum mélytanulásban való hasznosságát régóta ismerik. Lásd például :citet:`Sutskever.Martens.Dahl.ea.2013` részletes tárgyalását.

### Rosszul kondicionált probléma

A momentum módszer geometriai tulajdonságainak jobb megértéséhez visszatérünk a gradienscsökkenéshez, bár egy lényegesen kellemetlen célfüggvénnyel. Felidézve, a :numref:`sec_gd` szakaszban $f(\mathbf{x}) = x_1^2 + 2 x_2^2$-t, vagyis egy mérsékelten torzított ellipszoid célfüggvényt alkalmaztunk. Ezt a függvényt tovább torzítjuk az $x_1$ irányban való megnyújtással:

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

Mint korábban, $f$ minimuma $(0, 0)$-ban van. Ez a függvény az $x_1$ irányában *nagyon* lapos. Nézzük meg, mi történik, ha a gradienscsökkenést alkalmazzuk az új függvényen ugyanúgy, mint korábban. $0.4$-es tanulási rátát választunk.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

A szerkezeti felépítésből adódóan az $x_2$ irányú gradiens *sokkal* nagyobb, és sokkal gyorsabban változik, mint a vízszintes $x_1$ irányban. Tehát két kedvezőtlen választási lehetőség között rekedünk: ha kis tanulási rátát választunk, biztosítjuk, hogy a megoldás nem tér el az $x_2$ irányban, de lassú konvergenciával kell szembenéznünk az $x_1$ irányban. Ezzel szemben nagy tanulási rátával gyorsan haladunk az $x_1$ irányban, de eltérünk az $x_2$ irányban. Az alábbi példa bemutatja, mi történik még a tanulási ráta $0.4$-ről $0.6$-ra való enyhe növelésekor is. Az $x_1$ irányú konvergencia javul, de az összesített megoldás minősége sokkal rosszabb.

```{.python .input}
#@tab all
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

### A momentum módszer

A momentum módszer lehetővé teszi a fent leírt gradienscsökkenés probléma megoldását. Az optimalizálási nyomot nézve sejthetnénk, hogy a korábbi gradiensek átlagolása jól működne. Elvégre az $x_1$ irányban ez összegyűjti az egymáshoz igazított gradieneket, növelve az egyes lépéseknél megtett távolságot. Ezzel szemben az $x_2$ irányban, ahol a gradiensek oszcillálnak, az összesített gradiens csökkenti a lépésméretet az egymást kioltó oszcillációk miatt.
A $\mathbf{g}_t$ gradiens helyett $\mathbf{v}_t$ alkalmazása a következő frissítési egyenletekhez vezet:

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$

Megjegyezzük, hogy $\beta = 0$ esetén visszakapjuk a hagyományos gradienscsökkenést. Mielőtt mélyebben belemerülnénk a matematikai tulajdonságokba, nézzük meg gyorsan, hogyan viselkedik az algoritmus a gyakorlatban.

```{.python .input}
#@tab all
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Ahogy látható, ugyanolyan tanulási ráta mellett is a momentum jól konvergál. Nézzük meg, mi történik, ha csökkentjük a momentum hiperparamétert. Felezve $\beta = 0.25$-re alig konvergáló trajektóriát kapunk. Ennek ellenére sokkal jobb, mint momentum nélkül (amikor a megoldás eltér).

```{.python .input}
#@tab all
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Megjegyezzük, hogy a momentum kombinálható a sztochasztikus gradienscsökkenéssel, különösen a minibatch sztochasztikus gradienscsökkenéssel. Az egyetlen különbség az, hogy ekkor a $\mathbf{g}_{t, t-1}$ gradienst $\mathbf{g}_t$-vel helyettesítjük. Végül kényelmi okokból $\mathbf{v}_0 = 0$-val inicializálunk $t=0$ időpontban. Nézzük meg, mit tesz valójában a kiszivárgó átlagolás a frissítésekkel.

### Effektív mintasúly

Felidézve: $\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$. A határesetben a tagok összege $\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$. Más szóval: ahelyett, hogy $\eta$ méretű lépést tennénk a gradienscsökkenésben vagy a sztochasztikus gradienscsökkenésben, $\frac{\eta}{1-\beta}$ méretű lépést teszünk, miközben potenciálisan sokkal jobb viselkedésű ereszkedési iránnyal dolgozunk. Ez két előny egyszerre. A $\beta$ különböző értékeire vonatkozó súlyozás viselkedésének szemléltetéséhez tekintsük az alábbi ábrát.

```{.python .input}
#@tab all
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

## Gyakorlati kísérletek

Nézzük meg, hogyan működik a momentum a gyakorlatban, vagyis amikor egy megfelelő optimalizálóban alkalmazzák. Ehhez egy kissé skálázhatóbb implementációra van szükségünk.

### Implementálás alapoktól

A (minibatch) sztochasztikus gradienscsökkenéshez képest a momentum módszernek egy kiegészítő változókészletet kell fenntartania, vagyis a sebességet. Ugyanolyan alakú, mint a gradiensek (és az optimalizálási probléma változói). Az alábbi implementációban ezeket a változókat `states`-nek nevezzük.

```{.python .input}
#@tab mxnet,pytorch
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return (v_w, v_b)
```

```{.python .input}
#@tab tensorflow
def init_momentum_states(features_dim):
    v_w = tf.Variable(d2l.zeros((features_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    return (v_w, v_b)
```

```{.python .input}
#@tab mxnet
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + p.grad
        p[:] -= hyperparams['lr'] * v
```

```{.python .input}
#@tab pytorch
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd_momentum(params, grads, states, hyperparams):
    for p, v, g in zip(params, states, grads):
            v[:].assign(hyperparams['momentum'] * v + g)
            p[:].assign(p - hyperparams['lr'] * v)
```

Nézzük meg, hogyan működik ez a gyakorlatban.

```{.python .input}
#@tab all
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

Ha a `momentum` momentum hiperparamétert 0.9-re növeljük, ez lényegesen nagyobb effektív mintaméretet jelent: $\frac{1}{1 - 0.9} = 10$. A tanulási rátát kissé $0.01$-re csökkentjük, hogy a dolgok kézben maradjanak.

```{.python .input}
#@tab all
train_momentum(0.01, 0.9)
```

A tanulási ráta további csökkentése kezeli a nem sima optimalizálási problémák eseteit. $0.005$-re állítva jó konvergenciatulajdonságokat kapunk.

```{.python .input}
#@tab all
train_momentum(0.005, 0.9)
```

### Tömör implementáció

A Gluon-ban nagyon kevés tennivaló van, mivel a standard `sgd` megoldóban már be van építve a momentum. Megfelelő paraméterek beállítása nagyon hasonló trajektóriát eredményez.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

## Elméleti elemzés

Eddig az $f(x) = 0.1 x_1^2 + 2 x_2^2$ kétdimenziós példa kissé mesterkéltnek tűnt. Most látni fogjuk, hogy valójában ez meglehetősen reprezentatív azon problémák típusaira, amelyekkel szembesülhetünk, legalábbis konvex másodfokú célfüggvények minimalizálása esetén.

### Másodfokú konvex függvények

Tekintsük a következő függvényt:

$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$

Ez egy általános másodfokú függvény. Pozitív definit $\mathbf{Q} \succ 0$ mátrixokra, vagyis pozitív sajátértékű mátrixokra, a minimalizáló $\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$, a minimumérték $b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. Így $h$ átírható:

$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$

A gradiens: $\partial_{\mathbf{x}} h(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$. Vagyis a $\mathbf{x}$ és a minimalizáló közötti távolság, megszorozva $\mathbf{Q}$-val. Következésképpen a sebesség is $\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$ tagok lineáris kombinációja.

Mivel $\mathbf{Q}$ pozitív definit, felbontható sajátrendszerére $\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$, ahol $\mathbf{O}$ ortogonális (forgatási) mátrix, $\boldsymbol{\Lambda}$ pedig pozitív sajátértékek diagonális mátrixa. Ez lehetővé teszi számunkra, hogy $\mathbf{x}$-ről $\mathbf{z} \stackrel{\textrm{def}}{=} \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$-re váltsunk, lényegesen egyszerűsített kifejezést kapva:

$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$

ahol $b' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. Mivel $\mathbf{O}$ csak ortogonális mátrix, a gradieneket nem zavarja érdemlegesen. $\mathbf{z}$ szerint kifejezve a gradienscsökkenés:

$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$

Ennek a kifejezésnek az a fontos tulajdonsága, hogy a gradienscsökkenés *nem kever* különböző sajáttér-koordináták között. Vagyis $\mathbf{Q}$ sajátrendszerével kifejezve az optimalizálási probléma koordinátánként halad. Ez érvényes a következőre is:

$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$

Ezzel éppen bebizonyítottuk a következő tételt: a gradienscsökkenés momentummal és anélkül konvex másodfokú függvényre a másodfokú mátrix sajátvektorainak irányában koordinátánkénti optimalizálásra bontható.

### Skaláris függvények

A fenti eredmény alapján nézzük meg, mi történik, ha az $f(x) = \frac{\lambda}{2} x^2$ függvényt minimalizáljuk. Gradienscsökkenés esetén:

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$

Ha $|1 - \eta \lambda| < 1$, az optimalizálás exponenciális ütemben konvergál, mivel $t$ lépés után $x_t = (1 - \eta \lambda)^t x_0$. Ez megmutatja, hogyan javul kezdetben a konvergencia üteme a tanulási ráta $\eta$ növelésével, egészen $\eta \lambda = 1$-ig. Ezen túl az optimalizálás eltér, és $\eta \lambda > 2$ esetén divergál.

```{.python .input}
#@tab all
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = d2l.numpy(d2l.arange(20))
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

A momentum esetén a konvergencia elemzéséhez a frissítési egyenleteket két skalár szempontjából írjuk át: egyet $x$-re, egyet a $v$ sebességre. Ez adja:

$$
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$

A konvergencia viselkedését irányító $2 \times 2$-es mátrix jelölésére $\mathbf{R}$-t használtuk. $t$ lépés után a $[v_0, x_0]$ kezdőválasztás $\mathbf{R}(\beta, \eta, \lambda)^t [v_0, x_0]$-vá válik. Ezért a konvergencia sebessége $\mathbf{R}$ sajátértékeitől függ. Lásd :citet:`Goh.2017` [Distill-bejegyzését](https://distill.pub/2017/momentum/) a nagyszerű animációért, és :citet:`Flammarion.Bach.2015`-t a részletes elemzésért. Megmutatható, hogy $0 < \eta \lambda < 2 + 2 \beta$ esetén a sebesség konvergál. Ez nagyobb megvalósítható paramétertartomány a gradienscsökkenés $0 < \eta \lambda < 2$-jéhez képest. Azt is sugallja, hogy általában a $\beta$ nagy értékei kívánatosak. A részletes tárgyalás jelentős technikai részleteket igényel, és javasoljuk, hogy az érdeklődők az eredeti publikációkat olvassák el.

## Összefoglalás

* A momentum a gradieneket helyettesíti a korábbi gradiensek kiszivárgó átlagával. Ez jelentősen gyorsítja a konvergenciát.
* Mind a zajtalan gradienscsökkenés, mind a (zajos) sztochasztikus gradienscsökkenés számára kívánatos.
* A momentum megakadályozza az optimalizálási folyamat megakadását, ami a sztochasztikus gradienscsökkenésnél sokkal valószínűbb.
* A gradiensek effektív száma $\frac{1}{1-\beta}$, a múltbeli adatok exponenciálisan csökkenő súlyozása miatt.
* Konvex másodfokú problémák esetén ez részletesen, explicite elemezhető.
* Az implementáció meglehetősen egyszerű, de egy kiegészítő állapotvektort ($\mathbf{v}$ sebességet) kell tárolni.

## Gyakorló feladatok

1. Próbálj ki más momentum hiperparaméter és tanulási ráta kombinációkat, és figyeld meg és elemezd a különböző kísérleti eredményeket.
1. Próbáld ki a gradienscsökkenést és a momentumot másodfokú problémán, ahol több sajátértéke van: $f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$, pl. $\lambda_i = 2^{-i}$. Ábrázold, hogyan csökken $x$ értéke az $x_i = 1$ inicializálás esetén.
1. Határozd meg a $h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$ minimumértékét és minimalizálóját.
1. Mi változik, ha sztochasztikus gradienscsökkenést végzünk momentummal? Mi történik, ha minibatch sztochasztikus gradienscsökkenést alkalmazunk momentummal? Kísérletezz a paraméterekkel?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/354)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1070)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1071)
:end_tab:
