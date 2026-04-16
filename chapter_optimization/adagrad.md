# Adagrad
:label:`sec_adagrad`

Kezdjük ritkán előforduló jellemzőkkel rendelkező tanulási problémák vizsgálatával.


## Ritka jellemzők és tanulási sebességek

Képzeljük el, hogy egy nyelvi modellt tanítunk. A jó pontosság eléréséhez általában csökkenteni kell a tanulási sebességet a tanítás előrehaladtával, általában $\mathcal{O}(t^{-\frac{1}{2}})$ vagy ennél kisebb ütemben. Tekintsük most az olyan modell tanítását, amely ritka jellemzőkre épül, vagyis olyan jellemzőkre, amelyek csak ritkán fordulnak elő. Ez természetes nyelvben is elterjedt: például sokkal ritkábban találkozunk az *előkondicionálás* szóval, mint a *tanulás* szóval. Ez azonban más területeken is elterjedt, például a számítógépes hirdetésben és a személyre szabott kollaboratív szűrésben. Elvégre sok olyan dolog létezik, amely csak kevesek számára érdekes.

A ritkán előforduló jellemzőkhöz tartozó paraméterek csak akkor kapnak értelmes frissítéseket, ha ezek a jellemzők ténylegesen megjelennek. A csökkenő tanulási sebesség esetén előfordulhat, hogy a gyakori jellemzőkhöz tartozó paraméterek viszonylag gyorsan konvergálnak optimális értékeikre, míg a ritkán előfordulók esetén még nem figyeltük meg őket elégszer ahhoz, hogy meghatározható legyen az optimális érték. Más szóval a tanulási sebesség vagy túl lassan csökken a gyakori jellemzőknél, vagy túl gyorsan a ritka jellemzőknél.

A probléma kezelésére lehetséges trükk az egyes jellemzők megjelenéseinek megszámlálása, és ezt felhasználni a tanulási sebességek módosításakor. Vagyis $\eta = \frac{\eta_0}{\sqrt{t + c}}$ alakú tanulási sebesség helyett $\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$-t alkalmazhatnánk. A $s(i, t)$ az $i$ jellemző nem-nulla megjelenéseinek száma a $t$ időpontig. Ez valójában könnyen implementálható, lényeges overhead nélkül. Azonban csődöt mond, ha nem feltételezünk ritkaságot, hanem olyan adatokkal van dolgunk, amelyeknél a gradiensek általában nagyon kicsik, és csak ritkán nagyok. Elvégre nem egyértelmű, hol lenne a határ egy megfigyelt és egy nem megfigyelt jellemző között.

Az Adagrad :citet:`Duchi.Hazan.Singer.2011` ezt úgy kezeli, hogy a meglehetősen durva $s(i, t)$ számlálót felváltja a korábban megfigyelt gradiensek négyzetösszegével. Konkrétan: $s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$-t alkalmaz a tanulási sebesség módosítására. Ennek két előnye van: először nem kell dönteni, mikor elég nagy a gradiens. Másodszor, automatikusan skálázódik a gradiensek nagyságával. A rendszeresen nagy gradieneknek megfelelő koordinátákat lényegesen kisebbre skálázzák, míg mások kis gradienekkel sokkal enyhébb bánásmódban részesülnek. A gyakorlatban ez egy rendkívül hatékony optimalizálási eljárást eredményez számítógépes hirdetési és kapcsolódó problémáknál. De ez elrejt néhány további, az Adagradban rejlő előnyt, amelyeket az előkondicionálás kontextusában lehet a legjobban megérteni.


## Előkondicionálás

A konvex optimalizálási problémák jók az algoritmusok jellemzőinek elemzésére. Elvégre a legtöbb nemkonvex probléma esetén nehéz érdemleges elméleti garanciákat levezetni, de az *intuíció* és a *belátás* általában megmarad. Nézzük meg az $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$ minimalizálásának problémáját.

Ahogy a :numref:`sec_momentum` szakaszban láttuk, ez a probléma átírható sajátfelbontása $\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$ alapján, hogy egy lényegesen egyszerűsített problémát kapjunk, ahol minden koordináta egyenként megoldható:

$$f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b.$$

ahol $\bar{\mathbf{x}} = \mathbf{U} \mathbf{x}$, és következésképpen $\bar{\mathbf{c}} = \mathbf{U} \mathbf{c}$. A módosított probléma minimalizálója $\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$, minimumértéke $-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$. Ez sokkal könnyebben számítható, mivel $\boldsymbol{\Lambda}$ egy $\mathbf{Q}$ sajátértékeit tartalmazó diagonális mátrix.

Ha kissé megváltoztatjuk $\mathbf{c}$-t, azt várnánk, hogy $f$ minimalizálójában csak csekély változás következzen be. Sajnos ez nem így van. Bár $\mathbf{c}$ kis változásai egyenértékű kis változásokat okoznak $\bar{\mathbf{c}}$-ben, ez nem érvényes $f$ (és $\bar{f}$) minimalizálójára. Ha a $\boldsymbol{\Lambda}_i$ sajátértékek nagyok, csak kis változásokat látunk $\bar{x}_i$-ben és $\bar{f}$ minimumában. Ezzel szemben kis $\boldsymbol{\Lambda}_i$ esetén $\bar{x}_i$ változásai drámaiak lehetnek. A legnagyobb és a legkisebb sajátérték arányát az optimalizálási probléma kondíciószámának nevezik.

$$\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.$$

Ha a $\kappa$ kondíciószám nagy, nehéz pontosan megoldani az optimalizálási problémát. Biztosítani kell, hogy az értékek nagy dinamikus tartományát megfelelően kezeljük. Az elemzés egy nyilvánvaló, bár kissé naiv kérdéshez vezet: nem lehetne-e egyszerűen „javítani" a problémán azzal, hogy torzítjuk a teret úgy, hogy az összes sajátérték $1$ legyen? Elméletileg ez meglehetősen egyszerű: csupán a $\mathbf{Q}$ sajátértékeire és sajátvektoraira van szükségünk, hogy $\mathbf{x}$-ről $\mathbf{z} \stackrel{\textrm{def}}{=} \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$-re skálázzuk a problémát. Az új koordináta-rendszerben $\mathbf{x}^\top \mathbf{Q} \mathbf{x}$ egyszerűsíthetővé válna $\|\mathbf{z}\|^2$-re. Ez azonban meglehetősen praktikátlan javaslat. A sajátértékek és sajátvektorok kiszámítása általában *sokkal* drágább, mint maga a probléma megoldása.

Bár a sajátértékek pontos kiszámítása drága lehet, azok becslése vagy közelítő kiszámítása már sokkal jobb lehet, mintha semmit sem tennénk. Különösen: felhasználhatjuk $\mathbf{Q}$ átlós elemeit, és ennek megfelelően skálázhatjuk.

$$\tilde{\mathbf{Q}} = \textrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \textrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).$$

Ebben az esetben $\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$, és különösen $\tilde{\mathbf{Q}}_{ii} = 1$ minden $i$-re. A legtöbb esetben ez jelentősen csökkenti a kondíciószámot. Például a korábban tárgyalt esetekben ez teljesen kiküszöbölné a problémát, mivel a probléma tengelyirányba esik.

Sajnos egy újabb problémával szembesülünk: a mélytanulásban általában még a célfüggvény második deriváltjához sem férünk hozzá: $\mathbf{x} \in \mathbb{R}^d$ esetén a második derivált még egyetlen minibatch-en is $\mathcal{O}(d^2)$ tárhelyet és munkát igényelhet a kiszámításához, így a gyakorlatban kivitelezhetetlen. Az Adagrad zseniális ötlete az, hogy a Hesse-mátrix megfoghatatlan átlója helyett egy olyan helyettesítőt alkalmaz, amely viszonylag olcsón számítható és hatékony – magát a gradiens nagyságát.

Annak megértéséhez, hogy ez miért működik, vizsgáljuk meg $\bar{f}(\bar{\mathbf{x}})$-t. Az alábbi összefüggés áll fenn:

$$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),$$

ahol $\bar{\mathbf{x}}_0$ a $\bar{f}$ minimalizálója. Ezért a gradiens nagysága mind $\boldsymbol{\Lambda}$-tól, mind az optimalitástól való távolságtól függ. Ha $\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$ nem változna, ez minden lenne, amire szükség van. Elvégre ebben az esetben a $\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$ gradiens nagysága elegendő lenne. Mivel az Adagrad egy sztochasztikus gradienscsökkenés algoritmus, még az optimalitásnál is nem nulla varianciájú gradieneket látunk. Ennek eredményeként a gradiensek varianciáját biztonságosan alkalmazhatjuk a Hesse-mátrix skálájának olcsó proxijaként. Az alapos elemzés meghaladja e szakasz kereteit (több oldalt igényelne). A részletekért lásd :cite:`Duchi.Hazan.Singer.2011`.

## Az algoritmus

Formalizáljuk a fenti tárgyalást. A $\mathbf{s}_t$ változót alkalmazzuk a korábbi gradiens-variancia felhalmozására:

$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$

A műveletek koordinátánként alkalmazandók. Vagyis $\mathbf{v}^2$ elemei $v_i^2$. Hasonlóan $\frac{1}{\sqrt{v}}$ elemei $\frac{1}{\sqrt{v_i}}$, $\mathbf{u} \cdot \mathbf{v}$ elemei $u_i v_i$. Mint korábban, $\eta$ a tanulási sebesség, $\epsilon$ egy additív konstans, amely biztosítja, hogy ne osszunk $0$-val. Végül $\mathbf{s}_0 = \mathbf{0}$-val inicializálunk.

Ahogy a momentum eseténél, itt is szükség van egy kiegészítő változó nyilvántartására, ebben az esetben azért, hogy koordinátánként egyéni tanulási sebességet lehessen alkalmazni. Ez nem növeli jelentősen az Adagrad költségét az SGD-hez képest, egyszerűen azért, mert a fő költség általában az $l(y_t, f(\mathbf{x}_t, \mathbf{w}))$ és deriváltjának kiszámítása.

Fontos megjegyezni, hogy a négyzetes gradiensek felhalmozása $\mathbf{s}_t$-ben azt jelenti, hogy $\mathbf{s}_t$ lényegében lineáris ütemben nő (a gyakorlatban kissé lassabban a lineárisnál, mivel a gradiensek kezdetben csökkennek). Ez $\mathcal{O}(t^{-\frac{1}{2}})$ tanulási sebességhez vezet, bár koordinátánként módosítva. Konvex problémák esetén ez teljesen megfelelő. A mélytanulásban azonban a tanulási sebességet lassabban is csökkenthetjük. Ez számos Adagrad-variánshoz vezetett, amelyeket a következő fejezetekben tárgyalunk. Egyelőre nézzük meg, hogyan viselkedik egy másodfokú konvex problémán. Ugyanazt a problémát alkalmazzuk, mint korábban:

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

Az Adagradet a korábban használt tanulási sebességgel implementáljuk, vagyis $\eta = 0.4$. Ahogy látható, a független változó iteratív trajektóriája simább. Azonban $\boldsymbol{s}_t$ kumulatív hatása miatt a tanulási sebesség folyamatosan csökken, így a független változó az iteráció késői szakaszaiban nem mozdul annyit.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input}
#@tab all
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

Ha a tanulási sebességet $2$-re növeljük, sokkal jobb viselkedést tapasztalunk. Ez már jelzi, hogy a tanulási sebesség csökkentése talán túl agresszív, még zaj-mentes esetben is, és biztosítani kell a paraméterek megfelelő konvergenciáját.

```{.python .input}
#@tab all
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

## Implementálás alapoktól

A momentum módszerhez hasonlóan az Adagradnak is fenn kell tartania egy ugyanolyan alakú állapotváltozót, mint a paraméterek.

```{.python .input}
#@tab mxnet
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def init_adagrad_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)

def adagrad(params, grads, states, hyperparams):
    eps = 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(s + tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

A :numref:`sec_minibatch_sgd` szakaszban végzett kísérlethez képest
nagyobb tanulási sebességet alkalmazunk a modell tanításához.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

## Tömör implementáció

Az `adagrad` algoritmus `Trainer` példányát alkalmazva meghívhatjuk az Adagrad algoritmust a Gluon-ban.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adagrad', {'learning_rate': 0.1}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adagrad
d2l.train_concise_ch11(trainer, {'learning_rate' : 0.1}, data_iter)
```

## Összefoglalás

* Az Adagrad dinamikusan csökkenti a tanulási sebességet koordinátánként.
* A gradiens nagyságát alkalmazza annak mértékeként, hogy milyen gyorsan érjük el a haladást – a nagy gradieneknek megfelelő koordinátákat kisebb tanulási sebességgel kompenzálja.
* A pontos második derivált kiszámítása általában nem kivitelezhető a mélytanulásban memória- és számítási korlátok miatt. A gradiens hasznos közelítőként szolgálhat.
* Ha az optimalizálási problémának meglehetősen egyenetlen szerkezete van, az Adagrad segíthet csökkenteni a torzítást.
* Az Adagrad különösen hatékony ritka jellemzők esetén, ahol a tanulási sebességnek lassabban kell csökkenie a ritkán előforduló tagnál.
* A mélytanulási problémáknál az Adagrad néha túl agresszívan csökkenti a tanulási sebességet. A mérséklési stratégiákat a :numref:`sec_adam` kontextusában tárgyaljuk.

## Gyakorló feladatok

1. Bizonyítsd be, hogy ortogonális $\mathbf{U}$ mátrix és $\mathbf{c}$ vektor esetén a következő teljesül: $\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$. Miért jelenti ez azt, hogy a perturbációk nagysága nem változik az ortogonális változócsere után?
1. Próbáld ki az Adagradet az $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ függvényre, és a 45 fokkal elforgatott célfüggvényre is: $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. Másképpen viselkedik?
1. Bizonyítsd be a [Gerschgorin-körök tételét](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem), amely kimondja, hogy a $\mathbf{M}$ mátrix $\lambda_i$ sajátértékeire legalább egy $j$ választás esetén teljesül: $|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$.
1. Mit mond a Gerschgorin-tétel az átlósan előkondicionált $\textrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \textrm{diag}^{-\frac{1}{2}}(\mathbf{M})$ mátrix sajátértékeiről?
1. Próbáld ki az Adagradet egy megfelelő mély hálózaton, például a :numref:`sec_lenet` szakaszban a Fashion-MNIST adathalmazra alkalmazva.
1. Hogyan kellene módosítani az Adagradet, hogy kevésbé agresszív tanulási sebesség csökkenést érjen el?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/355)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1072)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1073)
:end_tab:
