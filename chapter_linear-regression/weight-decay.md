```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Súlycsökkentés (Weight Decay)
:label:`sec_weight_decay`

Most, hogy jellemeztük a túlillesztés problémáját,
bemutathatjuk az első *regularizációs* technikát.
Emlékeztetőül: a túlillesztést mindig enyhíthetjük
több tanítási adat gyűjtésével.
Ez azonban drága és időigényes lehet,
vagy teljesen kívül eshet az irányításunkon,
ami rövid távon lehetetlenné teszi.
Egyelőre feltételezhetjük, hogy már rendelkezünk
annyi kiváló minőségű adattal, amennyit az erőforrásaink lehetővé tesznek,
és összpontosítunk a rendelkezésre álló eszközökre,
amikor az adathalmazt adottnak vesszük.

Emlékeztető: a polinomiális regresszió példánkban
(:numref:`subsec_polynomial-curve-fitting`)
korlátozni tudtuk a modell kapacitását
az illesztett polinom fokának módosításával.
Valóban, a jellemzők számának korlátozása
egy népszerű technika a túlillesztés enyhítésére.
A jellemzők egyszerű eldobása azonban
túl durva eszköz lehet.
Maradjunk a polinomiális regresszió
példájánál, és gondoljuk meg, mi történhet
nagy dimenziójú bemenettel.
A polinomok természetes kiterjesztései
a többváltozós adatokra a *monómiumok*,
amelyek egyszerűen változók hatványainak szorzatai.
Egy monómium foka a hatványok összege.
Például $x_1^2 x_2$ és $x_3 x_5^2$
mindkettő 3-as fokú monómium.

Megjegyezzük, hogy a $d$ fokú tagok száma
gyorsan robban fel, ahogy $d$ növekszik.
Adott $k$ változó esetén a $d$ fokú monómiumok száma
${k - 1 + d} \choose {k - 1}$.
Még kis fokváltozások, mondjuk $2$-ről $3$-ra,
drámaian növelik a modellünk komplexitását.
Ezért gyakran finomabb eszközre van szükségünk
a függvény-komplexitás beállításához.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import optax
```

## Normák és súlycsökkentés

(**Ahelyett, hogy közvetlenül a paraméterek számát manipulálnánk,
a *súlycsökkentés* (weight decay) a paraméterek által felvehető értékek korlátozásával működik.**)
A deep learning körökön kívül általánosan $\ell_2$ regularizációnak nevezik,
mini-batch sztochasztikus gradienscsökkenéssel optimalizálva,
a súlycsökkentés talán a legelterjedtebb technika
a paraméteres gépi tanulási modellek regularizálásához.
A technika motivációja az az alapvető intuíció,
hogy az összes $f$ függvény közül
az $f = 0$ függvény
(amely minden bemenethez a $0$ értéket rendeli)
valamilyen értelemben a *legegyszerűbb*,
és a függvény komplexitása mérhető
a paramétereinek a nullától való távolságával.
De pontosan hogyan mérjük
a függvény és a nulla közötti távolságot?
Nincs egyetlen helyes válasz.
Valójában a matematika egész ágai,
beleértve a funkcionálanalízis
és a Banach-terek elméletének részeit is,
ilyen kérdésekkel foglalkoznak.

Egy egyszerű értelmezés lehet
egy lineáris függvény komplexitásának mérése
$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$
a súlyvektora valamilyen normájával, pl. $\| \mathbf{w} \|^2$.
Emlékeztetőül: az $\ell_2$ normát és az $\ell_1$ normát mutattuk be,
amelyek az általánosabb $\ell_p$ norma speciális esetei,
a :numref:`subsec_lin-algebra-norms` részben.
A kis súlyvektor biztosításának legáltalánosabb módszere
a normájának büntetési tagként való hozzáadása
a veszteség minimalizálásának problémájához.
Így az eredeti célkitűzésünket,
a *tanítási címkéken az előrejelzési veszteség minimalizálását*,
felváltjuk egy új célkitűzéssel:
az *előrejelzési veszteség és a büntetési tag összegének minimalizálása*.
Ha a súlyvektorunk most túl nagy lesz,
a tanulási algoritmusunk a $\| \mathbf{w} \|^2$ súlynorma minimalizálására
koncentrálhat a tanítási hiba minimalizálása helyett.
Pontosan ezt szeretnénk.
A kód illusztrálásához
visszatérünk korábbi példánkhoz
a :numref:`sec_linear_regression` lineáris regresszióból.
Ott a veszteségünk a következő volt:

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

Emlékeztető: $\mathbf{x}^{(i)}$ a jellemzők,
$y^{(i)}$ a címke bármely $i$ adatpéldányhoz, és $(\mathbf{w}, b)$
a súly- és eltolásparaméterek.
A súlyvektor méretének büntetéséhez
valahogyan hozzá kell adnunk a $\| \mathbf{w} \|^2$ értéket a veszteségfüggvényhez,
de hogyan egyenlítse ki a modell a
standard veszteséget ezzel az új additív büntetéssel?
A gyakorlatban ezt az egyensúlyt
a *regularizációs állandón* $\lambda$ keresztül jellemezzük,
egy nemnegatív hiperparamétert,
amelyet validációs adatokkal illesztünk be:

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2.$$


$\lambda = 0$ esetén visszakapjuk az eredeti veszteségfüggvényt.
$\lambda > 0$ esetén korlátozjuk a $\| \mathbf{w} \|$ méretét.
$2$-vel osztunk konvencióból:
amikor egy másodfokú függvény deriváltját vesszük,
a $2$ és az $1/2$ kiejti egymást, biztosítva, hogy a frissítési kifejezés
szépnek és egyszerűnek nézzen ki.
Az éles olvasó talán azon töpreng, miért a négyzetesen vett
normával dolgozunk, nem a standard normával (azaz az Euklideszi távolsággal).
Ezt számítási kényelem miatt tesszük.
Az $\ell_2$ norma négyzetbevételével eltávolítjuk a négyzetgyököt,
megmaradva a súlyvektor
minden összetevőjének négyzetek összegeként.
Ez megkönnyíti a büntetés deriváltjának kiszámítását:
a deriváltak összege egyenlő az összeg deriváltjával.


Ráadásul felvetődhet a kérdés, miért dolgozunk egyáltalán az $\ell_2$ normával,
és nem mondjuk az $\ell_1$ normával.
Valójában más választások is érvényesek és
népszerűek a statisztikában.
Míg az $\ell_2$-regularizált lineáris modellek
a klasszikus *gerinc-regresszió* (ridge regression) algoritmust alkotják,
az $\ell_1$-regularizált lineáris regresszió
hasonlóan alapvető módszer a statisztikában,
amelyet közismerten *lasso regressziónak* neveznek.
Az $\ell_2$ norma használatának egyik oka,
hogy az túlméretezett büntetést ró
a súlyvektor nagy összetevőire.
Ez a tanulási algoritmusunkat olyan modellek felé torzítja,
amelyek az általuk adott súlyt egyenletesebben osztják el
a jellemzők nagyobb száma között.
A gyakorlatban ez robusztusabbá teheti őket
egyetlen változóban lévő mérési hiba esetén.
Ezzel szemben az $\ell_1$ büntetések olyan modellekhez vezet,
amelyek a súlyokat a jellemzők kis halmazára koncentrálják,
a többi súlyt nullára törölve.
Ez hatékony módszert ad számunkra a *jellemzőválasztáshoz*,
amely más okokból is kívánatos lehet.
Például, ha a modellünk csak néhány jellemzőre támaszkodik,
esetleg nem kell adatokat gyűjtenünk, tárolnunk vagy továbbítanunk
a többi (elhagyott) jellemzőhöz.

A :eqref:`eq_linreg_batch_update` egyenlettel megegyező jelölést használva,
a mini-batch sztochasztikus gradienscsökkenés frissítései
az $\ell_2$-regularizált regresszióhoz:

$$\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}$$

Mint korábban, frissítjük $\mathbf{w}$-t a becslésünk
és a megfigyelés közötti különbség alapján.
Ugyanakkor csökkentjük is $\mathbf{w}$ méretét a nulla felé.
Ezért nevezik a módszert néha "súlycsökkentésnek":
a büntetési tag egyedüli hatásával
az optimalizálási algoritmusunk *csökkenti*
a súlyt a tanítás minden lépésében.
A jellemzőválasztással ellentétben
a súlycsökkentés egy mechanizmust kínál számunkra
a függvény komplexitásának folyamatos beállítására.
$\lambda$ kisebb értékei
kevésbé korlátozott $\mathbf{w}$-nek felelnek meg,
míg $\lambda$ nagyobb értékei
$\mathbf{w}$-t jelentősen korlátozzák.
Az, hogy megfelelő eltolás-büntetést $b^2$ is beleszámítunk-e,
implementációnként változhat,
és a neurális hálózat rétegei között is változhat.
Általában nem regularizáljuk az eltolás tagot.
Emellett,
bár az $\ell_2$ regularizáció nem feltétlenül egyenértékű a súlycsökkentéssel más optimalizálási algoritmusoknál,
a regularizáció ötlete a
súlyok méretének csökkentésén keresztül
továbbra is érvényes.

## Nagy dimenziójú lineáris regresszió

A súlycsökkentés előnyeit egy egyszerű szintetikus példán illusztrálhatjuk.

Először [**néhány adatot generálunk, mint korábban**]:

(**$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \textrm{ ahol }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$**)

Ebben a szintetikus adathalmazban a cimkénket
a bemenetünk egy alap lineáris függvénye adja,
amelyet Gauss-zajjal rontottunk meg,
amelynek várható értéke nulla és szórása 0.01.
Illusztrációs céljából
a túlillesztés hatásait hangsúlyossá tehetjük,
ha a probléma dimenzionalitását $d = 200$-ra növeljük,
és csak 20 példánnyal rendelkező kis tanítási halmazon dolgozunk.

```{.python .input}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()                
        n = num_train + num_val 
        if tab.selected('mxnet') or tab.selected('pytorch'):
            self.X = d2l.randn(n, num_inputs)
            noise = d2l.randn(n, 1) * 0.01
        if tab.selected('tensorflow'):
            self.X = d2l.normal((n, num_inputs))
            noise = d2l.normal((n, 1)) * 0.01
        if tab.selected('jax'):
            self.X = jax.random.normal(jax.random.PRNGKey(0), (n, num_inputs))
            noise = jax.random.normal(jax.random.PRNGKey(0), (n, 1)) * 0.01
        w, b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = d2l.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

## Implementáció nulláról

Most próbáljuk meg implementálni a súlycsökkentést nulláról.
Mivel a mini-batch sztochasztikus gradienscsökkenés
az optimalizálónk,
csupán hozzá kell adnunk a négyzetes $\ell_2$ büntetést
az eredeti veszteségfüggvényhez.

### (**Az $\ell_2$ norma büntetés definiálása**)

Talán a legkényelmesebb módja ennek a büntetésnek
az, ha minden tagot a helyén négyzetre emelünk és összeadjuk.

```{.python .input}
%%tab all
def l2_penalty(w):
    return d2l.reduce_sum(w**2) / 2
```

### A modell definiálása

A végső modellben
a lineáris regresszió és a négyzeteshiba nem változott a :numref:`sec_linear_scratch` óta,
tehát egyszerűen definiálunk egy alosztályt a `d2l.LinearRegressionScratch`-ből. Az egyetlen változás itt az, hogy a veszteségünk most tartalmazza a büntetési tagot.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()
        
    def loss(self, y_hat, y):
        return (super().loss(y_hat, y) +
                self.lambd * l2_penalty(self.w))
```

```{.python .input}
%%tab jax
class WeightDecayScratch(d2l.LinearRegressionScratch):
    lambd: int = 0
        
    def loss(self, params, X, y, state):
        return (super().loss(params, X, y, state) +
                self.lambd * l2_penalty(params['w']))
```

A következő kód illeszti a modellünket 20 példányból álló tanítási halmazra és kiértékeli 100 példányos validációs halmazon.

```{.python .input}
%%tab all
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):    
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        print('L2 norm of w:', float(l2_penalty(model.w)))
    if tab.selected('jax'):
        print('L2 norm of w:',
              float(l2_penalty(trainer.state.params['w'])))
```

### [**Tanítás regularizáció nélkül**]

Most futtatjuk ezt a kódot `lambd = 0`-val,
kikapcsolva a súlycsökkentést.
Vegyük észre, hogy súlyosan túlillesztünk,
csökkentve a tanítási hibát, de nem a
validációs hibát — ez a túlillesztés tankönyvi esete.

```{.python .input}
%%tab all
train_scratch(0)
```

### [**Súlycsökkentés alkalmazása**]

Az alábbiakban lényeges súlycsökkentéssel futtatunk.
Vegyük észre, hogy a tanítási hiba növekszik,
de a validációs hiba csökken.
Ez pontosan az a hatás,
amelyet a regularizációtól elvárunk.

```{.python .input}
%%tab all
train_scratch(3)
```

## [**Tömör implementáció**]

Mivel a súlycsökkentés mindenütt jelen van
a neurális hálózat optimalizálásában,
a deep learning keretrendszer különösen kényelmes módon integrálja,
a súlycsökkentést beépítve magába az optimalizálási algoritmusba
a könnyű használat érdekében, bármely veszteségfüggvénnyel kombinálva.
Ezenkívül ez az integráció számítási előnyt is jelent,
lehetővé téve az implementációs trükkök alkalmazását,
hogy a súlycsökkentést hozzáadják az algoritmushoz,
bármilyen extra számítási rezsi nélkül.
Mivel a frissítés súlycsökkentési részlete
csak minden egyes paraméter aktuális értékétől függ,
az optimalizálónak amúgy is meg kell érintenie minden paramétert egyszer.

:begin_tab:`mxnet`
Az alábbiakban
a súlycsökkentés hiperparamétert közvetlenül
a `wd`-n keresztül adjuk meg a `Trainer` példányosításakor.
Alapértelmezés szerint a Gluon egyszerre csökkenti
a súlyokat és az eltolásokat.
Megjegyezzük, hogy a `wd` hiperparaméter
meg lesz szorozva `wd_mult`-tal
a modell paramétereinek frissítésekor.
Tehát, ha a `wd_mult`-ot nullára állítjuk,
a $b$ eltolás paraméter nem fog csökkeni.
:end_tab:

:begin_tab:`pytorch`
Az alábbiakban
a súlycsökkentés hiperparamétert közvetlenül
a `weight_decay`-en keresztül adjuk meg az optimalizáló példányosításakor.
Alapértelmezés szerint a PyTorch egyszerre csökkenti
a súlyokat és az eltolásokat, de
beállíthatjuk az optimalizálót, hogy különböző paramétereket
különböző irányelvek szerint kezelje.
Itt csak a `weight_decay`-t állítjuk be
a súlyokhoz (a `net.weight` paraméterekhez), ezért az
eltolás (a `net.bias` paraméter) nem fog csökkeni.
:end_tab:

:begin_tab:`tensorflow`
Az alábbiakban egy $\ell_2$ regularizálót hozunk létre
a `wd` súlycsökkentés hiperparaméterrel, és alkalmazzuk a réteg súlyaira
a `kernel_regularizer` argumentumon keresztül.
:end_tab:

```{.python .input}
%%tab mxnet
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd
        
    def configure_optimizers(self):
        self.collect_params('.*bias').setattr('wd_mult', 0)
        return gluon.Trainer(self.collect_params(),
                             'sgd', 
                             {'learning_rate': self.lr, 'wd': self.wd})
```

```{.python .input}
%%tab pytorch
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)
```

```{.python .input}
%%tab tensorflow
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.net = tf.keras.layers.Dense(
            1, kernel_regularizer=tf.keras.regularizers.l2(wd),
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01)
        )
        
    def loss(self, y_hat, y):
        return super().loss(y_hat, y) + self.net.losses
```

```{.python .input}
%%tab jax
class WeightDecay(d2l.LinearRegression):
    wd: int = 0
    
    def configure_optimizers(self):
        # A súlycsökkentés nem érhető el közvetlenül az optax.sgd-n belül,
        # de az optax lehetővé teszi több transzformáció láncolását
        return optax.chain(optax.additive_weight_decay(self.wd),
                           optax.sgd(self.lr))
```

[**Az ábra hasonlónak tűnik, mint akkor,
amikor nulláról implementáltuk a súlycsökkentést**].
Ez a verzió azonban gyorsabb
és könnyebben implementálható,
olyan előnyök, amelyek egyre
hangsúlyosabbak lesznek, ahogy nagyobb problémákkal foglalkozol,
és ez a munka egyre rutinszerűbbé válik.

```{.python .input}
%%tab all
model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)

if tab.selected('jax'):
    print('L2 norm of w:', float(l2_penalty(model.get_w_b(trainer.state)[0])))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))
```

Eddig megérintettük az egyszerű lineáris függvény
egy fogalmát.
Azonban még egyszerű nemlineáris függvényeknél is a helyzet sokkal összetettebb lehet. Ennek megértéséhez a [reprodukáló kernel Hilbert-tér (RKHS)](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) fogalma
lehetővé teszi, hogy a lineáris függvényekhez bevezetett eszközöket
nemlineáris kontextusban alkalmazzuk.
Sajnos az RKHS alapú algoritmusok
hajlamosak rosszul skálázódni nagy, magas dimenziójú adatokhoz.
Ebben a könyvben az a közös heurisztika lesz az általános megközelítésünk,
amelynek értelmében a súlycsökkentést alkalmazzák
egy mély hálózat összes rétegére.

## Összefoglalás

A regularizáció egy általános módszer a túlillesztés kezelésére. A klasszikus regularizációs technikák büntetési tagot adnak a veszteségfüggvényhez (tanítás közben), hogy csökkentsék a tanult modell komplexitását.
Az $\ell_2$ büntetés alkalmazása az egyik különleges választás, amely egyszerűvé teszi a modellt. Ez súlycsökkentéshez vezet a mini-batch sztochasztikus gradienscsökkenés algoritmus frissítési lépéseiben.
A gyakorlatban a súlycsökkentés funkcionalitást a deep learning keretrendszerek optimalizálói biztosítják.
A különböző paraméterek különböző frissítési viselkedéssel rendelkezhetnek ugyanazon tanítási cikluson belül.



## Feladatok

1. Kísérletezz a $\lambda$ értékével a becslési problémában ebben a részben. Ábrázold a tanítási és validációs pontosságot a $\lambda$ függvényeként. Mit figyelsz meg?
1. Használj validációs halmazt a $\lambda$ optimális értékének megtalálásához. Ez valóban az optimális érték? Számít-e ez?
1. Hogyan néznének ki a frissítési egyenletek, ha $\|\mathbf{w}\|^2$ helyett $\sum_i |w_i|$-t használnánk büntetési tagként ($\ell_1$ regularizáció)?
1. Tudjuk, hogy $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$. Tudsz-e hasonló egyenletet találni mátrixokra (lásd a Frobenius-normát a :numref:`subsec_lin-algebra-norms` részben)?
1. Nézd át a tanítási hiba és az általánosítási hiba közötti kapcsolatot. A súlycsökkentésen, a megnövelt tanításon és a megfelelő komplexitású modell alkalmazásán kívül milyen egyéb módszerek segíthetnek nekünk a túlillesztés kezelésében?
1. A Bayes-statisztikában a prior és a valószínűség szorzatát használjuk a posteriori eléréséhez $P(w \mid x) \propto P(x \mid w) P(w)$ révén. Hogyan azonosíthatod $P(w)$-t a regularizációval?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/236)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17979)
:end_tab:
