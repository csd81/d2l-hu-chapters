# Gradienscsökkenés
:label:`sec_gd`

Ebben a szakaszban bemutatjuk a *gradienscsökkenés* mögött álló alapfogalmakat.
Bár a mélytanulásban ritkán alkalmazzák közvetlenül, a gradienscsökkenés megértése kulcsfontosságú a sztochasztikus gradienscsökkenés algoritmusok megértéséhez.
Például az optimalizálási probléma eltérhet a megoldástól a túlságosan nagy tanulási sebesség miatt. Ez a jelenség már a gradienscsökkenés esetén is megfigyelhető. Hasonlóképpen, az előkondicionálás a gradienscsökkenés egyik általánosan alkalmazott technikája, amely a fejlettebb algoritmusokra is érvényes.
Kezdjük egy egyszerű speciális esettel.


## Egydimenziós gradienscsökkenés

Az egydimenziós gradienscsökkenés kiváló példa annak magyarázatára, hogy a gradienscsökkenés algoritmus miért csökkentheti a célfüggvény értékét. Tekintsünk egy folytonosan differenciálható, valós értékű $f: \mathbb{R} \rightarrow \mathbb{R}$ függvényt. Taylor-kifejtéssel kapjuk:

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

Azaz, elsőrendű közelítésben $f(x+\epsilon)$ az $f(x)$ függvényértékkel és az $x$-beli $f'(x)$ első deriválttal adható meg. Nem megalapozatlan feltételezni, hogy kis $\epsilon$ esetén a negatív gradiens irányába haladva $f$ csökken. Az egyszerűség kedvéért rögzített $\eta > 0$ lépésközt választunk, és $\epsilon = -\eta f'(x)$-et választunk. Ezt behelyettesítve a fenti Taylor-kifejtésbe:

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$
:eqlabel:`gd-taylor-2`

Ha a derivált $f'(x) \neq 0$ nem tűnik el, haladást érünk el, mivel $\eta f'^2(x)>0$. Ráadásul $\eta$-t mindig elég kicsire választhatjuk, hogy a magasabb rendű tagok elhanyagolhatókká váljanak. Tehát:

$$f(x - \eta f'(x)) \lessapprox f(x).$$

Ez azt jelenti, hogy ha a

$$x \leftarrow x - \eta f'(x)$$

frissítési szabályt alkalmazzuk $x$ iterálásához, az $f(x)$ értéke csökkenhet. Ezért a gradienscsökkenés során először egy kezdeti $x$ értéket és egy $\eta > 0$ konstanst választunk, majd ezekkel folyamatosan iteráljuk $x$-et, amíg a leállítási feltétel nem teljesül – például amikor a gradiens nagysága $|f'(x)|$ elég kicsi, vagy az iterációk száma elér egy bizonyos értéket.

Az egyszerűség kedvéért az $f(x)=x^2$ célfüggvényt választjuk a gradienscsökkenés megvalósításának szemléltetéséhez. Bár tudjuk, hogy $x=0$ a megoldás $f(x)$ minimalizálásához, ezt az egyszerű függvényt mégis használjuk annak megfigyeléséhez, hogyan változik $x$.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
def f(x):  # Célfüggvény
    return x ** 2

def f_grad(x):  # A célfüggvény gradiense (deriváltja)
    return 2 * x
```

Ezután $x=10$-et használjuk kezdőértékként, és $\eta=0.2$-t feltételezünk. A gradienscsökkenés segítségével 10-szer iterálva $x$-et, láthatjuk, hogy $x$ értéke végül közelíti az optimális megoldást.

```{.python .input}
#@tab all
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)
```

Az $x$ optimalizálásának folyamata az alábbiak szerint ábrázolható.

```{.python .input}
#@tab all
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
```

### Tanulási sebesség
:label:`subsec_gd-learningrate`

A $\eta$ tanulási sebességet az algoritmus tervezője állíthatja be. Ha túl kis tanulási sebességet alkalmazunk, $x$ nagyon lassan frissül, és több iteráció szükséges a jobb megoldás eléréséhez. Ennek szemléltetéséhez tekintsük az ugyanolyan optimalizálási probléma haladását $\eta = 0.05$ esetén. Ahogy látható, még 10 lépés után is messze vagyunk az optimális megoldástól.

```{.python .input}
#@tab all
show_trace(gd(0.05, f_grad), f)
```

Ezzel szemben, ha túl nagy tanulási sebességet alkalmazunk, a $\left|\eta f'(x)\right|$ túl nagy lehet az elsőrendű Taylor-kifejtési képlethez. Vagyis a :eqref:`gd-taylor-2`-beli $\mathcal{O}(\eta^2 f'^2(x))$ tag jelentőssé válhat. Ebben az esetben nem garantálható, hogy $x$ iterálása csökkenti $f(x)$ értékét. Például ha $\eta=1.1$ tanulási sebességet állítunk be, $x$ túllő az $x=0$ optimális megoldáson, és fokozatosan eltér.

```{.python .input}
#@tab all
show_trace(gd(1.1, f_grad), f)
```

### Lokális minimumok

Annak szemléltetéséhez, hogy nemkonvex függvényeknél mi történik, tekintsük az $f(x) = x \cdot \cos(cx)$ esetet valamely $c$ konstansra. Ennek a függvénynek végtelen sok lokális minimuma van. A tanulási sebesség megválasztásától és a probléma kondicionáltságától függően a sok megoldás egyikéhez juthatunk. Az alábbi példa szemlélteti, hogy a (nem realisztikusan) nagy tanulási sebesség egy rossz lokális minimumhoz vezet.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Célfüggvény
    return x * d2l.cos(c * x)

def f_grad(x):  # A célfüggvény gradiense
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

show_trace(gd(2, f_grad), f)
```

## Többváltozós gradienscsökkenés

Most, hogy jobb intuíciónk van az egyváltozós esetről, tekintsük azt a helyzetet, amikor $\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$. Vagyis az $f: \mathbb{R}^d \to \mathbb{R}$ célfüggvény vektorokat képez skalárra. Ennek megfelelően a gradiense is többváltozós: $d$ parciális deriváltból álló vektor:

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$

A $\partial f(\mathbf{x})/\partial x_i$ parciális derivált elem a gradiensvektorban az $f$ változásának sebességét jelzi $\mathbf{x}$-nél az $x_i$ bemeneti koordináta tekintetében. Az egyváltozós esethez hasonlóan a megfelelő Taylor-közelítést alkalmazhatjuk többváltozós függvényekre is, hogy iránymutatást kapjunk. Konkrétan:

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \mathbf{\boldsymbol{\epsilon}}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`

Más szóval: $\boldsymbol{\epsilon}$-ban másodrendű tagokig a legmeredekebb ereszkedés iránya a negatív gradiens $-\nabla f(\mathbf{x})$. Megfelelő $\eta > 0$ tanulási sebesség megválasztásával kapjuk az archetipikus gradienscsökkenés algoritmust:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

Az algoritmus gyakorlati viselkedésének megfigyeléséhez konstruáljuk meg az $f(\mathbf{x})=x_1^2+2x_2^2$ célfüggvényt, amelynek bemenete a kétdimenziós $\mathbf{x} = [x_1, x_2]^\top$ vektor, kimenete egy skalár. A gradiens: $\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$. Megfigyeljük $\mathbf{x}$ trajektóriáját gradienscsökkenéssel a $[-5, -2]$ kezdőpozícióból indulva.

Kezdetként két segédfüggvényre van szükségünk. Az első egy frissítési függvényt alkalmaz 20-szor a kezdőértékre. A második segédfüggvény megjeleníti $\mathbf{x}$ trajektóriáját.

```{.python .input}
#@tab all
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """2D célfüggvény optimalizálása egyedi trainerrel."""
    # `s1` és `s2` belső állapotváltozók, amelyeket a Momentum, adagrad, RMSProp algoritmusokban használunk
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results
```

```{.python .input}
#@tab mxnet
def show_trace_2d(f, results):  #@save
    """2D változók trajektóriájának megjelenítése optimalizálás közben."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-55, 1, 1),
                          d2l.arange(-30, 1, 1))
    x1, x2 = x1.asnumpy()*0.1, x2.asnumpy()*0.1
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

```{.python .input}
#@tab tensorflow
def show_trace_2d(f, results):  #@save
    """2D változók trajektóriájának megjelenítése optimalizálás közben."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

```{.python .input}
#@tab pytorch
def show_trace_2d(f, results):  #@save
    """2D változók trajektóriájának megjelenítése optimalizálás közben."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

Ezután megfigyeljük a $\mathbf{x}$ optimalizálási változó trajektóriáját $\eta = 0.1$ tanulási sebesség mellett. Látható, hogy 20 lépés után $\mathbf{x}$ értéke közelíti a $[0, 0]$-beli minimumát. A haladás meglehetősen rendezett, bár eléggé lassú.

```{.python .input}
#@tab all
def f_2d(x1, x2):  # Célfüggvény
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # A célfüggvény gradiense
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
```

## Adaptív módszerek

Ahogy a :numref:`subsec_gd-learningrate` szakaszban láthattuk, a $\eta$ tanulási sebesség „tökéletes" meghatározása nehéz feladat. Ha túl kicsit választunk, alig haladunk. Ha túl nagyot választunk, az megoldás oszcillál, és a legrosszabb esetben akár el is térhet. Mi lenne, ha automatikusan meghatározhatnánk $\eta$-t, vagy teljesen el tudnánk kerülni a tanulási sebesség megválasztásának szükségességét?
A másodrendű módszerek, amelyek nemcsak a célfüggvény értékét és gradienst, hanem *görbületét* is figyelembe veszik, segíthetnek ebben. Bár ezek a módszerek közvetlenül nem alkalmazhatók a mélytanulásban a számítási költség miatt, hasznos intuíciót nyújtanak a fejlett optimalizálási algoritmusok tervezéséhez, amelyek az alább leírt algoritmusok sok kívánatos tulajdonságát utánozzák.


### Newton-módszer

Valamely $f: \mathbb{R}^d \rightarrow \mathbb{R}$ függvény Taylor-kifejtésének felülvizsgálatakor nem kell az első tagnál megállnunk. Valójában felírhatjuk:

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \nabla^2 f(\mathbf{x}) \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3).$$
:eqlabel:`gd-hot-taylor`

A jelölés egyszerűsítése érdekében definiáljuk $\mathbf{H} \stackrel{\textrm{def}}{=} \nabla^2 f(\mathbf{x})$-et mint $f$ Hesse-mátrixát, amely egy $d \times d$-es mátrix. Kis $d$ és egyszerű problémák esetén $\mathbf{H}$ könnyen számítható. Mély neurális hálózatoknál viszont $\mathbf{H}$ aránytalanul nagy lehet, mivel $\mathcal{O}(d^2)$ elemet kell tárolni. Ráadásul a visszaterjesztésen keresztüli kiszámítása is túl drága lehet. Egyelőre hagyjuk figyelmen kívül ezeket a szempontokat, és nézzük, milyen algoritmust kapnánk.

$f$ minimuma kielégíti a $\nabla f = 0$ feltételt.
A :numref:`subsec_calculus-grad` szakasz kalkulus-szabályait követve,
a :eqref:`gd-hot-taylor` deriváltját $\boldsymbol{\epsilon}$ szerint véve és a magasabb rendű tagokat elhanyagolva kapjuk:

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \textrm{ and hence }
\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

Vagyis az optimalizálási probléma részeként meg kell fordítanunk a $\mathbf{H}$ Hesse-mátrixot.

Egyszerű példaként, $f(x) = \frac{1}{2} x^2$ esetén $\nabla f(x) = x$ és $\mathbf{H} = 1$. Ebből bármely $x$-re $\epsilon = -x$ következik. Más szóval, *egyetlen* lépés elegendő a tökéletes konvergenciához, minden hangolás nélkül! Sajnos itt kissé szerencsések voltunk: a Taylor-kifejtés pontos volt, mivel $f(x+\epsilon)= \frac{1}{2} x^2 + \epsilon x + \frac{1}{2} \epsilon^2$.

Nézzük meg, mi történik más problémáknál.
Adott egy konvex hiperbolikus koszinusz függvény $f(x) = \cosh(cx)$ valamely $c$ konstansra; láthatjuk, hogy a $x=0$-beli globális minimum néhány iteráció után elérhető.

```{.python .input}
#@tab all
c = d2l.tensor(0.5)

def f(x):  # Célfüggvény
    return d2l.cosh(c * x)

def f_grad(x):  # A célfüggvény gradiense
    return c * d2l.sinh(c * x)

def f_hess(x):  # A célfüggvény Hesse-mátrixa
    return c**2 * d2l.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)
```

Most tekintsünk egy *nemkonvex* függvényt, például $f(x) = x \cos(c x)$-et valamely $c$ konstansra. Vegyük észre, hogy a Newton-módszerben a Hesse-mátrixszal osztunk. Ez azt jelenti, hogy ha a második derivált *negatív*, akkor $f$ értékét *növelő* irányba léphetünk.
Ez az algoritmus végzetes hibája.
Nézzük meg, mi történik a gyakorlatban.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Célfüggvény
    return x * d2l.cos(c * x)

def f_grad(x):  # A célfüggvény gradiense
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # A célfüggvény Hesse-mátrixa
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)
```

Ez látványosan rosszul sült el. Hogyan javíthatjuk? Az egyik módszer a Hesse-mátrix „javítása" az abszolút értékek használatával. Másik stratégia a tanulási sebesség visszavezetése. Ez ugyan látszólag ellentmond a célnak, de nem teljesen. A másodrendű információk lehetővé teszik, hogy óvatosak legyünk, ha a görbület nagy, és hosszabb lépéseket tegyünk, ha a célfüggvény simább.
Nézzük meg, hogyan működik ez egy kissé kisebb tanulási sebességgel, mondjuk $\eta = 0.5$. Ahogy látható, meglehetősen hatékony algoritmust kapunk.

```{.python .input}
#@tab all
show_trace(newton(0.5), f)
```

### Konvergenciaanalízis

A Newton-módszer konvergenciasebességét csak néhány konvex és háromszor differenciálható $f$ célfüggvényre elemezzük, ahol a második derivált nem nulla, vagyis $f'' > 0$. A többdimenziós bizonyítás az alábbi egydimenziós érvelés egyszerű kiterjesztése, és elhagyjuk, mivel az intuíció szempontjából nem sokat ad hozzá.

Jelöljük $x^{(k)}$-val $x$ értékét a $k$-adik iterációban, és legyen $e^{(k)} \stackrel{\textrm{def}}{=} x^{(k)} - x^*$ az optimalitástól való távolság a $k$-adik iterációban. Taylor-kifejtéssel az $f'(x^*) = 0$ feltétel felírható:

$$0 = f'(x^{(k)} - e^{(k)}) = f'(x^{(k)}) - e^{(k)} f''(x^{(k)}) + \frac{1}{2} (e^{(k)})^2 f'''(\xi^{(k)}),$$

ami valamely $\xi^{(k)} \in [x^{(k)} - e^{(k)}, x^{(k)}]$-ra teljesül. A fenti kifejtést $f''(x^{(k)})$-val osztva:

$$e^{(k)} - \frac{f'(x^{(k)})}{f''(x^{(k)})} = \frac{1}{2} (e^{(k)})^2 \frac{f'''(\xi^{(k)})}{f''(x^{(k)})}.$$

Felidézve, hogy a frissítés $x^{(k+1)} = x^{(k)} - f'(x^{(k)}) / f''(x^{(k)})$, és ezt behelyettesítve, mindkét oldal abszolút értékét véve:

$$\left|e^{(k+1)}\right| = \frac{1}{2}(e^{(k)})^2 \frac{\left|f'''(\xi^{(k)})\right|}{f''(x^{(k)})}.$$

Következésképpen, ha egy olyan területen vagyunk, ahol $\left|f'''(\xi^{(k)})\right| / (2f''(x^{(k)})) \leq c$ korlátos, akkor a hiba négyzetes mértékű csökkenést mutat:

$$\left|e^{(k+1)}\right| \leq c (e^{(k)})^2.$$


Megjegyezzük, hogy az optimalizálási kutatók ezt *lineáris* konvergenciának nevezik, míg az $\left|e^{(k+1)}\right| \leq \alpha \left|e^{(k)}\right|$ feltétel *konstans* konvergenciasebességnek felel meg.
Fontos megjegyezni, hogy ez az elemzés számos fenntartással él.
Először is, valójában nincs erős garanciánk arra, mikor érjük el a gyors konvergencia területét. Csak annyit tudunk, hogy ha elérjük, a konvergencia nagyon gyors lesz. Másodszor, ez az elemzés megköveteli, hogy $f$ jól viselkedjen a magasabb rendű deriváltak tekintetében. Lényegében azt kell biztosítani, hogy $f$-nek nincsenek „meglepő" tulajdonságai az értékeinek változása szempontjából.



### Előkondicionálás

Nem meglepő módon a teljes Hesse-mátrix kiszámítása és tárolása nagyon drága. Ezért kívánatos alternatívákat találni. Az egyik javítási lehetőség az *előkondicionálás*. Ez elkerüli a teljes Hesse-mátrix kiszámítását, és csak az *átlós* elemeket számítja ki. Ez az alábbi alakú frissítési algoritmushoz vezet:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \textrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}).$$


Bár ez nem olyan jó, mint a teljes Newton-módszer, mégis sokkal jobb, mintha egyáltalán nem alkalmaznánk.
Annak megértéséhez, hogy ez miért lehet jó ötlet, képzeljük el, hogy az egyik változó milliméteres magasságot, a másik kilométeres magasságot jelöl. Feltételezve, hogy mindkettőnél a természetes skála méterben van, szörnyű eltérés van a parameterizációban. Szerencsére az előkondicionálás alkalmazása megszünteti ezt. A gradienscsökkenéssel kombinált előkondicionálás lényegében különböző tanulási sebességet jelent az egyes változók (a $\mathbf{x}$ vektor koordinátái) számára.
Ahogy később látni fogjuk, az előkondicionálás számos innováció motorja a sztochasztikus gradienscsökkenés optimalizálási algoritmusokban.


### Gradienscsökkenés vonalkereséssel

A gradienscsökkenés egyik kulcsproblémája, hogy túllőhetünk a célon, vagy nem haladunk elég gyorsan. Egyszerű megoldás a vonalkeresés kombinálása a gradienscsökkenéssel. Vagyis a $\nabla f(\mathbf{x})$ által adott irányt használjuk, majd bináris keresést végzünk annak meghatározására, hogy melyik $\eta$ tanulási sebesség minimalizálja az $f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$ értéket.

Ez az algoritmus gyorsan konvergál (az elemzést és bizonyítást lásd például :citet:`Boyd.Vandenberghe.2004`). Azonban a mélytanulás szempontjából ez nem teljesen megvalósítható, mivel a vonalkeresés minden lépéséhez a célfüggvényt a teljes adathalmazon kellene kiértékelni. Ez túlságosan drága.

## Összefoglalás

* A tanulási sebesség számít. Ha túl nagy, eltérünk; ha túl kicsi, nem haladunk.
* A gradienscsökkenés beragadhat lokális minimumokba.
* Nagy dimenziókban a tanulási sebesség beállítása bonyolult.
* Az előkondicionálás segíthet a skálázási problémákon.
* A Newton-módszer sokkal gyorsabb, ha megfelelően működik konvex problémákon.
* Légy óvatos a Newton-módszer alkalmazásakor módosítások nélkül nemkonvex problémákon.

## Gyakorló feladatok

1. Kísérletezz különböző tanulási sebességekkel és célfüggvényekkel gradienscsökkenés esetén.
1. Valósítsd meg a vonalkeresést egy konvex függvény minimalizálásához a $[a, b]$ intervallumon.
    1. Szükséges-e derivált a bináris kereséshez, vagyis a $[a, (a+b)/2]$ vagy az $[(a+b)/2, b]$ interval kiválasztásához?
    1. Milyen gyors az algoritmus konvergenciasebessége?
    1. Valósítsd meg az algoritmust, és alkalmazd a $\log (\exp(x) + \exp(-2x -3))$ minimalizálásához.
1. Tervezz egy $\mathbb{R}^2$-en definiált célfüggvényt, amelynél a gradienscsökkenés rendkívül lassú. Tipp: skálázd a különböző koordinátákat eltérően.
1. Valósítsd meg a Newton-módszer könnyűsúlyú változatát előkondicionálással:
    1. Használd az átlós Hesse-mátrixot előkondicionálóként.
    1. Helyette annak abszolút értékeit alkalmazd (előjel nélkül).
    1. Alkalmazd ezt a fenti problémára.
1. Alkalmazd a fenti algoritmust számos célfüggvényre (konvex és nemkonvex esetekre egyaránt). Mi történik, ha $45$ fokkal elforgatod a koordinátákat?

[Discussions](https://discuss.d2l.ai/t/351)
