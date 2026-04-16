# Optimalizálás és mélytanulás
:label:`sec_optimization-intro`

Ebben a szakaszban az optimalizálás és a mélytanulás kapcsolatát, valamint az optimalizálás mélytanulásban való alkalmazásának kihívásait tárgyaljuk.
Egy mélytanulási probléma esetén általában először egy *veszteségfüggvényt* definiálunk. Miután rendelkezünk a veszteségfüggvénnyel, optimalizálási algoritmust alkalmazhatunk a veszteség minimalizálásának megkísérléséhez.
Az optimalizálásban a veszteségfüggvényt gyakran az optimalizálási probléma *célfüggvényének* nevezik. Hagyomány szerint a legtöbb optimalizálási algoritmus *minimalizálással* foglalkozik. Ha valaha maximalizálni kellene egy célfüggvényt, egyszerű megoldás létezik: csupán meg kell fordítani a célfüggvény előjelét.

## Az optimalizálás célja

Bár az optimalizálás lehetőséget nyújt a veszteségfüggvény minimalizálására a mélytanulásban, lényegét tekintve az optimalizálás és a mélytanulás céljai alapvetően különböznek.
Az előbbi elsősorban a célfüggvény minimalizálásával foglalkozik, míg az utóbbi egy véges mennyiségű adat alapján megfelelő modell megtalálásával.
A :numref:`sec_generalization_basics` szakaszban részletesen tárgyaltuk e két cél különbségét.
Például a tanítási hiba és az általánosítási hiba általában különbözik: mivel az optimalizálási algoritmus célfüggvénye általában a tanítóhalmazon alapuló veszteségfüggvény, az optimalizálás célja a tanítási hiba csökkentése.
A mélytanulás (vagy tágabb értelemben a statisztikai következtetés) célja azonban az általánosítási hiba csökkentése.
Az utóbbi eléréséhez a tanítási hiba csökkentése mellett figyelmet kell fordítani a túlillesztésre is.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

A fent említett eltérő célok szemléltetéséhez vizsgáljuk meg az empirikus kockázatot és a kockázatot.
Ahogy a :numref:`subsec_empirical-risk-and-risk` szakaszban leírtuk, az empirikus kockázat a tanítóhalmazon mért átlagos veszteség, míg a kockázat a teljes adatpopuláción mért várható veszteség.
Az alábbiakban két függvényt definiálunk: az `f` kockázatfüggvényt és a `g` empirikus kockázatfüggvényt.
Tegyük fel, hogy csak véges mennyiségű tanítóadattal rendelkezünk.
Ennek következtében a `g` kevésbé sima, mint az `f`.

```{.python .input}
#@tab all
def f(x):
    return x * d2l.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * d2l.cos(5 * np.pi * x)
```

Az alábbi grafikon szemlélteti, hogy a tanítóhalmazon mért empirikus kockázat minimuma más helyen lehet, mint a kockázat (általánosítási hiba) minimuma.

```{.python .input}
#@tab all
def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = d2l.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
```

## Optimalizálási kihívások a mélytanulásban

Ebben a fejezetben kifejezetten az optimalizálási algoritmusok teljesítményére összpontosítunk a célfüggvény minimalizálásában, nem pedig a modell általánosítási hibájára.
A :numref:`sec_linear_regression` szakaszban különbséget tettünk az analitikus és numerikus megoldások között az optimalizálási problémákban.
A mélytanulásban a legtöbb célfüggvény bonyolult, és nincs analitikus megoldása. Ehelyett numerikus optimalizálási algoritmusokat kell alkalmaznunk.
Az ebben a fejezetben szereplő optimalizálási algoritmusok mindegyike ebbe a kategóriába tartozik.

A mélytanulásban számos kihívással kell szembenézni az optimalizálás során. A leginkább zavaró problémák a lokális minimumok, a nyeregpontok és az eltűnő gradiensek.
Nézzük meg ezeket egyenként.


### Lokális minimumok

Bármely $f(x)$ célfüggvény esetén, ha $f(x)$ értéke $x$-nél kisebb, mint $f(x)$ értéke $x$ bármely más környékbeli pontján, akkor $f(x)$ lokális minimum lehet.
Ha $f(x)$ értéke $x$-nél a legkisebb az egész értelmezési tartomány felett, akkor $f(x)$ globális minimum.

Például az alábbi függvényt tekintve:

$$f(x) = x \cdot \textrm{cos}(\pi x) \textrm{ for } -1.0 \leq x \leq 2.0,$$

megközelíthetjük e függvény lokális és globális minimumát.

```{.python .input}
#@tab all
x = d2l.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
```

A mélytanulási modellek célfüggvényének általában sok lokális optimuma van.
Ha az optimalizálási probléma numerikus megoldása a lokális optimum közelében van, a végső iterációval kapott numerikus megoldás a célfüggvényt csak *lokálisan* minimalizálhatja, nem *globálisan*, mivel a célfüggvény megoldásainak gradiense közel nullává válik vagy nullává lesz.
Csupán némi zaj képes kilökni a paramétert a lokális minimumból. Ez egyébként a minibatch sztochasztikus gradienscsökkenés egyik előnyös tulajdonsága: a gradiensek természetes variabilitása a minibatchek között képes kiszabadítani a paramétereket a lokális minimumokból.


### Nyeregpontok

A lokális minimumok mellett a nyeregpontok egy másik ok arra, hogy a gradiensek eltűnjenek. A *nyeregpont* olyan hely, ahol egy függvény összes gradiense eltűnik, de sem globális, sem lokális minimum nem áll fenn.
Tekintsük az $f(x) = x^3$ függvényt. Első és második deriváltja egyaránt eltűnik $x=0$-ban. Az optimalizálás megakadhat ezen a ponton, annak ellenére, hogy ez nem minimum.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
```

A magasabb dimenziókban a nyeregpontok még alattomosabbak, ahogyan az alábbi példa is mutatja. Tekintsük az $f(x, y) = x^2 - y^2$ függvényt. Nyeregpontja a $(0, 0)$-ban van. Ez $y$ tekintetében maximum, $x$ tekintetében pedig minimum. Sőt, *kinézetre* olyan, mint egy nyereg – innen kapta ez a matematikai tulajdonság a nevét.

```{.python .input}
#@tab mxnet
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

```{.python .input}
#@tab pytorch, tensorflow
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

Feltételezzük, hogy egy függvény bemenete egy $k$-dimenziós vektor, kimenete pedig egy skalár, ezért Hesse-mátrixának $k$ sajátértéke lesz.
A függvény megoldása lokális minimum, lokális maximum vagy nyeregpont lehet azon a ponton, ahol a függvény gradiense nulla:

* Ha a függvény Hesse-mátrixának sajátértékei a nullagradiens-pozícióban mind pozitívak, a függvénynek lokális minimuma van.
* Ha a függvény Hesse-mátrixának sajátértékei a nullagradiens-pozícióban mind negatívak, a függvénynek lokális maximuma van.
* Ha a függvény Hesse-mátrixának sajátértékei a nullagradiens-pozícióban negatívak és pozitívak egyaránt, a függvénynek nyeregpontja van.

Nagy dimenziójú problémáknál meglehetősen nagy a valószínűsége annak, hogy legalább *néhány* sajátérték negatív. Ez a nyeregpontokat valószínűbbé teszi a lokális minimumoknál. A következő szakaszban, a konvexitás bevezetésekor, tárgyalni fogunk néhány kivételt. Röviden: a konvex függvények azok, amelyek Hesse-mátrixának sajátértékei soha nem negatívak. Sajnos azonban a legtöbb mélytanulási probléma nem ebbe a kategóriába esik. Ennek ellenére ez remek eszköz az optimalizálási algoritmusok tanulmányozásához.

### Eltűnő gradiensek

Valószínűleg a leginkább alattomosan fellépő probléma az eltűnő gradiens.
Idézzük fel a :numref:`subsec_activation-functions` szakaszban tárgyalt általánosan használt aktiválási függvényeket és azok deriváltjait.
Tegyük fel például, hogy az $f(x) = \tanh(x)$ függvényt szeretnénk minimalizálni, és $x = 4$ értékről indulunk el. Ahogyan látható, az $f$ gradiense közel nulla.
Konkrétabban: $f'(x) = 1 - \tanh^2(x)$, tehát $f'(4) = 0.0013$.
Következésképpen az optimalizálás hosszú ideig megrekedt, mielőtt haladást érnénk el. Ez egyébként az egyik oka annak, hogy a mélytanulási modellek betanítása meglehetősen nehézkes volt a ReLU aktiválási függvény bevezetése előtt.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [d2l.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
```

Ahogy láttuk, a mélytanulásban az optimalizálás tele van kihívásokkal. Szerencsére létezik algoritmusok egy robusztus köre, amelyek jól teljesítenek, és még kezdők számára is könnyen használhatók. Ráadásul valójában nem szükséges megtalálni a *legjobb* megoldást. A lokális optimumok, sőt azok közelítő megoldásai is rendkívül hasznosak lehetnek.

## Összefoglalás

* A tanítási hiba minimalizálása *nem* garantálja, hogy megtaláljuk azt a paraméterkészletet, amely minimalizálja az általánosítási hibát.
* Az optimalizálási problémáknak számos lokális minimuma lehet.
* A problémának még több nyeregpontja is lehet, mivel a problémák általában nem konvexek.
* Az eltűnő gradiensek megakadályozhatják az optimalizálást. Sokszor segít a probléma átparametrizálása. A paraméterek megfelelő inicializálása is előnyös lehet.


## Gyakorló feladatok

1. Tekintsünk egy egyszerű, egyetlen rejtett réteggel rendelkező MLP-t, ahol a rejtett rétegnek mondjuk $d$ dimenziója van és egyetlen kimenete. Mutassuk meg, hogy minden lokális minimumhoz legalább $d!$ ekvivalens megoldás létezik, amelyek azonos viselkedést mutatnak.
1. Tegyük fel, hogy van egy szimmetrikus véletlenszerű $\mathbf{M}$ mátrixunk, amelynek elemei $M_{ij} = M_{ji}$ mind valamilyen $p_{ij}$ valószínűségi eloszlásból kerülnek mintavételre. Tegyük fel továbbá, hogy $p_{ij}(x) = p_{ij}(-x)$, azaz az eloszlás szimmetrikus (lásd például :citet:`Wigner.1958`).
    1. Bizonyítsuk be, hogy a sajátértékek eloszlása is szimmetrikus. Vagyis bármely $\mathbf{v}$ sajátvektorra az ahhoz tartozó $\lambda$ sajátérték valószínűsége kielégíti: $P(\lambda > 0) = P(\lambda < 0)$.
    1. Miért *nem* jelenti a fenti állítás azt, hogy $P(\lambda > 0) = 0.5$?
1. Milyen más kihívásokat tudsz elképzelni a mélytanulás optimalizálásában?
1. Tegyük fel, hogy egy (valódi) labdát szeretnél egyensúlyba hozni egy (valódi) nyergen.
    1. Miért nehéz ez?
    1. Kihasználható-e ez a hatás az optimalizálási algoritmusoknál is?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/349)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/487)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/489)
:end_tab:
