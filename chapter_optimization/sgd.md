# Sztochasztikus gradienscsökkenés
:label:`sec_sgd`

A korábbi fejezetekben folyamatosan sztochasztikus gradienscsökkenést alkalmaztunk a tanítási eljárásunkban, anélkül azonban, hogy megmagyaráztuk volna, miért működik.
Ennek megvilágítása érdekében
a :numref:`sec_gd` szakaszban éppen bemutattuk a gradienscsökkenés alapelveit.
Ebben a szakaszban a *sztochasztikus gradienscsökkenés* részletesebb tárgyalásával folytatjuk.

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

## Sztochasztikus gradient frissítések

A mélytanulásban a célfüggvény általában a tanítóhalmaz egyes példáihoz tartozó veszteségfüggvények átlaga.
Adott egy $n$ példát tartalmazó tanítóhalmaz esetén,
feltételezzük, hogy $f_i(\mathbf{x})$ az $i$ indexű tanítási példához tartozó veszteségfüggvény,
ahol $\mathbf{x}$ a paramétervektort jelöli.
Ezután a célfüggvény:

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$

A célfüggvény gradiense $\mathbf{x}$-nél:

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

Ha gradienscsökkenést alkalmazunk, minden egyes független változó iteráció számítási költsége $\mathcal{O}(n)$, ami $n$-nel lineárisan nő. Ezért minél nagyobb a tanítóhalmaz, annál nagyobb a gradienscsökkenés iterációnkénti költsége.

A sztochasztikus gradienscsökkenés (SGD) csökkenti az iterációnkénti számítási költséget. A sztochasztikus gradienscsökkenés minden egyes iterációjában egyenletesen véletlenszerűen mintavételezünk egy $i\in\{1,\ldots, n\}$ indexet az adatpéldákhoz, és a $\nabla f_i(\mathbf{x})$ gradiensszámítással frissítjük $\mathbf{x}$-et:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}),$$

ahol $\eta$ a tanulási ráta. Látható, hogy az iterációnkénti számítási költség csökken a gradienscsökkenés $\mathcal{O}(n)$-jéről a konstans $\mathcal{O}(1)$-re. Ráadásul ki kell emelnünk, hogy a sztochasztikus gradiens $\nabla f_i(\mathbf{x})$ a teljes gradiens $\nabla f(\mathbf{x})$ torzítatlan becslése, mivel:

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$

Ez azt jelenti, hogy átlagosan a sztochasztikus gradiens jó közelítése a valódi gradiensnek.

Most összehasonlítjuk a gradienscsökkenéssel úgy, hogy 0 várható értékű és 1 varianciájú véletlenszerű zajt adunk a gradienhez, szimulálva a sztochasztikus gradienscsökkenést.

```{.python .input}
#@tab all
def f(x1, x2):  # Célfüggvény
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # A célfüggvény gradiense
    return 2 * x1, 4 * x2
```

```{.python .input}
#@tab mxnet
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Zajos gradiens szimulálása
    g1 += d2l.normal(0.0, 1, (1,))
    g2 += d2l.normal(0.0, 1, (1,))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab pytorch
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Zajos gradiens szimulálása
    g1 += torch.normal(0.0, 1, (1,)).item()
    g2 += torch.normal(0.0, 1, (1,)).item()
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab tensorflow
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Zajos gradiens szimulálása
    g1 += d2l.normal([1], 0.0, 1)
    g2 += d2l.normal([1], 0.0, 1)
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab all
def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # Állandó tanulási sebesség
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Ahogy látható, a sztochasztikus gradienscsökkenésben a változók trajektóriája sokkal zajosabb, mint a :numref:`sec_gd` szakaszban megfigyelt gradienscsökkenés esetén. Ez a gradiens sztochasztikus természetéből adódik. Vagyis még akkor is, amikor a minimum közelébe érünk, az $\eta \nabla f_i(\mathbf{x})$ pillanatnyi gradiens által bevezetett bizonytalanság továbbra is fennáll. Még 50 lépés után is meglehetősen gyenge a minőség. Sőt, további lépésekkel sem javul (bátorítunk arra, hogy több lépéssel is kísérletezz ennek megerősítéséhez). Ez csak egy alternatívát hagy: a $\eta$ tanulási ráta megváltoztatását. Ha azonban túl kicsit választunk, kezdetben nem érünk el érdemi haladást. Másrészt, ha túl nagyot választunk, nem kapunk jó megoldást, ahogy fentebb is láttuk. E két ellentétes cél feloldásának egyetlen módja a tanulási ráta *dinamikus* csökkentése az optimalizálás előrehaladtával.

Ez az oka annak is, hogy egy `lr` tanulási ráta függvényt adtunk hozzá az `sgd` lépésfüggvényhez. A fenti példában a tanulási ráta ütemező funkcionalitása inaktív, mivel az ahhoz tartozó `lr` függvényt állandóként állítottuk be.

## Dinamikus tanulási ráta

Az $\eta$ helyére egy időfüggő $\eta(t)$ tanulási ráta behelyettesítése növeli az optimalizálási algoritmus konvergenciájának szabályozásának összetettségét. Különösen azt kell kitalálni, hogy $\eta$-nak milyen gyorsan kell csökkenie. Ha túl gyorsan csökken, idő előtt befejezzük az optimalizálást. Ha túl lassan csökkentjük, túl sok időt pazarlunk az optimalizálásra. Az alábbiak néhány alapvető stratégia, amelyet az $\eta$ időbeli beállításához alkalmaznak (a fejlettebb stratégiákat később tárgyaljuk):

$$
\begin{aligned}
    \eta(t) & = \eta_i \textrm{ if } t_i \leq t \leq t_{i+1}  && \textrm{piecewise constant} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \textrm{exponential decay} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \textrm{polynomial decay}
\end{aligned}
$$

Az első *lépésenkénti konstans* forgatókönyvben csökkentjük a tanulási rátát, például amikor az optimalizálás haladása megáll. Ez egy általánosan alkalmazott stratégia mély hálózatok tanításánál. Alternatívaként sokkal agresszívabban is csökkenthetjük *exponenciális csillapítással*. Sajnos ez gyakran idő előtti leálláshoz vezet, mielőtt az algoritmus konvergált volna. Népszerű választás a $\alpha = 0.5$-ös *polinomiális csillapítás*. A konvex optimalizálás esetén számos bizonyíték létezik, amelyek igazolják, hogy ez az ütem jól viselkedik.

Nézzük meg, hogyan néz ki az exponenciális csillapítás a gyakorlatban.

```{.python .input}
#@tab all
def exponential_lr():
    # Globális változó, amely e függvényen kívül van definiálva és belül frissül
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
```

Ahogy várható, a paraméterek varianciája jelentősen csökkent. Ez azonban az optimális $\mathbf{x} = (0, 0)$ megoldáshoz való konvergencia rovására megy. Még 1000 iteráció után is messze vagyunk az optimális megoldástól. Valójában az algoritmus egyáltalán nem konvergál. Másrészt, ha polinomiális csillapítást alkalmazunk, ahol a tanulási ráta a lépések számának inverz négyzetgyökével csökken, a konvergencia már csak 50 lépés után is javul.

```{.python .input}
#@tab all
def polynomial_lr():
    # Globális változó, amely e függvényen kívül van definiálva és belül frissül
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Számos más lehetőség is létezik a tanulási ráta beállítására. Például kezdhetünk kis sebességgel, majd gyorsan növelhetjük, majd ismét csökkenthetjük, bár lassabban. Akár váltakozhatunk kisebb és nagyobb tanulási ráták között is. Sokféle ilyen ütemterv létezik. Egyelőre összpontosítsunk azokra a tanulási ráta ütemezőkre, amelyekre átfogó elméleti elemzés lehetséges, vagyis a konvex esetben alkalmazott tanulási rátákre. Általános nemkonvex problémáknál nagyon nehéz érdemleges konvergenciagaranciákat kapni, mivel általánosan nemlineáris nemkonvex problémák minimalizálása NP-nehéz. Áttekintésért lásd például Tibshirani 2015-ös kiváló [előadásjegyzeteit](https://www.stat.cmu.edu/%7Eryantibs/convexopt-F15/lectures/26-nonconvex.pdf).



## Konvergenciaanalízis konvex célfüggvényekre

A sztochasztikus gradienscsökkenés következő konvergenciaanalízise konvex célfüggvényekre opcionális, és elsősorban a probléma jobb megértéséhez szükséges intuíció átadását szolgálja.
A legegyszerűbb bizonyításokra szorítkozunk :cite:`Nesterov.Vial.2000`.
Lényegesen fejlettebb bizonyítási technikák is léteznek, például akkor, ha a célfüggvény különösen jól viselkedik.


Tegyük fel, hogy az $f(\boldsymbol{\xi}, \mathbf{x})$ célfüggvény konvex $\mathbf{x}$-ben
minden $\boldsymbol{\xi}$-re.
Konkrétabban
a sztochasztikus gradienscsökkenés frissítését tekintjük:

$$\mathbf{x}_{t+1} = \mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}),$$

ahol $f(\boldsymbol{\xi}_t, \mathbf{x})$
a célfüggvény
a $t$ lépésben valamely eloszlásból
mintavételezett $\boldsymbol{\xi}_t$ tanítási példa tekintetében, és $\mathbf{x}$ a modell paramétere.
Jelöljük

$$R(\mathbf{x}) = E_{\boldsymbol{\xi}}[f(\boldsymbol{\xi}, \mathbf{x})]$$

a várható kockázatot, és $R^*$-szal annak $\mathbf{x}$ tekintetében vett minimumát. Legyen $\mathbf{x}^*$ a minimalizáló (feltételezzük, hogy létezik az $\mathbf{x}$ értelmezési tartományán belül). Ebben az esetben nyomon követhetjük az aktuális $\mathbf{x}_t$ paraméter és a $\mathbf{x}^*$ kockázatminimalizáló közötti távolságot a $t$ időpontban, és megvizsgálhatjuk, hogy az idő múlásával javul-e:

$$\begin{aligned}    &\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \\ =& \|\mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}) - \mathbf{x}^*\|^2 \\    =& \|\mathbf{x}_{t} - \mathbf{x}^*\|^2 + \eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 - 2 \eta_t    \left\langle \mathbf{x}_t - \mathbf{x}^*, \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\right\rangle.   \end{aligned}$$
:eqlabel:`eq_sgd-xt+1-xstar`

Feltételezzük, hogy a sztochasztikus gradiens $\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})$ $\ell_2$-normája egy $L$ konstanssal korlátos, ebből következik:

$$\eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 \leq \eta_t^2 L^2.$$
:eqlabel:`eq_sgd-L`


Elsősorban az érdekel minket, hogyan változik $\mathbf{x}_t$ és $\mathbf{x}^*$ közötti távolság *várható értékben*. Valójában egy adott lépéssorozatnál a távolság növekedhet is, attól függően, hogy melyik $\boldsymbol{\xi}_t$-vel találkozunk. Ezért szükségünk van a belső szorzat korlátjára.
Mivel bármely konvex $f$ függvényre teljesül, hogy
$f(\mathbf{y}) \geq f(\mathbf{x}) + \langle f'(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$
minden $\mathbf{x}$-re és $\mathbf{y}$-ra,
konvexitás alapján:

$$f(\boldsymbol{\xi}_t, \mathbf{x}^*) \geq f(\boldsymbol{\xi}_t, \mathbf{x}_t) + \left\langle \mathbf{x}^* - \mathbf{x}_t, \partial_{\mathbf{x}} f(\boldsymbol{\xi}_t, \mathbf{x}_t) \right\rangle.$$
:eqlabel:`eq_sgd-f-xi-xstar`

A :eqref:`eq_sgd-L` és :eqref:`eq_sgd-f-xi-xstar` egyenlőtlenségeket behelyettesítve a :eqref:`eq_sgd-xt+1-xstar`-be, korlátot kapunk a $t+1$ időpontbeli paraméterek távolságára:

$$\|\mathbf{x}_{t} - \mathbf{x}^*\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \geq 2 \eta_t (f(\boldsymbol{\xi}_t, \mathbf{x}_t) - f(\boldsymbol{\xi}_t, \mathbf{x}^*)) - \eta_t^2 L^2.$$
:eqlabel:`eqref_sgd-xt-diff`

Ez azt jelenti, hogy addig haladunk, amíg az aktuális veszteség és az optimális veszteség különbsége meghaladja $\eta_t L^2/2$-t. Mivel ez a különbség szükségszerűen nullához tart, következik, hogy a $\eta_t$ tanulási rátának szintén el kell *tűnnie*.

Ezután a :eqref:`eqref_sgd-xt-diff` várható értékét vesszük. Ez adja:

$$E\left[\|\mathbf{x}_{t} - \mathbf{x}^*\|^2\right] - E\left[\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2\right] \geq 2 \eta_t [E[R(\mathbf{x}_t)] - R^*] -  \eta_t^2 L^2.$$

Az utolsó lépés a $t \in \{1, \ldots, T\}$-re vonatkozó egyenlőtlenségek összegzése. Mivel az összeg teleszkopikusan egyszerűsödik, és az alsó tagot elhagyva kapjuk:

$$\|\mathbf{x}_1 - \mathbf{x}^*\|^2 \geq 2 \left (\sum_{t=1}^T   \eta_t \right) [E[R(\mathbf{x}_t)] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$
:eqlabel:`eq_sgd-x1-xstar`

Megjegyezzük, hogy $\mathbf{x}_1$ adott, így a várható értéket el lehet hagyni. Végül definiáljuk:

$$\bar{\mathbf{x}} \stackrel{\textrm{def}}{=} \frac{\sum_{t=1}^T \eta_t \mathbf{x}_t}{\sum_{t=1}^T \eta_t}.$$

Mivel

$$E\left(\frac{\sum_{t=1}^T \eta_t R(\mathbf{x}_t)}{\sum_{t=1}^T \eta_t}\right) = \frac{\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)]}{\sum_{t=1}^T \eta_t} = E[R(\mathbf{x}_t)],$$

a Jensen-egyenlőtlenség alapján (ahol $i=t$, $\alpha_i = \eta_t/\sum_{t=1}^T \eta_t$ a :eqref:`eq_jensens-inequality`-ben) és $R$ konvexitásából következik, hogy $E[R(\mathbf{x}_t)] \geq E[R(\bar{\mathbf{x}})]$, tehát:

$$\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)] \geq \sum_{t=1}^T \eta_t  E\left[R(\bar{\mathbf{x}})\right].$$

Ezt behelyettesítve a :eqref:`eq_sgd-x1-xstar` egyenlőtlenségbe, kapjuk a korlátot:

$$
\left[E[\bar{\mathbf{x}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t},
$$

ahol $r^2 \stackrel{\textrm{def}}{=} \|\mathbf{x}_1 - \mathbf{x}^*\|^2$ a kezdeti paraméterválasztás és a végeredmény közötti távolságra vonatkozó korlát. Röviden: a konvergencia sebessége attól függ, hogy a sztochasztikus gradiens normája mennyire korlátos ($L$), és hogy a kezdeti paraméterérték mennyire távol van az optimalitástól ($r$). Megjegyzendő, hogy a korlát $\bar{\mathbf{x}}$-re vonatkozik, nem $\mathbf{x}_T$-re. Ennek oka, hogy $\bar{\mathbf{x}}$ az optimalizálási útvonal simított változata.
Ha $r$, $L$ és $T$ ismert, a tanulási ráta $\eta = r/(L \sqrt{T})$-vel választható. Ez az $rL/\sqrt{T}$ felső korlátot adja. Vagyis $\mathcal{O}(1/\sqrt{T})$ sebességgel konvergálunk az optimális megoldáshoz.


## Sztochasztikus gradiensek és véges minták

Eddig kissé lazán kezeltük a sztochasztikus gradienscsökkenésről szóló fejtegetést. Azt feltételeztük, hogy $x_i$ példányokat, általában $y_i$ címkékkel, valamely $p(x, y)$ eloszlásból mintavételezzük, és ezeket felhasználjuk a modell paramétereinek valamilyen módon történő frissítéséhez. Különösen a véges mintaméret esetén egyszerűen azt állítottuk, hogy a diszkrét $p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$ eloszlás
néhány $\delta_{x_i}$ és $\delta_{y_i}$ függvényre
lehetővé teszi a sztochasztikus gradienscsökkenés elvégzését.

Ez azonban nem igazán az, amit tettünk. Az aktuális szakasz játékpéldáiban egyszerűen zajt adtunk egy amúgy nem sztochasztikus gradienhez, vagyis úgy tettünk, mintha $(x_i, y_i)$ párjaink lennének. Kiderül, hogy ez itt indokolt (a részletes tárgyalásért lásd a gyakorló feladatokat). Aggasztóbb, hogy az összes korábbi tárgyalásunkban egyértelműen nem ezt tettük. Ehelyett pontosan *egyszer* iteráltunk az összes példán. Annak megértéséhez, hogy ez miért előnyösebb, vizsgáljuk meg a fordítottját: $n$ megfigyelést mintavételezünk a diszkrét eloszlásból *visszatevéssel*. Az $i$ elem véletlenszerű kiválasztásának valószínűsége $1/n$. Tehát *legalább egyszer* kiválasztani:

$$P(\textrm{choose~} i) = 1 - P(\textrm{omit~} i) = 1 - (1-1/n)^n \approx 1-e^{-1} \approx 0.63.$$

Hasonló logika alapján annak valószínűsége, hogy egy mintát (vagyis tanítási példát) *pontosan egyszer* veszünk ki:

$${n \choose 1} \frac{1}{n} \left(1-\frac{1}{n}\right)^{n-1} = \frac{n}{n-1} \left(1-\frac{1}{n}\right)^{n} \approx e^{-1} \approx 0.37.$$

A visszatevéssel való mintavételezés nagyobb varianciát és kisebb adathatékonyságot eredményez a *visszatevés nélküli* mintavételezéshez képest. Ezért a gyakorlatban az utóbbit alkalmazzuk (és ez az alapértelmezett választás az egész könyvben). Végül megjegyezzük, hogy a tanítóhalmazon végrehajtott ismételt átmenetek azt egy *különböző* véletlenszerű sorrendben járják be.


## Összefoglalás

* Konvex problémáknál bebizonyítható, hogy a tanulási ráták széles körű megválasztása esetén a sztochasztikus gradienscsökkenés konvergál az optimális megoldáshoz.
* A mélytanulásban ez általában nem igaz. A konvex problémák elemzése azonban hasznos betekintést nyújt az optimalizálás megközelítéséhez: a tanulási rátát fokozatosan, de nem túl gyorsan kell csökkenteni.
* Problémák lépnek fel, ha a tanulási ráta túl kicsi vagy túl nagy. A gyakorlatban a megfelelő tanulási rátát általában csak több kísérlet után találják meg.
* Ha a tanítóhalmazban több példa van, minden gradienscsökkenés iteráció kiszámítása drágább, ezért ezekben az esetekben a sztochasztikus gradienscsökkenés az előnyösebb.
* Az optimális garantált konvergencia nemkonvex esetekben általában nem elérhető, mivel az ellenőrizendő lokális minimumok száma exponenciálisan nagy lehet.



## Gyakorló feladatok

1. Kísérletezz különböző tanulási ráta ütemezőkkel a sztochasztikus gradienscsökkenéshez, és különböző számú iterációval. Különösen ábrázold az optimális megoldástól való $(0, 0)$ távolságot az iterációk számának függvényében.
1. Bizonyítsd be, hogy az $f(x_1, x_2) = x_1^2 + 2 x_2^2$ függvény esetén normális zajt hozzáadva a gradienhez egyenértékű az $f(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$ veszteségfüggvény minimalizálásával, ahol $\mathbf{x}$ normális eloszlásból van mintavételezve.
1. Hasonlítsd össze a sztochasztikus gradienscsökkenés konvergenciáját, ha visszatevéssel és visszatevés nélkül mintavételez az $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ halmazból.
1. Hogyan módosítanád a sztochasztikus gradienscsökkenés megoldót, ha egyes gradiensek (pontosabban a hozzájuk tartozó koordináták) következetesen nagyobbak lennének az összes többi gradiensnél?
1. Tegyük fel, hogy $f(x) = x^2 (1 + \sin x)$. Hány lokális minimuma van $f$-nek? Megváltoztatható-e $f$ úgy, hogy minimalizálásához az összes lokális minimumot ki kell értékelni?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/352)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/497)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1067)
:end_tab:
