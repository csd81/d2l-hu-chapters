```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Lineáris regresszió
:label:`sec_linear_regression`

A *regressziós* problémák minden olyan esetben felmerülnek, amikor numerikus értéket szeretnénk előrejelezni.
Tipikus példák az árak előrejelzése (lakások, részvények stb. esetén),
a tartózkodási idő becslése (kórházi betegeknél),
a kereslet előrejelzése (kiskereskedelmi értékesítéshez) és még sok más.
Nem minden előrejelzési probléma klasszikus regresszió.
Később bevezetjük az osztályozási problémákat,
ahol a cél egy kategóriahalmazhoz való tartozás előrejelzése.

Futó példaként tegyük fel, hogy
házak árát (dollárban) szeretnénk megbecsülni
alapterületük (négyzetméterben) és koruk (években) alapján.
A házárak előrejelzéséhez szükségünk van adatokra,
beleértve az eladási árat, az alapterületet és a kort
minden egyes háznál.
A gépi tanulás terminológiájában
az adathalmazt *tanítóadatnak* vagy *tanítóhalmaznak* nevezzük,
minden sort (amely egyetlen eladás adatait tartalmazza)
*példának* (vagy *adatpontnak*, *példánynak*, *mintának*) hívunk.
Azt, amit előre akarunk jelezni (az árat),
*címkének* (vagy *célnak*) nevezzük.
A változókat (kort és alapterületet),
amelyeken az előrejelzések alapulnak,
*jellemzőknek* (vagy *kovariánsoknak*) hívjuk.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np
import time
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import numpy as np
import time
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
import numpy as np
import time
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from jax import numpy as jnp
import math
import time
```

## Alapok

A *lineáris regresszió* egyszerre a legegyszerűbb
és a legelterjedtebb a regressziós problémák kezelésére szolgáló
standard eszközök között.
A 19. század elejéig visszanyúlva :cite:`Legendre.1805,Gauss.1809`,
a lineáris regresszió néhány egyszerű feltételezésből indul ki.
Először is feltesszük, hogy a jellemzők $\mathbf{x}$ és a cél $y$
közötti kapcsolat közelítőleg lineáris,
azaz a feltételes várható érték $E[Y \mid X=\mathbf{x}]$
kifejezhető a $\mathbf{x}$ jellemzők súlyozott összegeként.
Ez a felállítás megengedi, hogy a célérték
az észlelési zajból adódóan eltérjen a várható értékétől.
Ezután feltehetjük, hogy az ilyen zaj
jól viselkedik, és Gauss-eloszlást követ.
Általában $n$-nel jelöljük az adathalmazban lévő példák számát.
A minták és célok felsorolásához felső indexet,
a koordináták indexeléséhez alsó indexet használunk.
Konkrétabban,
$\mathbf{x}^{(i)}$ jelöli az $i$-edik mintát,
és $x_j^{(i)}$ annak $j$-edik koordinátáját.

### Modell
:label:`subsec_linear_model`

Minden megoldás szívében egy modell áll,
amely leírja, hogyan alakíthatók a jellemzők
a cél becslésévé.
A linearitás feltevése azt jelenti, hogy
a cél (ár) várható értéke kifejezhető
a jellemzők (alapterület és kor) súlyozott összegeként:

$$\textrm{price} = w_{\textrm{area}} \cdot \textrm{area} + w_{\textrm{age}} \cdot \textrm{age} + b.$$
:eqlabel:`eq_price-area`

Itt $w_{\textrm{area}}$ és $w_{\textrm{age}}$
*súlyoknak* nevezzük, $b$ pedig *eltolásnak*
(vagy *tengelymetszet-tagnak*).
A súlyok határozzák meg, hogy az egyes jellemzők
mennyire befolyásolják az előrejelzésünket.
Az eltolás határozza meg a becslés értékét
akkor, ha minden jellemző nulla.
Bár soha nem fogunk látni nullás alapterületű, újonnan épített házat,
mégis szükségünk van az eltolásra, mert nélküle
csak az origón átmenő egyeneseket tudnánk kifejezni
(nem az összes lineáris függvényt).
Szigorúan szólva, :eqref:`eq_price-area` a bemeneti jellemzők *affin transzformációja*, amelyet a jellemzők *lineáris transzformációja* (súlyozott összeg formájában) és egy *eltolás* (a hozzáadott eltolás révén) jellemez.
Adott adathalmaz esetén célunk
a $\mathbf{w}$ súlyok és $b$ eltolás megválasztása úgy,
hogy átlagosan a modell előrejelzései
a lehető legjobban illeszkedjenek az adatokban
megfigyelt valódi árakhoz.

Azokban a tudományágakban, ahol csak néhány jellemzős
adathalmazokkal foglalkoznak,
a modellek explicit, hosszú formában való kifejezése,
mint :eqref:`eq_price-area`-ban, megszokott.
A gépi tanulásban általában
magas dimenziójú adathalmazokkal dolgozunk,
ahol kényelmesebb a kompakt lineáris algebrai jelölés.
Ha bemeneteink $d$ jellemzőből állnak,
minden jellemzőnek adhatunk egy indexet ($1$ és $d$ között),
és az előrejelzésünket $\hat{y}$-nal fejezhetjük ki
(általában a „kalap" szimbólum becslést jelöl):

$$\hat{y} = w_1  x_1 + \cdots + w_d  x_d + b.$$

Az összes jellemzőt egy $\mathbf{x} \in \mathbb{R}^d$ vektorba,
az összes súlyt egy $\mathbf{w} \in \mathbb{R}^d$ vektorba gyűjtve,
a modellünket tömören kifejezhetjük
$\mathbf{w}$ és $\mathbf{x}$ skaláris szorzataként:

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b.$$
:eqlabel:`eq_linreg-y`

:eqref:`eq_linreg-y`-ban a $\mathbf{x}$ vektor
egyetlen példa jellemzőinek felel meg.
Kényelmes lesz az $n$ példából álló
teljes adathalmazunk jellemzőire
a *tervmátrix* $\mathbf{X} \in \mathbb{R}^{n \times d}$ révén hivatkozni.
Itt $\mathbf{X}$ minden példához tartalmaz egy sort
és minden jellemzőhöz egy oszlopot.
A $\mathbf{X}$ jellemzőgyűjteményhez tartozó
$\hat{\mathbf{y}} \in \mathbb{R}^n$ előrejelzések
mátrix–vektor szorzattal fejezhetők ki:

$${\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b,$$
:eqlabel:`eq_linreg-y-vec`

ahol az összegzésnél kiterjesztés (:numref:`subsec_broadcasting`) érvényesül.
Adott tanítóadatbázis $\mathbf{X}$ jellemzőivel
és a megfelelő (ismert) $\mathbf{y}$ címkékkel,
a lineáris regresszió célja megtalálni
a $\mathbf{w}$ súlyvektort és a $b$ eltolás tagot,
hogy a $\mathbf{X}$-ével azonos eloszlásból vett
új adatpélda esetén
annak címkéje (várható értékben)
a lehető legkisebb hibával legyen megjósolható.

Még ha el is hisszük, hogy az $\mathbf{x}$ alapján
$y$ legjobb modellje lineáris,
nem várhatjuk el, hogy $n$ valós példán
$y^{(i)}$ pontosan egyenlő legyen $\mathbf{w}^\top \mathbf{x}^{(i)}+b$-vel
minden $1 \leq i \leq n$-re.
Például bármilyen eszközzel is mérjük
az $\mathbf{X}$ jellemzőket és $\mathbf{y}$ címkéket,
kis mérési hiba mindig előfordulhat.
Ezért, még ha meg is vagyunk győződve
az alapvető összefüggés linearitásáról,
zaj tagot is beépítünk az ilyen hibák figyelembevételéhez.

Mielőtt a legjobb *paraméterek*
(vagy *modellparaméterek*) $\mathbf{w}$ és $b$ keresésébe fognánk,
még két dologra lesz szükségünk:
(i) egy adott modell minőségének mértékére;
és (ii) egy eljárásra a modell frissítésére a minőség javítása érdekében.

### Veszteségfüggvény
:label:`subsec_linear-regression-loss-function`

Természetesen a modell adatokhoz illesztéséhez
szükséges egy *illeszkedési mérték*
(vagy annak megfordítva, *nem-illeszkedési mérték*)
meghatározása.
A *veszteségfüggvények* a cél *valódi* és *előrejelzett*
értékei közötti távolságot mérik.
A veszteség általában egy nemnegatív szám,
ahol a kisebb értékek jobbak,
és a tökéletes előrejelzés vesztesége 0.
Regressziós problémáknál a legelterjedtebb veszteségfüggvény
a négyzetes hiba.
Ha az $i$-edik példára adott előrejelzésünk $\hat{y}^{(i)}$,
és a megfelelő valódi címke $y^{(i)}$,
a *négyzetes hiba* a következő:

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$
:eqlabel:`eq_mse`

A $\frac{1}{2}$ konstans valójában nem változtat semmin,
de jelölési szempontból hasznos,
mivel kiesik, ha a veszteség deriváltját vesszük.
Mivel a tanítóhalmazt adottnak tekintjük
(így nem áll módunkban megváltoztatni),
az empirikus hiba kizárólag a modellparaméterek függvénye.
:numref:`fig_fit_linreg`-ben egy egydimenziós bemenetre
illesztett lineáris regressziós modellt vizualizálunk.

![Lineáris regressziós modell illesztése egydimenziós adatokra.](../img/fit-linreg.svg)
:label:`fig_fit_linreg`

Figyeljük meg, hogy a $\hat{y}^{(i)}$ becslések
és $y^{(i)}$ célok közötti nagy különbségek
még nagyobb hozzájáruláshoz vezetnek a veszteségben,
a négyzetes alak miatt
(ez a négyzetes jelleg kétélű fegyver lehet: miközben arra ösztönzi a modellt, hogy kerülje a nagy hibákat,
túlzott érzékenységet is okozhat atipikus adatokra).
Az $n$ példából álló teljes adathalmaz modellminőségének méréséhez
egyszerűen átlagoljuk (vagy ekvivalensen összegezzük)
a tanítóhalmaz veszteségeit:

$$L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

A modell tanításakor olyan paramétereket ($\mathbf{w}^*, b^*$)
keresünk, amelyek minimalizálják az összes
tanítópéldán mért teljes veszteséget:

$$\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\  L(\mathbf{w}, b).$$

### Analitikus megoldás

A legtöbb modellel ellentétben, amelyekkel foglalkozni fogunk,
a lineáris regresszió meglepően egyszerű
optimalizálási feladatot kínál.
Különösen, az optimális paramétereket
(a tanítóadatokon értékelve)
analitikusan megtalálhatjuk egy egyszerű képlettel.
Először beolvaszthatjuk a $b$ eltolást a $\mathbf{w}$ paraméterbe
oly módon, hogy a tervmátrixhoz egy csupa 1-esből álló oszlopot fűzünk.
Ekkor az előrejelzési feladatunk $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$ minimalizálása.
Mindaddig, amíg a $\mathbf{X}$ tervmátrix teljes rangú
(egyik jellemző sem lineárisan függő a többitől),
a veszteségfelszínen egyetlen kritikus pont lesz,
és ez a teljes tartomány feletti veszteség minimumának felel meg.
A veszteség $\mathbf{w}$ szerinti deriváltját véve
és nullává téve kapjuk:

$$\begin{aligned}
    \partial_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 =
    2 \mathbf{X}^\top (\mathbf{X} \mathbf{w} - \mathbf{y}) = 0
    \textrm{ és ebből }
    \mathbf{X}^\top \mathbf{y} = \mathbf{X}^\top \mathbf{X} \mathbf{w}.
\end{aligned}$$

$\mathbf{w}$-re megoldva megkapjuk az optimalizálási feladat
optimális megoldását.
Ne feledjük, hogy ez a megoldás

$$\mathbf{w}^* = (\mathbf X^\top \mathbf X)^{-1}\mathbf X^\top \mathbf{y}$$

csak akkor egyértelmű,
ha $\mathbf X^\top \mathbf X$ mátrix invertálható,
azaz ha a tervmátrix oszlopai
lineárisan független :cite:`Golub.Van-Loan.1996`.

Bár az egyszerű feladatok, mint a lineáris regresszió,
analitikus megoldást engednek meg,
ne szokjunk hozzá az ilyen szerencséhez.
Bár az analitikus megoldások szép matematikai elemzést tesznek lehetővé,
az analitikus megoldás követelménye annyira korlátozó,
hogy szinte a deep learning összes izgalmas aspektusát kizárná.

### Minibatch sztochasztikus gradiens módszer

Szerencsére, még azokban az esetekben is, amikor a modelleket
nem tudjuk analitikusan megoldani,
a modelleket sok esetben hatékonyan be tudjuk tanítani a gyakorlatban.
Sőt, sok feladatnál azok a nehezen optimalizálható modellek
annyival jobbaknak bizonyulnak,
hogy érdemes rájönni, hogyan kell őket betanítani.

Szinte minden deep learning modell optimalizálásának kulcstechnikája,
amelyre a könyvben végig támaszkodunk,
a hiba iteratív csökkentéséből áll
a paraméterek frissítésével abba az irányba,
amely fokozatosan csökkenti a veszteségfüggvényt.
Ezt az algoritmust *gradiens módszernek* (gradienscsökkenés) nevezzük.

A gradiens módszer legalapvetőbb alkalmazása
a veszteségfüggvény deriváltjának kiszámítása,
amely az adathalmazban szereplő összes példán
kiszámított veszteségek átlaga.
A gyakorlatban ez rendkívül lassú lehet:
egyetlen frissítés előtt végig kell mennünk az egész adathalmazon,
még akkor is, ha a frissítési lépések nagyon hatékonyak lehetnek :cite:`Liu.Nocedal.1989`.
Ráadásul, ha a tanítóadatokban sok a redundancia,
a teljes frissítés előnye korlátozott.

A másik véglet az, hogy egyszerre csak egyetlen példát
veszünk figyelembe, és egy megfigyelés alapján frissítünk.
Az így kapott algoritmus, a *sztochasztikus gradiens módszer* (SGD)
hatékony stratégia lehet :cite:`Bottou.2010`, akár nagy adathalmazok esetén is.
Sajnos az SGD-nek számítási és statisztikai hátrányai is vannak.
Az egyik probléma abból ered, hogy a processzorok
sokkal gyorsabban végeznek szorzásokat és összeadásokat,
mint ahogy az adatokat a főmemóriából
a processzor gyorsítótárába mozgatják.
Egy mátrix–vektor szorzás elvégzése
nagyságrendekkel hatékonyabb,
mint a vektorok elemenkénti összeadásainak megfelelő száma.
Ez azt jelenti, hogy egy mintát egyszerre feldolgozni
sokkal tovább tarthat, mint egy teljes batch-et.
A második probléma, hogy egyes rétegek,
például a batchnormalizáció (lásd :numref:`sec_batch_norm`),
csak akkor működnek jól, ha egynél több megfigyeléshez van hozzáférésünk.

Mindkét problémára az a megoldás, hogy köztes stratégiát válasszunk:
ahelyett, hogy teljes batch-et vagy egyszerre csak egyetlen mintát vegyünk,
egy *mini-batch* megfigyelést veszünk :cite:`Li.Zhang.Chen.ea.2014`.
Az adott mini-batch méretének megválasztása számos tényezőtől függ,
például a memória mennyiségétől, a gyorsítók számától,
a rétegek választásától és a teljes adathalmaz méretétől.
Mindezek ellenére a 32 és 256 közötti érték,
lehetőleg egy nagy $2$-es hatvány többszöröse, jó kiindulópont.
Ez elvezet minket a *mini-batch sztochasztikus gradiens módszerhez*.

Legegyszerűbb formájában minden $t$ iterációban
először véletlenszerűen kiveszünk egy $\mathcal{B}_t$ mini-batch-et,
amely rögzített számú $|\mathcal{B}|$ tanítópéldából áll.
Ezután kiszámítjuk az átlagos veszteség deriváltját
(gradienst) a mini-batch felett a modellparaméterek szerint.
Végül a gradienst megszorozzuk egy előre meghatározott
kis pozitív $\eta$ értékkel,
amelyet *tanulási rátának* nevezünk,
és a kapott tagot kivonjuk az aktuális paraméterértékekből.
A frissítés a következőképpen fejezhető ki:

$$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).$$

Összefoglalva, a mini-batch SGD a következőképpen halad:
(i) a modellparaméterek értékeit inicializáljuk, általában véletlenszerűen;
(ii) iteratívan véletlenszerű mini-batch-eket veszünk ki az adatokból,
és a negatív gradiens irányában frissítjük a paramétereket.
Négyzetes veszteségek és affin transzformációk esetén
ennek zárt alakú kibontása van:

$$\begin{aligned} \mathbf{w} & \leftarrow \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) && = \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)\\ b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \partial_b l^{(i)}(\mathbf{w}, b) &&  = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}$$
:eqlabel:`eq_linreg_batch_update`

Mivel $\mathcal{B}$ mini-batch-et veszünk ki,
annak $|\mathcal{B}|$ méretével kell normalizálnunk.
A mini-batch mérete és a tanulási ráta felhasználó által meghatározott.
Az ilyen, a tanítási ciklus során nem frissített,
hangolható paramétereket *hiperparamétereknek* nevezzük.
Ezek számos technikával automatikusan hangolhatók, például Bayes-optimalizálással
:cite:`Frazier.2018`. Végül a megoldás minőségét
általában egy külön *validációs adathalmazon* (vagy *validációs készleten*) értékelik.

Miután egy előre meghatározott számú iterációig tanítottunk
(vagy valamilyen más leállási feltétel teljesül),
rögzítjük a becsült modellparamétereket,
amelyeket $\hat{\mathbf{w}}, \hat{b}$-vel jelölünk.
Megjegyezzük, hogy még ha a függvényünk valóban lineáris és zajmentes is,
ezek a paraméterek nem lesznek a veszteség pontos minimalizálói, és még csak nem is determinisztikusak.
Bár az algoritmus lassan konvergál a minimalizálók felé,
véges lépésszámban általában nem találja meg őket pontosan.
Ráadásul a paraméterek frissítésére használt
$\mathcal{B}$ mini-batch-eket véletlenszerűen választják ki.
Ez megszakítja a determinizmust.

A lineáris regresszió egy globális minimummal rendelkező
tanulási feladat
(feltéve, hogy $\mathbf{X}$ teljes rangú,
azaz $\mathbf{X}^\top \mathbf{X}$ invertálható).
A mély hálózatok veszteségfelszínein azonban sok nyeregpont és minimum található.
Szerencsére általában nem fontos, hogy
a paraméterek pontos halmazát megtaláljuk,
csupán olyan paraméterkészletet keresünk,
amely pontos előrejelzésekhez (és így alacsony veszteséghez) vezet.
A gyakorlatban a deep learning szakemberek
ritkán küzdenek azért, hogy megtalálják
a *tanítóhalmazokon* a veszteséget minimalizáló paramétereket
:cite:`Izmailov.Podoprikhin.Garipov.ea.2018,Frankle.Carbin.2018`.
A nehezebb feladat az, hogy olyan paramétereket találjunk,
amelyek korábban nem látott adatokra is pontos előrejelzést adnak,
ezt a kihívást *általánosításnak* nevezzük.
Ezekre a témákra a könyv folyamán visszatérünk.

### Előrejelzések

Az adott $\hat{\mathbf{w}}^\top \mathbf{x} + \hat{b}$ modellel
már *előrejelzéseket* készíthetünk új példákra,
például egy korábban nem látott ház
eladási árának becslésére
az $x_1$ alapterület és $x_2$ kor alapján.
A deep learning szakemberek az előrejelzési fázist *inferenciának* kezdték hívni,
de ez kissé félrevezető — az *inferencia* tágabb értelemben
bármilyen bizonyítékon alapuló következtetésre vonatkozik,
beleértve a paraméterértékeket
és egy nem látott példa valószínű címkéjét is.
Ha valami, a statisztikai irodalomban
az *inferencia* inkább paraméterbecslést jelöl,
és ez a terminológiai átfedés szükségtelen zavart okoz,
amikor deep learning szakemberek statisztikusokkal beszélnek.
A következőkben ahol lehetséges, az *előrejelzés* szót fogjuk használni.



## Vektorizáció a sebesség érdekében

A modellek tanításakor általában egész mini-batch-nyi példát
szeretnénk egyszerre feldolgozni.
Ennek hatékony elvégzéséhez szükség van arra, hogy (**mi**) (~~ne~~)
(**vektorizáljuk a számításokat, és gyors lineáris algebrai könyvtárakat
használjunk a drága Python for-ciklusok helyett.**)

Hogy megértsük, miért számít ez ennyire,
(**vizsgáljunk meg két módszert vektorok összeadására.**)
Kezdetnek hozzunk létre két, 10 000 dimenziós,
csupa 1-esből álló vektort.
Az első módszerben Python for-ciklussal
járjuk végig a vektorokat.
A másodikban egyetlen `+` hívásra támaszkodunk.

```{.python .input}
%%tab all
n = 10000
a = d2l.ones(n)
b = d2l.ones(n)
```

Most mérhetjük az egyes megközelítések teljesítményét.
Először [**összeadjuk őket koordinátánként,
for-ciklussal.**]

```{.python .input}
%%tab mxnet, pytorch
c = d2l.zeros(n)
t = time.time()
for i in range(n):
    c[i] = a[i] + b[i]
f'{time.time() - t:.5f} sec'
```

```{.python .input}
%%tab tensorflow
c = tf.Variable(d2l.zeros(n))
t = time.time()
for i in range(n):
    c[i].assign(a[i] + b[i])
f'{time.time() - t:.5f} sec'
```

```{.python .input}
%%tab jax
# A JAX tömbök megváltoztathatatlanok: létrehozásuk után tartalmuk
# nem módosítható. Egyes elemek frissítéséhez a JAX indexelt
# frissítési szintaxist biztosít, amely egy módosított másolatot ad vissza
c = d2l.zeros(n)
t = time.time()
for i in range(n):
    c = c.at[i].set(a[i] + b[i])
f'{time.time() - t:.5f} sec'
```

(**Alternatívaként a túlterhelt `+` operátorra támaszkodunk az elemenkénti összeg kiszámításához.**)

```{.python .input}
%%tab all
t = time.time()
d = a + b
f'{time.time() - t:.5f} sec'
```

A második módszer drámaian gyorsabb az elsőnél.
A kód vektorizálása gyakran nagyságrendnyi sebességnövekedést eredményez.
Ráadásul a matematika nagy részét a könyvtárra bízzuk,
így kevesebb számítást kell saját magunknak elvégeznünk,
csökkentve a hibalehetőségeket és növelve a kód hordozhatóságát.


## A normális eloszlás és a négyzetes veszteség
:label:`subsec_normal_distribution_and_squared_loss`

Eddig meglehetősen funkcionális motivációt adtunk
a négyzetes veszteség célfüggvényre:
az optimális paraméterek visszaadják
az $E[Y\mid X]$ feltételes várható értéket,
ha az alapul szolgáló minta valóban lineáris,
és a veszteség nagy büntetést szab ki a kiugró értékekre.
A négyzetes veszteség célfüggvényre formálisabb motivációt is adhatunk
a zaj eloszlásáról tett valószínűségi feltételezések révén.

A lineáris regressziót a 19. század fordulóján találták fel.
Bár régóta vita tárgya, hogy Gauss vagy Legendre
gondolta-e ki először az ötletet,
Gauss fedezte fel a normális eloszlást is
(amelyet *Gauss-eloszlásnak* is neveznek).
Kiderül, hogy a normális eloszlás
és a négyzetes veszteséggel rendelkező lineáris regresszió
mélyebb kapcsolatban áll egymással, mint a közös eredet.

Kezdjük azzal, hogy felidézzük a $\mu$ várható értékű és $\sigma^2$ varianciájú
(szórású: $\sigma$) normális eloszlást,
amelyet a következőképpen adunk meg:

$$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right).$$

Az alábbiakban [**definiálunk egy függvényt a normális eloszlás kiszámítására**].

```{.python .input}
%%tab all
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    if tab.selected('jax'):
        return p * jnp.exp(-0.5 * (x - mu)**2 / sigma**2)
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        return p * np.exp(-0.5 * (x - mu)**2 / sigma**2)
```

Most (**vizualizálhatjuk a normális eloszlásokat**).

```{.python .input}
%%tab mxnet
# Vizualizációhoz ismét NumPy-t használunk
x = np.arange(-7, 7, 0.01)

# Várható érték és szórás párok
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x.asnumpy(), [normal(x, mu, sigma).asnumpy() for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

```{.python .input}

%%tab pytorch, tensorflow, jax
if tab.selected('jax'):
    # Vizualizációhoz JAX NumPy-t használunk
    x = jnp.arange(-7, 7, 0.01)
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    # Vizualizációhoz ismét NumPy-t használunk
    x = np.arange(-7, 7, 0.01)

# Várható érték és szórás párok
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

Figyeljük meg, hogy a várható érték megváltoztatása
eltolást jelent az $x$-tengely mentén,
a variancia növelése pedig szétteríti az eloszlást,
csökkentve csúcsát.

A lineáris regresszió négyzetes veszteséggel való motiválásának egyik módja
az a feltevés, hogy a megfigyelések zajos mérésekből erednek,
ahol a $\epsilon$ zaj a $\mathcal{N}(0, \sigma^2)$ normális eloszlást követi:

$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon \textrm{ where } \epsilon \sim \mathcal{N}(0, \sigma^2).$$

Így most felírhatjuk egy adott $\mathbf{x}$-re
egy adott $y$ *valószínűségét* (likelihood):

$$P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right).$$

Így a likelihood faktorizálódik.
A *maximális likelihood elve* szerint
a $\mathbf{w}$ és $b$ paraméterek legjobb értékei azok,
amelyek maximalizálják a teljes adathalmaz *likelihood-ját*:

$$P(\mathbf y \mid \mathbf X) = \prod_{i=1}^{n} p(y^{(i)} \mid \mathbf{x}^{(i)}).$$

Az egyenlőség azért teljesül, mert minden $(\mathbf{x}^{(i)}, y^{(i)})$ pár
egymástól függetlenül lett megfigyelt.
A maximális likelihood elve szerint kiválasztott
becslőket *maximális likelihood becslőknek* nevezzük.
Bár sok exponenciális függvény szorzatának maximalizálása
nehéznek tűnhet,
lényegesen egyszerűsíthetjük a dolgokat anélkül, hogy a célt megváltoztatnánk,
ha a likelihood logaritmusát maximalizáljuk.
Történelmi okokból az optimalizálást inkább minimalizálásként,
nem maximalizálásként fejezik ki.
Ezért, anélkül, hogy bármit is változtatnánk,
*minimalizálhatjuk* a *negatív log-likelihood-ot*,
amelyet a következőképpen fejezhetünk ki:

$$-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2.$$

Ha feltesszük, hogy $\sigma$ rögzített,
az első tagot figyelmen kívül hagyhatjuk,
mert nem függ $\mathbf{w}$-től vagy $b$-től.
A második tag megegyezik a korábban bevezetett
négyzetes hiba veszteséggel,
kivéve a $\frac{1}{\sigma^2}$ szorzótényezőt.
Szerencsére a megoldás $\sigma$-tól sem függ.
Ebből következik, hogy az átlagos négyzetes hiba minimalizálása
egyenértékű a lineáris modell maximális likelihood becslésével
additív Gauss-zaj feltételezése mellett.


## Lineáris regresszió mint neurális hálózat

Bár a lineáris modellek nem elég gazdagok
ahhoz, hogy kifejezzék az ebben a könyvben
bemutatandó sok bonyolult hálózatot,
a (mesterséges) neurális hálózatok elég gazdagok
ahhoz, hogy magukban foglalják a lineáris modelleket
mint olyan hálózatokat,
amelyekben minden jellemzőt egy bemeneti neuron képvisel,
és ezek mindegyike közvetlenül a kimenethez kapcsolódik.

:numref:`fig_single_neuron` egy neurális hálózatként
ábrázolja a lineáris regressziót.
Az ábra kiemeli a kapcsolódási mintát,
például hogy minden bemenet hogyan kapcsolódik a kimenethez,
de nem mutatja a súlyok vagy eltolások konkrét értékeit.

![A lineáris regresszió egyetlen rétegű neurális hálózat.](../img/singleneuron.svg)
:label:`fig_single_neuron`

A bemenetek $x_1, \ldots, x_d$.
A $d$-t *bemenetek számának*
vagy a bemeneti réteg *jellemzők dimenziójának* nevezzük.
A hálózat kimenete $o_1$.
Mivel csupán egyetlen numerikus értéket próbálunk megjósolni,
csak egy kimeneti neuronunk van.
Megjegyezzük, hogy a bemeneti értékek mind *adottak*.
Egyetlen *számított* neuron létezik.
Összefoglalva, a lineáris regressziót
egyetlen rétegből álló teljesen összekötött neurális hálózatnak tekinthetjük.
Sokkal több réteggel rendelkező hálózatokkal
is találkozunk majd a következő fejezetekben.

### Biológia

Mivel a lineáris regresszió megelőzi
a számítási idegtudományt,
talán anakronisztikusnak tűnhet
a lineáris regressziót neurális hálózatok szempontjából leírni.
Ennek ellenére természetes kiindulópontot kínált,
amikor Warren McCulloch és Walter Pitts
kibernetikusok és neurofiziológusok
elkezdték fejleszteni a mesterséges neuronok modelljeit.
Gondoljunk a :numref:`fig_Neuron`-ban szereplő
biológiai neuron leegyszerűsített képére,
amely *dendriteket* (bemeneti végpontok),
a *sejtmagot* (CPU), az *axont* (kimeneti vezék)
és az *axon végpontokat* (kimeneti terminálisok)
tartalmaz, amelyek lehetővé teszik a más neuronokhoz
való kapcsolódást *szinapszisok* révén.

![A valódi neuron (forrás: az Egyesült Államok Nemzeti Rákkutató Intézetének Surveillance, Epidemiology and End Results (SEER) programjának „Anatomy and Physiology" kiadványa).](../img/neuron.svg)
:label:`fig_Neuron`

A más neuronoktól (vagy környezeti érzékelőktől)
érkező $x_i$ információ a dendritekben fogadódik.
Különösen, ez az információ *szinaptikus súlyok* $w_i$ által súlyozódik,
meghatározva a bemenetek hatását,
pl. aktiváció vagy gátlás a $x_i w_i$ szorzat révén.
A több forrásból érkező súlyozott bemenetek
a sejtmagban aggregálódnak
$y = \sum_i x_i w_i + b$ súlyozott összegként,
amelyre esetleg valamilyen nemlineáris utófeldolgozás
is vonatkozhat a $\sigma(y)$ függvény révén.
Ez az információ ezután az axonon keresztül az axon végpontokhoz jut,
ahol eléri a célját
(pl. egy végrehajtószervet, mint egy izom)
vagy betáplálódik egy másik neuronba a dendritjein keresztül.

Kétségtelenül az a magasabb szintű gondolat,
hogy ilyen egységek sokasága kombinálható —
feltéve, hogy helyes kapcsolattal és tanulási algoritmussal rendelkeznek —
sokkal érdekesebb és összetettebb viselkedést produkálva,
mint amit egyetlen neuron ki tudna fejezni,
a valódi biológiai idegrendszerek tanulmányozásából ered.
Ugyanakkor a mai deep learning kutatások nagy részét
sokkal szélesebb forrás ihleti.
:citet:`Russell.Norvig.2016`-ra hivatkozunk,
akik rámutattak arra, hogy bár a repülőgépeket a madarak
talán *ihlethették*,
az ornitológia évszázadok óta nem az aeronautikai innováció
elsődleges hajtóereje.
Hasonlóképpen, a deep learning mai inspirációja
legalább egyenlő mértékben, ha nem nagyobb mértékben
érkezik a matematikából, a nyelvészetből, a pszichológiából,
a statisztikából, a számítástudományból és sok más területről.

## Összefoglalás

Ebben a részben bevezettük
a hagyományos lineáris regressziót,
ahol egy lineáris függvény paramétereit
a tanítóhalmazon mért négyzetes veszteség minimalizálásával választjuk meg.
Ezt a célfüggvényt is motiváltuk
mind gyakorlati szempontok,
mind pedig a lineáris regresszió
mint maximális likelihood becslés értelmezése révén
linearitás és Gauss-zaj feltételezése mellett.
A számítási szempontok és a statisztikai összefüggések megvitatása után
megmutattuk, hogyan fejezhetők ki ilyen lineáris modellek
egyszerű neurális hálózatokként, ahol a bemenetek
közvetlenül a kimenet(ek)hez kapcsolódnak.
Bár hamarosan teljesen elhagyjuk a lineáris modelleket,
ezek elegendők ahhoz, hogy bevezessük
az összes modellünk által igényelt összetevőket:
paraméteres formák, differenciálható célfüggvények,
optimalizálás mini-batch sztochasztikus gradiens módszerrel,
és végső soron kiértékelés korábban nem látott adatokon.



## Gyakorlatok

1. Tegyük fel, hogy van néhány adatunk $x_1, \ldots, x_n \in \mathbb{R}$. Célunk egy $b$ konstans megtalálása, amely minimalizálja $\sum_i (x_i - b)^2$-t.
    1. Keresd meg $b$ optimális értékének analitikus megoldását.
    1. Hogyan kapcsolódik ez a feladat és megoldása a normális eloszláshoz?
    1. Mi történik, ha a veszteséget $\sum_i (x_i - b)^2$-ről $\sum_i |x_i-b|$-re változtatjuk? Meg tudod találni $b$ optimális megoldását?
1. Bizonyítsd be, hogy az $\mathbf{x}^\top \mathbf{w} + b$ által kifejezhető affin függvények egyenértékűek az $(\mathbf{x}, 1)$ fölötti lineáris függvényekkel.
1. Tegyük fel, hogy $\mathbf{x}$ másodfokú függvényeit szeretnéd megtalálni, azaz $f(\mathbf{x}) = b + \sum_i w_i x_i + \sum_{j \leq i} w_{ij} x_{i} x_{j}$. Hogyan fogalmaznád meg ezt egy mély hálózatban?
1. Emlékezz arra, hogy a lineáris regressziós feladat megoldhatóságának egyik feltétele az volt, hogy a $\mathbf{X}^\top \mathbf{X}$ tervmátrix teljes rangú legyen.
    1. Mi történik, ha ez nem teljesül?
    1. Hogyan lehetne javítani? Mi történik, ha kis koordinátánkénti független Gauss-zajt adunk $\mathbf{X}$ összes eleméhez?
    1. Mi a $\mathbf{X}^\top \mathbf{X}$ tervmátrix várható értéke ebben az esetben?
    1. Mi történik a sztochasztikus gradiens módszerrel, ha $\mathbf{X}^\top \mathbf{X}$ nem teljes rangú?
1. Tegyük fel, hogy az additív $\epsilon$ zajt irányító zajmodell az exponenciális eloszlás. Azaz $p(\epsilon) = \frac{1}{2} \exp(-|\epsilon|)$.
    1. Írd fel az adatok negatív log-likelihood-ját a modell szerint: $-\log P(\mathbf y \mid \mathbf X)$.
    1. Találsz zárt alakú megoldást?
    1. Javasolj mini-batch sztochasztikus gradiens módszer algoritmust ennek a feladatnak a megoldásához. Mi mehet félre (tipp: mi történik az álló pont közelében, miközben folyamatosan frissítjük a paramétereket)? Meg tudod javítani?
1. Tegyük fel, hogy két rétegből álló neurális hálózatot szeretnénk tervezni két lineáris réteg összetételével. Azaz az első réteg kimenete a második réteg bemenete lesz. Miért nem működne egy ilyen naiv összetétel?
1. Mi történik, ha regressziót szeretnél használni házak vagy részvényárak valósághű árbecsléséhez?
    1. Mutasd meg, hogy az additív Gauss-zaj feltételezése nem megfelelő. Tipp: lehetnek-e negatív árak? Mi a helyzet a fluktuációkkal?
    1. Miért lenne sokkal jobb az árlogaritmusra való regresszió, azaz $y = \log \textrm{price}$?
    1. Mivel kell foglalkoznod filléres részvények esetén, azaz nagyon alacsony árú részvényeknél? Tipp: kereskedhetsz-e az összes lehetséges áron? Miért nagyobb probléma ez olcsó részvényeknél? További információkért tekintsd át a részvényopciók árazásának jeles Black–Scholes modelljét :cite:`Black.Scholes.1973`.
1. Tegyük fel, hogy regressziót szeretnénk használni a szupermarketben eladott almák *számának* becslésére.
    1. Mik a problémák a Gauss additív zajmodellel? Tipp: almát árul, nem olajat.
    1. A [Poisson-eloszlás](https://en.wikipedia.org/wiki/Poisson_distribution) darabszám feletti eloszlásokat írja le. Az eloszlás: $p(k \mid \lambda) = \lambda^k e^{-\lambda}/k!$, ahol $\lambda$ a ráta-függvény és $k$ a látott események száma. Bizonyítsd be, hogy $\lambda$ a $k$ darabszámok várható értéke.
    1. Tervezz veszteségfüggvényt a Poisson-eloszláshoz.
    1. Tervezz veszteségfüggvényt $\log \lambda$ becslésére.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/40)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/258)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/259)
:end_tab:
