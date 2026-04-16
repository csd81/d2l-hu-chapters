# Aszinkron számítás
:label:`sec_async`

A mai számítógépek erősen párhuzamos rendszerek, amelyek több CPU magból (magánként gyakran több szállal), GPU-nként több feldolgozóelemből, és sokszor eszközönként több GPU-ból állnak. Röviden: egyszerre sok különböző dolgot tudunk feldolgozni, gyakran különböző eszközökön. Sajnos a Python nem a legjobb módja a párhuzamos és aszinkron kód írásának, legalábbis némi extra segítség nélkül. Elvégre a Python egyszálú, és ez valószínűleg nem fog megváltozni a jövőben. Az MXNet és a TensorFlow olyan mélytanulás keretrendszerek, amelyek *aszinkron programozási* modellt alkalmaznak a teljesítmény javítása érdekében,
míg a PyTorch Python saját ütemezőjét használja, ami eltérő teljesítménykompromisszumhoz vezet.
A PyTorch esetében alapértelmezés szerint a GPU műveletek aszinkronok. Amikor egy GPU-t használó függvényt hív meg, a műveletek az adott eszközre kerülnek sorba, de nem feltétlenül hajtódnak végre azonnal. Ez lehetővé teszi számunkra, hogy több számítást hajtsunk végre párhuzamosan, beleértve a CPU-n vagy más GPU-kon végzett műveleteket is.

Ezért az aszinkron programozás megértése segít hatékonyabb programok fejlesztésében, a számítási követelmények és a kölcsönös függőségek proaktív csökkentésével. Ez lehetővé teszi a memória-overhead csökkentését és a processzorkihasználtság növelését.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
import numpy, os, subprocess
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy, os, subprocess
import torch
from torch import nn
```

## Aszinkronitás a backend-en keresztül

:begin_tab:`mxnet`
Bemelegítésként tekintsük az alábbi egyszerű problémát: véletlenszerű mátrixot szeretnénk generálni és megszorozni. Tegyük ezt meg NumPy-ban és `mxnet.np`-ban is, hogy lássuk a különbséget.
:end_tab:

:begin_tab:`pytorch`
Bemelegítésként tekintsük az alábbi egyszerű problémát: véletlenszerű mátrixot szeretnénk generálni és megszorozni. Tegyük ezt meg NumPy-ban és PyTorch tenzorban is, hogy lássuk a különbséget.
Vegyük figyelembe, hogy a PyTorch `tensor` GPU-n van definiálva.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('mxnet.np'):
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
```

```{.python .input}
#@tab pytorch
# GPU számítás bemelegítése
device = d2l.try_gpu()
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
```

:begin_tab:`mxnet`
Az MXNet-en keresztüli benchmark kimenete nagyságrendekkel gyorsabb. Mivel mindkettő ugyanazon a processzoron fut, valami másnak kell történnie.
Ha az MXNet-et arra kényszerítjük, hogy a visszatérés előtt befejezzen minden backend számítást, kiderül, mi történt korábban: a számítást a backend hajtja végre, miközben a frontend visszaadja az irányítást a Pythonnak.
:end_tab:

:begin_tab:`pytorch`
A PyTorch-on keresztüli benchmark kimenete nagyságrendekkel gyorsabb.
A NumPy skaláris szorzat a CPU processzoron fut, míg
a PyTorch mátrixszorzat GPU-n hajtódik végre, ezért az utóbbi
várhatóan sokkal gyorsabb. De a hatalmas időbeli különbség arra utal, hogy
valami másnak is kell történnie.
Alapértelmezés szerint a GPU műveletek aszinkronok a PyTorch-ban.
Ha a PyTorch-ot arra kényszerítjük, hogy a visszatérés előtt befejezzen minden számítást, kiderül,
mi történt korábban: a számítást a backend hajtja végre,
miközben a frontend visszaadja az irányítást a Pythonnak.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark():
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark():
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)
```

:begin_tab:`mxnet`
Tágabb értelemben az MXNet rendelkezik egy frontend-del a felhasználókkal való közvetlen interakcióhoz (például Pythonon keresztül), valamint egy backend-del, amelyet a rendszer a számítás elvégzéséhez használ.
Ahogy a :numref:`fig_frontends` ábra mutatja, a felhasználók különböző frontend programozási nyelveken, például Pythonban, R-ben, Scalában és C++-ban írhatnak MXNet programokat. A használt frontend programozási nyelvtől függetlenül az MXNet programok végrehajtása elsősorban a C++ implementációk backend-jében történik. A frontend nyelv által kiadott műveletek átkerülnek a backend-be végrehajtásra.
A backend kezeli a saját szálait, amelyek folyamatosan gyűjtik és végrehajtják a sorba állított feladatokat. Vegyük figyelembe, hogy ehhez a backend-nek képesnek kell lennie nyomon követni a számítási gráf különböző lépései közötti függőségeket. Ezért nem lehetséges egymástól függő műveletek párhuzamosítása.
:end_tab:

:begin_tab:`pytorch`
Tágabb értelemben a PyTorch rendelkezik egy frontend-del a felhasználókkal való közvetlen interakcióhoz (például Pythonon keresztül), valamint egy backend-del, amelyet a rendszer a számítás elvégzéséhez használ.
Ahogy a :numref:`fig_frontends` ábra mutatja, a felhasználók különböző frontend programozási nyelveken, például Pythonban és C++-ban írhatnak PyTorch programokat. A használt frontend programozási nyelvtől függetlenül a PyTorch programok végrehajtása elsősorban a C++ implementációk backend-jében történik. A frontend nyelv által kiadott műveletek átkerülnek a backend-be végrehajtásra.
A backend kezeli a saját szálait, amelyek folyamatosan gyűjtik és végrehajtják a sorba állított feladatokat.
Vegyük figyelembe, hogy ehhez a backend-nek képesnek kell lennie nyomon követni a
számítási gráf különböző lépései közötti függőségeket.
Ezért nem lehetséges egymástól függő műveletek párhuzamosítása.
:end_tab:

![Programozási nyelv frontend-ek és mélytanulás keretrendszer backend-ek.](../img/frontends.png)
:width:`300px`
:label:`fig_frontends`

Nézzünk egy másik egyszerű példát, hogy jobban megértsük a függőségi gráfot.

```{.python .input}
#@tab mxnet
x = np.ones((1, 2))
y = np.ones((1, 2))
z = x * y + 2
z
```

```{.python .input}
#@tab pytorch
x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2
z
```

![A backend nyomon követi a számítási gráf különböző lépései közötti függőségeket.](../img/asyncgraph.svg)
:label:`fig_asyncgraph`



A fenti kódrészlet a :numref:`fig_asyncgraph` ábrán is látható.
Valahányszor a Python frontend szál végrehajtja az első három utasítás egyikét, egyszerűen visszaadja a feladatot a backend sornak. Amikor az utolsó utasítás eredményét *ki kell nyomtatni*, a Python frontend szál megvárja, amíg a C++ backend szál befejezi a `z` változó eredményének kiszámítását. Ennek a tervezésnek az egyik előnye, hogy a Python frontend szálnak nem kell tényleges számításokat végezni. Így kevés hatással van a program általános teljesítményére, függetlenül a Python teljesítményétől. A :numref:`fig_threading` ábra illusztrálja, hogyan lép kapcsolatba egymással a frontend és a backend.

![A frontend és a backend közötti kölcsönhatások.](../img/threading.svg)
:label:`fig_threading`




## Akadályok és blokkolók

:begin_tab:`mxnet`
Számos olyan művelet létezik, amely arra kényszeríti a Pythont, hogy megvárja a befejezést:

* A legnyilvánvalóbb: az `npx.waitall()` megvárja, amíg az összes számítás befejeződik, függetlenül attól, mikor adták ki a számítási utasításokat. A gyakorlatban rossz ötlet ezt az operátort használni, hacsak nem feltétlenül szükséges, mivel rossz teljesítményhez vezethet.
* Ha csak egy adott változó elérhetőségére akarunk várni, meghívhatjuk a `z.wait_to_read()` függvényt. Ebben az esetben az MXNet blokkolja a Pythonba való visszatérést, amíg a `z` változó ki nem számítódik. Más számítás ezután tovább folytatódhat.

Nézzük meg, hogyan működik ez a gyakorlatban.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('waitall'):
    b = np.dot(a, a)
    npx.waitall()

with d2l.Benchmark('wait_to_read'):
    b = np.dot(a, a)
    b.wait_to_read()
```

:begin_tab:`mxnet`
Mindkét művelet körülbelül ugyanannyi ideig tart. A nyilvánvaló blokkoló műveleteken kívül javasoljuk, hogy legyen tisztában a *implicit* blokkolókkal is. Egy változó kinyomtatása nyilván megköveteli a változó elérhetőségét, és így blokkoló. Végül a NumPy-ba való konverzió a `z.asnumpy()` segítségével, és a skalárba való konverzió a `z.item()` segítségével blokkolóak, mivel a NumPy-nak nincs fogalma az aszinkronitásról. Hozzá kell férnie az értékekhez, akárcsak a `print` függvénynek.

Az adatok kis mennyiségének gyakori másolása az MXNet hatóköréből a NumPy-ba és vissza tönkreteheti egy egyébként hatékony kód teljesítményét, mivel minden ilyen művelet megköveteli a számítási gráftól az összes szükséges közbenső eredmény kiértékelését, *mielőtt* bármi más elvégezhető lenne.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('numpy conversion'):
    b = np.dot(a, a)
    b.asnumpy()

with d2l.Benchmark('scalar conversion'):
    b = np.dot(a, a)
    b.sum().item()
```

## A számítás javítása

:begin_tab:`mxnet`
Egy erősen többszálú rendszeren (még a hétköznapi laptopoknak is 4 vagy több száluk van, és a többfoglalatú szervereken ez a szám meghaladhatja a 256-ot) a műveletek ütemezésének overhead-je jelentőssé válhat. Ezért nagyon kívánatos, hogy a számítás és az ütemezés aszinkron módon és párhuzamosan történjen. Ennek előnyét szemléltetendő nézzük meg, mi történik, ha egy változót 1-gyel növelünk többször egymás után, szekvenciálisan vagy aszinkron módon. A szinkron végrehajtást szimulálva egy `wait_to_read` akadályt szúrunk be minden összeadás közé.
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('synchronous'):
    for _ in range(10000):
        y = x + 1
        y.wait_to_read()

with d2l.Benchmark('asynchronous'):
    for _ in range(10000):
        y = x + 1
    npx.waitall()
```

:begin_tab:`mxnet`
A Python frontend szál és a C++ backend szál közötti kissé egyszerűsített interakció az alábbiak szerint foglalható össze:
1. A frontend megrendeli a backend-nek a `y = x + 1` számítási feladat sorba helyezését.
1. A backend ekkor fogadja a számítási feladatokat a sorból, és elvégzi a tényleges számításokat.
1. A backend ezután visszaküldi a számítási eredményeket a frontend-nek.
Tegyük fel, hogy ezek a három szakasz időtartama rendre $t_1, t_2$ és $t_3$. Ha nem használunk aszinkron programozást, akkor a 10 000 számítás elvégzéséhez szükséges teljes idő körülbelül $10000 (t_1+ t_2 + t_3)$. Ha aszinkron programozást használunk, akkor a 10 000 számítás elvégzéséhez szükséges teljes idő $t_1 + 10000 t_2 + t_3$-ra csökkenthető (feltéve, hogy $10000 t_2 > 9999t_1$), mivel a frontend-nek nem kell minden egyes ciklusnál megvárnia, hogy a backend visszaadja a számítási eredményeket.
:end_tab:


## Összefoglalás


* A mélytanulás keretrendszerek leválaszthatják a Python frontend-et egy végrehajtási backend-ről. Ez lehetővé teszi a parancsok gyors aszinkron beillesztését a backend-be és a kapcsolódó párhuzamosságot.
* Az aszinkronitás meglehetősen reszponzív frontend-et eredményez. Azonban óvatosan kell eljárni, hogy ne töltsük túl a feladatsort, mivel ez túlzott memóriafogyasztáshoz vezethet. Ajánlott minden mini-batchnél szinkronizálni, hogy a frontend és a backend közelítőleg szinkronban maradjanak.
* A chip gyártók kifinomult teljesítményelemző eszközöket kínálnak, amelyek sokkal részletesebb képet adnak a mélytanulás hatékonyságáról.

:begin_tab:`mxnet`
* Légy tudatában annak, hogy az MXNet memóriakezelésből Pythonba való konverziók arra kényszerítik a backend-et, hogy megvárja, amíg az adott változó készen áll. Az olyan függvények, mint a `print`, `asnumpy` és `item`, mind ezzel a hatással bírnak. Ez kívánatos lehet, de a szinkronizáció gondatlan használata tönkreteheti a teljesítményt.
:end_tab:


## Feladatok

:begin_tab:`mxnet`
1. Fentebb megemlítettük, hogy az aszinkron számítás használatával a 10 000 számítás elvégzéséhez szükséges teljes idő $t_1 + 10000 t_2 + t_3$-ra csökkenthető. Miért kell itt feltennünk, hogy $10000 t_2 > 9999 t_1$?
1. Mérd meg a `waitall` és a `wait_to_read` közötti különbséget. Tipp: hajts végre számos utasítást, és szinkronizálj egy közbenső eredményre.
:end_tab:

:begin_tab:`pytorch`
1. CPU-n mérd meg ugyanazokat a mátrixszorzási műveleteket ebben a szakaszban. Még mindig megfigyelhető-e aszinkronitás a backend-en keresztül?
:end_tab:

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/361)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2564)
:end_tab:
