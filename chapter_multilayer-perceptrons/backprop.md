# előreterjesztés, visszaterjesztés és számítási gráfok
:label:`sec_backprop`

Eddig mini-batch sztochasztikus gradienscsökkenés segítségével tanítottuk modelleinket.
Amikor azonban az algoritmust implementáltuk,
csak a modellen keresztüli *előreterjesztés* számításaival foglalkoztunk.
Amikor a gradiensek kiszámítására került sor,
egyszerűen meghívtuk a mélytanulási keretrendszer által biztosított visszaterjesztési függvényt.

A gradiensek automatikus kiszámítása
alapvetően leegyszerűsíti
a mélytanulási algoritmusok implementálását.
Az automatikus differenciálás előtt
még a bonyolult modellek kisebb változtatásai is megkövetelték
a bonyolult deriváltak kézi újraszámítását.
Meglepő módon az akadémiai cikkeknek sokszor
számos oldalt kellett szentelniük a frissítési szabályok levezetésének.
Bár továbbra is az automatikus differenciálásra kell támaszkodnunk,
hogy az érdekesebb részekre összpontosíthassunk,
tudni kell, hogyan számítják ki ezeket a gradienseket
a motorháztető alatt,
ha a mélytanulás felületes megértésén túl szeretnénk lépni.

Ebben a részben mélyebben belemerülünk
a *visszaterjesztés* részleteibe
(amelyet általánosabban *visszaterjesztésnek* neveznek).
A technikák és azok implementációjának szemléltetéséhez
néhány alapvető matematikán és számítási gráfon támaszkodunk.
Először egy egyetlen rejtett réteges MLP-re összpontosítunk
súlybomlással ($\ell_2$ regularizáció, amelyet a következő fejezetekben tárgyalunk).

## előreterjesztés

Az *előreterjesztés* (vagy *előremenet*) a köztes változók (beleértve a kimeneteket) kiszámítására és tárolására utal
egy neurális hálózatban, a bemeneti rétegtől a kimeneti réteg felé haladva.
Most lépésről lépésre végigjárjuk egy egyetlen rejtett réteges neurális hálózat mechanizmusait.
Ez kissé fáradságosnak tűnhet, de minden mélyebb megértésnek megvan az ára.


Az egyszerűség kedvéért tegyük fel,
hogy a bemeneti példa $\mathbf{x}\in \mathbb{R}^d$,
és hogy a rejtett réteg nem tartalmaz eltolás tagot.
Itt a köztes változó:

$$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x},$$

ahol $\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$
a rejtett réteg súlyparamétere.
A $\mathbf{z}\in \mathbb{R}^h$ köztes változót
az $\phi$ aktivációs függvényen átfuttatva
$h$ hosszú rejtett aktivációs vektort kapunk:

$$\mathbf{h}= \phi (\mathbf{z}).$$

A rejtett réteg $\mathbf{h}$ kimenete
szintén egy köztes változó.
Feltéve, hogy a kimeneti réteg paraméterei
csupán egy $\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$ súlyból állnak,
$q$ hosszú vektorral rendelkező kimeneti réteg változót kapunk:

$$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.$$

Feltéve, hogy a veszteségfüggvény $l$
és a példa címkéje $y$,
kiszámíthatjuk egy egyetlen adatpéldány veszteségtermét:

$$L = l(\mathbf{o}, y).$$

Amint a később bevezetendő $\ell_2$ regularizáció definíciójából látni fogjuk,
a $\lambda$ hiperparaméterrel a regularizációs tag:

$$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_\textrm{F}^2 + \|\mathbf{W}^{(2)}\|_\textrm{F}^2\right),$$
:eqlabel:`eq_forward-s`

ahol a mátrix Frobenius-normája
egyszerűen az $\ell_2$ norma,
amelyet a mátrix vektorrá lapítása után alkalmazunk.
Végül a modell regularizált vesztesége egy adott adatpéldányon:

$$J = L + s.$$

A következőkben $J$-t *célfüggvénynek* nevezzük.


## Az előreterjesztés számítási gráfja

A *számítási gráfok* ábrázolása segít vizualizálni
az operátorok és változók függőségeit a számítás során.
A :numref:`fig_forward` tartalmazza a fentebb leírt egyszerű hálózathoz tartozó gráfot,
ahol a négyzetek változókat, a körök operátorokat jelölnek.
A bal alsó sarok jelöli a bemenetet,
a jobb felső sarok a kimenetet.
Vegyük észre, hogy a nyilak iránya
(amelyek az adatáramlást illusztrálják)
elsősorban jobbra és felfelé mutat.

![Az előreterjesztés számítási gráfja.](../img/forward.svg)
:label:`fig_forward`

## Visszaterjesztés

A *visszaterjesztés* a neurális hálózat paramétereinek gradiensét kiszámító módszer.
Röviden, a módszer a hálózatot fordított sorrendben, a kimeneti rétegtől a bemeneti réteg felé járja be,
a matematikai analízis *láncszabálya* alapján.
Az algoritmus tárolja a szükséges köztes változókat
(parciális deriváltakat),
amelyek a gradiens kiszámításakor szükségesek egyes paraméterekre vonatkozóan.
Tegyük fel, hogy van két függvényünk:
$\mathsf{Y}=f(\mathsf{X})$
és $\mathsf{Z}=g(\mathsf{Y})$,
ahol a bemenet és a kimenet
$\mathsf{X}, \mathsf{Y}, \mathsf{Z}$
tetszőleges alakú tenzorok.
A láncszabály alkalmazásával
kiszámíthatjuk $\mathsf{Z}$ deriváltját $\mathsf{X}$-re vonatkozóan:

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \textrm{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$

Itt a $\textrm{prod}$ operátort használjuk az argumentumok szorzásához,
a szükséges műveletek (pl. transzponálás és bemeneti pozíciók felcserélése)
elvégzése után.
Vektorok esetén ez egyszerű:
ez csupán mátrix-mátrix szorzás.
Magasabb dimenziójú tenzorok esetén
a megfelelő analógot alkalmazzuk.
A $\textrm{prod}$ operátor elrejti az összes jelölési bonyodalmat.

Emlékeztetőül: az egyetlen rejtett réteges egyszerű hálózat paraméterei,
amelynek számítási gráfja a :numref:`fig_forward` ábrán látható,
$\mathbf{W}^{(1)}$ és $\mathbf{W}^{(2)}$.
A visszaterjesztés célja a
$\partial J/\partial \mathbf{W}^{(1)}$
és $\partial J/\partial \mathbf{W}^{(2)}$ gradiensek kiszámítása.
Ehhez alkalmazzuk a láncszabályt,
és sorban kiszámítjuk minden köztes változó és paraméter gradiensét.
A számítások sorrendje fordított
az előreterjesztésben elvégzettekhez képest,
mivel a számítási gráf kimenetéből kell kiindulni
és a paraméterek felé haladni.
Az első lépés a $J=L+s$ célfüggvény gradienseinek kiszámítása
a veszteségtagra $L$ és a regularizációs tagra $s$ vonatkozóan:

$$\frac{\partial J}{\partial L} = 1 \; \textrm{és} \; \frac{\partial J}{\partial s} = 1.$$

Ezután a láncszabály alapján kiszámítjuk a célfüggvény gradiensét
a kimeneti réteg $\mathbf{o}$ változójára vonatkozóan:

$$
\frac{\partial J}{\partial \mathbf{o}}
= \textrm{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q.
$$

Ezután kiszámítjuk a regularizációs tag gradiensét
mindkét paraméterre vonatkozóan:

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\; \textrm{és} \;
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$$

Most kiszámíthatjuk a kimeneti réteghez legközelebb lévő modellparaméterek
$\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$ gradiensét.
A láncszabály alkalmazása adja:

$$\frac{\partial J}{\partial \mathbf{W}^{(2)}}= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \textrm{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.$$
:eqlabel:`eq_backprop-J-h`

A $\mathbf{W}^{(1)}$-re vonatkozó gradiens megszerzéséhez
folytatni kell a visszaterjesztést
a kimeneti rétegtől a rejtett réteg felé.
A rejtett réteg kimenetére vonatkozó
$\partial J/\partial \mathbf{h} \in \mathbb{R}^h$ gradiens:


$$
\frac{\partial J}{\partial \mathbf{h}}
= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.
$$

Mivel az $\phi$ aktivációs függvény elemenként hat,
a $\mathbf{z}$ köztes változó $\partial J/\partial \mathbf{z} \in \mathbb{R}^h$ gradiensének kiszámításához
elemenként szorzási operátort kell használni,
amelyet $\odot$-val jelölünk:

$$
\frac{\partial J}{\partial \mathbf{z}}
= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).
$$

Végül megkaphatjuk a bemeneti réteghez legközelebb lévő modellparaméterek
$\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ gradiensét.
A láncszabály alapján:

$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}}
= \textrm{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \textrm{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.
$$



## Neurális hálózatok tanítása

Neurális hálózatok tanításakor
az előre irányú és a visszaterjesztés kölcsönösen függenek egymástól.
Különösen az előreterjesztéskor
a számítási gráfot a függőségek irányában járjuk be,
és kiszámítjuk az útvonalán lévő összes változót.
Ezeket aztán a visszaterjesztésnél alkalmazzuk,
ahol a gráf számítási sorrendje fordított.

Vegyük a fentebb említett egyszerű hálózatot szemléltetésként.
Egyrészt, a regularizációs tag :eqref:`eq_forward-s` kiszámítása
az előreterjesztés során
a $\mathbf{W}^{(1)}$ és $\mathbf{W}^{(2)}$ modellparaméterek aktuális értékeitől függ.
Ezeket az optimalizálási algoritmus a legutóbbi iteráció visszaterjesztése alapján adja meg.
Másrészt a :eqref:`eq_backprop-J-h` paraméter gradiens kiszámítása
a visszaterjesztés során
a $\mathbf{h}$ rejtett réteg kimenet aktuális értékétől függ,
amelyet az előreterjesztés ad meg.


Ezért neurális hálózatok tanításakor, amint a modell paraméterei inicializálva vannak,
felváltva hajtjuk végre az előreterjesztést és a visszaterjesztést,
a modell paramétereit a visszaterjesztés által adott gradiensekkel frissítve.
Vegyük észre, hogy a visszaterjesztés az előreterjesztésből tárolt köztes értékeket újrahasználja
az ismételt számítások elkerülése érdekében.
Ennek egyik következménye, hogy meg kell tartani
a köztes értékeket a visszaterjesztés befejezéséig.
Ez az egyik oka annak is, hogy a tanítás
lényegesen több memóriát igényel, mint az egyszerű előrejelzés.
Emellett az ilyen köztes értékek mérete nagyjából arányos
a hálózati rétegek számával és a batch méretével.
Ezért a mélyebb hálózatok nagyobb batch méretekkel való tanítása
könnyebben *memória kifogyási* hibákhoz vezet.


## Összefoglalás

Az előreterjesztés sorban kiszámítja és tárolja a neurális hálózat által meghatározott számítási gráf köztes változóit. A bemeneti rétegtől a kimeneti réteg felé halad.
A visszaterjesztés sorban kiszámítja és tárolja a neurális hálózat köztes változóinak és paramétereinek gradienseit fordított sorrendben.
mélytanulási modellek tanításakor az előreterjesztés és a visszaterjesztés egymástól függ,
és a tanítás lényegesen több memóriát igényel, mint az előrejelzés.


## Feladatok

1. Tegyük fel, hogy valamely skaláris $f$ függvény bemenetei $\mathbf{X}$, amelyek $n \times m$-es mátrixok. Mi az $f$ gradiensének dimenziója $\mathbf{X}$-re vonatkozóan?
1. Adj eltolást az ebben a részben leírt modell rejtett rétegéhez (nem kell eltolást belefoglalni a regularizációs tagba).
    1. Rajzold meg a megfelelő számítási gráfot.
    1. Vezedd le az előre irányú és visszaterjesztés egyenleteit.
1. Számítsd ki a memóriaigényt a jelen részben leírt modell tanításához és előrejelzéséhez.
1. Tegyük fel, hogy másodfokú deriváltakat szeretnél kiszámítani. Mi történik a számítási gráffal? Milyen hosszúra becsülöd a számítás idejét?
1. Tegyük fel, hogy a számítási gráf túl nagy a GPU-d számára.
    1. Szét lehet-e osztani több GPU között?
    1. Mik az előnyei és hátrányai a kisebb mini-batchcsel való tanításhoz képest?

[Discussions](https://discuss.d2l.ai/t/102)
