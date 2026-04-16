# Régióalapú konvolúciós neurális hálózatok (R-CNN)
:label:`sec_rcnn`

A :numref:`sec_ssd` fejezetben leírt egylépéses többdobozos felismerés mellett a régióalapú konvolúciós neurális hálózatok, más néven régiók CNN jellemzőkkel (R-CNN-ek) szintén az úttörő megközelítések közé tartoznak a deep learning objektumfelismerésre való alkalmazásában :cite:`Girshick.Donahue.Darrell.ea.2014`.
Ebben a fejezetben bemutatjuk az R-CNN-t és fejlesztéseinek sorozatát: a gyors R-CNN-t :cite:`Girshick.2015`, a gyorsabb R-CNN-t :cite:`Ren.He.Girshick.ea.2015` és a maszk R-CNN-t :cite:`He.Gkioxari.Dollar.ea.2017`.
Helyhiány miatt csak ezeknek a modelleknek a tervezésére összpontosítunk.



## R-CNN-ek


Az *R-CNN* először sok (pl. 2000) *régiójavaslatot* von ki a bemeneti képből (pl. a horgonydobozok is tekinthetők régiójavaslatnak), felcímkézve osztályaikat és befoglaló téglalapjaikat (pl. eltolásaikat). :cite:`Girshick.Donahue.Darrell.ea.2014`
Majd egy konvolúciós neurális hálózatot alkalmaznak az előre terjesztés elvégzésére minden egyes régiójavaslathoz annak jellemzőinek kinyeréséhez.
Ezután minden régióajánlat jellemzőit a régióajánlat osztályának és befoglaló téglalapjának jóslásához alkalmazzák.


![Az R-CNN modell.](../img/r-cnn.svg)
:label:`fig_r-cnn`

A :numref:`fig_r-cnn` ábra bemutatja az R-CNN modellt. Konkrétabban az R-CNN a következő négy lépésből áll:

1. *Szelektív keresés* elvégzése a bemeneti képen több, jó minőségű régióajánlat kinyeréséhez :cite:`Uijlings.Van-De-Sande.Gevers.ea.2013`. Ezeket a javasolt régiókat általában több léptéken választják ki, különböző alakokkal és méretekkel. Minden régióajánlatot egy osztállyal és egy valódi befoglaló téglalappal látnak el.
1. Válasszunk egy előtanított konvolúciós neurális hálózatot, és csonkítsuk a kimeneti réteg előtt. Méretezzük át minden régióajánlatot a hálózat által igényelt bemeneti méretre, és az előre terjesztésen keresztül adjuk ki a régióajánlat kinyert jellemzőit.
1. Vegyük minden régióajánlat kinyert jellemzőit és felcímkézett osztályát példaként. Tanítsunk több támaszvektor-gépet az objektumok osztályozásához, ahol minden támaszvektor-gép külön-külön állapítja meg, hogy a példa tartalmaz-e egy adott osztályt.
1. Vegyük minden régióajánlat kinyert jellemzőit és felcímkézett befoglaló téglalapját példaként. Tanítsunk egy lineáris regressziós modellt a valódi befoglaló téglalap jóslásához.


Bár az R-CNN modell hatékonyan vonja ki a képjellemzőket az előtanított konvolúciós neurális hálózatok segítségével, lassú.
Képzeljük el, hogy egyetlen bemeneti képből ezer régióajánlatot választunk ki: ehhez ezer konvolúciós neurális hálózat előre terjesztési lépésre van szükség az objektumfelismerés elvégzéséhez.
Ez a hatalmas számítási terhelés kivitelezhetetlenné teszi az R-CNN-ek széleskörű valós alkalmazási felhasználását.

## Gyors R-CNN

Az R-CNN fő teljesítménybeli szűk keresztmetszetje az egyes régióajánlatokhoz végzett független konvolúciós neurális hálózat előre terjesztésben rejlik, amely nem osztja meg a számításokat.
Mivel ezek a régiók általában átfednek, a független jellemzőkinyerés sok ismételt számítást eredményez.
A *gyors R-CNN* egyik fő fejlesztése az R-CNN-hez képest az, hogy a konvolúciós neurális hálózat előre terjesztését csak az egész képen végzik el :cite:`Girshick.2015`.

![A gyors R-CNN modell.](../img/fast-rcnn.svg)
:label:`fig_fast_r-cnn`

A :numref:`fig_fast_r-cnn` ábra bemutatja a gyors R-CNN modellt. Fő számításai a következők:


1. Az R-CNN-hez képest a gyors R-CNN esetén a jellemzőkinyeréshez szükséges konvolúciós neurális hálózat bemenete az egész kép, nem az egyes régióajánlatok. Emellett ez a konvolúciós neurális hálózat tanítható. Adott bemeneti képen, legyen a konvolúciós neurális hálózat kimenetének alakja $1 \times c \times h_1  \times w_1$.
1. Tegyük fel, hogy a szelektív keresés $n$ régióajánlatot generál. Ezek a régióajánlatok (különböző alakokkal) érdeklődési régiókat (különböző alakokkal) jelölnek meg a konvolúciós neurális hálózat kimenetén. Majd ezek az érdeklődési régiók további azonos alakú (pl. $h_2$ magasság és $w_2$ szélesség megadva) jellemzőket vonnak ki, hogy könnyen összefűzhetők legyenek. Ennek megvalósításához a gyors R-CNN bevezeti az *érdeklődési régió (RoI) pooling* réteget: a konvolúciós neurális hálózat kimenete és a régióajánlatok kerülnek be ebbe a rétegbe, amely $n \times c \times h_2 \times w_2$ alakú összefűzött jellemzőket ad ki, amelyeket az összes régióajánlathoz tovább vonnak ki.
1. Egy teljesen összekötött réteg segítségével az összefűzött jellemzőket $n \times d$ alakú kimenetté alakítják, ahol $d$ a modell tervétől függ.
1. Jósolják minden egyes $n$ régióajánlathoz az osztályt és a befoglaló téglalapot. Konkrétabban az osztály- és befoglaló téglalap-jóslásban a teljesen összekötött réteg kimenetét $n \times q$ alakú kimenetre ($q$ az osztályok száma) és $n \times 4$ alakú kimenetre alakítják. Az osztályjóslás softmax regressziót alkalmaz.


A gyors R-CNN-ben javasolt érdeklődési régió pooling réteg különbözik a :numref:`sec_pooling` fejezetben bemutatott pooling rétegtől.
A pooling rétegben közvetve szabályozzuk a kimenet alakját a pooling ablak, a párnázás és a lépésköz méretének megadásával.
Ezzel szemben az érdeklődési régió pooling rétegben közvetlenül megadhatjuk a kimenet alakját.

Például adjuk meg minden régió kimeneti magasságát és szélességét rendre $h_2$-nek és $w_2$-nek.
Bármely $h \times w$ alakú érdeklődési régióablak esetén ezt az ablakot $h_2 \times w_2$ rácsú részablakokra osztják, ahol minden részablak alakja hozzávetőlegesen $(h/h_2) \times (w/w_2)$.
A gyakorlatban bármely részablak magasságát és szélességét felfelé kerekítik, és a részablak kimeneteként a legnagyobb elemet alkalmazzák.
Ezért az érdeklődési régió pooling réteg azonos alakú jellemzőket tud kivonni, még akkor is, ha az érdeklődési régióknak különböző alakjaik vannak.


Szemléletes példaként a :numref:`fig_roi` ábrán egy $4 \times 4$ pixeles bemeneten a bal felső $3\times 3$ érdeklődési régiót választottuk ki.
Ehhez az érdeklődési régióhoz egy $2\times 2$-es érdeklődési régió pooling réteget alkalmazunk, hogy $2\times 2$-es kimenetet kapjunk.
Megjegyezzük, hogy a négy részablak mindegyike a 0, 1, 4 és 5 (5 a maximum); 2 és 6 (6 a maximum); 8 és 9 (9 a maximum); és 10 elemeket tartalmazza.

![Egy $2\times 2$-es érdeklődési régió pooling réteg.](../img/roi.svg)
:label:`fig_roi`

Az alábbiakban bemutatjuk az érdeklődési régió pooling réteg számítását. Tegyük fel, hogy a konvolúciós neurális hálózat által kinyert `X` jellemzők magassága és szélessége egyaránt 4, és csak egy csatorna van.

```{.python .input}
#@tab mxnet
from mxnet import np, npx

npx.set_np()

X = np.arange(16).reshape(1, 1, 4, 4)
X
```

```{.python .input}
#@tab pytorch
import torch
import torchvision

X = torch.arange(16.).reshape(1, 1, 4, 4)
X
```

Tegyük fel továbbá, hogy a bemeneti kép magassága és szélessége egyaránt 40 pixel, és a szelektív keresés két régióajánlatot generál ezen a képen.
Minden régióajánlatot öt elemmel fejeznek ki: az objektum osztálya, majd a bal felső és jobb alsó sarok $(x, y)$-koordinátái.

```{.python .input}
#@tab mxnet
rois = np.array([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
```

```{.python .input}
#@tab pytorch
rois = torch.Tensor([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
```

Mivel `X` magassága és szélessége a bemeneti kép magasságának és szélességének $1/10$-e, a két régióajánlat koordinátáit 0.1-szal megszorozzák a megadott `spatial_scale` argumentum szerint.
Majd a két érdeklődési régiót `X`-en `X[:, :, 0:3, 0:3]`-ként és `X[:, :, 1:4, 0:4]`-ként jelölik meg.
Végül a $2\times 2$-es érdeklődési régió poolingban minden érdeklődési régióból részablakok rácsát vonják ki, hogy azonos $2\times 2$ alakú jellemzőket kapjanak.

```{.python .input}
#@tab mxnet
npx.roi_pooling(X, rois, pooled_size=(2, 2), spatial_scale=0.1)
```

```{.python .input}
#@tab pytorch
torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=0.1)
```

## Gyorsabb R-CNN

A pontosabb objektumfelismeréshez a gyors R-CNN modellnek általában sok régióajánlatot kell generálnia a szelektív keresésben.
A régióajánlatok pontosság veszteség nélküli csökkentéséhez a *gyorsabb R-CNN* azt javasolja, hogy a szelektív keresést egy *régióajánlati hálózattal* váltsák fel :cite:`Ren.He.Girshick.ea.2015`.


![A gyorsabb R-CNN modell.](../img/faster-rcnn.svg)
:label:`fig_faster_r-cnn`


A :numref:`fig_faster_r-cnn` ábra a gyorsabb R-CNN modellt mutatja. A gyors R-CNN-hez képest a gyorsabb R-CNN csak a régióajánlati módszert változtatja meg a szelektív keresésről a régióajánlati hálózatra.
A modell többi része változatlan marad.
A régióajánlati hálózat a következő lépésekben működik:

1. Egy 1-es párnázású $3\times 3$-as konvolúciós réteg segítségével a konvolúciós neurális hálózat kimenetét $c$ csatornás új kimenetté alakítják. Így a konvolúciós neurális hálózat által kinyert jellemzőtérképek térbeli dimenzióinak minden egysége egy új $c$ hosszúságú jellemzővektort kap.
1. A jellemzőtérképek minden pixelét középpontként véve különböző méretű és képarányú horgonydobozokat generálnak és felcímkéznek.
1. Minden horgonydoboz középpontjában lévő $c$ hosszúságú jellemzővektort felhasználva jósolják a bináris osztályt (háttér vagy objektumok) és a befoglaló téglalapot ehhez a horgonydobozhoz.
1. Vegyük figyelembe azokat a jósolt befoglaló téglalapokat, amelyek jósolt osztályai objektumok. Távolítsuk el az átfedő eredményeket nem-maximum szuppresszióval. A maradék, objektumokat ábrázoló jósolt befoglaló téglalapok az érdeklődési régió pooling réteg által igényelt régióajánlatok.



Érdemes megjegyezni, hogy a gyorsabb R-CNN modell részeként a régióajánlati hálózatot a modell többi részével közösen tanítják.
Más szóval a gyorsabb R-CNN célfüggvénye nemcsak az osztály- és befoglaló téglalap-jóslást tartalmazza az objektumfelismerésben, hanem a horgonydobozok bináris osztályát és befoglaló téglalap-jóslását is a régióajánlati hálózatban.
A végponttól végpontig tartó tanítás eredményeképpen a régióajánlati hálózat megtanulja, hogyan generáljon magas minőségű régióajánlatokat, hogy pontosan maradjon az objektumfelismerésben az adatokból tanult, csökkentett számú régióajánlattal.


## Maszk R-CNN

A tanítóadathalmazban, ha az objektumok pixelszintű pozícióit is felcímkézik a képeken, a *maszk R-CNN* hatékonyan ki tudja használni ezeket a részletes címkéket az objektumfelismerés pontosságának további javítására :cite:`He.Gkioxari.Dollar.ea.2017`.


![A maszk R-CNN modell.](../img/mask-rcnn.svg)
:label:`fig_mask_r-cnn`

Ahogy a :numref:`fig_mask_r-cnn` ábrán látható, a maszk R-CNN a gyorsabb R-CNN alapján módosítva.
Konkrétabban a maszk R-CNN az érdeklődési régió pooling réteget az *érdeklődési régió (RoI) igazítási* réteggel váltja fel.
Ez az érdeklődési régió igazítási réteg bilineáris interpolációt alkalmaz a jellemzőtérképek térbeli információinak megőrzésére, ami jobban alkalmas a pixelszintű jósláshoz.
E réteg kimenete azonos alakú jellemzőtérképeket tartalmaz az összes érdeklődési régióhoz.
Ezeket a jellemzőtérképeket nemcsak minden érdeklődési régió osztályának és befoglaló téglalapjának jóslásához alkalmazzák, hanem az objektum pixelszintű pozíciójának jóslásához is egy kiegészítő teljesen konvolúciós hálózaton keresztül.
A teljesen konvolúciós hálózatok alkalmazásának részletei a kép pixelszintű szemantikájának jóslásához a fejezet következő részeiben kerülnek bemutatásra.


## Összefoglalás


* Az R-CNN sok régióajánlatot von ki a bemeneti képből, konvolúciós neurális hálózatot alkalmaz az előre terjesztés elvégzésére minden egyes régióajánlathoz annak jellemzőinek kinyeréséhez, majd ezeket a jellemzőket alkalmazza a régióajánlat osztályának és befoglaló téglalapjának jóslásához.
* A gyors R-CNN egyik fő fejlesztése az R-CNN-hez képest az, hogy a konvolúciós neurális hálózat előre terjesztését csak az egész képen végzik. Bevezeti az érdeklődési régió pooling réteget is, hogy azonos alakú jellemzőket vonhassanak ki az eltérő alakú érdeklődési régiókból.
* A gyorsabb R-CNN a gyors R-CNN-ben alkalmazott szelektív keresést közösen tanított régióajánlati hálózattal váltja fel, így az előbbi pontosan maradhat az objektumfelismerésben a csökkentett számú régióajánlattal.
* A gyorsabb R-CNN alapján a maszk R-CNN egy teljesen konvolúciós hálózatot is bevezet, hogy kihasználja a pixelszintű címkéket az objektumfelismerés pontosságának további javítására.


## Feladatok

1. Fel lehet-e fogni az objektumfelismerést egyetlen regressziós problémának, például befoglaló téglalapok és osztályvalószínűségek jóslásával? A YOLO modell tervét veheted alapul :cite:`Redmon.Divvala.Girshick.ea.2016`.
1. Hasonlítsd össze az egylépéses többdobozos felismerést az ebben a fejezetben bemutatott módszerekkel. Melyek a fő különbségeik? A :citet:`Zhao.Zheng.Xu.ea.2019` 2. ábrájára hivatkozhatsz.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/374)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1409)
:end_tab:
