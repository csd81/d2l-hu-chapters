# Sorozatokkal való munkavégzés
:label:`sec_sequence`

Eddig olyan modellekre összpontosítottunk, amelyek bemenetei
egyetlen $\mathbf{x} \in \mathbb{R}^d$ jellemzővektorból álltak.
A sorozatok feldolgozására képes modellek fejlesztésekor
a nézőpont fő változása az, hogy most
olyan bemenetekre összpontosítunk, amelyek jellemzővektorok
rendezett listájából állnak: $\mathbf{x}_1, \dots, \mathbf{x}_T$,
ahol minden $\mathbf{x}_t$ jellemzővektor
egy $t \in \mathbb{Z}^+$ időlépéshez van rendelve,
és $\mathbb{R}^d$-ben helyezkedik el.

Egyes adathalmazok egyetlen hatalmas sorozatból állnak.
Gondoljuk például az éghajlatkutatók rendelkezésére álló
rendkívül hosszú szenzorleolvasási adatfolyamokra.
Ilyen esetekben tanítási adathalmazokat hozhatunk létre
előre meghatározott hosszúságú részsorozatok véletlenszerű mintavételezésével.
Gyakrabban az adatok sorozatok gyűjteményeként érkeznek.
Tekintsük a következő példákat:
(i) dokumentumok gyűjteménye,
ahol minden dokumentum saját szósorozatként van ábrázolva,
és minden dokumentumnak saját $T_i$ hossza van;
(ii) betegek kórházi tartózkodásainak sorozatos ábrázolása,
ahol minden tartózkodás számos eseményből áll,
és a sorozat hossza nagyjából
a tartózkodás hosszától függ.


Korábban, egyedi bemenetekkel foglalkozva,
feltételeztük, hogy azok egymástól függetlenül kerülnek mintavételezésre
ugyanabból az alapeloszlásból $P(X)$.
Bár még mindig feltételezzük, hogy az egész sorozatok
(pl. teljes dokumentumok vagy beteg-trajektóriák)
egymástól függetlenül kerülnek mintavételezésre,
nem feltételezhetjük, hogy az egyes időlépésekben érkező adatok
egymástól függetlenek.
Például a dokumentum késői részein valószínűleg megjelenő szavak
erősen függnek a dokumentum korábbi részein előforduló szavaktól.
Az a gyógyszer, amelyet egy beteg valószínűleg
a kórházi tartózkodás 10. napján kap,
erősen függ attól, mi történt
az előző kilenc nap során.

Ez nem meglepő.
Ha nem hinnénk, hogy egy sorozat elemei összefüggnek egymással,
fel sem merülne, hogy sorozatként modellezzük őket.
Gondoljunk az automatikus kitöltési funkciók hasznosságára,
amelyek népszerűek a keresőeszközökben és a modern e-mail kliensekben.
Pontosan azért hasznosak, mert gyakran lehetséges
előrejelezni (tökéletlenül, de véletlenszerű találgatásnál jobban)
egy sorozat valószínű folytatásait,
adott néhány kezdeti előtagból.
A legtöbb sorozatmodellnél
nem követeljük meg a sorozatok függetlenségét,
sőt még a stacionaritásukat sem.
Ehelyett csak azt követeljük meg,
hogy maguk a sorozatok mintavételezése
valamely rögzített alapeloszlásból történjen
a teljes sorozatokra vonatkozóan.

Ez a rugalmas megközelítés lehetővé teszi az olyan jelenségeket, mint
(i) a dokumentumok, amelyek jelentősen eltérően néznek ki
az elején és a végén;
(ii) a betegek állapota, amely akár a gyógyulás,
akár a halál felé fejlődik a kórházi tartózkodás során;
(iii) a vásárlók ízlése, amely előre jelezhető módon változik
az ajánlórendszerrel való folyamatos interakció során.


Néha rögzített $y$ célt szeretnénk előrejelezni
sorozatosan strukturált bemenet alapján
(pl. hangulatelemzés filmkritika alapján).
Máskor sorozatosan strukturált célt szeretnénk előrejelezni
($y_1, \ldots, y_T$)
rögzített bemenet alapján (pl. képfeliratozás).
Még más esetekben a cél sorozatosan strukturált célok előrejelzése
sorozatosan strukturált bemenetek alapján
(pl. gépi fordítás vagy videófeliratozás).
Az ilyen sorozat-sorozat feladatok két formát öltenek:
(i) *igazított*: ahol minden időlépés bemenete
megfelel egy megfelelő célnak (pl. szófaji elemzés);
(ii) *nem igazított*: ahol a bemenet és a cél
nem feltétlenül mutat lépésenkénti megfeleltetést
(pl. gépi fordítás).

Mielőtt bármilyen célok kezeléséről aggódnánk,
a legegyszerűbb problémával foglalkozhatunk:
felügyelet nélküli sűrűségmodellezés (más néven *sorozatmodellezés*).
Adott sorozatgyűjtemény esetén
célunk a valószínűségi tömegfüggvény becslése,
amely megmondja, mennyire valószínű, hogy egy adott sorozatot látunk,
azaz $p(\mathbf{x}_1, \ldots, \mathbf{x}_T)$.

```{.python .input  n=6}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input  n=7}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon, init
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=8}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=9}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input  n=9}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
```

## Autoregresszív modellek


Mielőtt bevezetnénk a sorozatosan strukturált adatok kezelésére tervezett
speciális neurális hálózatokat,
nézzünk meg néhány tényleges sorozatadatot,
és alakítsunk ki néhány alapvető intuíciót és statisztikai eszközt.
Különösen az FTSE 100 indexből származó részvényárfolyam-adatokra
összpontosítunk (:numref:`fig_ftse100`).
Minden $t \in \mathbb{Z}^+$ *időlépésnél* megfigyeljük
az index $x_t$ árfolyamát abban az időpontban.


![Az FTSE 100 index körülbelül 30 éven át.](../img/ftse100.png)
:width:`400px`
:label:`fig_ftse100`


Tegyük fel, hogy egy kereskedő rövid távú ügyleteket szeretne kötni,
stratégiailag be- vagy kilépve az indexből,
attól függően, hogy úgy véli-e,
hogy az a következő időlépésben emelkedik vagy csökken.
Bármilyen más jellemző hiányában
(hírek, pénzügyi jelentési adatok stb.),
az egyetlen rendelkezésre álló jel az utóbbi
értékének előrejelzéséhez az eddigi árfolyam-előzmény.
A kereskedő tehát érdekelt a valószínűségi eloszlás megismerésében

$$P(x_t \mid x_{t-1}, \ldots, x_1)$$

az árfolyamokra vonatkozóan, amelyeket az index
a következő időlépésben felvehet.
Bár a teljes eloszlás becslése
egy folytonos értékű véletlen változóra nehéz lehet,
a kereskedő szívesen foglalkozna
az eloszlás néhány kulcsos statisztikájával,
különösen a várható értékkel és a varianciával.
A feltételes várható érték becslésének egy egyszerű stratégiája

$$\mathbb{E}[(x_t \mid x_{t-1}, \ldots, x_1)],$$

lineáris regressziós modell alkalmazása lenne
(vö. :numref:`sec_linear_regression`).
Az olyan modellek, amelyek egy jel értékét
ugyanazon jel korábbi értékeire regresszálják,
természetesen *autoregresszív modelleknek* nevezzük.
Csupán egy fő probléma van: a bemenetek száma,
$x_{t-1}, \ldots, x_1$, $t$-től függően változik.
Más szóval, a bemenetek száma növekszik
az általunk tapasztalt adatok mennyiségével.
Tehát ha a korábbi adatainkat tanítási halmazként kívánjuk kezelni,
azzal a problémával szembesülünk,
hogy minden példának eltérő számú jellemzője van.
Az ebben a fejezetben következő tartalom nagy részének
ezeknek a kihívásoknak az áthidalási technikái körül fog forogni
az ilyen *autoregresszív* modellezési problémák esetén,
ahol az érdeklődés tárgya
$P(x_t \mid x_{t-1}, \ldots, x_1)$
vagy ezen eloszlás valamilyen statisztikája.

Néhány stratégia visszatérően megjelenik.
Mindenekelőtt hihetnénk, hogy bár hosszú sorozatok
$x_{t-1}, \ldots, x_1$ elérhetők,
esetleg nem szükséges
ilyen messzire visszamenni az előzményben
a közeli jövő előrejelzésekor.
Ebben az esetben megelégedhetnénk
azzal, hogy csak egy $\tau$ hosszúságú ablakon kondicionálunk,
és csak az $x_{t-1}, \ldots, x_{t-\tau}$ megfigyeléseket vesszük figyelembe.
Az azonnali előny az, hogy az argumentumok száma
most mindig azonos, legalábbis $t > \tau$ esetén.
Ez lehetővé teszi bármely lineáris modell vagy deep network tanítását,
amelyek rögzített hosszúságú vektorokat igényelnek bemenetként.
Másodszor, fejleszthetünk olyan modelleket, amelyek fenntartanak
valamilyen $h_t$ összefoglalást a múltbeli megfigyelésekről
(ld. :numref:`fig_sequence-model`),
és ugyanakkor frissítik $h_t$-t
az $\hat{x}_t$ előrejelzés mellett.
Ez olyan modellekhez vezet, amelyek nem csak $x_t$-t becslik meg
$\hat{x}_t = P(x_t \mid h_{t})$ formájában,
hanem $h_t = g(h_{t-1}, x_{t-1})$ alakú frissítéseket is végeznek.
Mivel $h_t$ soha nem figyelhető meg,
ezeket a modelleket *látens autoregresszív modelleknek* is nevezik.

![Egy látens autoregresszív modell.](../img/sequence-model.svg)
:label:`fig_sequence-model`

A korábbi adatokból tanítási adatok létrehozásához
általában ablakok véletlenszerű mintavételezésével hozunk létre példákat.
Általánosságban nem várjuk el, hogy az idő megálljon.
Azonban azt gyakran feltételezzük, hogy bár
az $x_t$ konkrét értékei változhatnak,
az a dinamika, amely szerint minden egymást követő
megfigyelés a korábbi megfigyelések alapján keletkezik, nem változik.
A statisztikusok az olyan dinamikákat, amelyek nem változnak, *stacionáriusnak* nevezik.



## Sorozatmodellek

Néha, különösen nyelvi adatokkal dolgozva,
egy egész sorozat együttes valószínűségét kívánjuk becsülni.
Ez egy közönséges feladat, amikor diszkrét *tokenekből*,
például szavakból álló sorozatokkal dolgozunk.
Ezeket a becsült függvényeket általában *sorozatmodelleknek* nevezzük,
és természetes nyelvi adatok esetén *nyelvmodelleknek* hívják őket.
A sorozatmodellezés területét annyira meghatározta a természetes nyelvfeldolgozás,
hogy a sorozatmodelleket gyakran "nyelvmodelleknek" nevezzük,
még akkor is, amikor nem nyelvi adatokkal foglalkozunk.
A nyelvmodellek mindenféle okból hasznosnak bizonyulnak.
Néha mondatok valószínűségét szeretnénk értékelni.
Például összehasonlíthatjuk egy gépi fordítórendszer
vagy egy beszédfelismerő rendszer által generált
két jelöltkimenet természetességét.
A nyelvmodellezés azonban nem csak azt teszi lehetővé, hogy
*értékeljük* a valószínűséget,
hanem hogy *mintavételezzünk* sorozatokat,
sőt optimalizáljuk a legvalószínűbb sorozatokra.

Bár a nyelvmodellezés elsőre nem tűnik autoregresszív problémának,
a nyelvmodellezést redukálhatjuk autoregresszív előrejelzésre
azáltal, hogy egy sorozat $p(x_1, \ldots, x_T)$ együttes sűrűségét
feltételes sűrűségek szorzatára bontjuk
balról jobbra haladó módon,
alkalmazva a valószínűség láncolási szabályát:

$$P(x_1, \ldots, x_T) = P(x_1) \prod_{t=2}^T P(x_t \mid x_{t-1}, \ldots, x_1).$$

Megjegyezzük, hogy ha diszkrét jelzésekkel, például szavakkal dolgozunk,
akkor az autoregresszív modellnek valószínűségi osztályozónak kell lennie,
amely teljes valószínűségi eloszlást ad ki
a szókincs felett arra vonatkozóan, hogy melyik szó jön következőnek,
az addigi baloldali kontextus ismeretében.



### Markov-modellek
:label:`subsec_markov-models`


Most tegyük fel, hogy a fent említett stratégiát kívánjuk alkalmazni,
ahol csak az előző $\tau$ időlépésre kondicionálunk,
azaz $x_{t-1}, \ldots, x_{t-\tau}$-ra, nem pedig
az egész sorozat előzményére $x_{t-1}, \ldots, x_1$.
Ha az előző $\tau$ lépésen túli előzményt el tudjuk dobni
az előrejelzési erőből való veszteség nélkül,
akkor azt mondjuk, hogy a sorozat kielégíti a *Markov-feltételt*,
azaz *a jövő feltételesen független a múlttól,
az utóbbi előzmény ismeretében*.
Amikor $\tau = 1$, azt mondjuk, hogy az adatokat
*elsőrendű Markov-modell* jellemzi,
és ha $\tau = k$, akkor azt mondjuk, hogy az adatokat
$k^{\textrm{th}}$-rendű Markov-modell jellemzi.
Ha az elsőrendű Markov-feltétel teljesül ($\tau = 1$),
az együttes valószínűség faktorizációja szorzattá válik,
ahol minden szó valószínűsége az előző *szótól* függ:

$$P(x_1, \ldots, x_T) = P(x_1) \prod_{t=2}^T P(x_t \mid x_{t-1}).$$

Gyakran hasznosnak találjuk az olyan modelleket,
amelyek úgy viselkednek, mintha a Markov-feltétel teljesülne,
még akkor is, ha tudjuk, hogy ez csak *közelítőleg* igaz.
Valódi szöveges dokumentumoknál egyre több információt nyerünk,
ahogy egyre több baloldali kontextust veszünk figyelembe.
De ezek a nyereségek gyorsan csökkennek.
Ezért néha kompromisszumot kötünk, elkerülve a számítási és statisztikai nehézségeket
azáltal, hogy olyan modelleket tanítunk, amelyek érvényessége
egy $k^{\textrm{th}}$-rendű Markov-feltételtől függ.
Még a mai hatalmas, RNN- és Transformer-alapú nyelvmodellek is
ritkán tartalmaznak több ezernyi szónyi kontextust.


Diszkrét adatok esetén egy valódi Markov-modell
egyszerűen megszámolja, hányszor
fordult elő minden szó minden kontextusban, előállítva
$P(x_t \mid x_{t-1})$ relatív frekvencia becslését.
Amikor az adatok csak diszkrét értékeket vesznek fel
(mint a nyelvben),
a szavak legvalószínűbb sorozata hatékonyan kiszámítható
dinamikus programozással.


### A dekódolás sorrendje

Talán azon tűnődsz, miért ábrázoltuk
a szöveges sorozat $P(x_1, \ldots, x_T)$ faktorizációját
feltételes valószínűségek balról jobbra haladó láncolatként.
Miért ne jobbról balra, vagy valamilyen más, látszólag véletlen sorrendben?
Elvileg nincs semmi baj azzal, ha
$P(x_1, \ldots, x_T)$-t fordított sorrendben fejtjük ki.
Az eredmény érvényes faktorizáció:

$$P(x_1, \ldots, x_T) = P(x_T) \prod_{t=T-1}^1 P(x_t \mid x_{t+1}, \ldots, x_T).$$


Azonban számos oka van annak, hogy a szöveg faktorizálása
ugyanabban az irányban, amelyben olvassuk
(balról jobbra a legtöbb nyelvnél,
de jobbról balra arabul és héberül)
előnyösebb a nyelvmodellezési feladathoz.
Először is, ez csak természetesebb irány a gondolkodásunkhoz.
Végül is mindenki naponta olvas szövegeket,
és ez a folyamat irányítja azt a képességünket,
hogy előre jelezzük, milyen szavak és kifejezések
várhatóan következnek.
Gondolj csak arra, hányszor fejezted be
valaki más mondatát.
Tehát még ha nem is lenne más okunk az ilyen sorrendű dekódolások preferálásához,
hasznosak lennének, pusztán azért, mert jobb intuícióink vannak
arra, mi legyen valószínű, ha ebben a sorrendben jóslunk.

Másodszor, sorrendben faktorizálva
tetszőlegesen hosszú sorozatokhoz rendelhetünk valószínűségeket
ugyanazon nyelvmodell használatával.
Az $1$-től $t$-ig tartó lépések feletti valószínűség
$t+1$-es szóra való kiterjesztéséhez egyszerűen
megszorozzuk a következő token feltételes valószínűségével
az előzőekre kondicionálva:
$P(x_{t+1}, \ldots, x_1) = P(x_{t}, \ldots, x_1) \cdot P(x_{t+1} \mid x_{t}, \ldots, x_1)$.

Harmadszor, erősebb prediktív modelljeink vannak
a szomszédos szavak előrejelzéséhez, mint
a tetszőleges más helyen lévő szavak előrejelzéséhez.
Bár a faktorizáció minden sorrendje érvényes,
nem feltétlenül jelölnek egyformán könnyű
prediktív modellezési problémákat.
Ez nem csak a nyelvre igaz,
hanem más típusú adatokra is,
pl. amikor az adatoknak oksági struktúrájuk van.
Például úgy véljük, hogy a jövőbeli események nem befolyásolhatják a múltat.
Ezért ha megváltoztatjuk $x_t$-t, talán befolyásolhatjuk,
mi történik $x_{t+1}$-nél előre, de nem fordítva.
Vagyis ha megváltoztatjuk $x_t$-t, a múltbeli eseményeket lefedő eloszlás nem változik.
Egyes összefüggésekben ez megkönnyíti $P(x_{t+1} \mid x_t)$ becslését
szemben $P(x_t \mid x_{t+1})$ becslésével.
Például bizonyos esetekben megtalálhatjuk, hogy $x_{t+1} = f(x_t) + \epsilon$
valamely additív $\epsilon$ zajra,
míg a fordítottja nem igaz :cite:`Hoyer.Janzing.Mooij.ea.2009`.
Ez nagyszerű hír, mivel általában az előremutató irányban
vagyunk érdekeltek a becslésben.
A :citet:`Peters.Janzing.Scholkopf.2017` könyv bővebb tárgyalást tartalmaz erről a témáról.
Mi alig karcolgatjuk ennek felszínét.


## Tanítás

Mielőtt szöveges adatokra összpontosítanánk,
először próbáljuk ki ezt néhány
folytonos értékű szintetikus adattal.

(**Itt 1000 szintetikus adatpontunk a trigonometrikus `sin` függvényt
fogja követni, amelyet az időlépés 0,01-szeresére alkalmazunk.
Hogy a problémát egy kicsit érdekesebbé tegyük,
minden mintát additív zajjal rontunk meg.**)
Ebből a sorozatból tanítási példákat vonunk ki,
amelyek mindegyike jellemzőkből és egy címkéből áll.

```{.python .input  n=10}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, batch_size=16, T=1000, num_train=600, tau=4):
        self.save_hyperparameters()
        self.time = d2l.arange(1, T + 1, dtype=d2l.float32)
        if tab.selected('mxnet', 'pytorch'):
            self.x = d2l.sin(0.01 * self.time) + d2l.randn(T) * 0.2
        if tab.selected('tensorflow'):
            self.x = d2l.sin(0.01 * self.time) + d2l.normal([T]) * 0.2
        if tab.selected('jax'):
            key = d2l.get_key()
            self.x = d2l.sin(0.01 * self.time) + jax.random.normal(key,
                                                                   [T]) * 0.2
```

```{.python .input}
%%tab all
data = Data()
d2l.plot(data.time, data.x, 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

Kezdetnek olyan modellt próbálunk, amely úgy viselkedik, mintha
az adatok kielégítenék a $\tau^{\textrm{th}}$-rendű Markov-feltételt,
és így $x_t$-t csak az elmúlt $\tau$ megfigyelés alapján jósolja.
**Tehát minden időlépésnél egy példánk van
$y = x_t$ címkével és
$\mathbf{x}_t = [x_{t-\tau}, \ldots, x_{t-1}]$ jellemzőkkel.**
Az éles szemű olvasó észrevehette, hogy
ez $1000-\tau$ példát eredményez,
mivel nincs elegendő előzményünk $y_1, \ldots, y_\tau$-hoz.
Bár kitölthetnénk az első $\tau$ sorozatot nullákkal,
az egyszerűség kedvéért most kihagyjuk őket.
Az eredményül kapott adathalmaz $T - \tau$ példát tartalmaz,
ahol a modell minden bemenete $\tau$ sorozathosszú.
(**Az első 600 példán adatiterátort hozunk létre**),
amely a sin függvény egy periódusát fedi le.

```{.python .input}
%%tab all
@d2l.add_to_class(Data)
def get_dataloader(self, train):
    features = [self.x[i : self.T-self.tau+i] for i in range(self.tau)]
    self.features = d2l.stack(features, 1)
    self.labels = d2l.reshape(self.x[self.tau:], (-1, 1))
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader([self.features, self.labels], train, i)
```

Ebben a példában a modellünk standard lineáris regresszió lesz.

```{.python .input}
%%tab all
model = d2l.LinearRegression(lr=0.01)
trainer = d2l.Trainer(max_epochs=5)
trainer.fit(model, data)
```

## Előrejelzés

**A modell értékeléséhez először ellenőrizzük,
mennyire jól teljesít az egy lépéssel előre való előrejelzésnél**.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
onestep_preds = d2l.numpy(model(data.features))
d2l.plot(data.time[data.tau:], [data.labels, onestep_preds], 'time', 'x',
         legend=['labels', '1-step preds'], figsize=(6, 3))
```

```{.python .input}
%%tab jax
onestep_preds = model.apply({'params': trainer.state.params}, data.features)
d2l.plot(data.time[data.tau:], [data.labels, onestep_preds], 'time', 'x',
         legend=['labels', '1-step preds'], figsize=(6, 3))
```

Ezek az előrejelzések jól néznek ki,
még a végén is, $t=1000$-nél.

De mi van, ha csak a 604-es időlépésig (`n_train + tau`)
figyeltük meg a sorozatadatokat,
és több lépéssel előre szeretnénk előrejelzést adni?
Sajnos nem tudjuk közvetlenül kiszámítani
a 609-es időlépés egy lépéssel előre való előrejelzését,
mivel nem ismerjük a megfelelő bemeneteket,
csak $x_{604}$-ig láttunk adatokat.
Ezt a problémát úgy kezelhetjük, hogy
korábbi előrejelzéseinket bedugozzuk modellünk bemenetébe
a következő előrejelzések elkészítéséhez,
egy lépéssel előre haladva,
amíg el nem érjük a kívánt időlépést:

$$\begin{aligned}
\hat{x}_{605} &= f(x_{601}, x_{602}, x_{603}, x_{604}), \\
\hat{x}_{606} &= f(x_{602}, x_{603}, x_{604}, \hat{x}_{605}), \\
\hat{x}_{607} &= f(x_{603}, x_{604}, \hat{x}_{605}, \hat{x}_{606}),\\
\hat{x}_{608} &= f(x_{604}, \hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}),\\
\hat{x}_{609} &= f(\hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}, \hat{x}_{608}),\\
&\vdots\end{aligned}$$

Általánosan, az $x_1, \ldots, x_t$ megfigyelt sorozat esetén
a $t+k$ időlépésnél lévő $\hat{x}_{t+k}$ előrejelzett kimenet
$k$*-lépéssel előre való előrejelzésnek* nevezik.
Mivel $x_{604}$-ig figyeltük meg az adatokat,
a $k$-lépéssel előre való előrejelzés $\hat{x}_{604+k}$.
Más szóval, folyamatosan saját előrejelzéseinket kell
használnunk a több lépéssel előre való előrejelzésekhez.
Lássuk, hogyan megy ez.

```{.python .input}
%%tab mxnet, pytorch
multistep_preds = d2l.zeros(data.T)
multistep_preds[:] = data.x
for i in range(data.num_train + data.tau, data.T):
    multistep_preds[i] = model(
        d2l.reshape(multistep_preds[i-data.tau : i], (1, -1)))
multistep_preds = d2l.numpy(multistep_preds)
```

```{.python .input}
%%tab tensorflow
multistep_preds = tf.Variable(d2l.zeros(data.T))
multistep_preds[:].assign(data.x)
for i in range(data.num_train + data.tau, data.T):
    multistep_preds[i].assign(d2l.reshape(model(
        d2l.reshape(multistep_preds[i-data.tau : i], (1, -1))), ()))
```

```{.python .input}
%%tab jax
multistep_preds = d2l.zeros(data.T)
multistep_preds = multistep_preds.at[:].set(data.x)
for i in range(data.num_train + data.tau, data.T):
    pred = model.apply({'params': trainer.state.params},
                       d2l.reshape(multistep_preds[i-data.tau : i], (1, -1)))
    multistep_preds = multistep_preds.at[i].set(pred.item())
```

```{.python .input}
%%tab all
d2l.plot([data.time[data.tau:], data.time[data.num_train+data.tau:]],
         [onestep_preds, multistep_preds[data.num_train+data.tau:]], 'time',
         'x', legend=['1-step preds', 'multistep preds'], figsize=(6, 3))
```

Sajnos ebben az esetben látványosan kudarcot vallunk.
Az előrejelzések néhány lépés után
meglehetősen gyorsan konstansba csökkennek.
Miért teljesített az algoritmus annyival rosszabbul,
ha távolabb a jövőbe jósol?
Végső soron ez abból adódik, hogy a hibák felhalmozódnak.
Mondjuk, hogy az 1. lépés után $\epsilon_1 = \bar\epsilon$ hibánk van.
Most a 2. lépés *bemenete* $\epsilon_1$-vel perturbált,
ezért valamilyen $\epsilon_2 = \bar\epsilon + c \epsilon_1$
rendű hibát szenvedünk el valamely $c$ konstansra, és így tovább.
Az előrejelzések gyorsan eltérhetnek
a tényleges megfigyelésektől.
Talán már ismered ezt a közönséges jelenséget.
Például az időjárás-előrejelzések a következő 24 órára
meglehetősen pontosak, de azon túl
a pontosság gyorsan csökken.
A fejezet során és azon túl tárgyaljuk az ennek javítási módszereit.

Vizsgáljuk meg **közelebbről a $k$-lépéssel előre való előrejelzések nehézségeit**
az egész sorozaton való előrejelzések kiszámításával $k = 1, 4, 16, 64$-re.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
def k_step_pred(k):
    features = []
    for i in range(data.tau):
        features.append(data.x[i : i+data.T-data.tau-k+1])
    # A (i+tau)-adik elem tárolja az (i+1)-lépéses előrejelzéseket
    for i in range(k):
        preds = model(d2l.stack(features[i : i+data.tau], 1))
        features.append(d2l.reshape(preds, -1))
    return features[data.tau:]
```

```{.python .input}
%%tab jax
def k_step_pred(k):
    features = []
    for i in range(data.tau):
        features.append(data.x[i : i+data.T-data.tau-k+1])
    # A (i+tau)-adik elem tárolja az (i+1)-lépéses előrejelzéseket
    for i in range(k):
        preds = model.apply({'params': trainer.state.params},
                            d2l.stack(features[i : i+data.tau], 1))
        features.append(d2l.reshape(preds, -1))
    return features[data.tau:]
```

```{.python .input}
%%tab all
steps = (1, 4, 16, 64)
preds = k_step_pred(steps[-1])
d2l.plot(data.time[data.tau+steps[-1]-1:],
         [d2l.numpy(preds[k-1]) for k in steps], 'time', 'x',
         legend=[f'{k}-step preds' for k in steps], figsize=(6, 3))
```

Ez jól szemlélteti, hogyan változik az előrejelzés minősége,
ahogy távolabb próbálunk jósolni a jövőbe.
Míg a 4-lépéses előrejelzések még jól néznek ki,
azon túl szinte minden hasznavehetetlen.

## Összefoglalás

Jelentős különbség van a nehézségek tekintetében
az interpoláció és az extrapoláció között.
Következésképpen, ha sorozatod van, mindig tiszteld
az adatok időbeli sorrendjét tanítás közben,
azaz soha ne tanítsd a modellt jövőbeli adatokon.
Az ilyen adatok esetén
a sorozatmodellek speciális statisztikai eszközöket igényelnek a becsléshez.
Két népszerű választás az autoregresszív modellek
és a látens változó autoregresszív modellek.
Az oksági modellek esetén (pl. az idő előrehaladásával)
az előremutató irány becslése általában
sokkal könnyebb, mint a fordított irányé.
Egy adott sorozatnál, amelyet $t$ időlépésig figyeltünk meg,
a $t+k$ időlépésnél megjósolt kimenet
a $k$*-lépéssel előre való előrejelzés*.
Ahogy egyre távolabb jóslunk az idő előrehaladtával $k$ növelésével,
a hibák felhalmozódnak és az előrejelzés minősége romlik,
gyakran drámai mértékben.

## Feladatok

1. Javítsd a modellt ebben a fejezetben található kísérletben.
    1. Vegyél figyelembe négynél több múltbeli megfigyelést! Valójában mennyire van szükség?
    1. Mennyi múltbeli megfigyelésre lenne szükség, ha nem lenne zaj? Tipp: a $\sin$-t és $\cos$-t differenciálegyenletként is írhatod.
    1. Figyelembe vehetsz-e régebbi megfigyeléseket úgy, hogy a jellemzők teljes száma állandó marad? Ez javítja a pontosságot? Miért?
    1. Változtasd meg a neurális hálózat architektúráját, és értékeld a teljesítményt! Az új modellt több epochon is taníthatod. Mit figyelsz meg?
1. Egy befektető jó értékpapírt szeretne vásárolni.
   Megvizsgálja a múltbeli hozamokat, hogy eldöntse, melyik fog valószínűleg jól teljesíteni.
   Mi mehet félre ezzel a stratégiával?
1. Vonatkozik-e az oksálisság szövegre is? Milyen mértékben?
1. Adj egy példát arra, amikor látens autoregresszív modellre
   lehet szükség az adatok dinamikájának megragadásához.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/113)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/114)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1048)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18010)
:end_tab:
