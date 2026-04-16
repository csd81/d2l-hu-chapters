```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Házárak előrejelzése Kaggle-ön
:label:`sec_kaggle_house`

Most, hogy bemutattuk a mély hálózatok felépítéséhez és tanításához
szükséges néhány alapeszközt,
valamint regularizálási technikákat, beleértve
a súlybomlást és a dropoutot,
készen állunk arra, hogy mindezeket a tudásokat
egy Kaggle-versenyen való részvétellel a gyakorlatban alkalmazzuk.
A házárak előrejelzési verseny
remek kezdőpont.
Az adatok elég általánosak, és nem mutatnak egzotikus struktúrát,
ami speciális modelleket igényelne (ahogy az audió vagy videó esetén szükséges lehet).
Ez az adathalmaz, amelyet :citet:`De-Cock.2011` gyűjtött össze,
az iowai Ames városában a 2006--2010-es időszakban lévő házárakat tartalmazza.
Lényegesen nagyobb, mint Harrison és Rubinfeld (1978) híres
[Boston housing adathalmaza](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names),
több példányt és több jellemzőt kínálva.


Ebben a részben végigvezetünk az
adatok előfeldolgozásának, a modell tervezésének és a hiperparaméter-kiválasztásnak a részletein.
Reméljük, hogy egy gyakorlati megközelítésen keresztül
olyan intuíciókat szerezel, amelyek irányítani fognak
adattudósi karrieredben.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd

npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import pandas as pd
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import pandas as pd
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import pandas as pd
```

## Adatok letöltése

A könyv egészében különböző letöltött adathalmazokon
tanítjuk és teszteljük a modelleket.
Ezért (**két segédfüggvényt implementálunk**)
zip vagy tar fájlok letöltéséhez és kicsomagolásához.
Az ilyen segédfüggvények implementálási részleteit most átugorjuk.

```{.python .input  n=2}
%%tab all
def download(url, folder, sha1_hash=None):
    """Letölt egy fájlt a megadott mappába, és visszaadja a helyi fájlútvonalat."""

def extract(filename, folder):
    """Kicsomagol egy zip/tar fájlt a megadott mappába."""
```

## Kaggle

A [Kaggle](https://www.kaggle.com) egy népszerű platform,
amely gépi tanulási versenyeknek ad otthont.
Minden verseny egy adathalmaz köré épül, és sokat
érdekelt felek szponzorálnak, akik díjakat ajánlanak fel
a nyertes megoldásoknak.
A platform segíti a felhasználókat
fórumokon és megosztott kódon keresztüli interakcióban,
erősítve az együttműködést és a versenyt egyaránt.
Bár a ranglistán való előrejutás gyakran kicsúszik az irányítás alól,
mivel a kutatók rövidlátóan az előfeldolgozási lépésekre összpontosítanak
az alapvető kérdések feltevése helyett,
óriási értéke van a platform objektivitásának is,
amely közvetlen kvantitatív összehasonlításokat tesz lehetővé
a versengő megközelítések között, valamint kódmegosztást,
hogy mindenki tanulhasson abból, mi működött és mi nem.
Ha részt szeretnél venni egy Kaggle-versenyen,
először regisztrálnod kell egy fiókot
(lásd :numref:`fig_kaggle`).

![A Kaggle weboldal.](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

A házárak előrejelzési verseny oldalán, ahogy
a :numref:`fig_house_pricing` ábrán látható,
megtalálhatja az adathalmazt (az „Adatok" fülön),
benyújthatja az előrejelzéseit, és megtekintheti a rangsorát.
Az URL itt van:

> https://www.kaggle.com/c/house-prices-advanced-regression-techniques

![A házárak előrejelzési verseny oldala.](../img/house-pricing.png)
:width:`400px`
:label:`fig_house_pricing`

## Az adathalmaz elérése és beolvasása

Vegyük észre, hogy a verseny adatai
tanítási és teszt halmazokra vannak felosztva.
Minden rekord tartalmazza az ingatlan értékét
és olyan attribútumokat, mint az utca típusa, az építés éve,
a tető típusa, a pince állapota stb.
A jellemzők különböző adattípusokból állnak.
Például az építés éve
egész számmal van megadva,
a tető típusa diszkrét kategorikus értékekkel,
más jellemzők lebegőpontos számokkal.
És itt bonyolítja a valóság a dolgokat:
néhány példánynál egyes adatok teljesen hiányoznak,
a hiányzó értéket egyszerűen „na"-val jelölve.
A házak ára csak
a tanítóhalmazban szerepel
(ez mégiscsak egy verseny).
A tanítóhalmazt el szeretnénk osztani
egy validációs halmaz létrehozásához,
de a modelleket csak a hivatalos teszt halmazon értékelhetjük
az előrejelzések Kaggle-ra való feltöltése után.
A :numref:`fig_house_pricing` verseny oldal
„Adatok" fülén
linkek vannak az adatok letöltéséhez.

A kezdéshez [**beolvassuk és feldolgozzuk az adatokat
a `pandas` segítségével**], amelyet a :numref:`sec_pandas` részben mutattunk be.
Kényelmesen le tudjuk tölteni és gyorsítótárazni
a Kaggle-lakás adathalmazt.
Ha egy ehhez az adathalmazhoz tartozó fájl már létezik a gyorsítótár könyvtárban, és a SHA-1-je megegyezik a `sha1_hash`-sel, a kódunk a gyorsítótárazott fájlt fogja használni, hogy ne terheljük az internetet felesleges letöltésekkel.

```{.python .input  n=30}
%%tab all
class KaggleHouse(d2l.DataModule):
    def __init__(self, batch_size, train=None, val=None):
        super().__init__()
        self.save_hyperparameters()
        if self.train is None:
            self.raw_train = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_train.csv', self.root,
                sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
            self.raw_val = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_test.csv', self.root,
                sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))
```

A tanítási adathalmaz 1460 példányt,
80 jellemzőt és egy címkét tartalmaz, míg a validációs adat
1459 példányt és 80 jellemzőt tartalmaz.

```{.python .input  n=31}
%%tab all
data = KaggleHouse(batch_size=64)
print(data.raw_train.shape)
print(data.raw_val.shape)
```

## Adatok előfeldolgozása

[**Nézzük meg az első négy és az utolsó két jellemzőt
valamint a címkét (SalePrice)**] az első négy példányból.

```{.python .input  n=10}
%%tab all
print(data.raw_train.iloc[:4, [0, 1, 2, 3, -3, -2, -1]])
```

Láthatjuk, hogy minden példányban az első jellemző az azonosító.
Ez segít a modellnek azonosítani minden egyes tanítási példányt.
Bár ez kényelmes, nem hordoz
semmilyen információt előrejelzési célokra.
Ezért eltávolítjuk az adathalmazból
a modellbe való betáplálás előtt.
Ráadásul, tekintettel az adattípusok széles változatosságára,
az adatokat előfeldolgozni kell, mielőtt modellezni kezdhetjük.


Kezdjük a numerikus jellemzőkkel.
Először egy heurisztikát alkalmazunk,
[**az összes hiányzó értéket
a megfelelő jellemző átlagával pótolva.**]
Ezután, hogy az összes jellemzőt közös skálára hozzuk,
***standardizáljuk* az adatokat azáltal, hogy
a jellemzőket nulla átlagra és egységnyi varianciára méretezzük**:

$$x \leftarrow \frac{x - \mu}{\sigma},$$

ahol $\mu$ és $\sigma$ az átlagot és a szórást jelölik.
Annak igazolásához, hogy ez valóban úgy alakítja át
a jellemzőnket (változónkat), hogy nulla átlaga és egységnyi varianciája legyen,
vegyük észre, hogy $E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$
és hogy $E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$.
Intuíció szerint két okból standardizáljuk az adatokat.
Először kényelmes az optimalizálás szempontjából.
Másodszor, mivel *a priori* nem tudjuk,
melyik jellemzők lesznek relevánsak,
nem akarjuk jobban büntetni az egy jellemzőhöz rendelt együtthatókat
a többinél.

[**Ezután foglalkozunk a diszkrét értékekkel.**]
Ezek közé tartoznak olyan jellemzők, mint az „MSZoning".
(**Ezeket one-hot kódolással helyettesítjük**),
ugyanolyan módon, ahogy korábban
a többosztályos címkéket vektorokká alakítottuk (lásd :numref:`subsec_classification-problem`).
Például az „MSZoning" az „RL" és „RM" értékeket veszi fel.
Az „MSZoning" jellemzőt ejtve,
két új jelzőjellemzőt
„MSZoning_RL" és „MSZoning_RM" hozunk létre 0 vagy 1 értékekkel.
A one-hot kódolás szerint,
ha az „MSZoning" eredeti értéke „RL",
akkor „MSZoning_RL" értéke 1, az „MSZoning_RM" értéke 0.
A `pandas` csomag ezt automatikusan elvégzi helyettünk.

```{.python .input  n=32}
%%tab all
@d2l.add_to_class(KaggleHouse)
def preprocess(self):
    # Az azonosító és a címkeoszlop eltávolítása
    label = 'SalePrice'
    features = pd.concat(
        (self.raw_train.drop(columns=['Id', label]),
         self.raw_val.drop(columns=['Id'])))
    # Numerikus oszlopok standardizálása
    numeric_features = features.dtypes[features.dtypes!='object'].index
    features[numeric_features] = features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # NaN numerikus értékek nullával való pótlása
    features[numeric_features] = features[numeric_features].fillna(0)
    # Diszkrét jellemzők cseréje one-hot kódolásra
    features = pd.get_dummies(features, dummy_na=True)
    # Az előfeldolgozott jellemzők mentése
    self.train = features[:self.raw_train.shape[0]].copy()
    self.train[label] = self.raw_train[label]
    self.val = features[self.raw_train.shape[0]:].copy()
```

Láthatjuk, hogy ez az átalakítás
a jellemzők számát 79-ről 331-re növeli (az azonosító és a címke oszlopokat leszámítva).

```{.python .input  n=33}
%%tab all
data.preprocess()
data.train.shape
```

## Hibamérték

Kezdetnek egy lineáris modellt tanítunk négyzetes veszteséggel. Nem meglepő, hogy a lineáris modellünk nem fog versenynyertes beadványhoz vezetni, de biztonsági ellenőrzést nyújt annak megtekintéséhez, hogy van-e értelmes információ az adatokban. Ha nem tudunk jobban teljesíteni, mint a véletlenszerű tippelés, akkor valószínűleg adatfeldolgozási hibánk van. Ha pedig a dolgok működnek, a lineáris modell alaptervként fog szolgálni, adva némi intuíciót arról, milyen közel kerül az egyszerű modell a legjobb bejelentett modellekhez, és képet kapunk arról, mekkora javulást várhatunk a kifinomultabb modellektől.

A házárak esetén, akárcsak a részvényáraknál,
a relatív mennyiségekkel törődünk
inkább, mint az abszolút mennyiségekkel.
Ezért [**inkább a relatív hibáról
$\frac{y - \hat{y}}{y}$**]
kell gondoskodnunk,
mint az abszolút hibáról $y - \hat{y}$.
Például, ha az előrejelzésünk 100 000 dollárral tér el
egy ohiói vidéki ház árának becslésénél,
ahol egy tipikus ház értéke 125 000 dollár,
akkor valószínűleg nagyon rosszul teljesítünk.
Másrészt, ha ennyit tévedünk
a kaliforniai Los Altos Hillsben,
ez megdöbbentően pontos előrejelzést jelent
(ott a medián házár 4 millió dollárt meghaladja).

(**Az egyik módszer ennek a problémának a kezelésére az,
hogy az árbecslések logaritmusában mérjük a különbséget.**)
Valójában ez egyben a verseny által a beadványok minőségének értékelésére
használt hivatalos hibamérték is.
Végül is egy kis $\delta$ értékre $|\log y - \log \hat{y}| \leq \delta$
ez azt jelenti, hogy $e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$.
Ez a következő gyök átlagos négyzetes hibához (RMSLE) vezet az előrejelzett ár logaritmusa és a tényleges ár logaritmusa között:

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

```{.python .input  n=60}
%%tab all
@d2l.add_to_class(KaggleHouse)
def get_dataloader(self, train):
    label = 'SalePrice'
    data = self.train if train else self.val
    if label not in data: return
    get_tensor = lambda x: d2l.tensor(x.values.astype(float),
                                      dtype=d2l.float32)
    # Az árak logaritmusa
    tensors = (get_tensor(data.drop(columns=[label])),  # X
               d2l.reshape(d2l.log(get_tensor(data[label])), (-1, 1)))  # Y
    return self.get_tensorloader(tensors, train)
```

## $K$-szoros keresztvalidáció

Talán emlékszel, hogy bevezettük a [**keresztvalidációt**]
a :numref:`subsec_generalization-model-selection` részben, ahol megbeszéltük, hogyan kezeljük
a modellválasztást.
Ezt jól fogjuk hasznosítani a modell tervének kiválasztásához
és a hiperparaméterek beállításához.
Először szükségünk van egy olyan függvényre, amely visszaadja
az adatok $i$-edik adatrészét
egy $K$-szoros keresztvalidálási eljárásban.
Úgy halad végig, hogy az $i$-edik adatrészletet választja le
validációs adatként, a maradékot tanítási adatként visszaadva.
Vegyük észre, hogy ez nem a leghatékonyabb módszer az adatok kezelésére,
és valami sokkal okosabbat tennénk,
ha az adathalmazunk lényegesen nagyobb lenne.
De ez a hozzáadott összetettség feleslegesen zavarossá teheti a kódunkat,
ezért nyugodtan elhagyhatjuk itt, a problémánk egyszerűsége miatt.

```{.python .input}
%%tab all
def k_fold_data(data, k):
    rets = []
    fold_size = data.train.shape[0] // k
    for j in range(k):
        idx = range(j * fold_size, (j+1) * fold_size)
        rets.append(KaggleHouse(data.batch_size, data.train.drop(index=idx),  
                                data.train.loc[idx]))    
    return rets
```

[**Az átlagos validációs hiba kerül visszaadásra**],
amikor $K$-szor tanítunk a $K$-szoros keresztvalidációban.

```{.python .input}
%%tab all
def k_fold(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = d2l.LinearRegression(lr)
        model.board.yscale='log'
        if i != 0: model.board.display = False
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
    return models
```

## [**Modellválasztás**]

Ebben a példában egy nem hangolt hiperparaméter-készletet választunk,
és az olvasóra bízzuk a modell fejlesztését.
Egy jó választás megtalálása időt vehet igénybe,
attól függően, hány változót optimalizálunk.
Kellően nagy adathalmaz
és szokásos hiperparaméterek esetén
a $K$-szoros keresztvalidáció általában
kellően rugalmas a többszörös teszteléssel szemben.
Ha azonban ésszerűtlenül nagy számú opciót próbálunk ki,
azt tapasztalhatjuk, hogy a validációs
teljesítményünk már nem reprezentatív a valódi hibára nézve.

```{.python .input}
%%tab all
trainer = d2l.Trainer(max_epochs=10)
models = k_fold(trainer, data, k=5, lr=0.01)
```

Vegyük észre, hogy néha a tanítási hibák száma
egy adott hiperparaméter-készletnél nagyon alacsony lehet,
még akkor is, ha a $K$-szoros keresztvalidációs hibák száma
lényegesen magasabbra nő.
Ez azt jelzi, hogy túlilleszt.
A tanítás során mindkét számot figyelemmel kell követni.
Kevesebb túlillesztés jelezheti, hogy az adataink támogathatnak egy erősebb modellt.
Masszív túlillesztés azt sugallhatja, hogy regularizálási technikák bevezetésével nyerhetnénk.

##  [**Előrejelzések benyújtása a Kaggle-ra**]

Most, hogy tudjuk, mi lenne a hiperparaméterek jó megválasztása,
kiszámíthatjuk az átlagos előrejelzéseket
a teszt halmazra
az összes $K$ modell segítségével.
Az előrejelzések csv fájlba mentése
egyszerűsíti az eredmények Kaggle-ra való feltöltését.
A következő kód egy `submission.csv` nevű fájlt generál.

```{.python .input}
%%tab all
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds = [model(d2l.tensor(data.val.values.astype(float), dtype=d2l.float32))
             for model in models]
if tab.selected('jax'):
    preds = [model.apply({'params': trainer.state.params},
             d2l.tensor(data.val.values.astype(float), dtype=d2l.float32))
             for model in models]
# Az előrejelzések hatványozása a logaritmikus skálán
ensemble_preds = d2l.reduce_mean(d2l.exp(d2l.concat(preds, 1)), 1)
submission = pd.DataFrame({'Id':data.raw_val.Id,
                           'SalePrice':d2l.numpy(ensemble_preds)})
submission.to_csv('submission.csv', index=False)
```

Ezután, ahogy a :numref:`fig_kaggle_submit2` ábrán bemutatott,
benyújthatjuk az előrejelzéseinket a Kaggle-on,
és megnézhetjük, hogyan viszonyulnak a tényleges házárakhoz (címkékhez)
a teszt halmazon.
A lépések elég egyszerűek:

* Jelentkezz be a Kaggle weboldalra, és látogass el a házárak előrejelzési verseny oldalára.
* Kattints az „Előrejelzések benyújtása" vagy a „Késői benyújtás" gombra.
* Kattints az „Beadványfájl feltöltése" gombra a lap alján lévő szaggatott vonalas keretben, és válaszd ki a feltölteni kívánt előrejelzési fájlt.
* Kattints a „Beadvány benyújtása" gombra a lap alján az eredmények megtekintéséhez.

![Adatok benyújtása a Kaggle-ra.](../img/kaggle-submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## Összefoglalás és vita

A valódi adatok gyakran különböző adattípusok keverékét tartalmazzák, és előfeldolgozást igényelnek.
A valós értékű adatok nulla átlagra és egységnyi varianciára való átméretezése jó alapértelmezés. Ugyanez igaz a hiányzó értékek átlaggal való pótlására.
Ezenkívül a kategorikus jellemzők jelzőjellemzőkké alakítása lehetővé teszi, hogy one-hot vektorokként kezeljük őket.
Amikor inkább a relatív hibával foglalkozunk, mint az abszolút hibával, mérhetjük a különbséget az előrejelzés logaritmusában.
A modell kiválasztásához és a hiperparaméterek beállításához $K$-szoros keresztvalidációt alkalmazhatunk.



## Feladatok

1. Nyújtsd be az előrejelzéseidet erre a részre a Kaggle-on. Mennyire jók?
1. Mindig jó ötlet a hiányzó értékeket átlaggal helyettesíteni? Tipp: tudsz-e olyan helyzetet konstruálni, ahol az értékek nem véletlenszerűen hiányoznak?
1. Javítsd a pontszámot a hiperparaméterek hangolásával a $K$-szoros keresztvalidáción keresztül.
1. Javítsd a pontszámot a modell fejlesztésével (pl. rétegek, súlybomlás és dropout).
1. Mi történik, ha nem standardizáljuk a folytonos numerikus jellemzőket, ahogy ebben a részben tettük?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/106)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/107)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/237)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17988)
:end_tab:
