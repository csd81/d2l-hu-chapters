```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# batchnormalizáció
:label:`sec_batch_norm`

A mély neurális hálózatok tanítása nehéz feladat.
Ésszerű idő alatt való konvergáltatásuk trükkös lehet.
Ebben a szakaszban a *batchnormalizációt* írjuk le, egy népszerű és hatékony technikát, amely következetesen felgyorsítja a mély hálózatok konvergenciáját :cite:`Ioffe.Szegedy.2015`.
A reziduális blokkokkal együtt — amelyeket a :numref:`sec_resnet` részben tárgyalunk — a batchnormalizáció lehetővé tette a szakemberek számára, hogy rutinszerűen tanítsanak 100 rétegnél mélyebb hálózatokat.
A batchnormalizáció másodlagos (szerencsés) előnye a benne rejlő regularizáció.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from functools import partial
from jax import numpy as jnp
import jax
import optax
```

## Mély Hálózatok Tanítása

Amikor adatokkal dolgozunk, tanítás előtt gyakran előfeldolgozást végzünk.
Az adatok előfeldolgozásával kapcsolatos döntések sokszor óriási különbséget tesznek a végső eredményben.
Emlékezzünk az MLP-k lakásárak előrejelzésére való alkalmazására (:numref:`sec_kaggle_house`).
Az első lépés, amikor valós adatokkal dolgoztunk, a bemeneti jellemzők standardizálása volt, hogy nulla átlaguk $\boldsymbol{\mu} = 0$ és egységnyi varianciájuk $\boldsymbol{\Sigma} = \boldsymbol{1}$ legyen több megfigyelés esetén :cite:`friedman1987exploratory`, az utóbbit gyakran átméretezve úgy, hogy az átló egységnyi, azaz $\Sigma_{ii} = 1$.
Egy másik stratégia a vektorokat egységnyi hosszúra átméretezni, esetleg nulla átlaggal *megfigyelésenként*.
Ez jól működhet például térbeli érzékelőadatoknál. Ezek az előfeldolgozási technikák és sok más egyéb hasznos a becslési probléma jó kézben tartásához.
A jellemzők kiválasztásáról és kinyeréséről szóló áttekintéshez lásd például :citet:`guyon2008feature` cikkét.
A vektorok standardizálásának az a kellemes mellékhatása is van, hogy korlátozza az azokon ható függvények összetettségét. Például a support vector machines-ben alkalmazott klasszikus radius-margin korlát :cite:`Vapnik95` és a Perceptron Convergence Theorem :cite:`Novikoff62` korlátozott normájú bemenetekre támaszkodik.

Intuitívan ez a standardizálás jól működik az optimalizálóinkkal, mivel a paramétereket *a priori* hasonló skálán helyezi el.
Ezért természetes felvetni, hogy egy mély hálózaton *belüli* megfelelő normalizálási lépés vajon nem lenne-e hasznos. Bár ez nem egészen az a gondolat, amely a batchnormalizáció :cite:`Ioffe.Szegedy.2015` feltalálásához vezetett, mégis hasznos módja annak megértésének — és rokonával, a rétegnormalizációval :cite:`Ba.Kiros.Hinton.2016` együtt — egy egységes keretrendszeren belül.

Másodszor, egy tipikus MLP vagy CNN esetén tanítás közben a közbenső rétegek változóinak értékei (pl. az MLP affin transzformációjának kimenetei) nagyon eltérő nagyságrendű értékeket vehetnek fel: akár a bemenettől a kimenetig lévő rétegek mentén, akár az ugyanazon rétegben lévő egységek között, akár időben a modellparaméterek frissítésének következtében.
A batchnormalizáció feltalálói informálisan azt állították, hogy az ilyen változók eloszlásának ez az eltolódása akadályozhatja a hálózat konvergenciáját.
Intuitívan azt sejthetjük, hogy ha az egyik réteg változó aktivációi 100-szor akkorák, mint egy másik rétegéi, ez szükségessé teheti a tanulási ráták kompenzációs kiigazítását. Az adaptív megoldók, mint például az AdaGrad :cite:`Duchi.Hazan.Singer.2011`, az Adam :cite:`Kingma.Ba.2014`, a Yogi :cite:`Zaheer.Reddi.Sachan.ea.2018` vagy a Distributed Shampoo :cite:`anil2020scalable`, az optimalizálás szempontjából próbálnak megoldást találni erre a problémára, pl. másodrendű módszerek szempontjait beépítve.
Az alternatíva az, hogy egyszerűen adaptív normalizálással megelőzzük a probléma kialakulását.

Harmadszor, a mélyebb hálózatok összetettebb és hajlamosabbak a túlillesztésre.
Ez azt jelenti, hogy a regularizáció kritikusabbá válik. A regularizáció egyik általános technikája a zajinjektálás. Ez már régóta ismert, például a bemenetek zajinjektálásával kapcsolatban :cite:`Bishop.1995`. Ez képezi a dropout alapját is a :numref:`sec_dropout` részben. Mint kiderül, teljesen véletlenszerűen, a batchnormalizáció mindhárom előnnyel jár: előfeldolgozás, numerikus stabilitás és regularizáció.

A batchnormalizáció az egyes rétegekre, vagy opcionálisan az összesre alkalmazható:
Minden tanítási iterációban először normalizáljuk a bemeneteket (a batchnormalizáció bemeneteit) az átlaguk kivonásával és a szórásukkal való osztással, ahol mindkettőt az aktuális mini-batch statisztikái alapján becsüljük.
Ezután egy skálaegyütthatót és egy eltolást alkalmazunk az elveszített szabadsági fokok visszanyerésére. Pontosan ez a *batch* statisztikákon alapuló *normalizáció* adja a *batchnormalizáció* nevét.

Megjegyezzük, hogy ha 1-es méretű mini-batch-ekkel próbálnánk alkalmazni a batchnormalizációt, nem tudnánk semmit sem tanulni.
Ez azért van, mert az átlagok kivonása után minden rejtett egység 0 értéket venne fel.
Ahogy sejthető, mivel egy egész szakaszt szentelünk a batchnormalizációnak, elegendően nagy mini-batch-ekkel a megközelítés hatékonynak és stabilnak bizonyul.
Ebből az a tanulság, hogy a batchnormalizáció alkalmazásakor a batch méretének megválasztása még fontosabb, mint batchnormalizáció nélkül, vagy legalábbis megfelelő kalibrálás szükséges, ha a batch méretet módosítjuk.

Jelöljük $\mathcal{B}$-vel a mini-batch-t, és legyen $\mathbf{x} \in \mathcal{B}$ a batchnormalizáció ($\textrm{BN}$) bemenete. Ebben az esetben a batchnormalizáció a következőképpen definiált:

$$\textrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

A :eqref:`eq_batchnorm`-ben $\hat{\boldsymbol{\mu}}_\mathcal{B}$ a $\mathcal{B}$ mini-batch mintaátlaga, $\hat{\boldsymbol{\sigma}}_\mathcal{B}$ pedig a mintaszórása.
A standardizáció alkalmazása után a kapott mini-batch nulla átlagú és egységnyi varianciájú.
Az egységnyi variancia választása (más mágikus szám helyett) tetszőleges. Ezt a szabadsági fokot visszanyerjük egy elemenként értelmezett *skálaparaméter* $\boldsymbol{\gamma}$ és *eltolásparaméter* $\boldsymbol{\beta}$ beillesztésével, amelyek ugyanolyan alakúak, mint $\mathbf{x}$. Mindkét paramétert a modell tanítása során kell megtanulni.

A közbenső rétegek változóinak nagyságrendjei nem divergálhatnak tanítás közben, mivel a batchnormalizáció aktívan középre igazítja és átskálázza őket egy adott átlagra és méretre (a $\hat{\boldsymbol{\mu}}_\mathcal{B}$ és a ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ segítségével).
A gyakorlati tapasztalat megerősíti, hogy — ahogy a jellemzők átskálázásának tárgyalásakor már utaltunk rá — a batchnormalizáció láthatóan agresszívabb tanulási rátákat tesz lehetővé.
A $\hat{\boldsymbol{\mu}}_\mathcal{B}$ és ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ értékeket a :eqref:`eq_batchnorm`-ben a következőképpen számítjuk:

$$\hat{\boldsymbol{\mu}}_\mathcal{B} = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}
\textrm{ and }
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.$$

Megjegyezzük, hogy kis $\epsilon > 0$ konstanst adunk a varianciabecsléshez, hogy biztosítsuk, hogy soha ne próbáljunk nullával osztani, még azokban az esetekben sem, amikor az empirikus varianciabecslés nagyon kicsi lehet vagy eltűnhet.
A $\hat{\boldsymbol{\mu}}_\mathcal{B}$ és ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ becslések az átlag és a variancia zajos becslésének felhasználásával ellensúlyozzák a skálázási problémát.
Azt gondolhatnád, hogy ez a zajosság problémát okozhat.
Éppen ellenkezőleg, valójában előnyös.

Ez a mélytanulásban visszatérő témának bizonyult.
Elméletileg még nem jól jellemzett okokból az optimalizálásban különböző zajforrások gyakran gyorsabb tanításhoz és kisebb túlillesztéshez vezetnek: ez a változékonyság a regularizáció egyik formájaként tűnik hatni.
:citet:`Teye.Azizpour.Smith.2018` és :citet:`Luo.Wang.Shao.ea.2018` a batchnormalizáció tulajdonságait a Bayes-féle priorokhoz és büntetőtagokhoz kapcsolta.
Ez különösen rávilágít arra a talányra, hogy miért működik a batchnormalizáció a legjobban az 50–100-as tartományban lévő mérsékelt mini-batch-méreteknél.
Ez a méret úgy tűnik, hogy éppen a "megfelelő mennyiségű" zajt injektálja rétegenként, mind a $\hat{\boldsymbol{\sigma}}$-n keresztüli skálázás, mind a $\hat{\boldsymbol{\mu}}$-n keresztüli eltolás tekintetében: egy nagyobb mini-batch kevésbé regularizál a stabilabb becslések miatt, míg a nagyon kis mini-batch-ek hasznosítható jelzéseket pusztítanak el a nagy variancia miatt. Ennek az iránynak a további feltárása, alternatív előfeldolgozási és szűrési típusok figyelembevételével, talán más hatékony regularizációs típusokhoz vezet.

Ha betanított modellt vizsgálunk, talán azt gondolnánk, hogy a teljes adathalmazt szeretnénk felhasználni az átlag és a variancia becslésére.
Miután a tanítás befejeződött, miért akarnánk, hogy ugyanaz a kép különbözőképpen kerüljön besorolásra attól függően, hogy melyik batch-ben tartózkodik?
A tanítás során az ilyen pontos számítás megvalósíthatatlan, mivel az összes adatpéldánk közbenső változói minden egyes modellfrissítéskor megváltoznak.
Amint azonban a modell betanítottá válik, kiszámíthatjuk az egyes rétegek változóinak átlagát és varianciáját a teljes adathalmaz alapján.
Ez az általános gyakorlat a batchnormalizációt alkalmazó modelleknél; ezért a batchnormalizációs rétegek másképpen működnek *tanítási módban* (normalizálás mini-batch-statisztikák alapján), mint *predikciós módban* (normalizálás adathalmaz-statisztikák alapján).
Ebben a formában szorosan hasonlítanak a :numref:`sec_dropout` dropout regularizáció viselkedéséhez, ahol a zajt csak tanítás közben injektálják.


## batchnormalizációs Rétegek

A teljesen összekötött rétegek és a konvolúciós rétegek batchnormalizáció implementációja kissé eltérő.
Az egyik fő különbség a batchnormalizáció és más rétegek között az, hogy mivel az előbbi egyszerre egy teljes mini-batch-en dolgozik, nem hagyhatjuk figyelmen kívül a batch dimenziót, ahogy más rétegek bevezetésekor tettük.

### Teljesen Összekötött Rétegek

A batchnormalizáció teljesen összekötött rétegekre való alkalmazásakor :citet:`Ioffe.Szegedy.2015` az eredeti cikkben az affin transzformáció *után* és a nemlineáris aktivációs függvény *előtt* illesztette be a batchnormalizációt. Későbbi alkalmazások az aktivációs függvények *után* való beillesztéssel is kísérleteztek.
Jelöljük a teljesen összekötött réteg bemenetét $\mathbf{x}$-szel, az affin transzformációt $\mathbf{W}\mathbf{x} + \mathbf{b}$-vel (ahol $\mathbf{W}$ a súlyparaméter és $\mathbf{b}$ az eltolás-paraméter), és az aktivációs függvényt $\phi$-vel.
A batchnormalizáció által engedélyezett teljesen összekötött réteg kimenetét $\mathbf{h}$-val a következőképpen fejezhetjük ki:

$$\mathbf{h} = \phi(\textrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

Ne feledjük, hogy az átlagot és a varianciát *ugyanazon* mini-batch-en számítjuk, amelyre a transzformációt alkalmazzuk.

### Konvolúciós Rétegek

Hasonlóképpen, konvolúciós rétegek esetén a batchnormalizációt a konvolúció után, de a nemlineáris aktivációs függvény előtt alkalmazhatjuk. A teljesen összekötött rétegek batchnormalizációjától való fő különbség az, hogy a műveletet csatornánként, *az összes helyen* végezzük. Ez összhangban van az eltolási invariancia feltételezésével, amely a konvolúciókhoz vezetett: feltételeztük, hogy egy minta adott helyzete a képen nem kritikus a megértés szempontjából.

Tegyük fel, hogy mini-batch-jeink $m$ példát tartalmaznak, és minden csatornánál a konvolúció kimenete $p$ magasságú és $q$ szélességű.
Konvolúciós rétegek esetén minden batchnormalizációt a $m \cdot p \cdot q$ elemen hajtunk végre kimeneti csatornánként egyidejűleg.
Így az átlag és a variancia kiszámításakor az összes térbeli helyen összegyűjtjük az értékeket, és következésképpen ugyanazt az átlagot és varianciát alkalmazzuk egy adott csatornán belül az egyes térbeli helyek értékeinek normalizálására.
Minden csatornának saját skálaparamétere és eltolásparamétere van, amelyek mindkettő skalárok.

### Rétegnormalizáció
:label:`subsec_layer-normalization-in-bn`

Megjegyezzük, hogy a konvolúciók kontextusában a batchnormalizáció 1-es méretű mini-batch-ek esetén is jól definiált: elvégre az összes kép-helyet átlagolhatjuk. Következésképpen az átlag és a variancia jól definiált, még ha egyetlen megfigyelésen belül is. Ez a megfontolás vezette :citet:`Ba.Kiros.Hinton.2016`-ot a *rétegnormalizáció* (*layer normalization*) fogalmának bevezetéséhez. Pontosan úgy működik, mint a batchnormalizáció, csak egyetlen megfigyelésre alkalmazzák egyszerre. Ezért mind az eltolás, mind a skálázási tényező skalárok. Egy $n$-dimenziós $\mathbf{x}$ vektor esetén a rétegnormák a következők:

$$\mathbf{x} \rightarrow \textrm{LN}(\mathbf{x}) =  \frac{\mathbf{x} - \hat{\mu}}{\hat\sigma},$$

ahol a skálázás és az eltolás együtthatónként alkalmazott és a következőképpen adott:

$$\hat{\mu} \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n x_i \textrm{ and }
\hat{\sigma}^2 \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2 + \epsilon.$$

Mint korábban, kis $\epsilon > 0$ eltolást adunk hozzá a nullával való osztás megakadályozása érdekében. A rétegnormalizáció egyik fő előnye az, hogy megakadályozza a divergenciát. Elvégre, az $\epsilon$-t figyelmen kívül hagyva, a rétegnormalizáció kimenete skálafüggetlen. Vagyis $\textrm{LN}(\mathbf{x}) \approx \textrm{LN}(\alpha \mathbf{x})$ teljesül $\alpha \neq 0$ tetszőleges választásához. Ez egyenlőséggé válik $|\alpha| \to \infty$ esetén (a közelítő egyenlőség a variancia $\epsilon$ eltolásából fakad).

A rétegnormalizáció másik előnye, hogy nem függ a mini-batch méretétől. Szintén független attól, hogy tanítási vagy tesztelési módban vagyunk. Más szóval, csupán egy determinisztikus transzformáció, amely az aktivációkat adott skálára standardizálja. Ez nagyon hasznos lehet az optimalizálásbeli divergencia megakadályozásában. Kihagyjuk a további részleteket, és az érdeklődőknek az eredeti cikk elolvasását ajánljuk.

### batchnormalizáció Predikció Során

Ahogy korábban említettük, a batchnormalizáció tanítási módban jellemzően más viselkedést mutat, mint predikciós módban.
Először is, a mini-batch-enként becsült mintaátlagban és mintavarianciában lévő zaj már nem kívánatos, amint betanítottuk a modellt.
Másodszor, előfordulhat, hogy nem rendelkezünk az egyes batch-enkénti normalizációs statisztikák kiszámításának lehetőségével.
Például előfordulhat, hogy a modellt egyszerre csak egy predikció elvégzésére kell alkalmazni.

Jellemzően a tanítás után az összes adathalmazt használjuk a változóstatisztikák stabil becslésének kiszámításához, majd predikciós időben rögzítjük azokat.
Ezért a batchnormalizáció tanítás közben másképpen viselkedik, mint tesztelési időben.
Emlékezzünk arra, hogy a dropout is mutat ilyen jellemzőt.

## (**Implementáció Alapoktól**)

Hogy lássuk, hogyan működik a batchnormalizáció a gyakorlatban, alább az alapoktól implementáljuk.

```{.python .input}
%%tab mxnet
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Az autograd segítségével állapítjuk meg, hogy tanítási módban vagyunk-e
    if not autograd.is_training():
        # Predikciós módban a mozgó átlaggal kapott átlagot és varianciát használjuk
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # Teljesen összekötött réteg esetén az átlagot és a varianciát
            # a jellemző-dimenzió mentén számítjuk
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # Kétdimenziós konvolúciós réteg esetén az átlagot és a varianciát
            # a csatorna-dimenzió mentén számítjuk (axis=1). Itt meg kell
            # őriznünk X alakját, hogy a broadcasting művelet
            # később elvégezhető legyen
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # Tanítási módban az aktuális átlagot és varianciát használjuk 
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Az átlag és a variancia frissítése mozgó átlaggal
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Skálázás és eltolás
    return Y, moving_mean, moving_var
```

```{.python .input}
%%tab pytorch
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Az is_grad_enabled segítségével állapítjuk meg, hogy tanítási módban vagyunk-e
    if not torch.is_grad_enabled():
        # Predikciós módban a mozgó átlaggal kapott átlagot és varianciát használjuk
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # Teljesen összekötött réteg esetén az átlagot és a varianciát
            # a jellemző-dimenzió mentén számítjuk
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # Kétdimenziós konvolúciós réteg esetén az átlagot és a varianciát
            # a csatorna-dimenzió mentén számítjuk (axis=1). Itt meg kell
            # őriznünk X alakját, hogy a broadcasting művelet
            # később elvégezhető legyen
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # Tanítási módban az aktuális átlagot és varianciát használjuk 
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Az átlag és a variancia frissítése mozgó átlaggal
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Skálázás és eltolás
    return Y, moving_mean.data, moving_var.data
```

```{.python .input}
%%tab tensorflow
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # A mozgó variancia négyzetgyökének reciprokát számítjuk elemenként
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Skálázás és eltolás
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

```{.python .input}
%%tab jax
def batch_norm(X, deterministic, gamma, beta, moving_mean, moving_var, eps,
               momentum):
    # A `deterministic` segítségével állapítjuk meg, hogy az aktuális mód
    # tanítási mód vagy predikciós mód-e
    if deterministic:
        # Predikciós módban a mozgó átlaggal kapott átlagot és varianciát használjuk
        # A `linen.Module.variables` változóknak van egy `value` attribútumuk, amely a tömböt tartalmazza
        X_hat = (X - moving_mean.value) / jnp.sqrt(moving_var.value + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # Teljesen összekötött réteg esetén az átlagot és a varianciát
            # a jellemző-dimenzió mentén számítjuk
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # Kétdimenziós konvolúciós réteg esetén az átlagot és a varianciát
            # a csatorna-dimenzió mentén számítjuk (axis=1). Itt meg kell
            # őriznünk `X` alakját, hogy a broadcasting művelet
            # később elvégezhető legyen
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # Tanítási módban az aktuális átlagot és varianciát használjuk
        X_hat = (X - mean) / jnp.sqrt(var + eps)
        # Az átlag és a variancia frissítése mozgó átlaggal
        moving_mean.value = momentum * moving_mean.value + (1.0 - momentum) * mean
        moving_var.value = momentum * moving_var.value + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Skálázás és eltolás
    return Y
```

Most [**létrehozhatunk egy megfelelő `BatchNorm` réteget.**]
A rétegünk megfelelő paramétereket fog fenntartani a `gamma` skálához és a `beta` eltoláshoz, amelyeket a tanítás során frissítünk.
Ezen kívül a rétegünk mozgó átlagokat tart fenn az átlagokról és varianciákról a modell predikciója során való későbbi felhasználáshoz.

Az algoritmus részleteit félretéve, vegyük észre a rétegünk implementációjának alapjául szolgáló tervezési mintát.
Általában a matematikát egy különálló függvényben definiáljuk, mondjuk `batch_norm`.
Ezt a funkcionalitást ezután egy egyéni rétegbe integráljuk, amelynek kódja főként könyvvezetési kérdésekkel foglalkozik, mint például az adatok a megfelelő eszközkontextusba való áthelyezése, a szükséges változók kiosztása és inicializálása, a mozgó átlagok követése (itt az átlagra és a varianciára), stb.
Ez a minta lehetővé teszi a matematika és a sablonkód tiszta szétválasztását.
Megjegyezzük továbbá, hogy a kényelem kedvéért nem foglalkoztunk a bemeneti alak automatikus kikövetkeztetésével; ezért minden jellemző számát meg kell adnunk.
Mostanra minden modern mélytanulás keretrendszer automatikusan képes felismerni a méretet és az alakot a magas szintű batchnormalizációs API-kban (a gyakorlatban ezt fogjuk használni).

```{.python .input}
%%tab mxnet
class BatchNorm(nn.Block):
    # `num_features`: kimenetek száma teljesen összekötött réteg esetén,
    # vagy kimeneti csatornák száma konvolúciós réteg esetén. `num_dims`:
    # 2 teljesen összekötött réteg, 4 konvolúciós réteg esetén
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # A skálaparaméter és az eltolásparaméter (modellparaméterek)
        # inicializálása 1-re és 0-ra
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # A nem modellparaméter változók inicializálása 0-ra és
        # 1-re
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # Ha `X` nem a fő memóriában van, másoljuk a `moving_mean` és
        # `moving_var` változókat arra az eszközre, ahol `X` található
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # A frissített `moving_mean` és `moving_var` mentése
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.1)
        return Y
```

```{.python .input}
%%tab pytorch
class BatchNorm(nn.Module):
    # num_features: kimenetek száma teljesen összekötött réteg esetén,
    # vagy kimeneti csatornák száma konvolúciós réteg esetén. num_dims: 2
    # teljesen összekötött réteg, 4 konvolúciós réteg esetén
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # A skálaparaméter és az eltolásparaméter (modellparaméterek)
        # inicializálása 1-re és 0-ra
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # A nem modellparaméter változók inicializálása 0-ra és
        # 1-re
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # Ha X nem a fő memóriában van, másoljuk a moving_mean és moving_var
        # változókat arra az eszközre, ahol X található
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # A frissített moving_mean és moving_var mentése
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y
```

```{.python .input}
%%tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # A skálaparaméter és az eltolásparaméter (modellparaméterek)
        # inicializálása 1-re és 0-ra
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # A nem modellparaméter változók inicializálása 0-ra
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.1
        delta = (1.0 - momentum) * variable + momentum * value
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

```{.python .input}
%%tab jax
class BatchNorm(nn.Module):
    # `num_features`: kimenetek száma teljesen összekötött réteg esetén,
    # vagy kimeneti csatornák száma konvolúciós réteg esetén.
    # `num_dims`: 2 teljesen összekötött réteg, 4 konvolúciós réteg esetén
    # A `deterministic` segítségével állapítjuk meg, hogy az aktuális mód
    # tanítási mód vagy predikciós mód-e
    num_features: int
    num_dims: int
    deterministic: bool = False

    @nn.compact
    def __call__(self, X):
        if self.num_dims == 2:
            shape = (1, self.num_features)
        else:
            shape = (1, 1, 1, self.num_features)

        # A skálaparaméter és az eltolásparaméter (modellparaméterek)
        # inicializálása 1-re és 0-ra
        gamma = self.param('gamma', jax.nn.initializers.ones, shape)
        beta = self.param('beta', jax.nn.initializers.zeros, shape)

        # A nem modellparaméter változók inicializálása 0-ra és
        # 1-re. Mentés a 'batch_stats' gyűjteménybe
        moving_mean = self.variable('batch_stats', 'moving_mean', jnp.zeros, shape)
        moving_var = self.variable('batch_stats', 'moving_var', jnp.ones, shape)
        Y = batch_norm(X, self.deterministic, gamma, beta,
                       moving_mean, moving_var, eps=1e-5, momentum=0.9)

        return Y
```

A múltbeli átlag- és varianciabecsléseket a `momentum` segítségével aggregáljuk. Ez némileg félrevezető elnevezés, mivel semmi köze sincs az optimalizálás *momentum* tagjához. Mindazonáltal ez az általánosan elfogadott neve ennek a tagnak, és az API elnevezési konvenciónak való tisztelet jegyében ugyanazt a változónevet használjuk a kódunkban.

## [**LeNet batchnormalizációval**]

Hogy lássuk, hogyan alkalmazzuk a `BatchNorm`-ot kontextusban, alább egy hagyományos LeNet modellre alkalmazzuk (:numref:`sec_lenet`).
Ne feledjük, hogy a batchnormalizációt a konvolúciós rétegek vagy teljesen összekötött rétegek után, de a megfelelő aktivációs függvények előtt alkalmazzuk.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class BNLeNetScratch(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2), nn.Dense(120),
                BatchNorm(120, num_dims=2), nn.Activation('sigmoid'),
                nn.Dense(84), BatchNorm(84, num_dims=2),
                nn.Activation('sigmoid'), nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120),
                BatchNorm(120, num_dims=2), nn.Sigmoid(), nn.LazyLinear(84),
                BatchNorm(84, num_dims=2), nn.Sigmoid(),
                nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84), BatchNorm(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class BNLeNetScratch(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(6, kernel_size=(5, 5)),
            BatchNorm(6, num_dims=4, deterministic=not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(16, kernel_size=(5, 5)),
            BatchNorm(16, num_dims=4, deterministic=not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(120),
            BatchNorm(120, num_dims=2, deterministic=not self.training),
            nn.sigmoid,
            nn.Dense(84),
            BatchNorm(84, num_dims=2, deterministic=not self.training),
            nn.sigmoid,
            nn.Dense(self.num_classes)])
```

:begin_tab:`jax`
Mivel a `BatchNorm` rétegeknek ki kell számítaniuk a batch statisztikákat (átlag és variancia), a Flax nyomon követi a `batch_stats` szótárt, és minden mini-batch-csel frissíti azokat. Az olyan gyűjtemények, mint a `batch_stats`, tárolhatók a `TrainState` objektumban (a :numref:`oo-design-training` részben definiált `d2l.Trainer` osztályban) attribútumként, és a modell előreterjesztése során ezeket a `mutable` argumentumnak kell átadni, hogy a Flax visszaadja a módosított változókat.
:end_tab:

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    Y_hat, updates = state.apply_fn({'params': params,
                                     'batch_stats': state.batch_stats},
                                    *X, mutable=['batch_stats'],
                                    rngs={'dropout': state.dropout_rng})
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    return (fn(Y_hat, Y).mean(), updates) if averaged else (fn(Y_hat, Y), updates)
```

Mint korábban, [**a Fashion-MNIST adathalmazon fogjuk tanítani a hálózatunkat**].
Ez a kód lényegében azonos azzal, amellyel először tanítottuk a LeNet-et.

```{.python .input}
%%tab mxnet, pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNetScratch(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNetScratch(lr=0.5)
    trainer.fit(model, data)
```

[**Nézzük meg a `gamma` skálaparamétert és a `beta` eltolásparamétert**], amelyeket az első batchnormalizációs réteg tanult.

```{.python .input}
%%tab mxnet
model.net[1].gamma.data().reshape(-1,), model.net[1].beta.data().reshape(-1,)
```

```{.python .input}
%%tab pytorch
model.net[1].gamma.reshape((-1,)), model.net[1].beta.reshape((-1,))
```

```{.python .input}
%%tab tensorflow
tf.reshape(model.net.layers[1].gamma, (-1,)), tf.reshape(
    model.net.layers[1].beta, (-1,))
```

```{.python .input}
%%tab jax
trainer.state.params['net']['layers_1']['gamma'].reshape((-1,)), \
trainer.state.params['net']['layers_1']['beta'].reshape((-1,))
```

## [**Tömör Implementáció**]

Az imént definiált `BatchNorm` osztályhoz képest közvetlenül a mélytanulás keretrendszer magas szintű API-jaiból származó `BatchNorm` osztályt is használhatjuk.
A kód lényegében azonos a fenti implementációnkkal, kivéve, hogy már nem kell megadnunk az extra argumentumokat a dimenziók helyes meghatározásához.

```{.python .input}
%%tab pytorch, tensorflow, mxnet
class BNLeNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Dense(120), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(84), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(84), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class BNLeNet(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(6, kernel_size=(5, 5)),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(16, kernel_size=(5, 5)),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(120),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            nn.Dense(84),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            nn.Dense(self.num_classes)])
```

Alább [**ugyanazokat a hiperparamétereket használjuk a modell tanításához.**]
Megjegyezzük, hogy a szokásnak megfelelően a magas szintű API változat sokkal gyorsabban fut, mivel a kódja C++-ra vagy CUDA-ra lett fordítva, míg az egyéni implementációnkat Python-nak kell értelmezni.

```{.python .input}
%%tab mxnet, pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNet(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNet(lr=0.5)
    trainer.fit(model, data)
```

## Vita

Intuitívan a batchnormalizáció simítja az optimalizálási tájképet.
Azonban óvatosnak kell lennünk, hogy különbséget tegyünk a spekulatív intuíciók és a mély modellek tanításakor megfigyelt jelenségek tényleges magyarázatai között.
Emlékezzünk arra, hogy még azt sem értjük igazán, miért generalizálnak jól az egyszerűbb mély neurális hálózatok (MLP-k és hagyományos CNN-ek).
Még dropout-tal és súlycsökkentéssel is annyira rugalmasak, hogy az ismeretlen adatokra való általánosítási képességük valószínűleg lényegesen finomabb tanuláselméleti általánosítási garanciákat igényel.

A batchnormalizációt javasló eredeti cikk :cite:`Ioffe.Szegedy.2015`, amellett, hogy egy hatékony és hasznos eszközt mutatott be, magyarázatot is kínált arra, miért működik: az *internal covariate shift* (belső kovariancia-eltolódás) csökkentésével.
Feltehetőleg az *internal covariate shift* alatt olyasmit értettek, mint a fent kifejezett intuíció — az a gondolat, hogy a változóértékek eloszlása megváltozik a tanítás során.
Az magyarázatnak azonban két problémája volt:
i) Ez az eltolódás nagyon különbözik a *covariate shift*-től, ami félrevezető elnevezéssé teszi. Ha bármihez hasonlítható, inkább a fogalom-eltolódáshoz hasonlít.
ii) A magyarázat alulspecifikált intuíciót kínál, de nyitva hagyja azt a kérdést, hogy *pontosan miért működik ez a technika*, ami szigorú magyarázatot vár.
Egész könyvünkben arra törekszünk, hogy közvetítsük azokat az intuíciókat, amelyeket a szakemberek mély neurális hálózataik fejlesztésének irányítására használnak.
Azonban úgy gondoljuk, hogy fontos ezeket az iránymutatásul szolgáló intuíciókat elválasztani a megalapozott tudományos tényektől.
Végül, amikor elsajátítod ezt az anyagot és elkezded írni saját kutatási cikkeidet, szeretnéd majd egyértelműen elválasztani a technikai állításokat a sejtésektől.

A batchnormalizáció sikere nyomán az *internal covariate shift* általi magyarázata ismételten felmerült a technikai irodalomban zajló vitákban és a gépi tanulás kutatásának bemutatásával kapcsolatos tágabb diskurzusban.
Egy emlékezetes beszédben, amelyet a 2017-es NeurIPS konferencián a Test of Time Award átvételekor tartott, Ali Rahimi az *internal covariate shift*-et egy olyan érvelés fókuszpontjaként használta, amely a modern mélytanulás gyakorlatát az alkímiához hasonlította.
Ezt követően a példát részletesen újra megvizsgálták egy pozíciós cikkben, amely a gépi tanulás aggasztó tendenciáit vázolta fel :cite:`Lipton.Steinhardt.2018`.
Más szerzők alternatív magyarázatokat javasoltak a batchnormalizáció sikerére, néhányan :cite:`Santurkar.Tsipras.Ilyas.ea.2018` azt állítva, hogy a batchnormalizáció sikere ellenére bizonyos tekintetben az eredeti cikkben állított viselkedéssel ellentétes viselkedést mutat.

Megjegyezzük, hogy az *internal covariate shift* nem érdemel több kritikát, mint a gépi tanulás technikai irodalmában évente tett ezernyi hasonlóan homályos állítás bármelyike.
Valószínűleg a viták fókuszpontjaként való visszhangja a célközönség számára széles körben ismert jellegéből fakad.
A batchnormalizáció nélkülözhetetlen módszernek bizonyult, amelyet szinte az összes telepített képosztályozóban alkalmaznak, és a technikát bevezető cikk tízezer idézetet szerzett. Mindazonáltal azt sejtjük, hogy a zajos injektáláson keresztüli regularizáció, az átskálázáson keresztüli gyorsítás és végül az előfeldolgozás vezérelvei talán a rétegek és technikák további találmányaihoz vezetnek majd a jövőben.

A gyakorlatibb oldalon számos szempont érdemes megjegyezni a batchnormalizációval kapcsolatban:

* A modell tanítása során a batchnormalizáció folyamatosan állítja a hálózat közbenső kimenetét a mini-batch átlagának és szórásának felhasználásával, így a neurális hálózat egyes rétegein keresztüli közbenső kimenet értékei stabilabbak.
* A batchnormalizáció kissé eltérő a teljesen összekötött rétegek és a konvolúciós rétegek esetén. Valójában a konvolúciós rétegek esetén a rétegnormalizáció néha alternatívaként használható.
* Mint a dropout réteg, a batchnormalizációs rétegek is más viselkedést mutatnak tanítási módban, mint predikciós módban.
* A batchnormalizáció hasznos a regularizációhoz és az optimalizálásban való konvergencia javításához. Ezzel szemben az internal covariate shift csökkentésének eredeti motivációja nem tűnik érvényes magyarázatnak.
* Robusztusabb modelleknél, amelyek kevésbé érzékenyek a bemeneti perturbációkra, fontold meg a batchnormalizáció eltávolítását :cite:`wang2022removing`.

## Feladatok

1. El kell-e távolítani az eltolás-paramétert a teljesen összekötött rétegből vagy a konvolúciós rétegből a batchnormalizáció előtt? Miért?
1. Hasonlítsd össze a LeNet tanulási rátáit batchnormalizációval és anélkül.
    1. Ábrázold a validációs pontosság növekedését.
    1. Mekkora lehet a tanulási ráta, mielőtt az optimalizálás mindkét esetben meghibásodik?
1. Szükséges-e batchnormalizáció minden rétegben? Kísérletezz vele.
1. Implementáld a batchnormalizáció "lite" verzióját, amely csak az átlagot távolítja el, vagy alternatív módon olyat, amely csak a varianciát távolítja el. Hogyan viselkedik?
1. Rögzítsd a `beta` és a `gamma` paramétereket. Figyeld meg és elemezd az eredményeket.
1. Helyettesítheted-e a dropoutot batchnormalizációval? Hogyan változik a viselkedés?
1. Kutatási ötletek: gondolj más normalizációs transzformációkra, amelyeket alkalmazhatsz:
    1. Alkalmazhatod-e a valószínűségi integrál transzformációt?
    1. Használhatsz-e teljes rangú kovarianciabecslést? Miért valószínűleg nem kellene ezt tenned?
    1. Használhatsz-e más kompakt mátrixváltozatokat (blokk-diagonális, alacsony eltolású rang, Monarch, stb.)?
    1. Regularizálóként hat-e a ritka tömörítés?
    1. Vannak-e más vetítések (pl. konvex kúp, szimmetria-csoport-specifikus transzformációk), amelyeket használhatsz?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/83)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/84)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/330)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18005)
:end_tab:
