# Információelmélet
:label:`sec_information_theory`

Az univerzum tele van információval. Az információ közös nyelvet biztosít a tudományterületek között: Shakespeare szonettjétől a Cornell ArXiv kutatói cikkeiig, Van Gogh Csillagos éj nyomatától Beethoven 5. szimfóniájáig, az első programozási nyelv Plankalkültől a legkorszerűbb gépi tanulási algoritmusokig. Minden az információelmélet szabályait követi, formátumtól függetlenül. Az információelmélet segítségével mérhetjük és összehasonlíthatjuk, hogy mennyi információ van jelen különböző jelekben. Ebben a szakaszban az információelmélet alapfogalmait és a gépi tanulásban való alkalmazásait vizsgáljuk meg.

Mielőtt belekezdenénk, vázoljuk fel a gépi tanulás és az információelmélet kapcsolatát. A gépi tanulás célja érdekes jelek kinyerése az adatokból és fontos előrejelzések készítése. Az információelmélet ezzel szemben az információ kódolásával, dekódolásával, átvitelével és kezelésével foglalkozik. Ennek eredményeképpen az információelmélet alapvető nyelvet biztosít a gépileg tanult rendszerek információfeldolgozásának tárgyalásához. Például sok gépi tanulási alkalmazás a keresztentrópia-veszteséget használja, ahogy az a :numref:`sec_softmax` részben is látható. Ez a veszteség közvetlenül levezethető információelméleti megfontolásokból.


## Információ

Kezdjük az információelmélet „lelkével": az információval. Az *információ* bármibe kódolható, amely egy vagy több kódolási formátum adott sorozatát tartalmazza. Tegyük fel, hogy feladatunk az információ fogalmának meghatározása. Mi lehetne a kiindulópontunk?

Gondoljuk végig a következő gondolatkísérletet. Van egy barátunk egy kártyapakkal. Megkeveri a paklit, felforgat néhány lapot, és közöl velünk dolgokat a kártyákról. Megpróbáljuk megbecsülni az egyes kijelentések információtartalmát.

Először felforgat egy lapot, és azt mondja: „Látok egy kártyát." Ez semmilyen információt nem ad nekünk. Ezt eleve biztosra vettük, tehát az információ tartalma nulla kell legyen.

Ezután felforgat egy lapot, és azt mondja: „Szívet látok." Ez némi információt ad, de valójában csak $4$ különböző szín lehetséges, mindegyik egyforma valószínűséggel, tehát nem lepődünk meg az eredményen. Bármi legyen is az információ mértéke, ennek az eseménynek alacsony információtartalommal kell rendelkeznie.

Majd felforgat egy lapot, és azt mondja: „Ez a pikk 3-as." Ez több információ. Valóban $52$ egyforma valószínűségű lehetséges kimenetel volt, és barátunk megmondta, melyik volt. Ez közepes mennyiségű információnak felel meg.

Vigyük ezt a logikus végletéig. Tegyük fel, hogy végül felforgat minden lapot a pakliból, és felolvassa a megkevert pakli teljes sorrendjét. A paklinak $52!$ különböző sorrendje lehetséges, mindegyik egyforma valószínűséggel, tehát sok információra van szükségünk, hogy tudjuk, melyikről van szó.

Az általunk kidolgozott információfogalomnak meg kell felelnie ennek az intuíciónak. A következő szakaszokban megtanuljuk, hogyan számítsuk ki, hogy ezek az események rendre $0\textrm{ bit}$, $2\textrm{ bit}$, $~5{,}7\textrm{ bit}$ és $~225{,}6\textrm{ bit}$ információt tartalmaznak.

Ha végiggondoljuk ezeket a gondolatkísérleteket, egy természetes ötlet merül fel. Kiindulópontként, ahelyett hogy a tudásra koncentrálnánk, az esemény meglepetési fokát vagy elvont lehetőségét jelképező információra alapozhatjuk a gondolkodásunkat. Például ha egy szokatlan eseményt szeretnénk leírni, sok információra van szükségünk. Egy közönséges eseménynél kevés információra lehet szükség.

1948-ban Claude E. Shannon kiadta *A Mathematical Theory of Communication* :cite:`Shannon.1948` című munkáját, amellyel megalapozta az információelméletet. Cikkében Shannon elsőként vezette be az információentrópia fogalmát. Innen indítjuk utunkat.


### Saját-információ

Mivel az információ megtestesíti az esemény elvont lehetőségét, hogyan képezzük le a lehetőséget bitek számára? Shannon bevezette a *bit* terminológiát az információ egységeként, amelyet eredetileg John Tukey alkotott meg. Mi tehát a „bit", és miért használjuk az információ mérésére? Történelmileg egy régi adóberendezés csak kétféle kódot tudott küldeni vagy fogadni: $0$-t és $1$-et. A bináris kódolás ma is általánosan használatos minden modern digitális számítógépen. Ily módon minden információt $0$-k és $1$-ek sorozataként kódolunk. Így egy $n$ hosszúságú bináris számjegysor $n$ bit információt tartalmaz.

Tegyük fel, hogy bármely kódsorozatban minden $0$ vagy $1$ $\frac{1}{2}$ valószínűséggel fordul elő. Ezért egy $n$ hosszúságú kódsorozatból álló $X$ esemény $\frac{1}{2^n}$ valószínűséggel következik be. Ugyanakkor, ahogy korábban is megjegyeztük, ez a sorozat $n$ bit információt tartalmaz. Tehát általánosíthatjuk-e ezt egy matematikai függvényre, amely a $p$ valószínűséget bitek számává alakítja? Shannon a *saját-információ* meghatározásával adta meg a választ:

$$I(X) = - \log_2 (p),$$

mint az $X$ eseményből kapott *bit* mennyiségű információ. Megjegyzés: ebben a szakaszban mindig 2-es alapú logaritmust fogunk használni. Az egyszerűség kedvéért a szakasz többi részében elhagyjuk a logaritmus jelölésből a 2-es indexet, azaz a $\log(.)$ mindig $\log_2(.)$-t jelent. Például a „0010" kód saját-információja:

$$I(\textrm{"0010"}) = - \log (p(\textrm{"0010"})) = - \log \left( \frac{1}{2^4} \right) = 4 \textrm{ bit}.$$

A saját-információt az alábbiak szerint számíthatjuk ki. Ehhez először importáljuk az ebben a szakaszban szükséges összes csomagot.

```{.python .input}
#@tab mxnet
from mxnet import np
from mxnet.metric import NegativeLogLikelihood
from mxnet.ndarray import nansum
import random

def self_information(p):
    return -np.log2(p)

self_information(1 / 64)
```

```{.python .input}
#@tab pytorch
import torch
from torch.nn import NLLLoss

def nansum(x):
    # A `nansum` definiálása, mert a PyTorch nem kínál beépített változatot.
    return x[~torch.isnan(x)].sum()

def self_information(p):
    return -torch.log2(torch.tensor(p)).item()

self_information(1 / 64)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

def log2(x):
    return tf.math.log(x) / tf.math.log(2.)

def nansum(x):
    return tf.reduce_sum(tf.where(tf.math.is_nan(
        x), tf.zeros_like(x), x), axis=-1)

def self_information(p):
    return -log2(tf.constant(p)).numpy()

self_information(1 / 64)
```

## Entrópia

Mivel a saját-információ csak egyetlen diszkrét esemény információját méri, általánosabb mértékre van szükségünk bármely diszkrét vagy folytonos eloszlású véletlen változóhoz.


### Az entrópia motivációja

Próbáljuk meg pontosan meghatározni, mit szeretnénk. Ez egy informális leírása annak, amit *Shannon-entrópia axiómáinak* neveznek. Kiderül, hogy a következő, józan észre épülő állítások együttesen az információ egyértelmű definícióját kényszerítik ki. Az axiómák formális változata és néhány további axióma megtalálható a :citet:`Csiszar.2008` hivatkozásban.

1.  Az információ, amelyet egy véletlen változó megfigyelésével nyerünk, nem függ az elemek elnevezésétől, illetve a nulla valószínűségű további elemek jelenlététől.
2.  Az információ, amelyet két véletlen változó megfigyelésével nyerünk, nem több, mint az egyes változók külön megfigyelésével szerzett információk összege. Ha egymástól függetlenek, akkor pontosan az összeg.
3.  A (majdnem) biztos esemény megfigyelésekor szerzett információ (majdnem) nulla.

Bár ennek a tételnek a bizonyítása meghaladja szövegünk kereteit, fontos tudni, hogy ez egyértelműen meghatározza az entrópia alakját. Az egyetlen szabadságfok az alapegység megválasztásában rejlik, amelyet leggyakrabban azzal normálunk, hogy egy igazságos érmefejdobás által nyújtott információ egy bit legyen.

### Definíció

Bármely $P$ valószínűségeloszlást követő $X$ véletlen változóhoz, amelynek valószínűségsűrűség-függvénye (p.s.f.) vagy valószínűségtömeg-függvénye (v.t.f.) $p(x)$, az elvárt információmennyiséget *entrópiával* (vagy *Shannon-entrópiával*) mérjük:

$$H(X) = - E_{x \sim P} [\log p(x)].$$
:eqlabel:`eq_ent_def`

Pontosabban, ha $X$ diszkrét, akkor $$H(X) = - \sum_i p_i \log p_i \textrm{, ahol } p_i = P(X_i).$$

Ellenkező esetben, ha $X$ folytonos, az entrópiát *differenciális entrópiának* is nevezzük:

$$H(X) = - \int_x p(x) \log p(x) \; dx.$$

Az entrópiát az alábbiakban definiálhatjuk.

```{.python .input}
#@tab mxnet
def entropy(p):
    entropy = - p * np.log2(p)
    # A `nansum` összeadja a nem NaN értékeket.
    out = nansum(entropy.as_nd_ndarray())
    return out

entropy(np.array([0.1, 0.5, 0.1, 0.3]))
```

```{.python .input}
#@tab pytorch
def entropy(p):
    entropy = - p * torch.log2(p)
    # A `nansum` összeadja a nem NaN értékeket.
    out = nansum(entropy)
    return out

entropy(torch.tensor([0.1, 0.5, 0.1, 0.3]))
```

```{.python .input}
#@tab tensorflow
def entropy(p):
    return nansum(- p * log2(p))

entropy(tf.constant([0.1, 0.5, 0.1, 0.3]))
```

### Értelmezések

Talán kíváncsi vagy: az entrópia :eqref:`eq_ent_def` definíciójában miért használjuk a negatív logaritmus várható értékét? Íme néhány intuíció.

Először is, miért használunk *logaritmus* függvényt $\log$? Tegyük fel, hogy $p(x) = f_1(x) f_2(x) \ldots, f_n(x)$, ahol minden $f_i(x)$ komponensfüggvény független egymástól. Ez azt jelenti, hogy minden $f_i(x)$ önállóan járul hozzá a $p(x)$-ből szerzett összes információhoz. Ahogy fentebb tárgyaltuk, azt szeretnénk, hogy az entrópia-képlet additív legyen független véletlen változókra. Szerencsére a $\log$ természetesen képes a valószínűségi eloszlások szorzatát az egyes tagok összegévé alakítani.

Másodszor, miért használunk *negatív* $\log$-ot? Intuitív módon a gyakoribb eseményeknek kevesebb információt kell tartalmazniuk, mint a ritkábban előfordulóknak, hiszen szokatlan esetekből általában több információt nyerünk, mint a szokásosokból. A $\log$ azonban monoton növekvő a valószínűségekkel, és valóban negatív a $[0, 1]$ tartomány összes értékére. Monoton csökkenő kapcsolatot kell felépítenünk az esemény valószínűsége és entrópiája között, amely ideálisan mindig pozitív legyen (hiszen semmi, amit megfigyeltünk, nem kényszeríthet bennünket arra, hogy elfelejtsük, amit már tudtunk). Ezért adunk egy negatív előjelet a $\log$ függvény elé.

Végül, honnan ered a *várható érték* függvény? Tekintsük az $X$ véletlen változót. A saját-információt ($-\log(p)$) értelmezhetjük annak *meglepetési* mértékeként, amit egy adott kimenetel láttán érzünk. Valóban, ahogy a valószínűség nullához közelít, a meglepetés végtelenné válik. Hasonlóképpen az entrópiát értelmezhetjük az $X$ megfigyelésekor tapasztalt átlagos meglepetésként. Például képzeljük el, hogy egy nyerőgép rendszer statisztikailag egymástól független ${s_1, \ldots, s_k}$ szimbólumokat bocsát ki ${p_1, \ldots, p_k}$ valószínűségekkel. Ekkor ennek a rendszernek az entrópiája egyenlő az egyes kimenetek megfigyelésének átlagos saját-információjával, azaz:

$$H(S) = \sum_i {p_i \cdot I(s_i)} = - \sum_i {p_i \cdot \log p_i}.$$



### Az entrópia tulajdonságai

A fenti példák és értelmezések alapján levezethetjük az entrópia :eqref:`eq_ent_def` következő tulajdonságait. Itt $X$-et eseményként, $P$-t pedig $X$ valószínűségi eloszlásaként jelöljük.

* $H(X) \geq 0$ minden diszkrét $X$-re (folytonos $X$ esetén az entrópia negatív is lehet).

* Ha $X \sim P$ egy $p(x)$ valószínűségsűrűség-függvénnyel vagy valószínűségtömeg-függvénnyel, és megpróbáljuk $P$-t becsülni egy új $Q$ valószínűségi eloszlással, amelynek valószínűségsűrűség-függvénye vagy valószínűségtömeg-függvénye $q(x)$, akkor $$H(X) = - E_{x \sim P} [\log p(x)] \leq  - E_{x \sim P} [\log q(x)], \textrm{ és egyenlőség pontosan akkor áll fenn, ha } P = Q.$$  Alternatívan, $H(X)$ alsó korlátja a $P$-ből vett szimbólumok kódolásához szükséges átlagos bitek számának.

* Ha $X \sim P$, akkor $x$ akkor hordozza a maximális információmennyiséget, ha egyenletesen oszlik el az összes lehetséges kimenetel között. Pontosabban, ha a $P$ valószínűségi eloszlás diszkrét $k$-osztályú $\{p_1, \ldots, p_k \}$ eloszlás, akkor $$H(X) \leq \log(k), \textrm{ egyenlőség pontosan akkor áll fenn, ha } p_i = \frac{1}{k}, \forall i.$$ Ha $P$ folytonos véletlen változó, a helyzet jóval bonyolultabbá válik. Ha azonban kikötjük, hogy $P$ egy véges intervallumon értelmezett (az összes érték $0$ és $1$ között van), akkor $P$ entrópiája akkor a legnagyobb, ha az adott intervallumon egyenletes eloszlású.


## Kölcsönös információ

Korábban egy egyetlen $X$ véletlen változó entrópiáját definiáltuk; mi a helyzet egy $(X, Y)$ véletlen változópár entrópiájával? Ezeket a technikákat úgy foghatjuk fel, mint amelyek a következő típusú kérdésre próbálnak válaszolni: „Milyen információ van $X$-ben és $Y$-ban együtt, szemben az egyes változók önálló információjával? Van-e redundáns információ, vagy mindegyik egyedi?"

A következő tárgyalásban mindig $(X, Y)$-t tekintjük olyan véletlen változópárnak, amely egy $P$ közös valószínűségeloszlást követ, amelynek valószínűségsűrűség-függvénye vagy valószínűségtömeg-függvénye $p_{X, Y}(x, y)$, míg $X$ és $Y$ rendre $p_X(x)$ és $p_Y(y)$ valószínűségeloszlást követ.


### Együttes entrópia

Hasonlóan az egyetlen véletlen változó entrópiájához :eqref:`eq_ent_def`, az $(X, Y)$ véletlen változópár *együttes entrópiáját* $H(X, Y)$-ként definiáljuk:

$$H(X, Y) = -E_{(x, y) \sim P} [\log p_{X, Y}(x, y)]. $$
:eqlabel:`eq_joint_ent_def`

Pontosabban, ha $(X, Y)$ diszkrét véletlen változók párja, akkor:

$$H(X, Y) = - \sum_{x} \sum_{y} p_{X, Y}(x, y) \log p_{X, Y}(x, y).$$

Ha viszont $(X, Y)$ folytonos véletlen változók párja, akkor a *differenciális együttes entrópiát* a következőképpen definiáljuk:

$$H(X, Y) = - \int_{x, y} p_{X, Y}(x, y) \ \log p_{X, Y}(x, y) \;dx \;dy.$$

A :eqref:`eq_joint_ent_def` képletet úgy értelmezhetjük, mint a véletlen változópár teljes véletlenszerűségét. Két szélsőség esetén: ha $X = Y$ két azonos véletlen változó, akkor a párban lévő információ pontosan az egyikben lévő információval egyenlő, tehát $H(X, Y) = H(X) = H(Y)$. A másik szélsőségnél, ha $X$ és $Y$ független, akkor $H(X, Y) = H(X) + H(Y)$. Általánosan elmondható, hogy egy véletlen változópárban lévő információ nem kisebb bármelyik véletlen változó entrópiájánál, és nem nagyobb mindkettő összegénél.

$$
H(X), H(Y) \le H(X, Y) \le H(X) + H(Y).
$$

Valósítsuk meg az együttes entrópiát alapoktól.

```{.python .input}
#@tab mxnet
def joint_entropy(p_xy):
    joint_ent = -p_xy * np.log2(p_xy)
    # A `nansum` összeadja a nem NaN értékeket.
    out = nansum(joint_ent.as_nd_ndarray())
    return out

joint_entropy(np.array([[0.1, 0.5], [0.1, 0.3]]))
```

```{.python .input}
#@tab pytorch
def joint_entropy(p_xy):
    joint_ent = -p_xy * torch.log2(p_xy)
    # A `nansum` összeadja a nem NaN értékeket.
    out = nansum(joint_ent)
    return out

joint_entropy(torch.tensor([[0.1, 0.5], [0.1, 0.3]]))
```

```{.python .input}
#@tab tensorflow
def joint_entropy(p_xy):
    joint_ent = -p_xy * log2(p_xy)
    # A `nansum` összeadja a nem NaN értékeket.
    out = nansum(joint_ent)
    return out

joint_entropy(tf.constant([[0.1, 0.5], [0.1, 0.3]]))
```

Vegyük észre, hogy ez ugyanaz a *kód*, mint korábban, de most másképpen értelmezzük: a két véletlen változó közös eloszlásán dolgozunk.


### Feltételes entrópia

A fentebb definiált együttes entrópia a véletlen változópárban lévő információmennyiséget fejezi ki. Ez hasznos, de sokszor nem ez az, amivel foglalkozni szeretnénk. Tekintsük a gépi tanulás kontextusát. Legyen $X$ az a véletlen változó (vagy véletlen változók vektora), amely egy kép pixelértékeit írja le, és $Y$ az osztálycímkét jelölő véletlen változó. $X$-nek jelentős információt kell tartalmaznia — egy természetes kép összetett dolog. Azonban az $Y$-ban lévő információnak az után, hogy a kép megjelent, alacsonynak kell lennie. Valóban, egy számjegy képe már tartalmazza az információt arról, hogy melyik számjegyről van szó, kivéve ha a számjegy olvashatatlan. Ezért az információelmélet szókincsének bővítéséhez képesnek kell lennünk következtetni egy véletlen változó információtartalmára egy másik változóra feltételesen.

A valószínűségelméletben láttuk a *feltételes valószínűség* definícióját a változók közötti kapcsolat mérésére. Most hasonlóan szeretnénk definiálni a *feltételes entrópiát* $H(Y \mid X)$. Ezt a következőképpen írhatjuk:

$$ H(Y \mid X) = - E_{(x, y) \sim P} [\log p(y \mid x)],$$
:eqlabel:`eq_cond_ent_def`

ahol $p(y \mid x) = \frac{p_{X, Y}(x, y)}{p_X(x)}$ a feltételes valószínűség. Pontosabban, ha $(X, Y)$ diszkrét véletlen változók párja, akkor:

$$H(Y \mid X) = - \sum_{x} \sum_{y} p(x, y) \log p(y \mid x).$$

Ha $(X, Y)$ folytonos véletlen változók párja, a *differenciális feltételes entrópiát* hasonlóan definiáljuk:

$$H(Y \mid X) = - \int_x \int_y p(x, y) \ \log p(y \mid x) \;dx \;dy.$$


Most természetes kérdés, hogyan kapcsolódik a *feltételes entrópia* $H(Y \mid X)$ az $H(X)$ entrópiához és az $H(X, Y)$ együttes entrópiához? A fenti definíciók felhasználásával ezt egyszerűen kifejezhetjük:

$$H(Y \mid X) = H(X, Y) - H(X).$$

Ennek szemléletes értelmezése van: az $Y$-ban lévő, $X$-re feltételezett információ ($H(Y \mid X)$) megegyezik az $X$-ben és $Y$-ban együttesen lévő információval ($H(X, Y)$), csökkentve az $X$-ben már szereplő információval. Ez megadja az $Y$-ban lévő, de $X$-ben nem szereplő információt.

Most valósítsuk meg a feltételes entrópiát :eqref:`eq_cond_ent_def` alapoktól.

```{.python .input}
#@tab mxnet
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * np.log2(p_y_given_x)
    # A `nansum` összeadja a nem NaN értékeket.
    out = nansum(cond_ent.as_nd_ndarray())
    return out

conditional_entropy(np.array([[0.1, 0.5], [0.2, 0.3]]), np.array([0.2, 0.8]))
```

```{.python .input}
#@tab pytorch
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * torch.log2(p_y_given_x)
    # A `nansum` összeadja a nem NaN értékeket.
    out = nansum(cond_ent)
    return out

conditional_entropy(torch.tensor([[0.1, 0.5], [0.2, 0.3]]),
                    torch.tensor([0.2, 0.8]))
```

```{.python .input}
#@tab tensorflow
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * log2(p_y_given_x)
    # A `nansum` összeadja a nem NaN értékeket.
    out = nansum(cond_ent)
    return out

conditional_entropy(tf.constant([[0.1, 0.5], [0.2, 0.3]]),
                    tf.constant([0.2, 0.8]))
```

### Kölcsönös információ

Az $(X, Y)$ véletlen változópár korábbi tárgyalása után felmerülhet a kérdés: „Most, hogy tudjuk, mennyi információ van $Y$-ban, de nincs $X$-ben, hasonlóan megkérdezhetjük-e, hogy mennyi információ közös $X$-ben és $Y$-ban?" A válasz az $(X, Y)$ *kölcsönös információja* lesz, amelyet $I(X, Y)$-nak jelölünk.

Ahelyett, hogy rögtön a formális definícióba merülnénk, gyakoroljuk az intuíciónkat azzal, hogy először megpróbálunk levezetni egy kifejezést a kölcsönös információra kizárólag a korábban felépített fogalmak alapján. A két véletlen változó által közösen megosztott információt keressük. Ezt megtehetjük úgy, hogy az $X$-ben és $Y$-ban együttesen lévő összes információból kivonjuk azokat a részeket, amelyek nem közösek. Az $X$-ben és $Y$-ban együttesen lévő információt $H(X, Y)$-ként jelöljük. Ebből le kell vonni az $X$-ben, de nem $Y$-ban lévő, illetve a $Y$-ban, de nem $X$-ben lévő információt. Ahogy az előző szakaszban láttuk, ezt rendre $H(X \mid Y)$ és $H(Y \mid X)$ adja meg. Tehát a kölcsönös információnak a következőnek kell lennie:

$$
I(X, Y) = H(X, Y) - H(Y \mid X) - H(X \mid Y).
$$

Ez valóban érvényes definíciója a kölcsönös információnak. Ha kifejtjük e tagok definícióit és kombináljuk őket, egy kis algebrával megmutatható, hogy ez ugyanaz, mint:

$$I(X, Y) = E_{x} E_{y} \left\{ p_{X, Y}(x, y) \log\frac{p_{X, Y}(x, y)}{p_X(x) p_Y(y)} \right\}. $$
:eqlabel:`eq_mut_ent_def`


Ezeket a kapcsolatokat összefoglalhatjuk a :numref:`fig_mutual_information` ábrán. Kiváló intuíciós próba annak belátása, hogy a következő állítások mindegyike ekvivalens az $I(X, Y)$-nal.

* $H(X) - H(X \mid Y)$
* $H(Y) - H(Y \mid X)$
* $H(X) + H(Y) - H(X, Y)$

![A kölcsönös információ kapcsolata az együttes entrópiával és a feltételes entrópiával.](../img/mutual-information.svg)
:label:`fig_mutual_information`


Sok szempontból a kölcsönös információt :eqref:`eq_mut_ent_def` úgy tekinthetjük, mint a :numref:`sec_random_variables` részben látott korrelációs együttható elvszerű általánosítását. Ez lehetővé teszi számunkra, hogy ne csak lineáris kapcsolatokat keressünk változók között, hanem a két véletlen változó között bármilyen típusú, maximálisan megosztott információt.

Most valósítsuk meg a kölcsönös információt alapoktól.

```{.python .input}
#@tab mxnet
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * np.log2(p)
    # A `nansum` összeadja a nem NaN értékeket.
    out = nansum(mutual.as_nd_ndarray())
    return out

mutual_information(np.array([[0.1, 0.5], [0.1, 0.3]]),
                   np.array([0.2, 0.8]), np.array([[0.75, 0.25]]))
```

```{.python .input}
#@tab pytorch
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * torch.log2(p)
    # A `nansum` összeadja a nem NaN értékeket.
    out = nansum(mutual)
    return out

mutual_information(torch.tensor([[0.1, 0.5], [0.1, 0.3]]),
                   torch.tensor([0.2, 0.8]), torch.tensor([[0.75, 0.25]]))
```

```{.python .input}
#@tab tensorflow
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * log2(p)
    # A `nansum` összeadja a nem NaN értékeket.
    out = nansum(mutual)
    return out

mutual_information(tf.constant([[0.1, 0.5], [0.1, 0.3]]),
                   tf.constant([0.2, 0.8]), tf.constant([[0.75, 0.25]]))
```

### A kölcsönös információ tulajdonságai

Ahelyett, hogy megjegyeznénk a kölcsönös információ :eqref:`eq_mut_ent_def` definícióját, csak a legfontosabb tulajdonságait kell szem előtt tartanunk:

* A kölcsönös információ szimmetrikus, azaz $I(X, Y) = I(Y, X)$.
* A kölcsönös információ nemnegatív, azaz $I(X, Y) \geq 0$.
* $I(X, Y) = 0$ akkor és csak akkor, ha $X$ és $Y$ független. Például ha $X$ és $Y$ független, akkor $Y$ ismerete semmilyen információt nem ad $X$-ről, és fordítva, tehát kölcsönös információjuk nulla.
* Alternatívaként, ha $X$ invertálható függvénye $Y$-nak, akkor $Y$ és $X$ minden információt megoszt egymással, és $$I(X, Y) = H(Y) = H(X).$$

### Pontonkénti kölcsönös információ

Amikor a fejezet elején az entrópiával foglalkoztunk, meg tudtuk adni a $-\log(p_X(x))$ értelmezését annak kifejezésére, mennyire voltunk *meglepve* egy adott kimeneteltől. Hasonló értelmezést adhatunk a kölcsönös információban szereplő logaritmikus tagnak is, amelyet gyakran *pontonkénti kölcsönös információnak* neveznek:

$$\textrm{pmi}(x, y) = \log\frac{p_{X, Y}(x, y)}{p_X(x) p_Y(y)}.$$
:eqlabel:`eq_pmi_def`

A :eqref:`eq_pmi_def` képletet úgy értelmezhetjük, mint annak mértékét, hogy az $x$ és $y$ kimeneteleknek egy adott kombinációja mennyivel valószínűbb vagy kevésbé valószínű ahhoz képest, amit független véletlen kimeneteleknél várnánk. Ha nagy és pozitív, akkor ez a két kimenetel sokkal gyakrabban fordul elő, mint véletlenszerű esély esetén (*megjegyzés*: a nevező $p_X(x) p_Y(y)$, ami a két kimenetel valószínűsége egymástól függetlenül), míg ha nagy és negatív, akkor a két kimenetel sokkal ritkábban fordul elő, mint amit véletlenszerű eséllyel várnánk.

Ez lehetővé teszi számunkra, hogy a kölcsönös információt :eqref:`eq_mut_ent_def` úgy értelmezzük, mint azt az átlagos meglepetést, amellyel két kimenetel együttes előfordulását tapasztalva szembesülünk, összehasonlítva azzal, amit függetlenségük esetén várnánk.

### A kölcsönös információ alkalmazásai

A kölcsönös információ tiszta definíciójában kissé elvontnak tűnhet, de hogyan kapcsolódik a gépi tanuláshoz? A természetes nyelvfeldolgozásban az egyik legnehezebb probléma az *egyértelműsítés*, vagyis egy szó jelentésének a kontextusból való meghatározása. Például nemrégiben egy hírfolyamban az a cím jelent meg, hogy „Az Amazon lángban áll." Felmerülhet a kérdés, hogy az Amazon cégnek van-e égő épülete, vagy az Amazon esőerdő lángol-e.

Ebben az esetben a kölcsönös információ segíthet feloldani ezt a kétértelműséget. Először megkeressük azokat a szavakat, amelyek mindegyike viszonylag nagy kölcsönös információval rendelkezik az Amazon céggel kapcsolatban, például e-kereskedelem, technológia és online. Másodszor megkeresünk egy másik csoportot azokból a szavakból, amelyek mindegyike viszonylag nagy kölcsönös információval rendelkezik az Amazon esőerdővel kapcsolatban, például eső, erdő és trópusi. Amikor az „Amazon" szót egyértelműsíteni kell, összehasonlíthatjuk, hogy melyik csoport fordul elő többet az Amazon szó kontextusában. Ebben az esetben a cikk az erdőt folytatná leírni, és a kontextus egyértelművé válna.


## Kullback–Leibler-divergencia

Ahogy a :numref:`sec_linear-algebra` részben tárgyaltuk, normákat használhatunk két pont közötti távolság mérésére bármilyen dimenziójú térben. Hasonló feladatot szeretnénk elvégezni valószínűségi eloszlásokra is. Erre számos módszer létezik, de az információelmélet az egyik legelegánsabbat kínálja. Most a *Kullback–Leibler (KL)-divergenciát* vizsgáljuk meg, amely módot ad arra, hogy megállapítsuk, közel van-e egymáshoz két eloszlás.


### Definíció

Adott egy $P$ valószínűségeloszlást követő $X$ véletlen változó, amelynek valószínűségsűrűség-függvénye vagy valószínűségtömeg-függvénye $p(x)$, és $P$-t egy másik $Q$ valószínűségi eloszlással becsüljük, amelynek valószínűségsűrűség-függvénye vagy valószínűségtömeg-függvénye $q(x)$. Ekkor a $P$ és $Q$ közötti *Kullback–Leibler (KL)-divergencia* (vagy *relatív entrópia*) a következő:

$$D_{\textrm{KL}}(P\|Q) = E_{x \sim P} \left[ \log \frac{p(x)}{q(x)} \right].$$
:eqlabel:`eq_kl_def`

A pontonkénti kölcsönös információhoz :eqref:`eq_pmi_def` hasonlóan a logaritmikus tagnak ismét értelmezést adhatunk: $-\log \frac{q(x)}{p(x)} = -\log(q(x)) - (-\log(p(x)))$ nagy és pozitív lesz, ha $x$-et $P$ szerint sokkal gyakrabban látjuk, mint amit $Q$ esetén várnánk, és nagy és negatív lesz, ha a kimenetel sokkal ritkábban fordul elő a vártnál. Így értelmezhetjük azt mint a kimenetel megfigyelésekor tapasztalt *relatív* meglepetésünket, összehasonlítva azzal, mennyire lennénk meglepve a referencia-eloszlásból való megfigyelés esetén.

Valósítsuk meg a KL-divergenciát alapoktól.

```{.python .input}
#@tab mxnet
def kl_divergence(p, q):
    kl = p * np.log2(p / q)
    out = nansum(kl.as_nd_ndarray())
    return out.abs().asscalar()
```

```{.python .input}
#@tab pytorch
def kl_divergence(p, q):
    kl = p * torch.log2(p / q)
    out = nansum(kl)
    return out.abs().item()
```

```{.python .input}
#@tab tensorflow
def kl_divergence(p, q):
    kl = p * log2(p / q)
    out = nansum(kl)
    return tf.abs(out).numpy()
```

### A KL-divergencia tulajdonságai

Nézzük meg a KL-divergencia :eqref:`eq_kl_def` néhány tulajdonságát.

* A KL-divergencia nem szimmetrikus, azaz léteznek olyan $P, Q$ eloszlások, amelyekre $$D_{\textrm{KL}}(P\|Q) \neq D_{\textrm{KL}}(Q\|P).$$
* A KL-divergencia nemnegatív, azaz $$D_{\textrm{KL}}(P\|Q) \geq 0.$$ Az egyenlőség csak akkor áll fenn, ha $P = Q$.
* Ha létezik olyan $x$, amelyre $p(x) > 0$ és $q(x) = 0$, akkor $D_{\textrm{KL}}(P\|Q) = \infty$.
* Szoros kapcsolat áll fenn a KL-divergencia és a kölcsönös információ között. A :numref:`fig_mutual_information` ábrán látható összefüggésen túlmenően $I(X, Y)$ numerikusan ekvivalens a következő tagokkal is:
    1. $D_{\textrm{KL}}(P(X, Y)  \ \| \ P(X)P(Y))$;
    1. $E_Y \{ D_{\textrm{KL}}(P(X \mid Y) \ \| \ P(X)) \}$;
    1. $E_X \{ D_{\textrm{KL}}(P(Y \mid X) \ \| \ P(Y)) \}$.

  Az első tag esetén a kölcsönös információt úgy értelmezzük, mint $P(X, Y)$ és $P(X) \cdot P(Y)$ szorzata közötti KL-divergenciát, és így mérjük, mennyire tér el az együttes eloszlás a független eloszlástól. A második tag esetén a kölcsönös információ megmutatja az $Y$-ra vonatkozó bizonytalanság átlagos csökkenését, amely az $X$ eloszlásának megismeréséből fakad. Hasonlóan a harmadik taghoz.


### Példa

Nézzük végig egy egyszerű példán keresztül az aszimmetriát.

Először hozzunk létre és rendezzük sorba három, $10\,000$ hosszúságú tenzort: egy $p$ célvektort, amely $N(0, 1)$ normális eloszlást követ, és két jelöltvektort $q_1$ és $q_2$, amelyek rendre $N(-1, 1)$ és $N(1, 1)$ normális eloszlást követnek.

```{.python .input}
#@tab mxnet
random.seed(1)

nd_len = 10000
p = np.random.normal(loc=0, scale=1, size=(nd_len, ))
q1 = np.random.normal(loc=-1, scale=1, size=(nd_len, ))
q2 = np.random.normal(loc=1, scale=1, size=(nd_len, ))

p = np.array(sorted(p.asnumpy()))
q1 = np.array(sorted(q1.asnumpy()))
q2 = np.array(sorted(q2.asnumpy()))
```

```{.python .input}
#@tab pytorch
torch.manual_seed(1)

tensor_len = 10000
p = torch.normal(0, 1, (tensor_len, ))
q1 = torch.normal(-1, 1, (tensor_len, ))
q2 = torch.normal(1, 1, (tensor_len, ))

p = torch.sort(p)[0]
q1 = torch.sort(q1)[0]
q2 = torch.sort(q2)[0]
```

```{.python .input}
#@tab tensorflow
tensor_len = 10000
p = tf.random.normal((tensor_len, ), 0, 1)
q1 = tf.random.normal((tensor_len, ), -1, 1)
q2 = tf.random.normal((tensor_len, ), 1, 1)

p = tf.sort(p)
q1 = tf.sort(q1)
q2 = tf.sort(q2)
```

Mivel $q_1$ és $q_2$ szimmetrikusak az $y$-tengelyre (azaz $x=0$-ra), hasonló KL-divergencia értéket várunk $D_{\textrm{KL}}(p\|q_1)$ és $D_{\textrm{KL}}(p\|q_2)$ esetén. Ahogy az alábbiakban látható, $D_{\textrm{KL}}(p\|q_1)$ és $D_{\textrm{KL}}(p\|q_2)$ eltérése kevesebb mint 3%.

```{.python .input}
#@tab all
kl_pq1 = kl_divergence(p, q1)
kl_pq2 = kl_divergence(p, q2)
similar_percentage = abs(kl_pq1 - kl_pq2) / ((kl_pq1 + kl_pq2) / 2) * 100

kl_pq1, kl_pq2, similar_percentage
```

Ezzel szemben azt tapasztaljuk, hogy $D_{\textrm{KL}}(q_2 \|p)$ és $D_{\textrm{KL}}(p \| q_2)$ jelentősen eltér egymástól, körülbelül 40%-os eltéréssel, ahogy az alábbiakban látható.

```{.python .input}
#@tab all
kl_q2p = kl_divergence(q2, p)
differ_percentage = abs(kl_q2p - kl_pq2) / ((kl_q2p + kl_pq2) / 2) * 100

kl_q2p, differ_percentage
```

## Keresztentrópia

Ha kíváncsi vagy az információelmélet mélytanulás-beli alkalmazásaira, itt egy gyors példa. Definiáljuk a $p(x)$ valószínűségeloszlású igaz $P$ eloszlást és a $q(x)$ valószínűségeloszlású becsült $Q$ eloszlást, amelyeket a szakasz hátralévő részében használunk.

Tegyük fel, hogy $n$ adatpéldányból $\{x_1, \ldots, x_n\}$ bináris osztályozási problémát kell megoldanunk. Tegyük fel, hogy $1$-gyel és $0$-val kódoljuk a pozitív és negatív osztálycímkét $y_i$-vel rendre, és neurális hálózatunk $\theta$-val paraméterezhető. Ha olyan legjobb $\theta$-t keresünk, hogy $\hat{y}_i= p_{\theta}(y_i \mid x_i)$, természetes a maximális log-valószínűségi megközelítés alkalmazása, ahogy a :numref:`sec_maximum_likelihood` részben is láthattuk. Pontosabban, az igaz $y_i$ címkékre és a $\hat{y}_i= p_{\theta}(y_i \mid x_i)$ előrejelzésekre a pozitívként való osztályozás valószínűsége $\pi_i= p_{\theta}(y_i = 1 \mid x_i)$. Ezért a log-valószínűségi függvény a következő:

$$
\begin{aligned}
l(\theta) &= \log L(\theta) \\
  &= \log \prod_{i=1}^n \pi_i^{y_i} (1 - \pi_i)^{1 - y_i} \\
  &= \sum_{i=1}^n y_i \log(\pi_i) + (1 - y_i) \log (1 - \pi_i). \\
\end{aligned}
$$

Az $l(\theta)$ log-valószínűségi függvény maximalizálása azonos a $- l(\theta)$ minimalizálásával, és így innen megtalálhatjuk a legjobb $\theta$-t. A fenti veszteség általánosítása tetszőleges eloszlásokra: a $-l(\theta)$ veszteséget *keresztentrópia-veszteségnek* $\textrm{CE}(y, \hat{y})$ is nevezzük, ahol $y$ az igaz $P$ eloszlást, $\hat{y}$ pedig a becsült $Q$ eloszlást követi.

Mindez a maximális valószínűség szemszögéből lett levezetve. Ha azonban közelebbről megvizsgáljuk, láthatjuk, hogy olyan tagok, mint $\log(\pi_i)$ kerültek a számításba, ami egyértelmű jele annak, hogy a kifejezést információelméleti szempontból is értelmezhetjük.


### Formális definíció

A KL-divergenciához hasonlóan egy $X$ véletlen változóra szintén mérhetjük a becsült $Q$ eloszlás és az igaz $P$ eloszlás közötti eltérést *keresztentrópiával*:

$$\textrm{CE}(P, Q) = - E_{x \sim P} [\log(q(x))].$$
:eqlabel:`eq_ce_def`

A fentebb tárgyalt entrópia-tulajdonságok felhasználásával értelmezhetjük ezt a $H(P)$ entrópia és $P$, $Q$ közötti KL-divergencia összegeként is, azaz:

$$\textrm{CE} (P, Q) = H(P) + D_{\textrm{KL}}(P\|Q).$$


A keresztentrópia-veszteséget az alábbiakban valósíthatjuk meg.

```{.python .input}
#@tab mxnet
def cross_entropy(y_hat, y):
    ce = -np.log(y_hat[range(len(y_hat)), y])
    return ce.mean()
```

```{.python .input}
#@tab pytorch
def cross_entropy(y_hat, y):
    ce = -torch.log(y_hat[range(len(y_hat)), y])
    return ce.mean()
```

```{.python .input}
#@tab tensorflow
def cross_entropy(y_hat, y):
    # A `tf.gather_nd` a tenzor megadott indexeinek kiválasztására szolgál.
    ce = -tf.math.log(tf.gather_nd(y_hat, indices = [[i, j] for i, j in zip(
        range(len(y_hat)), y)]))
    return tf.reduce_mean(ce).numpy()
```

Most definiáljunk két tenzort a címkékre és az előrejelzésekre, majd számítsuk ki a keresztentrópia-veszteségüket.

```{.python .input}
#@tab mxnet
labels = np.array([0, 2])
preds = np.array([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

```{.python .input}
#@tab pytorch
labels = torch.tensor([0, 2])
preds = torch.tensor([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

```{.python .input}
#@tab tensorflow
labels = tf.constant([0, 2])
preds = tf.constant([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

### Tulajdonságok

Ahogy a szakasz elején utaltunk rá, a keresztentrópia :eqref:`eq_ce_def` használható veszteségfüggvény definiálására az optimalizálási feladatban. Kiderül, hogy a következők ekvivalensek:

1. $Q$ prediktív valószínűségének maximalizálása a $P$ eloszlásra (azaz $E_{x
\sim P} [\log (q(x))]$);
1. A keresztentrópia $\textrm{CE} (P, Q)$ minimalizálása;
1. A KL-divergencia $D_{\textrm{KL}}(P\|Q)$ minimalizálása.

A keresztentrópia definíciója közvetve bizonyítja a 2. és 3. célkitűzés ekvivalens kapcsolatát, amennyiben az igazi adatok $H(P)$ entrópiája állandó.


### A keresztentrópia mint többosztályos osztályozás célfüggvénye

Ha mélyebbre ásunk a keresztentrópia-veszteséggel $\textrm{CE}$ rendelkező osztályozási célfüggvénybe, azt találjuk, hogy a $\textrm{CE}$ minimalizálása ekvivalens az $L$ log-valószínűségi függvény maximalizálásával.

Mindenekelőtt tegyük fel, hogy adott egy $n$ példányból álló adathalmaz, amelyet $k$ osztályba lehet sorolni. Minden $i$ adatpéldányra bármely $k$-osztályú $\mathbf{y}_i = (y_{i1}, \ldots, y_{ik})$ címkét *one-hot kódolással* ábrázolunk. Pontosabban, ha az $i$ példány a $j$ osztályhoz tartozik, akkor a $j$-edik elemet $1$-re, az összes többi komponenst $0$-ra állítjuk, azaz:

$$ y_{ij} = \begin{cases}1 & j \in J; \\ 0 &\textrm{egyébként.}\end{cases}$$

Például ha egy többosztályos osztályozási feladat három osztályt tartalmaz: $A$, $B$ és $C$, akkor az $\mathbf{y}_i$ címkék a következőképpen kódolhatók: {$A: (1, 0, 0); B: (0, 1, 0); C: (0, 0, 1)$}.


Tegyük fel, hogy neurális hálózatunk $\theta$-val paraméterezhető. Az igaz $\mathbf{y}_i$ címkevektorokra és az előrejelzésekre $$\hat{\mathbf{y}}_i= p_{\theta}(\mathbf{y}_i \mid \mathbf{x}_i) = \sum_{j=1}^k y_{ij} p_{\theta} (y_{ij}  \mid  \mathbf{x}_i).$$

Ezért a *keresztentrópia-veszteség* a következő:

$$
\textrm{CE}(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{i=1}^n \mathbf{y}_i \log \hat{\mathbf{y}}_i
 = - \sum_{i=1}^n \sum_{j=1}^k y_{ij} \log{p_{\theta} (y_{ij}  \mid  \mathbf{x}_i)}.\\
$$

Másrészt a problémát a maximális valószínűség becslésén keresztül is megközelíthetjük. Először röviden bemutatjuk a $k$-osztályú multinoulli-eloszlást. Ez a Bernoulli-eloszlás kiterjesztése bináris osztályról többosztályosra. Ha egy $\mathbf{z} = (z_{1}, \ldots, z_{k})$ véletlen változó $\mathbf{p} =$ ($p_{1}, \ldots, p_{k}$) valószínűségekkel rendelkező $k$-osztályú *multinoulli-eloszlást* követ, azaz $$p(\mathbf{z}) = p(z_1, \ldots, z_k) = \textrm{Multi} (p_1, \ldots, p_k), \textrm{ ahol } \sum_{i=1}^k p_i = 1,$$ akkor $\mathbf{z}$ együttes valószínűségtömeg-függvénye (v.t.f.) a következő:
$$\mathbf{p}^\mathbf{z} = \prod_{j=1}^k p_{j}^{z_{j}}.$$


Belátható, hogy minden adatpéldány $\mathbf{y}_i$ címkéje $\boldsymbol{\pi} =$ ($\pi_{1}, \ldots, \pi_{k}$) valószínűségekkel rendelkező $k$-osztályú multinoulli-eloszlást követ. Ezért minden $\mathbf{y}_i$ adatpéldány együttes v.t.f.-je $\mathbf{\pi}^{\mathbf{y}_i} = \prod_{j=1}^k \pi_{j}^{y_{ij}}.$
Ezért a log-valószínűségi függvény a következő:

$$
\begin{aligned}
l(\theta)
 = \log L(\theta)
 = \log \prod_{i=1}^n \boldsymbol{\pi}^{\mathbf{y}_i}
 = \log \prod_{i=1}^n \prod_{j=1}^k \pi_{j}^{y_{ij}}
 = \sum_{i=1}^n \sum_{j=1}^k y_{ij} \log{\pi_{j}}.\\
\end{aligned}
$$

Mivel a maximális valószínűség becslésnél a $\pi_{j} = p_{\theta} (y_{ij}  \mid  \mathbf{x}_i)$ beállítással maximalizáljuk az $l(\theta)$ célfüggvényt, ezért bármely többosztályos osztályozásnál a fenti $l(\theta)$ log-valószínűségi függvény maximalizálása ekvivalens a $\textrm{CE}(y, \hat{y})$ keresztentrópia-veszteség minimalizálásával.


A fenti bizonyítás teszteléséhez alkalmazzuk a beépített `NegativeLogLikelihood` mértéket. A korábbi példában szereplő `labels` és `preds` értékekkel ugyanolyan numerikus veszteséget kapunk, mint az előző példában, 5 tizedesjegy pontossággal.

```{.python .input}
#@tab mxnet
nll_loss = NegativeLogLikelihood()
nll_loss.update(labels.as_nd_ndarray(), preds.as_nd_ndarray())
nll_loss.get()
```

```{.python .input}
#@tab pytorch
# A PyTorch keresztentrópia-veszteség-implementációja a `nn.LogSoftmax()`
# és a `nn.NLLLoss()` kombinációja.
nll_loss = NLLLoss()
loss = nll_loss(torch.log(preds), labels)
loss
```

```{.python .input}
#@tab tensorflow
def nll_loss(y_hat, y):
    # A címkéket one-hot vektorokká alakítjuk.
    y = tf.keras.utils.to_categorical(y, num_classes= y_hat.shape[1])
    # Nem a definícióból számoljuk a negatív log-valószínűséget.
    # Inkább körkörös érvelést használunk. Mivel az NLL ugyanaz, mint a
    # `cross_entropy`, a keresztentrópia kiszámítása megadná az NLL-t.
    cross_entropy = tf.keras.losses.CategoricalCrossentropy(
        from_logits = True, reduction = tf.keras.losses.Reduction.NONE)
    return tf.reduce_mean(cross_entropy(y, y_hat)).numpy()

loss = nll_loss(tf.math.log(preds), labels)
loss
```

## Összefoglalás

* Az információelmélet az információ kódolásával, dekódolásával, átvitelével és kezelésével foglalkozó tudományterület.
* Az entrópia az egysége annak mérésére, hogy mennyi információ van jelen különböző jelekben.
* A KL-divergencia szintén képes mérni a két eloszlás közötti eltérést.
* A keresztentrópia többosztályos osztályozás célfüggvényeként értelmezhető. A keresztentrópia-veszteség minimalizálása ekvivalens a log-valószínűségi függvény maximalizálásával.


## Feladatok

1. Ellenőrizd, hogy az első szakasz kártyapéldái valóban a megadott entrópiával rendelkeznek-e.
1. Mutasd meg, hogy a KL-divergencia $D(p\|q)$ minden $p$ és $q$ eloszlásra nemnegatív. Tipp: használd Jensen-egyenlőtlenséget, azaz azt a tényt, hogy $-\log x$ konvex függvény.
1. Számítsuk ki az entrópiát néhány adatforrásból:
    * Tegyük fel, hogy figyeljük egy írógépen gépelő majom által generált kimenetet. A majom véletlenszerűen nyomja meg az írógép $44$ billentyűjének bármelyikét (feltételezhetjük, hogy még nem fedezte fel a speciális billentyűket vagy a Shift billentyűt). Hány bit véletlenszerűséget figyelünk meg karakterenként?
    * A majommal elégedetlenül lévén egy részeg szedővel helyettesítjük. Képes szavakat generálni, bár nem koherensen. Ehelyett véletlenszerűen választ egy szót egy $2000$ szavas szókincsből. Feltételezzük, hogy az angol szavak átlagos hossza $4{,}5$ betű. Hány bit véletlenszerűséget figyelünk meg most karakterenként?
    * Az eredménnyel még mindig elégedetlenek lévén a szedőt egy kiváló minőségű nyelvi modellel helyettesítjük. A nyelvi modell jelenleg szavanként akár $15$ pontos perplexitást is el tud érni. A nyelvi modell karakter *perplexitása* a valószínűségek egy halmazának geometriai átlagának reciprokaként van definiálva, ahol minden valószínűség a szó egy karakteréhez tartozik. Pontosabban, ha egy adott szó hossza $l$, akkor $\textrm{PPL}(\textrm{szó}) = \left[\prod_i p(\textrm{karakter}_i)\right]^{ -\frac{1}{l}} = \exp \left[ - \frac{1}{l} \sum_i{\log p(\textrm{karakter}_i)} \right].$ Tegyük fel, hogy a tesztszónak 4,5 betűje van; hány bit véletlenszerűséget figyelünk meg most karakterenként?
1. Magyarázd el intuitívan, miért teljesül $I(X, Y) = H(X) - H(X \mid Y)$. Majd mutasd meg, hogy ez igaz, mindkét oldalt a közös eloszlásra vonatkozó várható értékként kifejezve.
1. Mi a KL-divergencia a két Gauss-eloszlás $\mathcal{N}(\mu_1, \sigma_1^2)$ és $\mathcal{N}(\mu_2, \sigma_2^2)$ között?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/420)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1105)
:end_tab:
