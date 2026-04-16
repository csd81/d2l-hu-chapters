# Szóbeágyazás globális vektorokkal (GloVe)
:label:`sec_glove`


A szó–szó együttes előfordulások
a kontextusablakokban
gazdag szemantikai információt hordozhatnak.
Például
egy nagy korpuszban
a „solid" szó valószínűleg gyakrabban fordul elő
az „ice" szóval együtt, mint a „steam" szóval,
míg a „gas" szó
feltehetőleg az „steam" szóval
fordul elő együtt
„ice"-nél sűrűbben.
Emellett
az ilyen együttes előfordulások
globális korpuszstatisztikái
előre kiszámíthatók:
ez hatékonyabb tanítást tehet lehetővé.
Hogy a teljes korpusz statisztikai
információit fel tudjuk használni
szóbeágyazáshoz,
először tekintsük át újra
a skip-gram modellt (:numref:`subsec_skip-gram`),
de értelmezzük azt
a globális korpuszstatisztikák –
például az együttes előfordulási számok –
segítségével.

## A Skip-Gram modell globális korpuszstatisztikákkal
:label:`subsec_skipgram-global`

Jelöljük $q_{ij}$-vel
a $P(w_j\mid w_i)$
feltételes valószínűséget,
amely a skip-gram modellben
a $w_j$ szó valószínűségét adja meg a $w_i$ szó feltételével:

$$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \exp(\mathbf{u}_k^\top \mathbf{v}_i)},$$

ahol
bármely $i$ indexre a $\mathbf{v}_i$ és $\mathbf{u}_i$ vektorok
a $w_i$ szót
rendre középső szóként és kontextusszóként reprezentálják,
valamint $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$
a szótár indexkészlete.

Tekintsük a $w_i$ szót,
amely a korpuszban több alkalommal is előfordulhat.
A teljes korpuszban
minden olyan kontextusszó,
amelynek középső szava $w_i$,
egy *multihalmaz* $\mathcal{C}_i$-t alkot
a szóindexekből,
amely *megenged ismétlődő elemeket*.
Bármely elem
előfordulásainak száma az elem *multiplicitása*.
Szemléltetésként:
tegyük fel, hogy $w_i$ kétszer szerepel a korpuszban,
és azok a kontextusszó-indexek,
amelyeknek $w_i$ a középső szavuk
a két kontextusablakban,
rendre
$k, j, m, k$ és $k, l, k, j$.
Ekkor a $\mathcal{C}_i = \{j, j, k, k, k, k, l, m\}$ multihalmaz
elemeinek $j, k, l, m$ multiplicitása
rendre 2, 4, 1, 1.

Jelöljük most $x_{ij}$-vel a $j$ elem multiplicitását
a $\mathcal{C}_i$ multihalmazban.
Ez a $w_j$ szó (mint kontextusszó)
és a $w_i$ szó (mint középső szó)
globális együttes előfordulási száma
azonos kontextusablakban
a teljes korpuszban.
Ezeket a globális korpuszstatisztikákat felhasználva
a skip-gram modell veszteségfüggvénye
ekvivalens a következővel:

$$-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-x_ij`

Jelöljük továbbá $x_i$-vel
a $w_i$ szó középső szóként szereplő
kontextusablakain belüli
összes kontextusszó számát,
ami ekvivalens $|\mathcal{C}_i|$-vel.
Legyen $p_{ij}$
az $x_{ij}/x_i$ feltételes valószínűség,
amely a $w_j$ kontextusszó generálásának valószínűsége
a $w_i$ középső szó feltételével;
ekkor :eqref:`eq_skipgram-x_ij` átírható:

$$-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-p_ij`

A :eqref:`eq_skipgram-p_ij` egyenletben $-\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$
a globális korpuszstatisztikák
$p_{ij}$ feltételes eloszlásának
és a modell $q_{ij}$ előrejelzési eloszlásának
kereszt-entrópiáját számítja ki.
Ezt a veszteséget
a fentiek szerint $x_i$ súlyozza.
A :eqref:`eq_skipgram-p_ij`-beli veszteségfüggvény minimalizálása
lehetővé teszi, hogy az előrejelzett feltételes eloszlás
közelítse a globális korpuszstatisztikák
feltételes eloszlását.


Bár a kereszt-entrópia veszteségfüggvényt
általánosan használják
valószínűségi eloszlások közötti távolság mérésére,
itt nem feltétlenül a legjobb választás.
Egyrészt, ahogy :numref:`sec_approx_train`-ben is megjegyeztük,
$q_{ij}$ megfelelő normalizálásának
a teljes szótáron vett összeg a költsége,
ami számítási szempontból rendkívül drága lehet.
Másrészt
nagy korpusz esetén
számos ritka eseményt a kereszt-entrópia veszteség
aránytalanul nagy súllyal kezel.

## A GloVe modell

Erre tekintettel
a *GloVe* modell három változtatást vezet be
a négyzetesen veszteségű skip-gram modellhez képest :cite:`Pennington.Socher.Manning.2014`:

1. A nem valószínűségi eloszlásként kezelt $p'_{ij}=x_{ij}$ és $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$ változókat mindkét oldalon logaritmálják, így a négyzetes veszteségtag $\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$ lesz.
2. Minden $w_i$ szóhoz két skaláris modellparamétert adnak: a középső szó $b_i$ és a kontextusszó $c_i$ torzítását.
3. Az egyes veszteségtagok súlyát a $h(x_{ij})$ súlyfüggvénnyel helyettesítik, ahol $h(x)$ a $[0, 1]$ intervallumon monoton növekvő.

Mindezeket egybefoglalva, a GloVe tanítása az alábbi veszteségfüggvény minimalizálásából áll:

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$
:eqlabel:`eq_glove-loss`

A súlyfüggvényre egy javasolt választás:
$h(x) = (x/c) ^\alpha$ (pl. $\alpha = 0{,}75$), ha $x < c$ (pl. $c = 100$); egyébként $h(x) = 1$.
Ebben az esetben,
mivel $h(0)=0$,
minden $x_{ij}=0$-ra vonatkozó négyzetes veszteségtag elhagyható
a számítási hatékonyság érdekében.
Például
minibatch sztochasztikus gradienscsökkenés alkalmazásakor
minden iterációban
véletlenszerűen mintavételezünk egy minibatch *nullától különböző* $x_{ij}$-t,
hogy kiszámítsuk a gradienseket
és frissítsük a modell paramétereit.
Fontos megjegyezni, hogy ezek a nemnulla $x_{ij}$ értékek
előre kiszámított globális korpuszstatisztikák;
ezért nevezzük a modellt GloVe-nak,
a *Global Vectors* rövidítéseként.

Kiemelendő, hogy
ha a $w_i$ szó szerepel a $w_j$ szó kontextusablakában,
akkor *fordítva is igaz*.
Ezért $x_{ij}=x_{ji}$.
Ellentétben a word2vec modellel,
amely az aszimmetrikus $p_{ij}$ feltételes valószínűséget illeszti,
a GloVe a szimmetrikus $\log \, x_{ij}$ értéket illeszti.
Következésképpen bármely szó középső szó vektora
és kontextusszó vektora matematikailag ekvivalens a GloVe modellben.
A gyakorlatban azonban az eltérő inicializálási értékek miatt
ugyanaz a szó a tanítás után
e két vektorban mégis különböző értékeket vehet fel:
a GloVe ezeket összegzi kimeneti vektorrá.



## A GloVe modell értelmezése az együttes előfordulási valószínűségek arányán keresztül


A GloVe modellt egy másik nézőpontból is értelmezhetjük.
A :numref:`subsec_skipgram-global`-ban bevezetett jelölésrendszert használva
legyen $p_{ij} \stackrel{\textrm{def}}{=} P(w_j \mid w_i)$ a $w_j$ kontextusszó generálásának feltételes valószínűsége,
feltéve, hogy $w_i$ a középső szó a korpuszban.
A :numref:`tab_glove` táblázat
néhány együttes előfordulási valószínűséget
és azok arányát mutatja be
az „ice" és a „steam" szavakhoz viszonyítva,
egy nagy korpusz statisztikái alapján.


:Szó–szó együttes előfordulási valószínűségek és arányaik egy nagy korpuszból (a :citet:`Pennington.Socher.Manning.2014` 1. táblázata alapján)
:label:`tab_glove`

|$w_k$=|solid|gas|water|fashion|
|:--|:-|:-|:-|:-|
|$p_1=P(w_k\mid \textrm{ice})$|0.00019|0.000066|0.003|0.000017|
|$p_2=P(w_k\mid\textrm{steam})$|0.000022|0.00078|0.0022|0.000018|
|$p_1/p_2$|8.9|0.085|1.36|0.96|



A :numref:`tab_glove` táblázatból a következők figyelhetők meg:

* Egy olyan $w_k$ szóra, amely az „ice"-hez kapcsolódik, de a „steam"-hez nem – például $w_k=\textrm{solid}$ –, nagyobb együttes előfordulási arány várható, például 8,9.
* Egy olyan $w_k$ szóra, amely a „steam"-hez kapcsolódik, de az „ice"-hez nem – például $w_k=\textrm{gas}$ –, kisebb együttes előfordulási arány várható, például 0,085.
* Egy olyan $w_k$ szóra, amely mind az „ice"-hez, mind a „steam"-hez kapcsolódik – például $w_k=\textrm{water}$ –, az arány 1-hez közeli értéket vesz fel, például 1,36.
* Egy olyan $w_k$ szóra, amely sem az „ice"-hez, sem a „steam"-hez nem kapcsolódik – például $w_k=\textrm{fashion}$ –, az arány szintén 1-hez közeli, például 0,96.




Belátható, hogy az együttes előfordulási valószínűségek aránya
szemléletesen kifejezi a szavak közötti összefüggéseket.
Ezért tervezhetünk egy háromszó-vektoros függvényt,
amely ezt az arányt illeszti.
A ${p_{ij}}/{p_{ik}}$ együttes előfordulási valószínűség-arányhoz –
ahol $w_i$ a középső szó,
$w_j$ és $w_k$ pedig a kontextusszavak –
keressük azt az $f$ függvényt, amely:

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$
:eqlabel:`eq_glove-f`

Az $f$-re vonatkozó számos lehetséges tervből
az alábbiakban csupán egy ésszerű választást mutatunk be.
Mivel az együttes előfordulási valószínűségek aránya skalár,
megköveteljük, hogy
$f$ skaláris függvény legyen, például:
$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = f\left((\mathbf{u}_j - \mathbf{u}_k)^\top {\mathbf{v}}_i\right)$.
Ha :eqref:`eq_glove-f`-ben felcseréljük a $j$ és $k$ indexeket,
teljesülnie kell, hogy
$f(x)f(-x)=1$,
amiből egyik lehetséges megoldás $f(x)=\exp(x)$,
azaz:

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = \frac{\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right)}{\exp\left(\mathbf{u}_k^\top {\mathbf{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$

Válasszuk most az
$\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right) \approx \alpha p_{ij}$
közelítést,
ahol $\alpha$ egy konstans.
Mivel $p_{ij}=x_{ij}/x_i$, mindkét oldalt logaritmálva kapjuk: $\mathbf{u}_j^\top {\mathbf{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$.
A $- \log\, \alpha + \log\, x_i$ tagok illesztésére további torzítási tagokat használhatunk, mint a középső szó $b_i$ és a kontextusszó $c_j$ torzítása:

$$\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log\, x_{ij}.$$
:eqlabel:`eq_glove-square`

A :eqref:`eq_glove-square` négyzethibájának
súlyozással való mérésével
megkapjuk a :eqref:`eq_glove-loss`-beli GloVe veszteségfüggvényt.



## Összefoglalás

* A skip-gram modell értelmezhető globális korpuszstatisztikák – például szó–szó együttes előfordulási számok – segítségével.
* A kereszt-entrópia veszteségfüggvény két valószínűségi eloszlás különbségének mérésére nem feltétlenül a legjobb választás, különösen nagy korpusz esetén. A GloVe négyzetesen veszteségű illesztést alkalmaz az előre kiszámított globális korpuszstatisztikákra.
* A GloVe modellben bármely szó középső szó vektora és kontextusszó vektora matematikailag ekvivalens.
* A GloVe értelmezhető a szó–szó együttes előfordulási valószínűségek arányán keresztül.


## Feladatok

1. Ha $w_i$ és $w_j$ szavak ugyanabban a kontextusablakban fordulnak elő, hogyan lehetne felhasználni a szövegsorozatban elfoglalt távolságukat a $p_{ij}$ feltételes valószínűség kiszámításának módosítására? Útmutatás: lásd a GloVe-cikk 4.2. szakaszát :cite:`Pennington.Socher.Manning.2014`.
1. Bármely szóra: matematikailag ekvivalens-e a GloVe modellben a középső szó torzítása és a kontextusszó torzítása? Miért?


[Discussions](https://discuss.d2l.ai/t/385)
