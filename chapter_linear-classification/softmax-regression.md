# Softmax regresszió
:label:`sec_softmax`

A :numref:`sec_linear_regression` szakaszban bemutattuk a lineáris regressziót,
amelyet :numref:`sec_linear_scratch` szakaszban nulláról valósítottunk meg,
majd :numref:`sec_linear_concise` szakaszban egy mélytanulási keretrendszer
magas szintű API-jait felhasználva is elkészítettük.

A regresszió az az eszköz, amelyet akkor veszünk elő, amikor arra vagyunk kíváncsiak,
hogy *mennyi?* vagy *hány?* Például ha megjósolnánk egy ház eladási árát,
egy baseball-csapat várható győzelmeinek számát, vagy azt, hogy
egy beteg hány napig marad kórházban elbocsátása előtt,
akkor valószínűleg regressziós modellt keresünk.
Azonban még a regressziós modellek között is vannak fontos különbségek.
Például egy ház ára sosem lehet negatív, és a változások sokszor az alap árhoz
képest relatívak. Ezért hatékonyabb lehet az ár logaritmusán végezni a regressziót.
Hasonlóan, a kórházban töltött napok száma *diszkrét, nemnegatív* véletlen változó,
tehát a legkisebb négyzetek módszere sem feltétlenül ideális.
Az ilyen típusú esemény-idő modellezés számos további bonyodalommal jár,
amelyekkel az ún. *túlélési elemzés* nevű szakterületen foglalkoznak.

A lényeg nem az, hogy elijesszenek ezek a finomságok, hanem csupán annak érzékeltetése,
hogy a becslésnek számos aspektusa van a négyzethibák minimalizálásán túl.
Tágabb értelemben is igaz, hogy a felügyelt tanulás jóval több, mint regresszió.
Ebben a szakaszban az *osztályozási* problémákra összpontosítunk,
ahol a *mennyi?* típusú kérdések helyett *melyik kategória?* típusú kérdéseket teszünk fel.



* Ez az e-mail a spam mappába vagy a bejövők közé tartozik?
* Valószínűbb-e, hogy az adott ügyfél feliratkozik az előfizetéses szolgáltatásra, vagy sem?
* Ez a kép szamarat, kutyát, macskát vagy kakast ábrázol?
* Melyik filmet fogja Aston legközelebb megnézni?
* A könyv melyik fejezetét fogod legközelebb olvasni?

A gépi tanulás területén az *osztályozás* szó általánosan két, egymástól kissé eltérő problémát jelöl:
(i) azokat, ahol csupán a példák kategóriákhoz való kemény hozzárendelése érdekel minket;
és (ii) azokat, ahol puha hozzárendelésre törekszünk, vagyis minden egyes kategória valószínűségét is meg kívánjuk becsülni.
A kettő közötti határ sokszor elmosódik, részben azért, mert még ha csak a kemény hozzárendelések érdekelnek is minket,
a modelljeink általában puha hozzárendeléseket adnak.

Sőt, vannak esetek, amikor egynél több címke is igaz lehet.
Például egy hírcsikk egyszerre foglalkozhat szórakoztatással, üzlettel és az űrhajózással,
de nem feltétlenül az orvostudománnyal vagy a sporttal.
Ilyenkor egyetlen kategóriába sorolni önmagában nem lenne különösebben hasznos.
Ez a probléma [többcímkés osztályozásként](https://en.wikipedia.org/wiki/Multi-label_classification) ismert.
Áttekintésért lásd :citet:`Tsoumakas.Katakis.2007`,
és képek osztályozásánál hatékony algoritmusért :citet:`Huang.Xu.Yu.2015`.

## Osztályozás
:label:`subsec_classification-problem`

Hogy kézzel fogjuk a témát, kezdjük egy egyszerű képosztályozási feladattal.
Minden bemeneti adat egy $2\times2$ méretű szürkeárnyalatos kép.
Minden pixelértéket egyetlen skalárral ábrázolhatunk,
így négy jellemzőhöz jutunk: $x_1, x_2, x_3, x_4$.
Tételezzük fel, hogy minden kép a „macska", „csirke" és „kutya" kategóriák egyikébe tartozik.

Ezután el kell döntenünk, hogyan ábrázoljuk a címkéket.
Két nyilvánvaló lehetőségünk van.
A legkézenfekvőbb talán az lenne, ha $y \in \{1, 2, 3\}$ értékeket választanánk,
ahol az egészek rendre $\{\textrm{kutya}, \textrm{macska}, \textrm{csirke}\}$ kategóriákat jelölnek.
Ez remek módszer az információ *tárolására* számítógépen.
Ha a kategóriák között valamilyen természetes sorrend lenne,
például ha $\{\textrm{csecsemő}, \textrm{kisgyermek}, \textrm{serdülő}, \textrm{fiatal felnőtt}, \textrm{felnőtt}, \textrm{idős}\}$
kategóriákat próbálnánk megjósolni, akkor akár értelmes is lenne
[ordinális regressziós](https://en.wikipedia.org/wiki/Ordinal_regression) problémaként kezelni ezt.
Különböző rangsorolási veszteségfüggvények áttekintéséért lásd :citet:`Moon.Smola.Chang.ea.2010`,
bayesi megközelítésért egynél több módusú válaszok esetére lásd :citet:`Beutel.Murray.Faloutsos.ea.2014`.

Általában az osztályozási feladatokban nincsen természetes sorrend az osztályok között.
Szerencsére a statisztikusok már régen találtak egy egyszerű módot a kategorikus adatok ábrázolására: az *egy-forró kódolást*.
Az egy-forró kódolás egy olyan vektor, amelynek annyi komponense van, ahány kategória létezik.
Az adott példányhoz tartozó kategória komponensét 1-re állítjuk, az összes többi komponens értéke 0 lesz.
A mi esetünkben a $y$ címke egy háromdimenziós vektor,
ahol $(1, 0, 0)$ a „macskát", $(0, 1, 0)$ a „csirkét" és $(0, 0, 1)$ a „kutyát" jelenti:

$$y \in \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}.$$

### Lineáris modell

Az összes lehetséges osztályhoz tartozó feltételes valószínűségek becsléséhez
egy több kimenetű modellre van szükségünk, osztályonként egy kimenettel.
A lineáris modellekkel végzett osztályozáshoz annyi affin függvényre van szükségünk,
ahány kimenetünk van. Szigorúan véve eggyel kevesebb is elegendő lenne,
mivel az utolsó kategória valószínűsége az $1$ és a többi kategória összegének különbsége,
de a szimmetria kedvéért egy kissé redundáns paraméterezetést alkalmazunk.
Minden kimenet saját affin függvénynek felel meg.
A mi esetünkben, ahol 4 jellemző és 3 kimeneti kategória van,
12 skalárra van szükségünk a súlyok ábrázolásához ($w$ alsó indexekkel)
és 3 skalárra az eltolások megadásához ($b$ alsó indexekkel):

$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}
$$

A megfelelő neurális hálózat diagramja a :numref:`fig_softmaxreg` ábrán látható.
A lineáris regresszióhoz hasonlóan egyrétegű neurális hálózatot használunk.
Mivel minden kimenet ($o_1, o_2$ és $o_3$) minden bemenettől ($x_1$, $x_2$, $x_3$ és $x_4$) függ,
a kimeneti réteg *teljesen összekötött rétegként* is leírható.

![A softmax regresszió egy egyrétegű neurális hálózat.](../img/softmaxreg.svg)
:label:`fig_softmaxreg`

Tömörebb jelöléssel vektorokat és mátrixokat használunk:
$\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$
sokkal praktikusabb matematikailag és kódban is.
Megjegyezzük, hogy az összes súlyt egy $3 \times 4$-es mátrixba gyűjtöttük,
az eltolások pedig $\mathbf{b} \in \mathbb{R}^3$ vektorban szerepelnek.

### A Softmax
:label:`subsec_softmax_operation`

Megfelelő veszteségfüggvénnyel megpróbálhatnánk közvetlenül minimalizálni
$\mathbf{o}$ és a $\mathbf{y}$ címkék közötti különbséget.
Bár az osztályozás vektoros értékű regressziós problémaként való kezelése meglepően jól működik,
az alábbi szempontok miatt mégsem kielégítő:

* Nincs garancia arra, hogy a $o_i$ kimenetek összege $1$ lesz, ahogyan azt a valószínűségektől elvárjuk.
* Nincs garancia arra sem, hogy a $o_i$ kimenetek nemnegatívak lesznek, még akkor sem, ha összegük $1$, vagy hogy egyik sem haladja meg az $1$-et.

Mindkét tényező megnehezíti a becslési feladatot, és a megoldás erősen érzékennyé válik a kiugró értékekre.
Például ha feltételezzük, hogy pozitív lineáris összefüggés van a hálószobák száma
és annak valószínűsége között, hogy valaki megvesz egy házat,
a valószínűség meghaladhatja az $1$-et, ha egy kastélyról van szó!
Ezért szükségünk van egy mechanizmusra, amely „összenyomja" a kimeneteket.

Számos módszer létezik erre a célra.
Például feltételezhetjük, hogy a $\mathbf{o}$ kimenetek a $\mathbf{y}$ zajos változatai,
ahol a zaj egy normális eloszlásból vett $\boldsymbol{\epsilon}$ értékkel adódik hozzá.
Más szóval $\mathbf{y} = \mathbf{o} + \boldsymbol{\epsilon}$,
ahol $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$.
Ez az ún. [probit modell](https://en.wikipedia.org/wiki/Probit_model),
amelyet először :citet:`Fechner.1860` vezetett be.
Bár vonzó lehetőség, a softmax-hoz képest nem vezet olyan jó eredményekre,
és az optimalizálási feladat sem válik különösen széppé.

Egy másik megközelítés (amely a nemnegatívitást is biztosítja)
az exponenciális függvény alkalmazása: $P(y = i) \propto \exp o_i$.
Ez valóban kielégíti azt a követelményt, hogy a feltételes osztályvalószínűség
növekszik $o_i$ növekedésével, monoton, és minden valószínűség nemnegatív.
Ezeket az értékeket $1$-re normalizálhatjuk úgy, hogy mindegyiket elosztjuk az összegükkel.
Ezt a folyamatot *normalizálásnak* nevezzük.
A két lépés összekapcsolásával kapjuk a *softmax* függvényt:

$$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o}) \quad \textrm{ahol}\quad \hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}.$$
:eqlabel:`eq_softmax_y_and_o`

Vegyük észre, hogy $\mathbf{o}$ legnagyobb koordinátája felel meg a $\hat{\mathbf{y}}$ szerint
legvalószínűbb osztálynak.
Ráadásul mivel a softmax művelet megőrzi argumentumai sorrendjét,
nem kell kiszámítanunk a softmax-ot ahhoz, hogy megállapítsuk, melyik osztálynak van a legnagyobb valószínűsége:

$$
\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j.
$$


A softmax ötlete :citet:`Gibbs.1902` fizikából átvett gondolatain alapul.
Még korábbra visszanyúlva Boltzmann, a modern statisztikus fizika atyja
ezt a trükköt arra használta, hogy a gázmolekulák energiaállapotain értelmezett eloszlást modellezze.
Konkrétan felfedezte, hogy egy termodinamikai rendszerben,
például egy gáz molekuláiban, az egyes energiaállapotok előfordulása
arányos $\exp(-E/kT)$-vel,
ahol $E$ az adott állapot energiája, $T$ a hőmérséklet, $k$ pedig a Boltzmann-állandó.
Amikor a statisztikusok egy statisztikai rendszer „hőmérsékletének" növeléséről vagy csökkentéséről beszélnek,
$T$ megváltoztatására utalnak, ami az alacsonyabb vagy magasabb energiaállapotok kedvezményezéséhez vezet.
Gibbs ötletét követve az energia a hibának felel meg.
Az energiaalapú modellek :cite:`Ranzato.Boureau.Chopra.ea.2007`
ezt a nézőpontot alkalmazzák a mélytanulás problémáinak leírásában.

### Vektorizáció
:label:`subsec_softmax_vectorization`

A számítási hatékonyság javítása érdekében adatok mini-batch-eket használva vektorizálunk.
Tegyük fel, hogy adott egy $\mathbf{X} \in \mathbb{R}^{n \times d}$ mini-batch,
amely $n$ példát tartalmaz, mindegyik $d$ dimenziós (azaz $d$ bemeneti jellemzőjű).
Továbbá tegyük fel, hogy $q$ kimeneti kategóriánk van.
Ekkor a súlyokra $\mathbf{W} \in \mathbb{R}^{d \times q}$
és az eltolásokra $\mathbf{b} \in \mathbb{R}^{1\times q}$ teljesül.

$$ \begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned} $$
:eqlabel:`eq_mini-batch_softmax_reg`

Ez az $\mathbf{X} \mathbf{W}$ mátrix-mátrix szorzatba tömöríti a domináns műveletet.
Mivel $\mathbf{X}$ minden sora egy adatpéldányt képvisel,
a softmax műveletet *soronként* végezhetjük el:
$\mathbf{O}$ minden sorában hatványozzuk az összes elemet, majd normalizáljuk összegükkel.
Fontos azonban ügyelni arra, hogy nagy számok hatványozását és logaritmálását elkerüljük,
mivel ez numerikus túlcsordulást vagy alulcsordulást okozhat.
A mélytanulási keretrendszerek ezt automatikusan kezelik.

## Veszteségfüggvény
:label:`subsec_softmax-regression-loss-func`

Most, hogy megvan a leképezésünk a $\mathbf{x}$ jellemzőktől a $\mathbf{\hat{y}}$ valószínűségekig,
szükségünk van egy módszerre, amellyel optimalizáljuk e leképezés pontosságát.
A maximális valószínűség becslésére (maximum likelihood estimation) támaszkodunk,
amellyel már találkoztunk, amikor a négyzethibás veszteség valószínűségi megalapozását adtuk meg
a :numref:`subsec_normal_distribution_and_squared_loss` szakaszban.

### Log-valószínűség

A softmax függvény egy $\hat{\mathbf{y}}$ vektort ad,
amelyet az egyes osztályok (becsült) feltételes valószínűségeként értelmezhetünk
egy adott $\mathbf{x}$ bemenetre, például $\hat{y}_1$ = $P(y=\textrm{macska} \mid \mathbf{x})$.
A következőkben feltételezzük, hogy az $\mathbf{X}$ jellemzőkből álló adathalmaz
$\mathbf{Y}$ címkéit egy-forró kódolású vektorral ábrázoljuk.
A becsléseinket az adatokhoz hasonlíthatjuk annak ellenőrzésével, hogy modellünk szerint
mekkora valószínűséggel fordulnak elő a tényleges osztályok a jellemzők alapján:

$$
P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}).
$$

A faktorizációt azért alkalmazhatjuk, mert feltételezzük, hogy minden egyes címke
a saját $P(\mathbf{y}\mid\mathbf{x}^{(i)})$ eloszlásából függetlenül kerül kisorsolásra.
Mivel tagok szorzatának maximalizálása kényelmetlen,
a negatív logaritmust vesszük, és az ekvivalens negatív log-valószínűség minimalizálási
feladatot oldjuk meg:

$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
= \sum_{i=1}^n l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}),
$$

ahol bármely $\mathbf{y}$ címke és $\hat{\mathbf{y}}$ modellbecslés párra
$q$ osztály esetén a veszteségfüggvény $l$:

$$ l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j. $$
:eqlabel:`eq_l_cross_entropy`

A később megmagyarázott okokból az :eqref:`eq_l_cross_entropy`-beli veszteségfüggvényt
általánosan *keresztentrópia-veszteségnek* nevezik.
Mivel $\mathbf{y}$ egy $q$ hosszú egy-forró vektor,
az összes $j$ koordinátára vett összeg egyetlen tagot kivéve nullává válik.
Megjegyezzük, hogy a $l(\mathbf{y}, \hat{\mathbf{y}})$ veszteség alulról $0$-val korlátos,
ha $\hat{\mathbf{y}}$ valószínűségi vektor: egyetlen elem sem nagyobb $1$-nél,
tehát azok negatív logaritmusa sem lehet $0$-nál kisebb;
$l(\mathbf{y}, \hat{\mathbf{y}}) = 0$ kizárólag akkor áll fenn, ha a tényleges címkét *biztonsággal* megjósoljuk.
Ez véges súlybeállításokkal soha nem fordulhat elő, mivel egy softmax kimenet $1$-hez való közelítéséhez
a megfelelő $o_i$ bemenetet a végtelenbe (vagy az összes többi $o_j$ kimenetet negatív végtelenbe)
kellene vinni ($j \neq i$ esetén).
Még ha modellünk $0$ kimeneti valószínűséget is rendelhetne,
ilyen magas bizonyossággal tett bármely hiba végtelen veszteséget vonna maga után ($-\log 0 = \infty$).


### Softmax és keresztentrópia-veszteség
:label:`subsec_softmax_and_derivatives`

Mivel a softmax függvény és a hozzá tartozó keresztentrópia-veszteség annyira elterjedt,
érdemes jobban megérteni a kiszámításuk módját.
Behelyettesítve :eqref:`eq_softmax_y_and_o`-t a :eqref:`eq_l_cross_entropy`-beli
veszteség definíciójába, és felhasználva a softmax definícióját:

$$
\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j \\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
\end{aligned}
$$

A folyamat jobb megértéséhez vizsgáljuk meg bármely $o_j$ logit szerinti deriváltat:

$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.
$$

Más szóval a derivált a modellünk által a softmax művelettel kifejezett valószínűség
és a tényleges esemény — amelyet az egy-forró vektor elemei fejeznek ki — különbsége.
Ez nagyon hasonlít arra, amit a regressziónál láttunk,
ahol a gradiens az $y$ megfigyelés és a $\hat{y}$ becslés különbsége volt.
Ez nem véletlen.
Bármely exponenciális családbeli modellben a log-valószínűség gradiensét pontosan ez a tag adja meg.
Ez a tény megkönnyíti a gradiensek kiszámítását a gyakorlatban.

Most vizsgáljuk meg azt az esetet, amikor nem egyetlen kimenetelt figyelünk meg,
hanem a kimeneteleknek egy teljes eloszlását.
Ugyanolyan jelöléssel ábrázolhatjuk a $\mathbf{y}$ címkét, mint korábban.
Az egyetlen különbség, hogy csupán bináris elemeket tartalmazó vektor helyett,
mint $(0, 0, 1)$, most egy általános valószínűségi vektort kapunk,
például $(0.1, 0.2, 0.7)$.
Az :eqref:`eq_l_cross_entropy`-ban a $l$ veszteség definiálásához korábban alkalmazott matematika
ugyanúgy működik, csupán az értelmezés általánosabb.
Ez a veszteség a veszteség várható értéke egy címkék feletti eloszlásban.
Ezt a veszteséget *keresztentrópia-veszteségnek* nevezzük, és az osztályozási problémák
egyik legelterjedtebb veszteségfüggvénye.
A név misztikumát feloldhatjuk az információelmélet alapjainak rövid bemutatásával.
Röviden: megmutatja, hány bitre van szükségünk ahhoz, hogy kódoljuk, amit látunk, $\mathbf{y}$,
ahhoz képest, amit megjósolunk, $\hat{\mathbf{y}}$.
A következőkben egy nagyon egyszerű magyarázatot adunk.
Az információelméletről részletesebben lásd :citet:`Cover.Thomas.1999` vagy :citet:`mackay2003information`.



## Az információelmélet alapjai
:label:`subsec_info_theory_basics`

Számos mélytanulással foglalkozó cikk az információelméletből vett fogalmakra és intuíciókra támaszkodik.
Hogy ezeket megérthessük, szükségünk van néhány közös fogalomra.
Ez egyfajta gyors útmutató.
Az *információelmélet* az információ (más szóval adatok) kódolásának,
dekódolásának, átvitelének és kezelésének problémájával foglalkozik.

### Entrópia

Az információelmélet alapgondolata az adatokban rejlő információ mennyiségének mérése.
Ez korlátot szab az adatok tömöríthetőségére.
Egy $P$ eloszlás *entrópiája*, $H[P]$, a következőképpen definiált:

$$H[P] = \sum_j - P(j) \log P(j).$$
:eqlabel:`eq_softmax_reg_entropy`

Az információelmélet egyik alaptétele kimondja, hogy ahhoz, hogy a $P$ eloszlásból
véletlenszerűen vett adatokat kódoljuk, legalább $H[P]$ „nat"-ra van szükségünk :cite:`Shannon.1948`.
Ha kíváncsi vagy, mi a „nat": ez a bit megfelelője, de $e$ alapú kód esetén (nem 2 alapú).
Így egy nat $\frac{1}{\log(2)} \approx 1.44$ bitnek felel meg.


### Meglepetés (Surprisal)

Talán azon töprengsz, mi köze van a tömörítésnek a jósláshoz.
Képzelj el egy adatfolyamot, amelyet tömöríteni szeretnél.
Ha mindig könnyen meg tudjuk jósolni a következő tokent,
akkor az adatok könnyen tömöríthetők.
Vegyük a szélső esetet, amikor a folyam minden tokenje mindig ugyanazt az értéket veszi fel.
Ez egy igen unalmas adatfolyam!
Nemcsak unalmas, de könnyű is megjósolni.
Mivel a tokenek mindig ugyanolyanok, nem kell semmilyen információt átvinni
az adatfolyam tartalmának közléséhez.
Könnyű megjósolni, könnyű tömöríteni.

Ha azonban nem tudjuk tökéletesen megjósolni minden eseményt,
időnként meglepetés ér minket.
Meglepettségünk annál nagyobb, minél kisebb valószínűséget rendeltünk az adott eseményhez.
Claude Shannon a $\log \frac{1}{P(j)} = -\log P(j)$ formulát választotta
a $j$ esemény megfigyelésekor tapasztalt *meglepettség* mérésére,
amennyiben (szubjektíven) $P(j)$ valószínűséget rendeltünk hozzá.
Az :eqref:`eq_softmax_reg_entropy`-ban definiált entrópia
ekkor a *várható meglepettség*, amikor helyes valószínűségeket rendelünk az eseményekhez,
amelyek valóban illeszkednek az adatgeneráló folyamathoz.


### Keresztentrópia újra

Ha tehát az entrópia a meglepettség szintje, amelyet az tapasztal,
aki ismeri a valódi valószínűségeket, akkor felmerül a kérdés: mi a keresztentrópia?
A $P$-től $Q$-ig mért keresztentrópia, $H(P, Q)$ jelöléssel,
egy szubjektív $Q$ valószínűségekkel rendelkező megfigyelő várható meglepettségét adja meg,
amikor valójában $P$ valószínűségek szerint generált adatokat lát.
Ez $H(P, Q) \stackrel{\textrm{def}}{=} \sum_j - P(j) \log Q(j)$ formulával írható le.
A legkisebb lehetséges keresztentrópia akkor érhető el, ha $P=Q$.
Ebben az esetben a $P$-től $P$-ig mért keresztentrópia $H(P, P)= H(P)$.

Röviden: a keresztentrópia alapú osztályozási célkitűzést kétféleképpen értelmezhetjük:
(i) a megfigyelt adatok valószínűségének maximalizálásaként;
és (ii) a meglepettségünk (és így a szükséges bitek számának) minimalizálásaként,
amelyeket a címkék közléséhez kellene felhasználni.

## Összefoglalás és vita

Ebben a szakaszban találkoztunk az első nem triviális veszteségfüggvénnyel,
amely lehetővé teszi az optimalizálást *diszkrét* kimeneti tereken.
A tervezés kulcsa egy valószínűségi megközelítés volt:
a diszkrét kategóriákat egy valószínűségi eloszlásból vett minták példányainak tekintettük.
Mellékhatásként megismertük a softmax-ot, egy hasznos aktivációs függvényt,
amely egy közönséges neurális hálózati réteg kimenetét érvényes diszkrét valószínűségi eloszlássá alakítja.
Láttuk, hogy a keresztentrópia-veszteség deriváltja softmax-szal kombinálva
nagyon hasonlóan viselkedik a négyzethibás veszteség deriváltjához:
a várható viselkedés és a becslés különbségéből adódik.
Bár csak a felszínt karcolhattuk meg,
izgalmas kapcsolatokat fedeztünk fel a statisztikus fizikával és az információelmélettel.

Bár ez elegendő az elinduláshoz, és remélhetőleg felkeltette az étvágyad,
korántsem merültünk mélyre.
Sok mindenre nem tértünk ki, például a számítási szempontokra.
Konkrétan: minden $d$ bemenetű és $q$ kimenetű teljesen összekötött réteg esetén
a paraméteres és számítási igény $\mathcal{O}(dq)$, ami a gyakorlatban tiltóan magas lehet.
Szerencsére a $d$ bemenet $q$ kimenetté való átalakításának e terhe csökkenthető
közelítéssel és tömörítéssel.
Például a Deep Fried Convnets :cite:`Yang.Moczulski.Denil.ea.2015`
permutációk, Fourier-transzformációk és skálázás kombinációját alkalmazza
a költség másodfokúról log-lineárissá csökkentésére.
Hasonló technikák működnek fejlettebb strukturális mátrix-közelítéseknél is :cite:`sindhwani2015structured`.
Végül kvaternió-szerű dekompozíciókat is alkalmazhatunk a költség $\mathcal{O}(\frac{dq}{n})$-re csökkentéséhez,
ha hajlandók vagyunk kis pontosságot feláldozni számítási és tárolási megtakarításért :cite:`Zhang.Tay.Zhang.ea.2021`,
egy $n$ tömörítési faktorral.
Ez egy aktívan kutatott terület.
A kihívást az jelenti, hogy nem feltétlenül a legtömörebb ábrázolást
vagy a legkevesebb lebegőpontos műveletet célozzuk meg,
hanem azt a megoldást, amely a leghatékonyabban hajtható végre modern GPU-kon.

## Feladatok

1. Vizsgáld meg mélyebben az exponenciális családok és a softmax közötti kapcsolatot!
    1. Számítsd ki a softmax keresztentrópia-veszteségének $l(\mathbf{y},\hat{\mathbf{y}})$ második deriváltját!
    1. Számítsd ki a $\mathrm{softmax}(\mathbf{o})$ eloszlás varianciáját, és mutasd meg, hogy egyezik a fent kiszámított második deriválttal!
1. Tegyük fel, hogy három osztályunk van, amelyek egyenlő valószínűséggel fordulnak elő, azaz a valószínűségi vektor $(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$.
    1. Mi a probléma, ha bináris kódot próbálunk tervezni erre?
    1. Tudsz-e jobb kódot tervezni? Tipp: mi történik, ha két független megfigyelést próbálunk kódolni? Mi van, ha $n$ megfigyelést egyszerre kódolunk?
1. Amikor fizikai kábelen átvitt jeleket kódolnak, a mérnökök nem mindig bináris kódokat alkalmaznak. Például a [PAM-3](https://en.wikipedia.org/wiki/Ternary_signal) három jelszintet $\{-1, 0, 1\}$ használ a két szint $\{0, 1\}$ helyett. Hány ternáris egységre van szükség ahhoz, hogy egy egész számot kódoljunk a $\{0, \ldots, 7\}$ tartományban? Elektronikai szempontból miért lehet jobb ötlet ez?
1. A [Bradley--Terry modell](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)
logisztikus modellt alkalmaz a preferenciák megragadásához. Ahhoz, hogy egy felhasználó az almák és narancsok között válasszon, $o_{\textrm{alma}}$ és $o_{\textrm{narancs}}$ pontszámokat feltételezünk. Elvárásunk, hogy a nagyobb pontszámok nagyobb valószínűséggel vezessenek az adott elem kiválasztásához, és a legnagyobb pontszámú elem legyen a legvalószínűbben kiválasztott :cite:`Bradley.Terry.1952`.
    1. Mutasd meg, hogy a softmax eleget tesz ennek a feltételnek!
    1. Mi történik, ha lehetővé akarjuk tenni az almák vagy narancsok helyett a „semmi" alapértelmezett lehetőség választását? Tipp: most a felhasználónak három lehetősége van.
1. A Softmax a nevét a következő leképezéstől kapta: $\textrm{RealSoftMax}(a, b) = \log (\exp(a) + \exp(b))$.
    1. Bizonyítsd be, hogy $\textrm{RealSoftMax}(a, b) > \mathrm{max}(a, b)$!
    1. Mennyire kicsire lehet csökkenteni a két függvény különbségét? Tipp: általánosság elvesztése nélkül felvehetjük, hogy $b = 0$ és $a \geq b$.
    1. Bizonyítsd be, hogy ez fennáll $\lambda^{-1} \textrm{RealSoftMax}(\lambda a, \lambda b)$-re is, feltéve, hogy $\lambda > 0$!
    1. Mutasd meg, hogy $\lambda \to \infty$ esetén $\lambda^{-1} \textrm{RealSoftMax}(\lambda a, \lambda b) \to \mathrm{max}(a, b)$!
    1. Szerkessz egy analóg softmin függvényt!
    1. Terjeszd ki kettőnél több számra!
1. A $g(\mathbf{x}) \stackrel{\textrm{def}}{=} \log \sum_i \exp x_i$ függvényt néha [log-partíciós függvénynek](https://en.wikipedia.org/wiki/Partition_function_(mathematics)) is nevezik.
    1. Bizonyítsd be, hogy a függvény konvex! Tipp: ehhez használd fel, hogy az első derivált a softmax valószínűségeiből adódik, és mutasd meg, hogy a második derivált a variancia.
    1. Mutasd meg, hogy $g$ eltolás-invariáns, azaz $g(\mathbf{x} + b) = g(\mathbf{x})$!
    1. Mi történik, ha az $x_i$ koordináták némelyike nagyon nagy? Mi van, ha mind nagyon kicsik?
    1. Mutasd meg, hogy ha $b = \mathrm{max}_i x_i$-t választunk, numerikusan stabil implementációt kapunk!
1. Tegyük fel, hogy van egy $P$ valószínűségi eloszlásunk. Válasszunk egy másik $Q$ eloszlást, ahol $Q(i) \propto P(i)^\alpha$, $\alpha > 0$ esetén.
    1. Melyik $\alpha$ érték felel meg a hőmérséklet megduplázásának? Melyik a felezésének?
    1. Mi történik, ha a hőmérsékletet $0$-hoz közelítjük?
    1. Mi történik, ha a hőmérsékletet $\infty$-hez közelítjük?

[Megbeszélések](https://discuss.d2l.ai/t/46)
