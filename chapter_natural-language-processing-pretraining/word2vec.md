# Szóbeágyazás (word2vec)
:label:`sec_word2vec`


A természetes nyelv egy összetett rendszer, amelyet jelentések kifejezésére használunk.
Ebben a rendszerben a szavak a jelentés alapegységei.
Ahogy a neve is utal rá,
a *szóvektorok* szavak reprezentálására használt vektorok,
amelyek egyben a szavak jellemzővektorainak vagy reprezentációinak is tekinthetők.
A szavak valós vektorokra való leképezésének technikáját *szóbeágyazásnak* nevezzük.
Az elmúlt években
a szóbeágyazás fokozatosan a természetes nyelvfeldolgozás
alapismeretévé vált.


## A One-Hot Vektorok Rossz Választás

A :numref:`sec_rnn-scratch` fejezetben one-hot vektorokat használtunk a szavak (karakterek mint szavak) reprezentálására.
Tegyük fel, hogy a szótárban szereplő különböző szavak száma (a szótár mérete) $N$,
és minden szó egy $0$-tól $N-1$-ig terjedő egész számnak (indexnek) felel meg.
Az $i$ indexű szó one-hot vektoros reprezentációjához
létrehozunk egy $N$ hosszú, csupa nullából álló vektort,
és az $i$ pozíción lévő elemet 1-re állítjuk.
Ily módon minden szó egy $N$ hosszú vektorral van reprezentálva,
amely közvetlenül felhasználható neurális hálózatokban.


Bár a one-hot szóvektorokat könnyű megalkotni,
általában nem jó választás.
Egyik fő ok, hogy a one-hot szóvektorok nem tudják pontosan kifejezni a különböző szavak közötti hasonlóságot, például a *koszinusz-hasonlóságot*, amelyet gyakran alkalmazunk.
A $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ vektorok esetén a koszinusz-hasonlóság a köztük lévő szög koszinusza:


$$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} \in [-1, 1].$$


Mivel bármely két különböző szó one-hot vektorának koszinusz-hasonlósága 0,
a one-hot vektorok nem képesek kódolni a szavak közötti hasonlóságot.


## Az Önfelügyelt word2vec

A [word2vec](https://code.google.com/archive/p/word2vec/) eszközt a fenti probléma megoldására javasolták.
Minden szót rögzített hosszú vektorra képez le, és ezek a vektorok jobban ki tudják fejezni a különböző szavak közötti hasonlóság és analógia viszonyát.
A word2vec eszköz két modellt tartalmaz: a *skip-gram* :cite:`Mikolov.Sutskever.Chen.ea.2013` és a *continuous bag of words* (CBOW) :cite:`Mikolov.Chen.Corrado.ea.2013` modelleket.
A szemantikailag értelmes reprezentációkhoz
a tanításuk feltételes valószínűségekre támaszkodik,
amelyek úgy értelmezhetők, mint egyes szavak megjóslása
a korpuszban lévő környező szavaik segítségével.
Mivel a felügyelet a felirat nélküli adatokból ered,
mind a skip-gram, mind a continuous bag of words
önfelügyelt modellek.

Az alábbiakban bemutatjuk ezt a két modellt és tanítási módszereiket.


## A Skip-Gram Modell
:label:`subsec_skip-gram`

A *skip-gram* modell azt feltételezi, hogy egy szó felhasználható a szövegsorozatban lévő környező szavak generálására.
Vegyük például a „the", „man", „loves", „his", „son" szövegsorozatot.
Válasszuk a „loves" szót *középső szónak*, és állítsuk a kontextusablak méretét 2-re.
Ahogy az :numref:`fig_skip_gram` ábra mutatja,
a „loves" középső szó esetén
a skip-gram modell azt a feltételes valószínűséget vizsgálja, amellyel generálhatók a *kontextusszavak*: „the", „man", „his" és „son",
amelyek legfeljebb 2 szóra találhatók a középső szótól:

$$P(\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}\mid\textrm{"loves"}).$$

Tegyük fel, hogy
a kontextusszavak a középső szó ismeretében egymástól függetlenül generálódnak (feltételes függetlenség).
Ebben az esetben a fenti feltételes valószínűség átírható:

$$P(\textrm{"the"}\mid\textrm{"loves"})\cdot P(\textrm{"man"}\mid\textrm{"loves"})\cdot P(\textrm{"his"}\mid\textrm{"loves"})\cdot P(\textrm{"son"}\mid\textrm{"loves"}).$$

![A skip-gram modell a középső szóhoz adott, környező kontextusszavak generálásának feltételes valószínűségét vizsgálja.](../img/skip-gram.svg)
:label:`fig_skip_gram`

A skip-gram modellben minden szóhoz két $d$-dimenziós vektoros reprezentáció tartozik
a feltételes valószínűségek kiszámításához.
Konkrétan,
a szótárban $i$ indexű bármely szónál jelöljük $\mathbf{v}_i\in\mathbb{R}^d$-vel
és $\mathbf{u}_i\in\mathbb{R}^d$-vel
a két vektort,
amelyeket *középső* illetve *kontextus* szóként való használatkor alkalmazunk.
Annak feltételes valószínűsége, hogy bármely $w_o$ kontextusszó (a szótárban $o$ indexszel) generálódik a $w_c$ középső szóhoz ($c$ indexszel a szótárban), modellezhető a vektorskaláris szorzatokon végzett softmax művelettel:


$$P(w_o \mid w_c) = \frac{\exp(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)},$$
:eqlabel:`eq_skip-gram-softmax`

ahol a szókincs indexhalmaza $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$.
Adott egy $T$ hosszú szövegsorozat, ahol a $t$ időlépésbeli szót $w^{(t)}$-vel jelöljük.
Feltételezzük, hogy
a kontextusszavak egymástól függetlenül generálódnak
bármely középső szóhoz adottan.
Az $m$ méretű kontextusablak esetén
a skip-gram modell likelihood-függvénye
az összes kontextusszó generálásának valószínűsége
bármely középső szóhoz adottan:


$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

ahol bármely $1$-nél kisebb vagy $T$-nél nagyobb időlépés kihagyható.

### Tanítás

A skip-gram modell paraméterei a szókincs minden szavának középső szóvektora és kontextusszóvektora.
A tanítás során a likelihood-függvény maximalizálásával (azaz maximális likelihood-becsléssel) tanítjuk a modell paramétereit. Ez ekvivalens a következő veszteségfüggvény minimalizálásával:

$$ - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \textrm{log}\, P(w^{(t+j)} \mid w^{(t)}).$$

Amikor stochastic gradient descent-et alkalmazunk a veszteség minimalizálásához,
minden iterációban
véletlenszerűen mintavételezhetünk egy rövidebb részsorozatot,
hogy kiszámítsuk a (sztochasztikus) gradienst ehhez a részsorozathoz, és frissítsük a modell paramétereit.
Ehhez a (sztochasztikus) gradienshez
meg kell kapnunk
a logaritmikus feltételes valószínűség gradienseit
a középső szóvektorhoz és a kontextusszóvektorhoz képest.
Általánosan, az :eqref:`eq_skip-gram-softmax` szerint
a $w_c$ középső szót és $w_o$ kontextusszót érintő
logaritmikus feltételes valószínűség:


$$\log P(w_o \mid w_c) =\mathbf{u}_o^\top \mathbf{v}_c - \log\left(\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)\right).$$
:eqlabel:`eq_skip-gram-log`

Differenciálással megkaphatjuk a $\mathbf{v}_c$ középső szóvektorhoz képesti gradienst:

$$\begin{aligned}\frac{\partial \textrm{log}\, P(w_o \mid w_c)}{\partial \mathbf{v}_c}&= \mathbf{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\mathbf{u}_j^\top \mathbf{v}_c)\mathbf{u}_j}{\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\right) \mathbf{u}_j\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \mathbf{u}_j.\end{aligned}$$
:eqlabel:`eq_skip-gram-grad`


Megjegyezzük, hogy az :eqref:`eq_skip-gram-grad` számításhoz szükség van a szótár összes szavának feltételes valószínűségére, ha $w_c$ a középső szó.
A többi szóvektor gradienseit hasonlóképpen kapjuk meg.


A tanítás után a szótárban $i$ indexű bármely szóhoz megkapjuk mind a $\mathbf{v}_i$ (középső szóként) és $\mathbf{u}_i$ (kontextusszóként) vektort.
A természetes nyelvfeldolgozási alkalmazásokban a skip-gram modell középső szóvektorait általában
szóreprezentációként használják.


## A Continuous Bag of Words (CBOW) Modell


A *continuous bag of words* (CBOW) modell hasonló a skip-gram modellhez.
A fő különbség a skip-gram modellhez képest az, hogy
a continuous bag of words modell
azt feltételezi, hogy egy középső szó
a szövegsorozatban lévő környező kontextusszavai alapján generálódik.
Például
ugyanabban a „the", „man", „loves", „his", „son" szövegsorozatban, ahol a „loves" a középső szó és a kontextusablak mérete 2,
a continuous bag of words modell
azt a feltételes valószínűséget vizsgálja, amellyel a „loves" középső szó generálódik a „the", „man", „his" és „son" kontextusszavak alapján (ahogy az :numref:`fig_cbow` ábra mutatja):

$$P(\textrm{"loves"}\mid\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}).$$

![A continuous bag of words modell azt a feltételes valószínűséget vizsgálja, amellyel a középső szó generálódik a környező kontextusszavakhoz adottan.](../img/cbow.svg)
:label:`fig_cbow`


Mivel a continuous bag of words modellben több kontextusszó is szerepel,
ezeket a kontextusszóvektorokat átlagolják
a feltételes valószínűség kiszámításakor.
Konkrétan,
a szótárban $i$ indexű bármely szónál jelöljük $\mathbf{v}_i\in\mathbb{R}^d$-vel
és $\mathbf{u}_i\in\mathbb{R}^d$-vel
a két vektort,
amelyeket *kontextus* szóként és *középső* szóként való használatkor alkalmazunk
(a szerepek felcserélődtek a skip-gram modellhez képest).
A $w_c$ középső szó (a szótárban $c$ indexszel) generálásának feltételes valószínűsége
a $w_{o_1}, \ldots, w_{o_{2m}}$ környező kontextusszavakhoz adottan (a szótárban $o_1, \ldots, o_{2m}$ indexekkel) modellezhető:



$$P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\exp\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \exp\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots + \mathbf{v}_{o_{2m}}) \right)}.$$
:eqlabel:`fig_cbow-full`


A rövidség kedvéért legyen $\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\}$ és $\bar{\mathbf{v}}_o = \left(\mathbf{v}_{o_1} + \ldots + \mathbf{v}_{o_{2m}} \right)/(2m)$. Ekkor az :eqref:`fig_cbow-full` egyenlet egyszerűsíthető:

$$P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\mathbf{u}_c^\top \bar{\mathbf{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)}.$$

Adott egy $T$ hosszú szövegsorozat, ahol a $t$ időlépésbeli szót $w^{(t)}$-vel jelöljük.
Az $m$ méretű kontextusablak esetén
a continuous bag of words modell likelihood-függvénye
az összes középső szó generálásának valószínűsége
a kontextusszavaikhoz adottan:


$$ \prod_{t=1}^{T}  P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

### Tanítás

A continuous bag of words modellek tanítása
szinte megegyezik
a skip-gram modellek tanításával.
A continuous bag of words modell maximális likelihood-becslése ekvivalens a következő veszteségfüggvény minimalizálásával:



$$  -\sum_{t=1}^T  \textrm{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

Megjegyezzük, hogy

$$\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right).$$

Differenciálással megkaphatjuk bármely $\mathbf{v}_{o_i}$ kontextusszóvektorhoz képesti gradienst ($i = 1, \ldots, 2m$):


$$\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} = \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right) = \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right).$$
:eqlabel:`eq_cbow-gradient`


A többi szóvektor gradienseit hasonlóképpen kapjuk meg.
A skip-gram modellel ellentétben
a continuous bag of words modell általában
a kontextusszóvektorokat használja szóreprezentációként.




## Összefoglalás

* A szóvektorok szavak reprezentálására használt vektorok, amelyek egyben a szavak jellemzővektorainak vagy reprezentációinak is tekinthetők. A szavak valós vektorokra való leképezésének technikáját szóbeágyazásnak nevezzük.
* A word2vec eszköz tartalmazza mind a skip-gram, mind a continuous bag of words modelleket.
* A skip-gram modell azt feltételezi, hogy egy szó felhasználható a szövegsorozatban lévő környező szavak generálására; míg a continuous bag of words modell azt feltételezi, hogy egy középső szó a környező kontextusszavai alapján generálódik.



## Gyakorló feladatok

1. Mekkora a számítási bonyolultság az egyes gradiensek kiszámításánál? Mi lehet a probléma, ha a szótár mérete hatalmas?
1. Az angolban néhány rögzített kifejezés több szóból áll, például „new york". Hogyan tanítható a szóvektoruk? Tipp: lásd a word2vec cikk 4. szakaszát :cite:`Mikolov.Sutskever.Chen.ea.2013`.
1. Gondoljuk át a word2vec tervezést a skip-gram modell példáján keresztül. Milyen kapcsolat áll fenn a skip-gram modellbeli két szóvektor skaláris szorzata és a koszinusz-hasonlóság között? Miért lehet magas a hasonló szemantikájú szópárok szóvektorainak koszinusz-hasonlósága (amelyeket a skip-gram modellel tanítottak)?

[Discussions](https://discuss.d2l.ai/t/381)
