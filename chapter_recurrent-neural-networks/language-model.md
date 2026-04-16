# Nyelvmodellek
:label:`sec_language-model`

A :numref:`sec_text-sequence` fejezetben láttuk, hogyan lehet szöveges sorozatokat tokenekre leképezni, ahol ezek a tokenek diszkrét megfigyelések sorozataként tekinthetők, például szavakként vagy karakterekként. Tegyük fel, hogy a $T$ hosszúságú szöveges sorozat tokenei rendre $x_1, x_2, \ldots, x_T$.
A *nyelvmodellek* célja
a teljes sorozat együttes valószínűségének becslése:

$$P(x_1, x_2, \ldots, x_T),$$

ahol a :numref:`sec_sequence` statisztikai eszközei alkalmazhatók.

A nyelvmodellek hihetetlenül hasznosak. Például egy ideális nyelvmodellnek képesnek kellene lennie természetes szöveget generálni magától, egyszerűen egy tokent húzva egyszerre $x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1)$.
Teljesen ellentétben az írógépen gépelő majommal, az ilyen modellből eredő összes szöveg természetes nyelvként jelenne meg, pl. angol szövegként. Sőt, elegendő lenne egy értelmes párbeszéd generálásához, egyszerűen a szöveget korábbi párbeszédrészletekre kondicionálva.
Nyilvánvalóan még nagyon messze vagyunk egy ilyen rendszer tervezésétől, mivel a szöveget *meg kellene értenie*, nem csak grammatikailag értelmes tartalmat generálnia.

Ennek ellenére a nyelvmodellek korlátozott formájukban is nagy hasznot hajtanak.
Például a "felismerni a beszédet" és "felrázkódtatni egy szép strandot" mondatok nagyon hasonlóan hangoznak.
Ez kétértelműséget okozhat a beszédfelismerésben,
amelyet egy nyelvmodell könnyen felold azzal, hogy elutasítja a második fordítást mint képtelen.
Hasonlóan, dokumentumösszefoglaló algoritmusnál érdemes tudni, hogy "kutya megharapja az embert" sokkal gyakoribb, mint "ember megharapja a kutyát", vagy hogy az "enni akarom a nagymamát" egy meglehetősen zavaró kijelentés, míg az "enni akarok, nagymama" sokkal ártatlanabb.

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from jax import numpy as jnp
```

## Nyelvmodellek tanítása

A nyilvánvaló kérdés az, hogyan kellene modellezni egy dokumentumot, vagy akár egy tokensorozatot.
Tegyük fel, hogy a szöveges adatokat szó szinten tokenizáljuk.
Kezdjük az alapvető valószínűségi szabályok alkalmazásával:

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1}).$$

Például
egy négy szót tartalmazó szöveges sorozat valószínűsége így adható meg:

$$\begin{aligned}&P(\textrm{deep}, \textrm{learning}, \textrm{is}, \textrm{fun}) \\
=&P(\textrm{deep}) P(\textrm{learning}  \mid  \textrm{deep}) P(\textrm{is}  \mid  \textrm{deep}, \textrm{learning}) P(\textrm{fun}  \mid  \textrm{deep}, \textrm{learning}, \textrm{is}).\end{aligned}$$

### Markov-modellek és $n$-gramok
:label:`subsec_markov-models-and-n-grams`

A :numref:`sec_sequence` sorozatmodell-elemzései közül
alkalmazzuk a Markov-modelleket a nyelvmodellezésre.
Egy sorozat feletti eloszlás kielégíti az elsőrendű Markov-tulajdonságot, ha $P(x_{t+1} \mid x_t, \ldots, x_1) = P(x_{t+1} \mid x_t)$. A magasabb rendek hosszabb függőségeknek felelnek meg. Ez számos közelítéshez vezet, amelyeket egy sorozat modellezéséhez alkalmazhatnánk:

$$
\begin{aligned}
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2) P(x_3) P(x_4),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3).
\end{aligned}
$$

Az egy, két és három változót tartalmazó valószínűségi képleteket általában rendre *unigram*, *bigram* és *trigram* modelleknek nevezzük.
A nyelvmodell kiszámításához szükségünk van a szavak valószínűségének
és egy szó feltételes valószínűségének kiszámítására
az előző néhány szó ismeretében.
Megjegyezzük, hogy
az ilyen valószínűségek
a nyelvmodell paraméterei.



### Szógyakoriság

Feltételezzük, hogy a tanítási adathalmaz egy nagy szöveges korpusz, mint például az összes
Wikipedia bejegyzés, a [Project Gutenberg](https://en.wikipedia.org/wiki/Project_Gutenberg),
és az interneten közzétett összes szöveg.
A szavak valószínűsége kiszámítható egy adott szó relatív szógyakoriságából
a tanítási adathalmazban.
Például a $\hat{P}(\textrm{deep})$ becslés kiszámítható
annak valószínűségeként, hogy bármely mondat a "deep" szóval kezdődik.
Egy kissé kevésbé pontos megközelítés az lenne,
ha megszámolnánk a "deep" szó összes előfordulását, és elosztanánk
a korpuszban lévő szavak teljes számával.
Ez meglehetősen jól működik, különösen a
gyakori szavak esetén. Továbblépve megpróbálhatjuk becsülni

$$\hat{P}(\textrm{learning} \mid \textrm{deep}) = \frac{n(\textrm{deep, learning})}{n(\textrm{deep})},$$

ahol $n(x)$ és $n(x, x')$ rendre az egyes szavak
és az egymás melletti szópárok előfordulásainak száma.
Sajnos
a szópár valószínűségének becslése valamivel nehezebb, mivel a
"deep learning" előfordulásai sokkal ritkábbak.
Különösen szokatlan szókombinációknál nehéz lehet
elegendő előfordulást találni pontos becslések eléréséhez.
Ahogy a :numref:`subsec_natural-lang-stat` empirikus eredményei is mutatják,
a háromszavas kombinációknál és azon túl a helyzet rosszabb lesz.
Sok ésszerű háromszavas kombináció lesz, amelyeket valószínűleg nem fogunk látni az adathalmazunkban.
Hacsak nem biztosítunk valamiféle megoldást arra, hogy ilyen szókombinációkhoz nullától eltérő számot rendeljünk, nem fogjuk tudni ezeket használni egy nyelvmodellben. Ha az adathalmaz kicsi, vagy ha a szavak nagyon ritkák, lehet, hogy egyet sem találunk belőlük.

### Laplace-simítás

Egy általános stratégia valamilyen *Laplace-simítás* elvégzése.
A megoldás az,
hogy egy kis konstanst adunk hozzá minden számhoz.
Jelöljük $n$-nel a tanítóhalmaz szavainak teljes számát
és $m$-mel az egyedi szavak számát.
Ez a megoldás segít az egyes szavakon, pl. az alábbi formában:

$$\begin{aligned}
	\hat{P}(x) & = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1}, \\
	\hat{P}(x' \mid x) & = \frac{n(x, x') + \epsilon_2 \hat{P}(x')}{n(x) + \epsilon_2}, \\
	\hat{P}(x'' \mid x,x') & = \frac{n(x, x',x'') + \epsilon_3 \hat{P}(x'')}{n(x, x') + \epsilon_3}.
\end{aligned}$$

Itt $\epsilon_1,\epsilon_2$ és $\epsilon_3$ hiperparaméterek.
Vegyük $\epsilon_1$-et példának:
ha $\epsilon_1 = 0$, nem alkalmazunk simítást;
ha $\epsilon_1$ pozitív végtelenbe tart,
$\hat{P}(x)$ az egyenletes $1/m$ valószínűséghez közelít.
A fenti egy meglehetősen primitív változata annak,
amit más technikák el tudnak érni :cite:`Wood.Gasthaus.Archambeau.ea.2011`.


Sajnos az ehhez hasonló modellek meglehetősen gyorsan kezelhetetlenné válnak
a következő okokból.
Először,
ahogy a :numref:`subsec_natural-lang-stat`-ban tárgyaltuk,
sok $n$-gram nagyon ritkán fordul elő,
így a Laplace-simítás meglehetősen alkalmatlan a nyelvmodellezésre.
Másodszor, minden számot tárolni kell.
Harmadszor, ez teljesen figyelmen kívül hagyja a szavak jelentését. Például
a "macska" és a "cica" hasonló kontextusokban kellene, hogy előforduljon.
Meglehetősen nehéz ilyen modelleket kiegészítő kontextusokhoz igazítani,
míg a deep learning alapú nyelvmodellek jól alkalmasak
ennek figyelembevételére.
Végül, a hosszú szósorozatok szinte biztosan újszerűek lesznek, ezért egy modell, amely egyszerűen
megszámolja a korábban látott szósorozatok gyakoriságát, ott rosszul fog teljesíteni.
Ezért a fejezet hátralévő részében a neurális hálózatokra összpontosítunk a nyelvmodellezés céljából.


## Perplexitás
:label:`subsec_perplexity`

Ezután tárgyaljuk, hogyan mérhetjük a nyelvmodell minőségét, amelyet majd a következő szakaszokban modelljeink értékelésére fogunk használni.
Az egyik módszer az, hogy ellenőrizzük, mennyire meglepő a szöveg.
Egy jó nyelvmodell képes magas pontossággal megjósolni a következő tokeneket.
Tekintsük az "Esik az eső" kifejezés következő folytatásait, amelyeket különböző nyelvmodellek javasolnak:

1. "Esik az eső odakint"
1. "Esik az eső banánfa"
1. "Esik az eső piouw;kcj pwepoiut"

Minőség szempontjából az 1. példa egyértelműen a legjobb. A szavak értelmesek és logikailag összefüggők.
Bár nem feltétlenül tükrözi pontosan, hogy melyik szó következik szemantikailag ("San Franciscóban" és "télen" teljesen ésszerű folytatások lettek volna), a modell képes megragadni, milyen típusú szó következik.
A 2. példa lényegesen rosszabb, mivel értelmetlen folytatást produkál. Mindazonáltal legalább a modell megtanulta a szavak helyesírását és a szavak közötti valamilyen fokú korrelációt. Végül a 3. példa egy rosszul tanított modellt jelez, amely nem illeszkedik megfelelően az adatokra.

A modell minőségét a sorozat valószínűségének kiszámításával is mérhetjük.
Sajnos ez egy olyan szám, amely nehezen értelmezhető és nehezen hasonlítható össze.
Végül is a rövidebb sorozatok sokkal valószínűbbek, mint a hosszabbak,
ezért a modell Tolsztoj remekművén,
a *Háború és béke* c. könyvén való kiértékelése elkerülhetetlenül sokkal kisebb valószínűséget eredményez, mint mondjuk Saint-Exupéry novelláján, *A kis hercegen*. Hiányzik az átlagnak megfelelő valami.

Az információelmélet hasznos itt.
Az entrópiát, a meglepetést és a keresztentrópiát
a softmax regresszió bevezetésekor definiáltuk
(:numref:`subsec_info_theory_basics`).
Ha szöveget szeretnénk tömöríteni, megkérdezhetjük a következő token előrejelzéséről
az aktuális tokenek ismeretében.
Egy jobb nyelvmodellnek lehetővé kell tennie, hogy pontosabban megjósoljuk a következő tokent.
Tehát lehetővé kell tennie, hogy kevesebb bitet használjunk a sorozat tömörítéséhez.
Tehát mérhetjük a sorozat összes $n$ tokenjén átlagolt keresztentrópia-veszteséggel:

$$\frac{1}{n} \sum_{t=1}^n -\log P(x_t \mid x_{t-1}, \ldots, x_1),$$
:eqlabel:`eq_avg_ce_for_lm`

ahol $P$-t egy nyelvmodell adja meg, és $x_t$ a $t$ időlépésnél megfigyelt tényleges token a sorozatból.
Ez lehetővé teszi a különböző hosszúságú dokumentumokon való teljesítmény összehasonlíthatóságát. Történelmi okokból a természetes nyelvfeldolgozás tudósai előnyben részesítenek egy *perplexitásnak* nevezett mennyiséget. Röviden, ez a :eqref:`eq_avg_ce_for_lm` exponenciálisa:

$$\exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right).$$

A perplexitás legjobban a valódi választások számának geometriai közepének reciprokaként értelmezhető, amikor a következő tokent kell kiválasztani. Nézzünk meg néhány esetet:

* A legjobb esetben a modell mindig tökéletesen becsli a céltoken valószínűségét 1-nek. Ebben az esetben a modell perplexitása 1.
* A legrosszabb esetben a modell mindig 0-ra jósolja a céltoken valószínűségét. Ebben a helyzetben a perplexitás pozitív végtelen.
* Az alapvonalon a modell egyenletes eloszlást jósol a szókincs összes elérhető tokenjére. Ebben az esetben a perplexitás egyenlő a szókincs egyedi tokenjének számával. Valójában, ha a sorozatot bármilyen tömörítés nélkül kellene tárolnunk, ez lenne a legjobb, amit az kódolásánál elérhetnénk. Ezért ez egy nem triviális felső korlátot biztosít, amelyet minden hasznos modellnek meg kell haladnia.

## Sorozatok felosztása
:label:`subsec_partitioning-seqs`

Neurális hálózatokat fogunk alkalmazni a nyelvmodellek tervezéséhez,
és perplexitást fogunk használni a modell minőségének értékelésére
a következő token előrejelzésében
az adott tokenek ismeretében szöveges sorozatokban.
A modell bevezetése előtt
tegyük fel, hogy egyszerre
előre meghatározott hosszúságú sorozatok egy mini-batch-ét dolgozza fel.
Most az a kérdés, hogyan lehet **véletlenszerűen olvasni a bemeneti sorozatok és a célsorozatok mini-batch-eit**.


Tegyük fel, hogy az adathalmaz egy $T$ tokenindexből álló sorozat formájában van a `corpus`-ban.
Ezt részsorozatokra fogjuk
felosztani, ahol minden részsorozatnak $n$ tokenje (időlépése) van.
Ahhoz, hogy az egész adathalmaz (szinte) összes tokenjét bejárjuk
minden epochban,
és megkapjuk az összes lehetséges $n$ hosszúságú részsorozatot,
véletlenszerűséget alkalmazhatunk.
Konkrétabban,
minden epoch elején
eldobjuk az első $d$ tokent,
ahol $d\in [0,n)$ egyenletesen és véletlenszerűen van mintavételezve.
A sorozat többi részét
$m=\lfloor (T-d)/n \rfloor$ részsorozatra osztjuk fel.
Jelöljük $\mathbf x_t = [x_t, \ldots, x_{t+n-1}]$-gyel az $x_t$ tokennél a $t$ időlépéstől induló $n$ hosszúságú részsorozatot.
Az így kapott $m$ felosztott részsorozat:
$\mathbf x_d, \mathbf x_{d+n}, \ldots, \mathbf x_{d+n(m-1)}.$
Minden részsorozat bemeneti sorozatként kerül be a nyelvmodellbe.


A nyelvmodellezésnél
a cél a következő token megjóslása az eddig látott tokenek alapján; ezért a célok (címkék) az eredeti sorozat, egy tokennel eltolva.
A bármely $\mathbf x_t$ bemeneti sorozathoz tartozó célsorozat
$\mathbf x_{t+1}$ $n$ hosszal.

![Öt pár bemeneti sorozat és célsorozat megszerzése felosztott 5 hosszúságú részsorozatokból.](../img/lang-model-data.svg)
:label:`fig_lang_model_data`

A :numref:`fig_lang_model_data` egy példát mutat öt pár bemeneti sorozat és célsorozat megszerzésére $n=5$ és $d=2$ esetén.

```{.python .input  n=5}
%%tab all
@d2l.add_to_class(d2l.TimeMachine)  #@save
def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000):
    super(d2l.TimeMachine, self).__init__()
    self.save_hyperparameters()
    corpus, self.vocab = self.build(self._download())
    array = d2l.tensor([corpus[i:i+num_steps+1] 
                        for i in range(len(corpus)-num_steps)])
    self.X, self.Y = array[:,:-1], array[:,1:]
```

A nyelvmodellek tanításához
véletlenszerűen fogunk mintavételezni
bemeneti sorozatok és célsorozatok párjait mini-batch-ekben.
A következő adatbetöltő minden alkalommal véletlenszerűen generál egy mini-batch-et az adathalmazból.
A `batch_size` argumentum meghatározza az egyes mini-batch-ben lévő részsorozat-példányok számát,
a `num_steps` pedig a részsorozat hosszát tokenekben.

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(d2l.TimeMachine)  #@save
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(
        self.num_train, self.num_train + self.num_val)
    return self.get_tensorloader([self.X, self.Y], train, idx)
```

Ahogy a következőkben láthatjuk,
egy célsorozatokból álló mini-batch megszerezhető
a bemeneti sorozatok egy tokennel való eltolásával.

```{.python .input  n=7}
%%tab all
data = d2l.TimeMachine(batch_size=2, num_steps=10)
for X, Y in data.train_dataloader():
    print('X:', X, '\nY:', Y)
    break
```

## Összefoglalás és megbeszélés

A nyelvmodellek egy szöveges sorozat együttes valószínűségét becslik meg. Hosszú sorozatoknál az $n$-gramok kényelmes modellt biztosítanak a függőség csonkításával. Azonban sok struktúra van, de nincs elég gyakoriság a ritka szókombinációk hatékony kezeléséhez Laplace-simítással. Ezért a következő szakaszokban a neurális nyelvmodellezésre fogunk összpontosítani.
A nyelvmodellek tanításához véletlenszerűen mintavételezhetünk bemeneti sorozatok és célsorozatok párjait mini-batch-ekben. A tanítás után perplexitással mérjük a nyelvmodell minőségét.

A nyelvmodellek skálázhatók az adatméret, a modellméret és a tanítási számítás mennyiségének növelésével. A nagy nyelvmodellek elvégezhetik a kívánt feladatokat azzal, hogy a bemeneti szöveges utasítások alapján kimeneti szöveget jósolnak. Ahogy a :numref:`sec_large-pretraining-transformers` fejezetben is tárgyaljuk,
jelenleg
a nagy nyelvmodellek képezik az élvonalbeli rendszerek alapját a változatos feladatokon.


## Feladatok

1. Tegyük fel, hogy a tanítási adathalmazban 100 000 szó van. Mennyi szógyakoriságot és többszavas szomszédos gyakoriságot kell egy négygramnak tárolnia?
1. Hogyan modelleznél egy párbeszédet?
1. Milyen más módszereket tudsz elképzelni a hosszú sorozatadatok olvasásához?
1. Gondolj arra a módszerünkre, amellyel minden epoch elején eldobunk néhány tokent véletlenszerűen.
    1. Valóban egyenletes eloszlást eredményez-e a dokumentum sorozatai felett?
    1. Mit kellene tenned, hogy még egyenletesebbé tedd?
1. Ha azt szeretnénk, hogy egy sorozatpélda teljes mondat legyen, milyen problémát okoz ez a mini-batch mintavételezésénél? Hogyan javíthatjuk ki?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1049)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18012)
:end_tab:
