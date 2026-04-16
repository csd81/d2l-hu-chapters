```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Valószínűség és statisztika
:label:`sec_prob`

A gépi tanulás lényegében a bizonytalanságról szól.
A felügyelt tanulásban valamit ismeretlent (a *célváltozót*)
szeretnénk megjósolni valamiből, ami ismert (a *jellemzőkből*).
A célunktól függően megpróbálhatjuk megjósolni
a célváltozó legvalószínűbb értékét.
Vagy megjósolhatjuk azt az értéket,
amelynek a céltól való várható távolsága a legkisebb.
Néha nem csupán egy adott értéket szeretnénk megjósolni,
hanem *bizonytalanságunkat is számszerűsíteni*.
Például egy pácienst leíró jellemzők alapján
tudni szeretnénk, *milyen valószínűséggel*
kaphat szívrohamot a következő évben.
A felügyelet nélküli tanulásban
szintén foglalkozunk a bizonytalansággal.
Ahhoz, hogy megállapítsuk, egy méréssorozat rendhagyó-e,
hasznos tudni, mekkora valószínűséggel figyelhetők meg
ilyen értékek az adott populációban.
Továbbá a megerősítéses tanulásban
olyan ügynököket szeretnénk fejleszteni,
amelyek különféle környezetekben intelligensen cselekszenek.
Ehhez arra kell következtetni,
hogyan várható a környezet megváltozása,
és milyen jutalmakra lehet számítani
az egyes lehetséges cselekvések hatására.

A *valószínűség* az a matematikai terület,
amely a bizonytalanság alatti következtetéssel foglalkozik.
Egy folyamat valószínűségi modellje alapján
következtethetünk különféle eseményekre vonatkozó likelihood értékekre.
A valószínűségek használata az ismételhető eseményeket
(mint az érmefeldob­ások) leíró frekvenciák jellemzésére
meglehetősen egyértelmű.
Valóban, a *frequentista* tudósok ragaszkodnak
a valószínűség olyan értelmezéséhez,
amely *kizárólag* az ilyen ismételhető eseményekre vonatkozik.
Ezzel szemben a *bayesiánus* tudósok
a valószínűség fogalmát tágabban alkalmazzák
a bizonytalanság alatti következtetés formalizálásához.
A bayesiánus valószínűséget két sajátos vonás jellemzi:
(i) a nem ismételhető eseményekhez meggyőződési fokot rendel,
pl. mi a *valószínűsége* egy gát összeomlásának?;
és (ii) szubjektivitás. Míg a bayesiánus
valószínűség egyértelmű szabályokat ad arra,
hogyan kell valakinek frissítenie meggyőződését
új bizonyítékok fényében,
lehetővé teszi, hogy különböző személyek
különböző *prior* meggyőződésekkel induljanak.
A *statisztika* segít visszafelé következtetni:
adatgyűjtéssel és -rendszerezéssel kezdünk,
majd visszakövetkeztetünk arra,
milyen következtetéseket vonhatunk le
az adatokat generáló folyamatról.
Valahányszor egy adatkészletet elemzünk,
és olyan mintákat keresünk,
amelyek remélhetőleg egy tágabb populációt jellemeznek,
statisztikai gondolkodást alkalmazunk.
Számos tanfolyamot, szakot, disszertációt, karriert, tanszéket,
vállalatot és intézményt szenteltek
a valószínűség-elmélet és a statisztika tanulmányozásának.
Bár ez a szakasz csupán a felszínt karcolja,
megadjuk a modellek megalkotásához szükséges alapokat.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.numpy.random import multinomial
import random
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import random
import torch
from torch.distributions.multinomial import Multinomial
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import random
import tensorflow as tf
from tensorflow_probability import distributions as tfd
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import random
import jax
from jax import numpy as jnp
import numpy as np
```

## Egy egyszerű példa: érmefeldob­ás

Képzeljük el, hogy egy érmét szeretnénk feldobni,
és számszerűsíteni szeretnénk,
mekkora valószínűséggel látunk fejet (vs. írást).
Ha az érme *igazságos*,
akkor mindkét kimenetel
(fej és írás)
egyforma valószínűségű.
Ha ráadásul $n$-szer tervezzük feldobni az érmét,
akkor a fejek *várható* aránya
pontosan egyezni fog
az írások *várható* arányával.
Ennek egyik szemléletes magyarázata a szimmetria:
minden lehetséges kimenetelre,
ahol $n_\textrm{h}$ fej és $n_\textrm{t} = (n - n_\textrm{h})$ írás van,
létezik egy egyformán valószínű kimenetel
$n_\textrm{t}$ fejjel és $n_\textrm{h}$ írással.
Megjegyezzük, hogy ez csak akkor lehetséges,
ha átlagosan $1/2$ arányban fejre
és $1/2$ arányban írásra számíthatunk.
Természetesen ha ezt a kísérletet
sokszor elvégzed $n=1000000$ dobással,
előfordulhat, hogy soha nem látsz olyan próbát,
ahol $n_\textrm{h} = n_\textrm{t}$ pontosan.


Formálisan az $1/2$ értéket *valószínűségnek* nevezzük,
és az adott dobás fejre esésének bizonyosságát fejezi ki.
A valószínűségek $0$ és $1$ közötti értékeket rendelnek
az érdeklő­dési körbe eső kimenetelekhez, amelyeket *eseményeknek* nevezünk.
Az érdeklő esemény itt a $\textrm{heads}$,
és a megfelelő valószínűséget $P(\textrm{heads})$-szel jelöljük.
Az $1$ valószínűség abszolút bizonyosságot jelent
(képzeljük el egy trükkös érmét, amelynek mindkét oldala fej),
a $0$ valószínűség pedig lehetetlenséget jelent
(pl. ha mindkét oldal írás).
Az $n_\textrm{h}/n$ és $n_\textrm{t}/n$ frekvenciák nem valószínűségek,
hanem *statisztikák*.
A valószínűségek *elméleti* mennyiségek,
amelyek az adatgenerálási folyamat hátterében állnak.
Az $1/2$ valószínűség magának az érmének egy tulajdonsága.
Ezzel szemben a statisztikák *empirikus* mennyiségek,
amelyek a megfigyelt adatok függvényeiként számíthatók.
Valószínűségi és statisztikai mennyiségek iránti érdeklő­désünk
elválaszthatatlanul összefonódik.
Gyakran speciális statisztikákat tervezünk, amelyeket *becslőknek* nevezünk,
és amelyek egy adatkészlet alapján a modellparaméterek
(például valószínűségek) *becsléseit* adják.
Amikor ezek a becslők kielégítenek
egy *konzisztencia* nevű szép tulajdonságot,
a becslések konvergálnak a megfelelő valószínűséghez.
Ezek a levezetett valószínűségek viszont
tájékoztatnak ugyanolyan populációból
a jövőben tapasztalható adatok valószínű statisztikai tulajdonságairól.

Tegyük fel, hogy egy valódi érmére bukkanunk,
amelynek nem ismerjük a valódi $P(\textrm{heads})$ értékét.
Ennek statisztikai módszerekkel való vizsgálatához
(i) adatokat kell gyűjtenünk;
és (ii) becslőt kell terveznünk.
Az adatgyűjtés itt könnyű;
az érmét sokszor feldobhatjuk,
és rögzíthetjük az összes kimenetelt.
Formálisan az egyes realizációk kihúzását
valamilyen mögöttes véletlen folyamatból
*mintavételnek* nevezzük.
Ahogy sejthető,
az egyik természetes becslő
a megfigyelt *fejek* száma
és az összes dobás arányának hányadosa.

Tegyük fel most, hogy az érme valóban igazságos,
azaz $P(\textrm{heads}) = 0.5$.
Egy igazságos érme feldobásainak szimulálásához
bármilyen véletlenszám-generátort használhatunk.
Egy $0.5$ valószínűségű eseményből való mintavételre
több egyszerű módszer is létezik.
Például a Python `random.random` függvénye
$[0,1]$ intervallumból ad vissza számokat,
ahol bármely $[a, b] \subset [0,1]$ részintervallumba
esés valószínűsége $b-a$.
Így `0`-t és `1`-et egyenként `0.5` valószínűséggel kapunk,
ha megvizsgáljuk, hogy a visszaadott lebegőpontos szám nagyobb-e `0.5`-nél:

```{.python .input}
%%tab all
num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print("heads, tails: ", [heads, tails])
```

Általánosabban, bármely véges számú lehetséges kimenetellel
rendelkező változóból
(mint egy érmefeldobás vagy kockadobás)
több húzást szimulálhatunk a multinomiális függvény hívásával,
ahol az első argumentum a húzások száma,
a második pedig az egyes lehetséges kimenetelekhez
tartozó valószínűségek listája.
Egy igazságos érme tíz feldobásának szimulálásához
a `[0.5, 0.5]` valószínűségvektort adjuk meg,
ahol a 0-s index a fejet,
az 1-es index az írást jelöli.
A függvény egy olyan vektort ad vissza,
amelynek hossza megegyezik a lehetséges kimenetelekkel
(itt 2),
ahol az első komponens megadja a fejek számát,
a második komponens pedig az írások számát.

```{.python .input}
%%tab mxnet
fair_probs = [0.5, 0.5]
multinomial(100, fair_probs)
```

```{.python .input}
%%tab pytorch
fair_probs = torch.tensor([0.5, 0.5])
Multinomial(100, fair_probs).sample()
```

```{.python .input}
%%tab tensorflow
fair_probs = tf.ones(2) / 2
tfd.Multinomial(100, fair_probs).sample()
```

```{.python .input}
%%tab jax
fair_probs = [0.5, 0.5]
# a jax.random-ban nincs multinomiális eloszlás implementálva
np.random.multinomial(100, fair_probs)
```

Valahányszor futtatjuk ezt a mintavételezési folyamatot,
új véletlen értéket kapunk,
amely eltérhet az előző kimeneteltől.
Ha elosztjuk a dobások számával,
megkapjuk az egyes kimeneteleknek
az adatainkban mért *frekvenciáját*.
Vegyük észre, hogy ezek a frekvenciák
– csakúgy, mint az általuk becsülni kívánt valószínűségek –
összege $1$.

```{.python .input}
%%tab mxnet
multinomial(100, fair_probs) / 100
```

```{.python .input}
%%tab pytorch
Multinomial(100, fair_probs).sample() / 100
```

```{.python .input}
%%tab tensorflow
tfd.Multinomial(100, fair_probs).sample() / 100
```

```{.python .input}
%%tab jax
np.random.multinomial(100, fair_probs) / 100
```

Bár a szimulált érménk igazságos
(mi magunk állítottuk be a `[0.5, 0.5]` valószínűségeket),
a fejek és írások száma nem feltétlenül egyezik meg.
Ez azért van, mert csak viszonylag kevés mintát húztunk.
Ha nem mi valósítottuk volna meg a szimulációt,
hanem csupán a kimenetet láttuk volna,
honnan tudnánk, hogy az érme enyhén igazságtalan-e,
vagy a $1/2$-től való esetleges eltérés
csupán a kis mintaméret következménye?
Nézzük meg, mi történik, ha 10 000 dobást szimulálunk.

```{.python .input}
%%tab mxnet
counts = multinomial(10000, fair_probs).astype(np.float32)
counts / 10000
```

```{.python .input}
%%tab pytorch
counts = Multinomial(10000, fair_probs).sample()
counts / 10000
```

```{.python .input}
%%tab tensorflow
counts = tfd.Multinomial(10000, fair_probs).sample()
counts / 10000
```

```{.python .input}
%%tab jax
counts = np.random.multinomial(10000, fair_probs).astype(np.float32)
counts / 10000
```

Általában ismételt események átlagai esetén
(mint az érmefeldobások)
az ismétlések számának növekedésével
becslésünk garantáltan konvergál
a valódi mögöttes valószínűségekhez.
E jelenség matematikai megfogalmazása
a *nagy számok törvénye*,
a *centrális határeloszlás tétel* pedig
azt mondja ki, hogy sok esetben
a mintaméret $n$ növekedésével
ezek a hibák $(1/\sqrt{n})$ arányban csökkennek.
Szerezzünk intuitív képet arról,
hogyan alakul a becslésünk,
miközben a dobások számát 1-ről 10 000-re növeljük.

```{.python .input}
%%tab pytorch
counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
estimates = estimates.numpy()

d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
%%tab mxnet
counts = multinomial(1, fair_probs, size=10000)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)
```

```{.python .input}
%%tab tensorflow
counts = tfd.Multinomial(1, fair_probs).sample(10000)
cum_counts = tf.cumsum(counts, axis=0)
estimates = cum_counts / tf.reduce_sum(cum_counts, axis=1, keepdims=True)
estimates = estimates.numpy()
```

```{.python .input}
%%tab jax
counts = np.random.multinomial(1, fair_probs, size=10000).astype(np.float32)
cum_counts = counts.cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)
```

```{.python .input}
%%tab mxnet, tensorflow, jax
d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

Minden folytonos görbe az érme egyik lehetséges értékének felel meg,
és megmutatja, hogy az adott kísérleti csoportok után
mekkora becsült valószínűséggel esik az érme arra az értékre.
A szaggatott fekete vonal a valódi mögöttes valószínűséget jelöli.
Ahogy több kísérlet elvégzésével több adathoz jutunk,
a görbék közelítenek a valódi valószínűséghez.
Talán már körvonalazódnak azok a mélyebb kérdések,
amelyek a statisztikusokat foglalkoztatják:
Milyen gyorsan zajlik ez a konvergencia?
Ha ugyanabból a gyárból már számos érmét teszteltünk volna,
hogyan lehetne ezt az információt beépíteni?

## Formálisabb tárgyalás

Már elég messzire jutottunk: felállítottunk
egy valószínűségi modellt,
szintetikus adatokat generáltunk,
futtattunk egy statisztikai becslőt,
empirikusan megvizsgáltuk a konvergenciát,
és hibamértékeket számoltunk (ellenőriztük az eltérést).
Ahhoz azonban, hogy továbblépjünk,
pontosabbnak kell lennünk.


A véletlennel foglalkozva a lehetséges kimeneteleket
$\mathcal{S}$-sel jelöljük,
és *mintaterének* vagy *kimeneti térnek* nevezzük.
Minden elem egy különálló lehetséges *kimenetel*.
Egyetlen érme feldobásakor
$\mathcal{S} = \{\textrm{heads}, \textrm{tails}\}$.
Egyetlen kockadobásnál $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$.
Két érme feldobásakor a lehetséges kimenetelők:
$\{(\textrm{heads}, \textrm{heads}), (\textrm{heads}, \textrm{tails}), (\textrm{tails}, \textrm{heads}),  (\textrm{tails}, \textrm{tails})\}$.
Az *események* a mintatér részhalmazai.
Például az „az első érmefeldobás fejet ad" esemény
a $\{(\textrm{heads}, \textrm{heads}), (\textrm{heads}, \textrm{tails})\}$ halmaznak felel meg.
Ha egy véletlen kísérlet $z$ kimenetele kielégíti
a $z \in \mathcal{A}$ feltételt, akkor $\mathcal{A}$ esemény bekövetkezett.
Egyetlen kockadobásnál definiálhatjuk az
„$5$-öt látunk" ($\mathcal{A} = \{5\}$)
és a „páratlan számot látunk" ($\mathcal{B} = \{1, 3, 5\}$) eseményeket.
Ha a kocka $5$-öt mutat,
azt mondjuk, hogy mind $\mathcal{A}$, mind $\mathcal{B}$ bekövetkezett.
Ha viszont $z = 3$,
akkor $\mathcal{A}$ nem következett be,
de $\mathcal{B}$ igen.


A *valószínűségi* függvény az eseményeket
valós értékekre képezi le: ${P: \mathcal{A} \subseteq \mathcal{S} \rightarrow [0,1]}$.
Az adott $\mathcal{S}$ mintaterű $\mathcal{A}$ esemény
$P(\mathcal{A})$-val jelölt valószínűsége
a következő tulajdonságokkal rendelkezik:

* Bármely $\mathcal{A}$ esemény valószínűsége nemnegatív valós szám, azaz $P(\mathcal{A}) \geq 0$;
* A teljes mintatér valószínűsége $1$, azaz $P(\mathcal{S}) = 1$;
* Bármely megszámlálható $\mathcal{A}_1, \mathcal{A}_2, \ldots$ eseménysorozat esetén, amelyek *kizárják egymást* (azaz $\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ minden $i \neq j$ esetén), annak valószínűsége, hogy valamelyikük bekövetkezik, egyenlő az egyedi valószínűségeik összegével, azaz $P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$.

A valószínűség-elméletnek ezeket az axiómáit
:citet:`Kolmogorov.1933` javasolta,
és alkalmazásukkal számos fontos következmény vezethető le gyorsan.
Például azonnal következik,
hogy bármely $\mathcal{A}$ esemény
*vagy* komplementerének $\mathcal{A}'$-nek bekövetkezési valószínűsége 1
(mivel $\mathcal{A} \cup \mathcal{A}' = \mathcal{S}$).
Bizonyítható az is, hogy $P(\emptyset) = 0$,
mivel $1 = P(\mathcal{S} \cup \mathcal{S}') = P(\mathcal{S} \cup \emptyset) = P(\mathcal{S}) + P(\emptyset) = 1 + P(\emptyset)$.
Következésképpen bármely $\mathcal{A}$ esemény
*és* komplementerének $\mathcal{A}'$-nek egyidejű bekövetkezési valószínűsége
$P(\mathcal{A} \cap \mathcal{A}') = 0$.
Informálisan: a lehetetlen eseményeknek
nulla a bekövetkezési valószínűsége.



## Véletlen változók

Amikor olyan eseményekről beszéltünk,
mint a kockadobás páratlan kimenetele
vagy az első érmefeldobás fejre esése,
a *véletlen változó* fogalmát idéztük.
Formálisan a véletlen változók leképezések
a mögöttes mintaterről
(esetleg sok) értékek halmazára.
Felmerülhet a kérdés, miben különbözik a véletlen változó a mintaterétől,
hiszen mindkettő kimenetelekből áll.
Fontos, hogy a véletlen változók
sokkal durvábbak lehetnek a nyers mintaternél.
Definiálhatunk például egy „0,5-nél nagyobb" bináris véletlen változót
akkor is, ha a mögöttes mintatér végtelen,
pl. a $0$ és $1$ közötti szakasz pontjai.
Emellett több véletlen változó
osztozhat ugyanazon a mögöttes mintaterén.
Például a „megszólal-e az otthoni riasztóm"
és a „betörtek-e a házamba"
mindkét bináris véletlen változó
ugyanazon mögöttes mintaterű.
Következésképpen az egyik véletlen változó értékének ismerete
árulhat valamit a másik véletlen változó várható értékéről.
Ha tudjuk, hogy a riasztó megszólalt,
gyaníthatjuk, hogy valószínűleg betörtek a házba.


A véletlen változó által felvett minden érték
a mögöttes mintatér egy részhalmazának felel meg.
Így az $X=v$-vel jelölt esemény,
ahol az $X$ véletlen változó a $v$ értéket veszi fel,
egy *esemény*, és $P(X=v)$ jelöli annak valószínűségét.
Ez a jelölés néha nehézkes,
és ha a kontextus egyértelmű, megengedjük magunknak jelölésrendszerünk lazítását.
Például $P(X)$-szel utalhatunk
$X$ *eloszlására*, azaz
arra a függvényre, amely megadja,
mekkora valószínűséggel vesz fel $X$ bármely adott értéket.
Máskor olyan kifejezéseket írunk,
mint $P(X,Y) = P(X) P(Y)$,
rövidítésként egy olyan állításra,
amely igaz az $X$ és $Y$ véletlen változók
által felvehető összes értékre, azaz:
minden $i,j$ esetén $P(X=i \textrm{ és } Y=j) = P(X=i)P(Y=j)$.
Máskor $P(v)$-vel jelölünk,
ha a véletlen változó egyértelmű a kontextusból.
Mivel a valószínűség-elméletben egy esemény a mintatér kimeneteleinek egy halmaza,
megadhatunk egy értéktartományt, amelybe a véletlen változó eshet.
Például $P(1 \leq X \leq 3)$ a $\{1 \leq X \leq 3\}$ esemény valószínűségét jelöli.


Vegyük észre, hogy finom különbség van
a *diszkrét* véletlen változók –
mint az érmefeldobások vagy kockadobások –
és a *folytonos* véletlen változók között,
mint például a populációból véletlenszerűen kiválasztott
személy súlya és magassága.
Ez utóbbi esetben ritkán törődünk
valaki pontos magasságával.
Ha elég precíz méréseket végzünk,
azt találjuk, hogy a Földön
nincs két egyforma magasságú ember.
Sőt, elég finom mérőeszközzel
reggelente és lefekvéskor is más magasságunk van.
Kevés értelme van azt kérdezni,
hogy pontosan mekkora a valószínűsége annak,
hogy valaki pontosan 1,801392782910287192 méter magas.
Helyette általában arra vagyunk kíváncsiak,
hogy valakinek a magassága egy adott intervallumba esik-e,
mondjuk 1,79 és 1,81 méter közé.
Ilyen esetekben valószínűségi *sűrűségekkel* dolgozunk.
Pontosan 1,80 méteres magasságnak nincs valószínűsége, de nem nulla a sűrűsége.
Egy intervallumhoz rendelt valószínűség meghatározásához
a sűrűséget kell *integrálni*
az adott intervallumon.

## Több véletlen változó

Észrevehettük, hogy az előző szakaszon
sem tudtunk végigmenni anélkül,
hogy ne tennénk állításokat
több véletlen változó közötti kölcsönhatásokról
(gondoljunk a $P(X,Y) = P(X) P(Y)$ összefüggésre).
A gépi tanulás nagy része éppen ezekkel a kapcsolatokkal foglalkozik.
A mintatér itt az érdeklő populáció lenne,
például egy vállalkozással tranzakciókat lebonyolító ügyfelek,
az interneten lévő fényképek
vagy a biológusok által ismert fehérjék.
Minden véletlen változó
egy-egy jellemző (ismeretlen) értékét képviselné.
Valahányszor egy egyedet mintavételezünk a populációból,
az egyes véletlen változók egy-egy realizációját figyeljük meg.
Mivel a véletlen változók által felvett értékek
a mintatér olyan részhalmazainak felelnek meg,
amelyek átfedhetnek, részben fedhetik egymást,
vagy teljesen diszjunktak lehetnek,
az egyik véletlen változó értékének ismerete
befolyásolhatja a másik véletlen változó
valószínű értékeiről alkotott képünket.
Ha egy beteg belép a kórházba,
és azt figyeljük meg, hogy légzési nehézségei vannak
és elvesztette a szaglását,
valószínűbbnek tartjuk, hogy COVID-19-ben szenved,
mint ha nem lenne légzési gondja
és teljesen normális szaglása volna.


Több véletlen változóval dolgozva
konstruálhatunk eseményeket,
amelyek a változók által közösen felvehető
értékkombinációk mindegyikének felelnek meg.
Az egyes kombinációkhoz
(pl. $A=a$ és $B=b$) valószínűséget rendelő
valószínűségi függvényt *együttes valószínűségi* függvénynek nevezzük,
és egyszerűen a mintatér megfelelő részhalmazainak
metszetéhez rendelt valószínűséget adja vissza.
Az $A$ és $B$ véletlen változók
rendre $a$ és $b$ értékét felvevő eseményhez rendelt
*együttes valószínűséget* $P(A = a, B = b)$-vel jelöljük,
ahol a vessző „és"-t jelent.
Vegyük észre, hogy bármely $a$ és $b$ értékre fennáll:

$$P(A=a, B=b) \leq P(A=a) \textrm{ és } P(A=a, B=b) \leq P(B = b),$$

hiszen ahhoz, hogy $A=a$ és $B=b$ egyaránt teljesüljön,
$A=a$-nak *és* $B=b$-nek is be kell következnie.
Érdekesség, hogy az együttes valószínűség
mindent elmond e véletlen változókról
valószínűségi szempontból,
és belőle számos más hasznos mennyiség levezethető,
köztük az egyedi $P(A)$ és $P(B)$ eloszlások visszanyerése.
$P(A=a)$ visszanyeréséhez csupán összegezzük
$P(A=a, B=v)$-t a $B$ véletlen változó
összes $v$ értékére:
$P(A=a) = \sum_v P(A=a, B=v)$.


A $\frac{P(A=a, B=b)}{P(A=a)} \leq 1$ arány
rendkívül fontosnak bizonyul.
Ezt *feltételes valószínűségnek* nevezzük,
és a „$\mid$" szimbólummal jelöljük:

$$P(B=b \mid A=a) = P(A=a,B=b)/P(A=a).$$

Megmutatja a $B=b$ eseményhez rendelt új valószínűséget,
ha feltételezzük, hogy $A=a$ bekövetkezett.
Ezt a feltételes valószínűséget úgy is felfoghatjuk,
hogy figyelmünket csak az $A=a$-hoz kapcsolódó
mintaterű részhalmazra korlátozzuk,
majd renormalizálunk, hogy az összes valószínűség összege 1 legyen.
A feltételes valószínűségek valójában
egyszerűen közönséges valószínűségek,
és így tiszteletben tartják az összes axiómát,
mindaddig, amíg minden tagot ugyanarra az eseményre kondicionálunk,
és figyelmünket ugyanarra a mintaterére korlátozzuk.
Például diszjunkt $\mathcal{B}$ és $\mathcal{B}'$ eseményekre fennáll:
$P(\mathcal{B} \cup \mathcal{B}' \mid A = a) = P(\mathcal{B} \mid A = a) + P(\mathcal{B}' \mid A = a)$.


A feltételes valószínűségek definíciójából levezethető
a *Bayes-tétel* nevű híres eredmény.
A konstrukció alapján $P(A, B) = P(B\mid A) P(A)$
és $P(A, B) = P(A\mid B) P(B)$.
A két egyenlet összevonásával kapjuk:
$P(B\mid A) P(A) = P(A\mid B) P(B)$, tehát

$$P(A \mid B) = \frac{P(B\mid A) P(A)}{P(B)}.$$






Ez az egyszerű egyenlet mély következményekkel jár,
mivel lehetővé teszi a kondicionálás sorrendjének megfordítását.
Ha tudjuk, hogyan becsüljük $P(B\mid A)$-t, $P(A)$-t és $P(B)$-t,
akkor megbecsülhetjük $P(A\mid B)$-t is.
Sokszor az egyik tagot könnyebb közvetlenül megbecsülni,
a másikat nem, és ilyenkor a Bayes-tétel segítségünkre siet.
Például ha ismerjük egy adott betegség tüneteinek előfordulási arányát,
és a betegség, illetve a tünetek általános előfordulási arányát,
megállapíthatjuk, milyen valószínűséggel rendelkezik valaki a betegséggel
a tünetei alapján.
Egyes esetekben nem feltétlenül férünk hozzá közvetlenül $P(B)$-hez,
például a tünetek előfordulási arányához.
Ilyenkor jól jön a Bayes-tétel egy egyszerűsített változata:

$$P(A \mid B) \propto P(B \mid A) P(A).$$

Mivel tudjuk, hogy $P(A \mid B)$-t 1-re kell normalizálni, azaz $\sum_a P(A=a \mid B) = 1$,
segítségével kiszámíthatjuk:

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{\sum_a P(B \mid A=a) P(A = a)}.$$

A bayesiánus statisztikában a megfigyelőt úgy képzeljük el,
mint aki bizonyos (szubjektív) prior meggyőződésekkel rendelkezik
a rendelkezésre álló hipotézisek valószínűségéről,
amelyeket a *prior* $P(H)$ kódol,
és egy *likelihood-függvénnyel*, amely megadja,
mekkora valószínűséggel figyeljük meg
az összegyűjtött bizonyítékok bármely értékét
a $P(E \mid H)$ osztályba tartozó hipotézisek mindegyikére.
A Bayes-tételt úgy értelmezzük,
hogy megmondja, hogyan frissítsük az eredeti *prior* $P(H)$-t
az elérhető $E$ bizonyíték fényében,
hogy *posterior* meggyőződéseket kapjunk:
$P(H \mid E) = \frac{P(E \mid H) P(H)}{P(E)}$.
Informálisan: „a posterior egyenlő a priorral szorozva a likelihooddal, osztva a bizonyítékkal".
Mivel a $P(E)$ bizonyíték minden hipotézisre azonos,
elegendő csupán a hipotézisek felett normalizálni.

Vegyük észre, hogy $\sum_a P(A=a \mid B) = 1$ lehetővé teszi a véletlen változók *marginalizálását* is. Azaz elhagyhatunk változókat egy együttes eloszlásból, mint $P(A, B)$. Végeredményben:

$$\sum_a P(B \mid A=a) P(A=a) = \sum_a P(B, A=a) = P(B).$$

A *függetlenség* egy másik alapvetően fontos fogalom,
amely számos statisztikai gondolat gerincét alkotja.
Röviden: két változó *független*,
ha az $A$ értékére való kondicionálás
semmilyen változást nem okoz
a $B$-hez tartozó valószínűségi eloszlásban és fordítva.
Formálisabban, az $A \perp B$-vel jelölt függetlenség
megköveteli, hogy $P(A \mid B) = P(A)$, következésképpen
$P(A,B) = P(A \mid B) P(B) = P(A) P(B)$.
A függetlenség sokszor megalapozott feltevés.
Ha például az $A$ véletlen változó
egy igazságos érme feldobásának kimenetelét képviseli,
a $B$ véletlen változó pedig egy másik érméét,
akkor annak ismerete, hogy $A$ fejet adott-e,
nem befolyásolhatja
annak valószínűségét, hogy $B$ fejet ad-e.


A függetlenség különösen hasznos,
ha teljesül az adataink egymást követő húzásai között
valamilyen mögöttes eloszlásból
(erős statisztikai következtetésekre ad lehetőséget),
vagy ha az adatainkban szereplő különböző változók között áll fenn,
lehetővé téve, hogy egyszerűbb modellekkel dolgozzunk,
amelyek kódolják ezt a függetlenségi struktúrát.
Másrészt a véletlen változók közötti függőségek becslése
sokszor maga a tanulás célja.
Azért becsüljük a betegség valószínűségét a tünetek alapján,
mert hisszük, hogy a betegségek és a tünetek *nem* függetlenek.


Mivel a feltételes valószínűségek valódi valószínűségek,
a függetlenség és a függőség fogalmai rájuk is alkalmazhatók.
Két véletlen változó, $A$ és $B$, *feltételesen független*
egy harmadik $C$ változó feltételezése esetén
akkor és csak akkor, ha $P(A, B \mid C) = P(A \mid C)P(B \mid C)$.
Érdekesség, hogy két változó általánosan független lehet,
de egy harmadikra való kondicionáláskor függővé válhat.
Ez gyakran olyankor fordul elő, amikor a két véletlen változó, $A$ és $B$,
egy harmadik $C$ változó okainak felelnek meg.
Például a csonttörések és a tüdőrák független lehet az általános populációban,
de ha a kórházban való tartózkodásra kondicionálunk,
azt tapasztalhatjuk, hogy a csonttörések negatívan korrelálnak a tüdőrákkal.
Ez azért van, mert a csonttörés *megmagyarázza*, hogy valaki miért van kórházban,
és így csökkenti annak valószínűségét,
hogy tüdőrák miatt hospitalizálták.


Ezzel szemben két egymástól függő véletlen változó
egy harmadikra való kondicionáláskor függetlenné válhat.
Ez általában akkor fordul elő,
amikor két egyébként összefüggéstelen eseménynek közös oka van.
A cipőméret és az olvasási szint erősen korrelál
az általános iskolás tanulók körében,
de ez a korreláció eltűnik, ha az életkorra kondicionálunk.



## Egy példa
:label:`subsec_probability_hiv_app`

Tegyük próbára tudásunkat.
Tegyük fel, hogy egy orvos HIV-tesztet végez egy betegen.
Ez a teszt meglehetősen pontos: csak 1% valószínűséggel hibás,
ha a beteg egészséges, de beteget jelez,
azaz az egészséges betegek 1%-ánál pozitív eredményt ad.
Ráadásul soha nem marad el a HIV kimutatása,
ha a beteg valóban fertőzött.
$D_1 \in \{0, 1\}$-gyel jelöljük a diagnózist
($0$ ha negatív, $1$ ha pozitív),
$H \in \{0, 1\}$-gyel pedig a HIV-státuszt.

| Conditional probability | $H=1$ | $H=0$ |
|:------------------------|------:|------:|
| $P(D_1 = 1 \mid H)$        |     1 |  0.01 |
| $P(D_1 = 0 \mid H)$        |     0 |  0.99 |

Vegyük észre, hogy az oszlopok összege mindig 1 (a soroké nem feltétlenül),
mivel feltételes valószínűségekről van szó.
Számítsuk ki annak valószínűségét, hogy a betegnek HIV-je van,
ha a teszt pozitív eredményt ad, azaz $P(H = 1 \mid D_1 = 1)$.
Intuitívan ez függ attól, mennyire elterjedt a betegség,
mivel ez befolyásolja a téves riasztások számát.
Tegyük fel, hogy a populáció viszonylag mentes a betegségtől, pl. $P(H=1) = 0.0015$.
A Bayes-tétel alkalmazásához marginalizálást kell végeznünk
a meghatározáshoz:

$$\begin{aligned}
P(D_1 = 1)
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1 \mid H=0) P(H=0) + P(D_1=1 \mid H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$

Ebből következik:

$$P(H = 1 \mid D_1 = 1) = \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} = 0.1306.$$

Más szóval, csupán 13,06% annak esélye,
hogy a betegnek valóban HIV-je van,
annak ellenére, hogy a teszt meglehetősen pontos.
Amint látjuk, a valószínűség ellentmondásos lehet.
Mit tegyen egy beteg, ha ilyen rémisztő hírt kap?
Valószínűleg arra kérné az orvost,
hogy végezzen el egy második tesztet a tisztázás érdekében.
A második teszt eltérő jellemzőkkel bír,
és nem olyan jó, mint az első.

| Conditional probability | $H=1$ | $H=0$ |
|:------------------------|------:|------:|
| $P(D_2 = 1 \mid H)$          |  0.98 |  0.03 |
| $P(D_2 = 0 \mid H)$          |  0.02 |  0.97 |

Sajnos a második teszt is pozitív eredményt ad.
Számítsuk ki a Bayes-tétel alkalmazásához szükséges valószínűségeket,
feltételezve a feltételes függetlenséget:

$$\begin{aligned}
P(D_1 = 1, D_2 = 1 \mid H = 0)
& = P(D_1 = 1 \mid H = 0) P(D_2 = 1 \mid H = 0)
=& 0.0003, \\
P(D_1 = 1, D_2 = 1 \mid H = 1)
& = P(D_1 = 1 \mid H = 1) P(D_2 = 1 \mid H = 1)
=& 0.98.
\end{aligned}
$$

Most már alkalmazhatjuk a marginalizálást
annak valószínűségének meghatározásához,
hogy mindkét teszt pozitív eredményt ad:

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1)\\
&= P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
&= P(D_1 = 1, D_2 = 1 \mid H = 0)P(H=0) + P(D_1 = 1, D_2 = 1 \mid H = 1)P(H=1)\\
&= 0.00176955.
\end{aligned}
$$

Végül annak valószínűsége, hogy a betegnek HIV-je van,
feltéve, hogy mindkét teszt pozitív, a következő:

$$P(H = 1 \mid D_1 = 1, D_2 = 1)
= \frac{P(D_1 = 1, D_2 = 1 \mid H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)}
= 0.8307.$$

Vagyis a második teszt révén lényegesen nagyobb bizonyossággal állíthatjuk,
hogy valami nincs rendben.
Annak ellenére, hogy a második teszt jóval kevésbé pontos, mint az első,
mégis jelentősen javította a becslésünket.
A két teszt feltételes egymástól való függetlenségének feltételezése
döntő fontosságú volt a pontosabb becslés elkészítéséhez.
Vegyük a szélső esetet, amikor ugyanazt a tesztet kétszer futtatjuk.
Ebben a helyzetben mindkét alkalommal ugyanolyan eredményt várnánk,
tehát ugyanazon teszt megismétlése nem hoz újabb információt.
Az éles szemű olvasó észrevehette, hogy a diagnózis úgy viselkedett,
mint egy szabad szemmel is látható osztályozó,
ahol a beteg egészségi állapotáról való döntési képességünk
növekszik, ahogy több jellemzőt (teszteredményt) kapunk.


## Várható értékek

A döntéshozatalnál sokszor nem elég
az egyes eseményekhez rendelt valószínűségeket külön vizsgálni;
hasznos aggregátumokká kell őket összegyúrni,
amelyek iránymutatást nyújtanak.
Például ha véletlen változók folytonos skaláris értékeket vesznek fel,
sokszor arra vagyunk kíváncsiak, milyen értéket várhatunk *átlagosan*.
Ezt a mennyiséget formálisan *várható értéknek* nevezzük.
Befektetések esetén az első érdekes mennyiség
a várható hozam lehet,
az összes lehetséges kimenetelt átlagolva
(és a megfelelő valószínűségekkel súlyozva).
Például tegyük fel, hogy 50% valószínűséggel
egy befektetés teljesen meghiúsulhat,
40% valószínűséggel 2$\times$-os hozamot,
10% valószínűséggel pedig 10$\times$-es hozamot adhat.
A várható hozam kiszámításához összeadjuk az összes hozamot,
mindegyiket megszorozva a bekövetkezési valószínűségével.
Ez $0.5 \cdot 0 + 0.4 \cdot 2 + 0.1 \cdot 10 = 1.8$ várható értéket ad.
Tehát a várható hozam 1.8$\times$.


Általánosan az $X$ véletlen változó *várható értékét* (vagy átlagát)
a következőképpen definiáljuk:

$$E[X] = E_{x \sim P}[x] = \sum_{x} x P(X = x).$$

Sűrűségek esetén $E[X] = \int x \;dp(x)$.
Néha $x$ valamilyen függvényének várható értéke érdekel.
Ezek kiszámíthatók:

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x) \textrm{ és } E_{x \sim P}[f(x)] = \int f(x) p(x) \;dx$$

diszkrét valószínűségek, illetve sűrűségek esetén.
Visszatérve a fenti befektetési példára,
$f$ a hozamhoz kapcsolódó *hasznosság* (boldogság) lehet.
A viselkedési közgazdászok régóta megfigyelték,
hogy az emberek a pénz elvesztéséhez nagyobb negatív hasznosságot rendelnek,
mint amennyi pozitív hasznosságot egy dollár megkeresése hoz az alapállapothoz képest.
Ráadásul a pénz értéke szublineáris.
100 ezer dolláros vagyon a semmihez képest
a különbséget jelentheti a lakbér fizetése,
a jó étkezés és a minőségi egészségügyi ellátás
és a hajléktalanság között.
Ezzel szemben a 200 ezer és a 100 ezer dolláros vagyon különbsége
kevésbé drámai.
Az ilyen megfontolások alapozzák meg azt a közkeletű mondást,
hogy „a pénz hasznosságfüggvénye logaritmikus".


Ha a teljes veszteséghez kapcsolódó hasznosság $-1$ lenne,
és az $1$-, $2$-, illetve $10$-szeres hozamhoz társuló hasznosságok
rendre $1$, $2$ és $4$,
akkor a befektetés várható boldogsága
$0.5 \cdot (-1) + 0.4 \cdot 2 + 0.1 \cdot 4 = 0.7$ lenne
(a hasznosság várható 30%-os csökkenése).
Ha valóban ez lenne a hasznossági függvényünk,
lehet, hogy jobban járnánk, ha a pénzt bankban tartanánk.

Pénzügyi döntéseknél szintén érdemes mérni,
mennyire *kockázatos* egy befektetés.
Ekkor nem csak a várható értékre vagyunk kíváncsiak,
hanem arra is, mennyire *változnak* a tényleges értékek
ehhez képest.
Fontos, hogy nem vehetjük egyszerűen
a tényleges és a várható értékek különbségének várható értékét.
Ennek az az oka, hogy egy különbség várható értéke
a várható értékek különbsége,
azaz $E[X - E[X]] = E[X] - E[E[X]] = 0$.
Helyette megnézhetjük e különbség
bármely nemnegatív függvényének várható értékét.
Egy véletlen változó *szórásnégyzete* a *négyzetes* különbségek
várható értékéből számítható:

$$\textrm{Var}[X] = E\left[(X - E[X])^2\right] = E[X^2] - E[X]^2.$$

Az egyenlőség $(X - E[X])^2 = X^2 - 2 X E[X] + E[X]^2$
kifejtéséből és minden tag várható értékének vételéből adódik.
A szórásnégyzet négyzetgyöke egy másik hasznos mennyiség,
amelyet *szórásnak* nevezünk.
Ez és a szórásnégyzet ugyanazt az információt hordozza
(mindkettő kiszámítható a másikból),
de a szórás azzal az előnyös tulajdonsággal rendelkezik,
hogy ugyanolyan mértékegységben fejezhető ki,
mint a véletlen változó által képviselt eredeti mennyiség.

Végül egy véletlen változó függvényének szórásnégyzete
analóg módon definiálható:

$$\textrm{Var}_{x \sim P}[f(x)] = E_{x \sim P}[f^2(x)] - E_{x \sim P}[f(x)]^2.$$

Visszatérve a befektetési példánkhoz,
most már kiszámíthatjuk a befektetés szórásnégyzetét.
Ez $0.5 \cdot 0 + 0.4 \cdot 2^2 + 0.1 \cdot 10^2 - 1.8^2 = 8.36$.
Minden szempontból kockázatos befektetésről van szó.
Megjegyezzük, hogy matematikai szokás szerint
az átlagot és a szórásnégyzetet
sokszor $\mu$-val, illetve $\sigma^2$-tel jelölik.
Ez különösen jellemző, ha Gauss-eloszlás paraméterezésére használjuk.

Ahogy a *skaláris* véletlen változók esetén
bevezettük a várható értéket és a szórásnégyzetet,
ugyanúgy megtehetjük ezt vektorértékűeknél is.
A várható értékek egyszerűek, mivel elemenként alkalmazhatók.
Például a $\boldsymbol{\mu} \stackrel{\textrm{def}}{=} E_{\mathbf{x} \sim P}[\mathbf{x}]$
koordinátái $\mu_i = E_{\mathbf{x} \sim P}[x_i]$.
A *kovariancák* bonyolultabbak.
Ezeket a véletlen változók és átlaguk különbségének
*külső szorzatából* vett várható értékkel definiáljuk:

$$\boldsymbol{\Sigma} \stackrel{\textrm{def}}{=} \textrm{Cov}_{\mathbf{x} \sim P}[\mathbf{x}] = E_{\mathbf{x} \sim P}\left[(\mathbf{x} - \boldsymbol{\mu}) (\mathbf{x} - \boldsymbol{\mu})^\top\right].$$

Ezt a $\boldsymbol{\Sigma}$ mátrixot kovariancamátrixnak nevezzük.
Hatásának szemléltetéséhez vegyünk
egy $\mathbf{x}$-szel azonos méretű $\mathbf{v}$ vektort.
Fennáll, hogy

$$\mathbf{v}^\top \boldsymbol{\Sigma} \mathbf{v} = E_{\mathbf{x} \sim P}\left[\mathbf{v}^\top(\mathbf{x} - \boldsymbol{\mu}) (\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{v}\right] = \textrm{Var}_{x \sim P}[\mathbf{v}^\top \mathbf{x}].$$

Így $\boldsymbol{\Sigma}$ lehetővé teszi a szórásnégyzet kiszámítását
$\mathbf{x}$ bármely lineáris függvényére
egyszerű mátrixszorzással.
Az átlón kívüli elemek megmutatják, mennyire korrelálnak a koordináták:
a 0 értéke korrelálatlanságot jelent,
míg a nagyobb pozitív érték
erősebb korrelációt jelöl.



## Megbeszélés

A gépi tanulásban rengeteg dologban lehetünk bizonytalanok!
Bizonytalanok lehetünk egy bemenethez tartozó címke értékében.
Bizonytalanok lehetünk egy paraméter becsült értékében.
Sőt, abban is bizonytalanok lehetünk, hogy a bevetés során érkező adatok
egyáltalán ugyanolyan eloszlásból származnak-e, mint a tanítási adatok.

*Aleatorikus bizonytalanságon* azt értjük,
amely a feladatban rejlik,
és valódi véletlenszerűségből adódik,
amelyet a megfigyelt változók nem fednek le.
*Episztemikus bizonytalanságon* a modell paramétereire vonatkozó bizonytalanságot értjük,
amelyet remélhetőleg több adat gyűjtésével csökkenthetünk.
Episztemikus bizonytalanságunk lehet
egy érme fejre esési valószínűségével kapcsolatban,
de még ha ismerjük is ezt a valószínűséget,
aleatorikus bizonytalanság marad
bármely jövőbeli dobás kimenetelével kapcsolatban.
Bármeddig figyeljük is, amint valaki egy igazságos érmét dob,
soha nem leszünk 50%-nál jobban vagy kevésbé biztosak abban,
hogy a következő dobás fejet ad.
Ezek a kifejezések a mechanikai modellezésből származnak
(lásd pl. :citet:`Der-Kiureghian.Ditlevsen.2009` a [bizonytalanságszámítás](https://en.wikipedia.org/wiki/Uncertainty_quantification) ezen aspektusának áttekintéséért).
Érdemes megjegyezni azonban, hogy e kifejezések némi fogalmi torzítást tartalmaznak.
Az *episztemikus* szó mindent érint, ami a *tudáshoz* kapcsolódik,
és így filozófiai értelemben minden bizonytalanság episztemikus.


Láttuk, hogy ismeretlen valószínűségi eloszlásból
vett mintaadatok olyan információt adhatnak,
amellyel becsülhetők az adatgeneráló eloszlás paraméterei.
Azonban ennek üteme meglehetősen lassú lehet.
Az érmefeldobásos példánkban (és sok másban)
a legjobb becslők $1/\sqrt{n}$ arányban konvergálnak,
ahol $n$ a mintaméret (pl. a dobások száma).
Ez azt jelenti, hogy 10-ről 1000 megfigyelésre növelve (általában könnyen teljesíthető feladat)
tízszeres bizonytalanságcsökkenést tapasztalunk,
míg a következő 1000 megfigyelés aránylag keveset segít,
mindössze 1,41-szeres csökkentést nyújtva.
Ez a gépi tanulás tartós jellemzője:
bár sokszor vannak könnyen elérhető javulások,
a további haladáshoz nagyon sok adatra,
és sokszor hatalmas számítási kapacitásra is szükség van.
A nagy méretű nyelvi modellek esetén ennek empirikus áttekintéséhez lásd: :citet:`Revels.Lubin.Papamarkou.2016`.

Finomítottuk a statisztikai modellezés nyelvét és eszközkészletét is.
Ennek során megismertük a feltételes valószínűségeket
és a statisztika egyik legfontosabb egyenletét – a Bayes-tételt.
Ez hatékony eszköz az adatok által közvetített információ szétválasztásához
egy $P(B \mid A)$ likelihood-tagon keresztül, amely azt méri,
mennyire egyeznek a $B$ megfigyelések az $A$ paraméterválasztással,
és egy $P(A)$ prior valószínűségen keresztül,
amely meghatározza, mennyire valószínű egy adott $A$ választás eleve.
Különösen láttuk, hogyan alkalmazható ez a szabály
diagnózisok valószínűségének meghatározására,
a teszt hatékonysága *és*
magának a betegségnek az előfordulási aránya (azaz a priorunk) alapján.

Végül bevezettük az első nem triviális kérdéseket
egy adott valószínűségi eloszlás hatásáról,
mégpedig a várható értékeket és a szórásnégyzeteket.
Bár egy valószínűségi eloszlásnál sokkal több mennyiség létezik
a lineáris és kvadratikus várható értékeken túl,
ez a kettő máris sokat elárul az eloszlás lehetséges viselkedéséről.
Például a [Csebisev-egyenlőtlenség](https://en.wikipedia.org/wiki/Chebyshev%27s_inequality)
kimondja, hogy $P(|X - \mu| \geq k \sigma) \leq 1/k^2$,
ahol $\mu$ a várható érték, $\sigma^2$ az eloszlás szórásnégyzete,
és $k > 1$ az általunk választott konfidenciaparaméter.
Ez azt mondja, hogy az eloszlásból vett húzások legalább 50% valószínűséggel
a várható érték körüli
$[-\sqrt{2} \sigma, \sqrt{2} \sigma]$ intervallumon belül esnek.




## Feladatok

1. Adj egy példát arra, ahol több adat megfigyelése a kimenetelre vonatkozó bizonytalanságot tetszőlegesen alacsony szintre csökkentheti.
1. Adj egy példát arra, ahol több adat megfigyelése a bizonytalanságot csak egy bizonyos pontig csökkenti, azon túl már nem. Magyarázd meg, miért van ez így, és hol várható ez a pont.
1. Empirikusan bemutattuk az átlaghoz való konvergenciát egy érme feldobásakor. Számítsd ki a $n$ minta húzása után a fej megjelenési valószínűségének becsléséhez tartozó szórásnégyzetet.
    1. Hogyan skálázódik a szórásnégyzet a megfigyelések számával?
    1. Használd a Csebisev-egyenlőtlenséget a várható értéktől való eltérés korlátának meghatározásához.
    1. Hogyan kapcsolódik ez a centrális határeloszlás tételhez?
1. Tegyük fel, hogy $m$ mintát húzunk $x_i$ értékekkel egy nulla várható értékű, egységnyi szórásnégyzetű eloszlásból. Számítsd ki a $z_m \stackrel{\textrm{def}}{=} m^{-1} \sum_{i=1}^m x_i$ átlagokat. Alkalmazhatjuk-e a Csebisev-egyenlőtlenséget minden $z_m$-re külön-külön? Miért nem?
1. Adott két $P(\mathcal{A})$ és $P(\mathcal{B})$ valószínűségű esemény; számítsd ki $P(\mathcal{A} \cup \mathcal{B})$ és $P(\mathcal{A} \cap \mathcal{B})$ felső és alsó korlátait. Tipp: ábrázold a helyzetet [Venn-diagram](https://en.wikipedia.org/wiki/Venn_diagram) segítségével.
1. Tegyük fel, hogy van egy véletlen változósorozatunk: $A$, $B$ és $C$, ahol $B$ csak $A$-tól függ, $C$ pedig csak $B$-től. Egyszerűsíthető-e az együttes valószínűség $P(A, B, C)$? Tipp: ez egy [Markov-lánc](https://en.wikipedia.org/wiki/Markov_chain).
1. A :numref:`subsec_probability_hiv_app`-ban tegyük fel, hogy a két teszt kimenetelei nem függetlenek egymástól. Különösen tegyük fel, hogy önmagában mindkét teszt hamis pozitív aránya 10%, hamis negatív aránya 1%. Azaz legyen $P(D =1 \mid H=0) = 0.1$ és $P(D = 0 \mid H=1) = 0.01$. Továbbá tegyük fel, hogy $H = 1$ (fertőzött) esetén a teszteredmények feltételesen függetlenek, azaz $P(D_1, D_2 \mid H=1) = P(D_1 \mid H=1) P(D_2 \mid H=1)$, de egészséges betegeknél az eredmények összefüggenek: $P(D_1 = D_2 = 1 \mid H=0) = 0.02$.
    1. Írd fel a $D_1$ és $D_2$ együttes valószínűségtáblázatát $H=0$ esetén az eddig ismert információk alapján.
    1. Vezessük le annak valószínűségét, hogy a beteg fertőzött ($H=1$), miután az egyik teszt pozitív eredményt adott. Feltételezheted ugyanazt az alapvalószínűséget: $P(H=1) = 0.0015$.
    1. Vezessük le annak valószínűségét, hogy a beteg fertőzött ($H=1$), miután mindkét teszt pozitív eredményt adott.
1. Tegyük fel, hogy befektetési bank vagyonkezelője vagy, és $s_i$ részvények közül választhatsz, amelyekbe befektethetsz. A portfóliónak $\alpha_i$ súlyokkal kell összeadódnia $1$-re minden részvényre. A részvényeknek átlagos hozama $\boldsymbol{\mu} = E_{\mathbf{s} \sim P}[\mathbf{s}]$ és kovariancája $\boldsymbol{\Sigma} = \textrm{Cov}_{\mathbf{s} \sim P}[\mathbf{s}]$.
    1. Számítsd ki egy adott $\boldsymbol{\alpha}$ portfólió várható hozamát.
    1. Ha maximalizálni szeretnéd a portfólió hozamát, hogyan válassz befektetést?
    1. Számítsd ki a portfólió *szórásnégyzetét*.
    1. Fogalmazz meg egy optimalizálási feladatot, amely maximalizálja a hozamot, miközben a szórásnégyzetet egy felső korlát alatt tartja. Ez a Nobel-díjas [Markowitz-portfólió](https://en.wikipedia.org/wiki/Markowitz_model) :cite:`Mangram.2013`. A megoldáshoz kvadratikus programozási megoldóra lesz szükség, ami messze meghaladja e könyv kereteit.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/36)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/37)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/198)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17971)
:end_tab:
