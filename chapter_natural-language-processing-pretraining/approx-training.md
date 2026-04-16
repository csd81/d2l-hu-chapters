# Közelítő Tanítás
:label:`sec_approx_train`

Idézzük fel a :numref:`sec_word2vec` fejezetbeli tárgyalásunkat.
A skip-gram modell alapötlete az,
hogy softmax műveletekkel számítsuk ki
a $w_o$ kontextusszó generálásának feltételes valószínűségét
az adott $w_c$ középső szóhoz
az :eqref:`eq_skip-gram-softmax` képlet szerint,
amelynek megfelelő logaritmikus veszteségét az :eqref:`eq_skip-gram-log` ellentéte adja.



A softmax művelet természetéből adódóan,
mivel a kontextusszó bármelyik szó lehet a
$\mathcal{V}$ szótárból,
az :eqref:`eq_skip-gram-log` ellentéte
tartalmaz egy összegzést,
amelynek tagjai annyi elemből állnak, mint az egész szókincs mérete.
Ennek következtében
a skip-gram modell gradienszámítása
az :eqref:`eq_skip-gram-grad`-ban
és a continuous bag-of-words modellé
az :eqref:`eq_cbow-gradient`-ban
mindkettő tartalmaz
összegzést.
Sajnos
az ilyen gradiensek számítási költsége,
amelyek egy nagy szótáron
(általában
százezer vagy millió szóval)
összegeznek,
óriási!

A fent említett számítási bonyolultság csökkentése érdekében ez a szakasz két közelítő tanítási módszert mutat be:
a *negatív mintavételezést* és a *hierarchikus softmax-ot*.
A skip-gram modell és
a continuous bag of words modell közötti hasonlóság miatt
csak a skip-gram modellt vesszük példaként
e két közelítő tanítási módszer bemutatásához.

## Negatív Mintavételezés
:label:`subsec_negative-sampling`


A negatív mintavételezés módosítja az eredeti célfüggvényt.
A $w_c$ középső szó kontextusablakát adva,
azt, hogy bármely (kontextus) $w_o$ szó
ebből a kontextusablakból származik,
olyan eseményként kezeljük, amelynek valószínűségét a következő képlet adja:


$$P(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c),$$

ahol $\sigma$ a sigmoid aktivációs függvény definícióját alkalmazza:

$$\sigma(x) = \frac{1}{1+\exp(-x)}.$$
:eqlabel:`eq_sigma-f`

Kezdjük azzal, hogy
maximalizáljuk az összes ilyen esemény együttes valószínűségét a szövegsorozatokban
a szóbeágyazások tanításához.
Konkrétan,
adott egy $T$ hosszú szövegsorozat,
jelöljük $w^{(t)}$-vel a $t$ időlépésbeli szót,
legyen a kontextusablak mérete $m$,
és maximalizáljuk a következő együttes valószínűséget:


$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(D=1\mid w^{(t)}, w^{(t+j)}).$$
:eqlabel:`eq-negative-sample-pos`


Azonban
az :eqref:`eq-negative-sample-pos`
csak azokat az eseményeket veszi figyelembe,
amelyek pozitív példákat érintenek.
Ennek következtében
az :eqref:`eq-negative-sample-pos`-ban lévő együttes valószínűség
csak akkor maximalizálható 1-re,
ha az összes szóvektor végtelen.
Természetesen
az ilyen eredmények értelmetlenek.
Ahhoz, hogy a célfüggvény
értelmesebbé váljon,
a *negatív mintavételezés*
egy előre meghatározott eloszlásból mintavételezett
negatív példákat ad hozzá.

Jelöljük $S$-sel
azt az eseményt, hogy
egy $w_o$ kontextusszó
a $w_c$ középső szó kontextusablakából származik.
Ehhez a $w_o$-t érintő eseményhez
egy $P(w)$ előre meghatározott eloszlásból
mintavételezünk $K$ *zajszót*,
amelyek nem ebből a kontextusablakból valók.
Jelöljük $N_k$-val
azt az eseményt, hogy
a $w_k$ zajszó ($k=1, \ldots, K$)
nem a $w_c$ kontextusablakából
származik.
Feltételezzük, hogy
ezek az események,
amelyek mind a pozitív példát, mind a negatív példákat érintik
($S, N_1, \ldots, N_K$), egymástól függetlenek.
A negatív mintavételezés
az együttes valószínűséget (csak pozitív példákat tartalmazva)
az :eqref:`eq-negative-sample-pos`-ban átírja:

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

ahol a feltételes valószínűséget az $S, N_1, \ldots, N_K$ eseményeken keresztül közelítjük:

$$ P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k).$$
:eqlabel:`eq-negative-sample-conditional-prob`

Jelöljük
$i_t$-vel és $h_k$-val
a $w^{(t)}$ szó indexét a szövegsorozat $t$ időlépésében
és a $w_k$ zajszó indexét.
Az :eqref:`eq-negative-sample-conditional-prob`-ban lévő feltételes valószínűségekkel kapcsolatos logaritmikus veszteség:

$$
\begin{aligned}
-\log P(w^{(t+j)} \mid w^{(t)})
=& -\log P(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim P(w)}^K \log P(D=0\mid w^{(t)}, w_k)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\left(1-\sigma\left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right)\right)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\sigma\left(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right).
\end{aligned}
$$


Láthatjuk, hogy
most minden tanítási lépésnél
a gradiensek számítási költsége
nem függ a szótár méretétől,
hanem lineárisan függ $K$-tól.
Amikor a $K$ hiperparamétert
kisebb értékre állítjuk,
a negatív mintavételezéssel végzett
egyes tanítási lépések gradienszámításának
számítási költsége kisebb.




## Hierarchikus Softmax

Alternatív közelítő tanítási módszerként
a *hierarchikus softmax*
bináris fát használ,
amely egy adatstruktúra,
ahogy az :numref:`fig_hi_softmax` ábra is bemutatja,
ahol a fa minden levélcsúcsa
a $\mathcal{V}$ szótárban egy szót képvisel.

![Hierarchikus softmax közelítő tanításhoz, ahol a fa minden levélcsúcsa a szótárban egy szót képvisel.](../img/hi-softmax.svg)
:label:`fig_hi_softmax`

Jelöljük $L(w)$-vel
a bináris fában
a gyökércsúcstól a $w$ szót képviselő levélcsúcsig vezető úton lévő csúcsok számát (mindkét végpontot beleértve).
Legyen $n(w,j)$ az úton lévő $j$-ik csúcs,
amelynek kontextusszóvektora
$\mathbf{u}_{n(w, j)}$.
Például
$L(w_3) = 4$ az :numref:`fig_hi_softmax` ábrán.
A hierarchikus softmax közelíti az :eqref:`eq_skip-gram-softmax`-ban lévő feltételes valószínűséget:


$$P(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma\left( [\![  n(w_o, j+1) = \textrm{leftChild}(n(w_o, j)) ]\!] \cdot \mathbf{u}_{n(w_o, j)}^\top \mathbf{v}_c\right),$$

ahol a $\sigma$ függvényt az :eqref:`eq_sigma-f` definiálja,
és $\textrm{leftChild}(n)$ az $n$ csúcs bal gyerekcsúcsa: ha $x$ igaz, $[\![x]\!] = 1$; egyébként $[\![x]\!] = -1$.

Szemléltetésképpen
számítsuk ki
a $w_3$ szó generálásának feltételes valószínűségét
a $w_c$ szóhoz adottan az :numref:`fig_hi_softmax` ábrán.
Ehhez szükség van a
$w_c$ szó $\mathbf{v}_c$ szóvektora és
a gyökértől $w_3$-ig vezető úton lévő
(az :numref:`fig_hi_softmax` ábrán félkövér úton)
nem levél csúcsvektorok közötti skaláris szorzatokra,
amelyet balra, jobbra, majd balra haladva járunk be:


$$P(w_3 \mid w_c) = \sigma(\mathbf{u}_{n(w_3, 1)}^\top \mathbf{v}_c) \cdot \sigma(-\mathbf{u}_{n(w_3, 2)}^\top \mathbf{v}_c) \cdot \sigma(\mathbf{u}_{n(w_3, 3)}^\top \mathbf{v}_c).$$

Mivel $\sigma(x)+\sigma(-x) = 1$,
teljesül, hogy
a $\mathcal{V}$ szótárban lévő összes szó generálásának feltételes valószínűsége
bármely $w_c$ szóhoz adottan
összege egy:

$$\sum_{w \in \mathcal{V}} P(w \mid w_c) = 1.$$
:eqlabel:`eq_hi-softmax-sum-one`

Szerencsére, mivel $L(w_o)-1$ a bináris fa struktúra miatt $\mathcal{O}(\textrm{log}_2|\mathcal{V}|)$ nagyságrendű,
amikor a $\mathcal{V}$ szótár mérete hatalmas,
a hierarchikus softmax-ot alkalmazó minden egyes tanítási lépés számítási költsége
jelentősen csökken a közelítő tanítás nélkülihez képest.

## Összefoglalás

* A negatív mintavételezés a veszteségfüggvényt úgy konstruálja, hogy kölcsönösen független, pozitív és negatív példákat egyaránt tartalmazó eseményeket vesz figyelembe. A tanítás számítási költsége minden lépésnél lineárisan függ a zajszavak számától.
* A hierarchikus softmax a veszteségfüggvényt a bináris fa gyökércsúcsától a levélcsúcsig vezető úton keresztül konstruálja. A tanítás számítási költsége minden lépésnél a szótár méretének logaritmusától függ.

## Gyakorló feladatok

1. Hogyan mintavételezhetünk zajszavakat a negatív mintavételezésben?
1. Igazold, hogy az :eqref:`eq_hi-softmax-sum-one` teljesül.
1. Hogyan tanítható a continuous bag of words modell negatív mintavételezéssel és hierarchikus softmax-szal?

[Discussions](https://discuss.d2l.ai/t/382)
