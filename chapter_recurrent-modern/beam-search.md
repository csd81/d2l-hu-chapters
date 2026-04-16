# Beam Search
:label:`sec_beam-search`

A :numref:`sec_seq2seq` fejezetben bemutattuk a kódoló–dekódoló architektúrát,
és a végponttól végpontig való tanításuk szokásos technikáit. Azonban a teszt idején való előrejelzésnél
csak a *mohó* stratégiát említettük,
amelynél minden időlépésnél
azt a tokent választjuk ki, amelyhez
a legmagasabb becsült valószínűség tartozik a következő tokenként,
egészen addig, amíg valamely időlépésnél
azt találjuk, hogy megjósoltuk
a speciális sorozatvégi "&lt;eos&gt;" tokent.
Ebben a részben először formalizáljuk ezt a *mohó keresési* stratégiát,
és azonosítunk néhány problémát,
amelyekkel a szakemberek szoktak találkozni.
Ezt követően összehasonlítjuk ezt a stratégiát
két alternatívával:
*teljes keresés* (szemléltetés céljából, de nem praktikus)
és *beam search* (a standard módszer a gyakorlatban).

Kezdjük a matematikai jelölés felállításával,
a :numref:`sec_seq2seq` fejezetből kölcsönözve.
Bármely $t'$ időlépésnél a dekódoló kimenete
az egyes szókincs-tokenek valószínűségét képviseli
a sorozatban következőként
(a $y_{t'+1}$ valószínű értéke),
kondicionálva az előző tokenekre
$y_1, \ldots, y_{t'}$ és
a $\mathbf{c}$ kontextusvariábilisra,
amelyet a kódoló állít elő
a bemeneti sorozat reprezentálásához.
A számítási költség mennyiségi kifejezéséhez
jelöljük $\mathcal{Y}$-nal
a kimeneti szókincset
(beleértve a speciális sorozatvégi "&lt;eos&gt;" tokent).
Legyen $T'$ a kimeneti sorozat tokenjeinek maximális száma.
Célunk, hogy megkeressük az ideális kimenetet az összes
$\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$
lehetséges kimeneti sorozat közül.
Vegyük figyelembe, hogy ez kissé túlbecsüli
a különböző kimenetek számát,
mivel a "&lt;eos&gt;" token után nincsenek következő tokenek.
Azonban céljainkhoz
ez a szám nagyjából megragadja a keresési tér méretét.


## Mohó Keresés

Vizsgáljuk meg az egyszerű *mohó keresési* stratégiát a :numref:`sec_seq2seq` fejezetből.
Ebben bármely $t'$ időlépésnél
egyszerűen kiválasztjuk azt a tokent,
amelyhez a legmagasabb feltételes valószínűség tartozik
$\mathcal{Y}$-ból, azaz

$$y_{t'} = \operatorname*{argmax}_{y \in \mathcal{Y}} P(y \mid y_1, \ldots, y_{t'-1}, \mathbf{c}).$$

Amint a modell kiadja a "&lt;eos&gt;" tokent
(vagy elérjük a maximális $T'$ hosszt),
a kimeneti sorozat befejezettnek tekinthető.

Ez a stratégia ésszerűnek tűnhet,
és valóban nem is olyan rossz!
Tekintve, hogy számítási szempontból milyen keveset igényel,
nehéz lenne jobbat kihozni ugyanolyan erőfeszítéssel.
Azonban, ha félretesszük a hatékonyságot egy pillanatra,
ésszerűbbnek tűnhet a *legvalószínűbb sorozatot* keresni,
nem a (mohón kiválasztott) *legvalószínűbb tokenek* sorozatát.
Kiderül, hogy ez a két objektum meglehetősen különböző lehet.
A legvalószínűbb sorozat az, amely maximalizálja a
$\prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$ kifejezést.
A gépi fordítás példájában,
ha a dekódoló valóban visszaállítja az alapul szolgáló generatív folyamat valószínűségeit,
akkor ez adná a legvalószínűbb fordítást.
Sajnos nincs garancia arra,
hogy a mohó keresés ezt a sorozatot adná.

Szemléltessük egy példával.
Tegyük fel, hogy négy token van
"A", "B", "C" és "&lt;eos&gt;" a kimeneti szótárban.
A :numref:`fig_s2s-prob1` ábrán
az egyes időlépések alatti négy szám
az "A", "B", "C"
és "&lt;eos&gt;" generálásának feltételes valószínűségeit képviseli az adott időlépésnél.

![Minden időlépésnél a mohó keresés a legmagasabb feltételes valószínűségű tokent választja ki.](../img/s2s-prob1.svg)
:label:`fig_s2s-prob1`

Minden időlépésnél a mohó keresés
a legmagasabb feltételes valószínűségű tokent választja ki.
Ennélfogva az "A", "B", "C" és "&lt;eos&gt;" kimeneti sorozatot
fogjuk megjósolni (:numref:`fig_s2s-prob1`).
Ennek a kimeneti sorozatnak a feltételes valószínűsége
$0.5\times0.4\times0.4\times0.6 = 0.048$.


Következőleg nézzünk egy másik példát a :numref:`fig_s2s-prob2` ábrán.
Ellentétben a :numref:`fig_s2s-prob1` ábrával,
a 2. időlépésnél a "C" tokent választjuk,
amelynek *második* legmagasabb a feltételes valószínűsége.

![Az egyes időlépések alatti négy szám az "A", "B", "C" és "&lt;eos&gt;" generálásának feltételes valószínűségeit képviseli az adott időlépésnél.
A 2. időlépésnél a "C" tokent, amely a második legmagasabb feltételes valószínűségű,
választják ki.](../img/s2s-prob2.svg)
:label:`fig_s2s-prob2`

Mivel az 1. és 2. időlépések kimeneti alsorozatai,
amelyekre a 3. időlépés épül,
megváltoztak "A"-ról és "B"-ről a :numref:`fig_s2s-prob1` ábrán
"A"-ra és "C"-re a :numref:`fig_s2s-prob2` ábrán,
minden token feltételes valószínűsége
a 3. időlépésnél is megváltozott a :numref:`fig_s2s-prob2` ábrán.
Tegyük fel, hogy a 3. időlépésnél a "B" tokent választjuk.
Most a 4. időlépés az első három időlépés
"A", "C" és "B" kimeneti alsorozatára feltételezett,
amely megváltozott az "A", "B" és "C"-ről a :numref:`fig_s2s-prob1` ábrán.
Ezért a 4. időlépésnél az egyes tokenek generálásának feltételes valószínűsége
a :numref:`fig_s2s-prob2` ábrán
is különbözik a :numref:`fig_s2s-prob1` ábrától.
Ennek eredményeként az "A", "C", "B" és "&lt;eos&gt;" kimeneti sorozat
feltételes valószínűsége a :numref:`fig_s2s-prob2` ábrán
$0.5\times0.3 \times0.6\times0.6=0.054$,
amely nagyobb, mint a mohó keresésé a :numref:`fig_s2s-prob1` ábrán.
Ebben a példában az "A", "B", "C" és "&lt;eos&gt;" kimeneti sorozat,
amelyet a mohó keresés kapott, nem optimális.




## Teljes Keresés

Ha a cél a legvalószínűbb sorozat megszerzése,
fontolóra vehetjük a *teljes keresés* alkalmazását:
felsoroljuk az összes lehetséges kimeneti sorozatot
a feltételes valószínűségeikkel együtt,
majd azt adjuk ki, amelynek a legmagasabb becsült valószínűsége van.


Bár ez biztosan megadná, amit kívánunk,
a számítási költsége tiltó mértékű lenne:
$\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$,
exponenciálisan növekvő a sorozat hosszával, és rendkívül nagy alap a szókincs méretéből adódóan.
Például, ha $|\mathcal{Y}|=10000$ és $T'=10$,
mindkettő kis szám a valós alkalmazásokhoz képest, $10000^{10} = 10^{40}$ sorozatot kellene kiértékelnünk, ami már meghaladja bármely belátható számítógép képességeit.
A mohó keresés számítási költsége viszont
$\mathcal{O}(\left|\mathcal{Y}\right|T')$:
csodálatosan olcsó, de messze nem optimális.
Például, ha $|\mathcal{Y}|=10000$ és $T'=10$,
mindössze $10000\times10=10^5$ sorozatot kell kiértékelnünk.


## Beam Search

A sorozatdekódolási stratégiákat egy skálán helyezhetjük el,
ahol a *beam search* kompromisszumot kínál
a mohó keresés hatékonysága
és a teljes keresés optimalitása között.
A beam search legegyszerűbb változatát
egyetlen hiperparaméter jellemez,
a *beam méret*, $k$.
Magyarázzuk el ezt a terminológiát.
Az 1. időlépésnél kiválasztjuk a $k$ tokent,
amelyhez a legmagasabb becsült valószínűségek tartoznak.
Mindegyik a $k$ jelölt kimeneti sorozat első tokenje lesz.
Minden következő időlépésnél,
az előző időlépésnél lévő $k$ jelölt kimeneti sorozat alapján,
folytatjuk a $k$ jelölt kimeneti sorozat kiválasztását,
amelyekhez a legmagasabb becsült valószínűségek tartoznak
$k\left|\mathcal{Y}\right|$ lehetséges választásból.

![A beam search folyamata (beam méret $=2$; kimeneti sorozat maximális hossza $=3$). A jelölt kimeneti sorozatok: $\mathit{A}$, $\mathit{C}$, $\mathit{AB}$, $\mathit{CE}$, $\mathit{ABD}$ és $\mathit{CED}$.](../img/beam-search.svg)
:label:`fig_beam-search`


A :numref:`fig_beam-search` egy példán keresztül szemlélteti
a beam search folyamatát.
Tegyük fel, hogy a kimeneti szókincs
csak öt elemet tartalmaz:
$\mathcal{Y} = \{A, B, C, D, E\}$,
ahol az egyik "&lt;eos&gt;".
Legyen a beam mérete kettő és
a kimeneti sorozat maximális hossza három.
Az 1. időlépésnél
tegyük fel, hogy a legmagasabb feltételes valószínűséggel rendelkező tokenek
$P(y_1 \mid \mathbf{c})$ az $A$ és $C$.
A 2. időlépésnél minden $y_2 \in \mathcal{Y}$ esetén
kiszámítjuk

$$\begin{aligned}P(A, y_2 \mid \mathbf{c}) = P(A \mid \mathbf{c})P(y_2 \mid A, \mathbf{c}),\\ P(C, y_2 \mid \mathbf{c}) = P(C \mid \mathbf{c})P(y_2 \mid C, \mathbf{c}),\end{aligned}$$  

és kiválasztjuk a legnagyobb kettőt e tíz érték közül, mondjuk
$P(A, B \mid \mathbf{c})$ és $P(C, E \mid \mathbf{c})$.
Ezután a 3. időlépésnél minden $y_3 \in \mathcal{Y}$ esetén kiszámítjuk

$$\begin{aligned}P(A, B, y_3 \mid \mathbf{c}) = P(A, B \mid \mathbf{c})P(y_3 \mid A, B, \mathbf{c}),\\P(C, E, y_3 \mid \mathbf{c}) = P(C, E \mid \mathbf{c})P(y_3 \mid C, E, \mathbf{c}),\end{aligned}$$ 

és kiválasztjuk a legnagyobb kettőt e tíz érték közül, mondjuk
$P(A, B, D \mid \mathbf{c})$ és $P(C, E, D \mid  \mathbf{c}).$
Ennek eredményeként hat jelölt kimeneti sorozatot kapunk:
(i) $A$; (ii) $C$; (iii) $A$, $B$; (iv) $C$, $E$; (v) $A$, $B$, $D$; és (vi) $C$, $E$, $D$.


Végül a végső jelölt kimeneti sorozatok halmazát
ezekre a hat sorozatra alapozva kapjuk meg (pl. elvetve a "&lt;eos&gt;" tokent és az utána következő részeket).
Ezután azt a kimeneti sorozatot választjuk, amely maximalizálja a következő pontszámot:

$$ \frac{1}{L^\alpha} \log P(y_1, \ldots, y_{L}\mid \mathbf{c}) = \frac{1}{L^\alpha} \sum_{t'=1}^L \log P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c});$$
:eqlabel:`eq_beam-search-score`

ahol $L$ a végső jelölt sorozat hossza
és $\alpha$ értéke általában 0.75.
Mivel egy hosszabb sorozatnak több logaritmikus tagja van
a :eqref:`eq_beam-search-score` összegzésében,
a nevező $L^\alpha$ tagja bünteti
a hosszú sorozatokat.

A beam search számítási költsége $\mathcal{O}(k\left|\mathcal{Y}\right|T')$.
Ez az eredmény a mohó keresés és a teljes keresés eredménye között van.
A mohó keresés a beam search speciális esetének tekinthető,
amikor a beam méret 1-re van állítva.




## Összefoglalás

A sorozatkeresési stratégiák közé tartozik
a mohó keresés, a teljes keresés és a beam search.
A beam search kompromisszumot nyújt a pontosság és
a számítási költség között a beam méret rugalmas megválasztásán keresztül.


## Feladatok

1. Tekinthető-e a teljes keresés a beam search egy speciális típusaként? Miért, vagy miért nem?
1. Alkalmazd a beam search-t a gépi fordítás problémájára a :numref:`sec_seq2seq` fejezetben. Hogyan befolyásolja a beam méret a fordítási eredményeket és az előrejelzési sebességet?
1. A :numref:`sec_rnn-scratch` fejezetben a felhasználó által megadott előtagokat követő szöveg generálásához nyelvi modellezést alkalmaztunk. Milyen keresési stratégiát használ? Tudod javítani?

[Discussions](https://discuss.d2l.ai/t/338)
