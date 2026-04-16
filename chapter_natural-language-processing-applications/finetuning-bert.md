# A BERT finomhangolása szekvencia- és token-szintű alkalmazásokhoz
:label:`sec_finetuning-bert`


A fejezet előző szakaszaiban különböző modelleket terveztünk természetes nyelvfeldolgozási alkalmazásokhoz, például RNN-eken, CNN-eken, figyelemen és MLP-ken alapuló megközelítésekkel. Ezek a modellek hasznosak, ha tér- vagy időbeli korlátok állnak fenn, azonban minden egyes természetes nyelvfeldolgozási feladathoz külön modellt készíteni a gyakorlatban kivitelezhetetlen. A :numref:`sec_bert` szakaszban bevezettük a BERT előtanítási modellt, amely minimális architekturális változtatásokat igényel a természetes nyelvfeldolgozási feladatok széles köréhez. Egyrészt a javaslata idején a BERT számos természetes nyelvfeldolgozási feladaton javította a legjobb eredményeket. Másrészt, ahogyan azt a :numref:`sec_bert-pretraining` szakasz megjegyzi, az eredeti BERT modell két verziója 110 millió, illetve 340 millió paramétert tartalmaz. Így ha elegendő számítási erőforrás áll rendelkezésre, érdemes lehet a BERT-et finomhangolni az alsóbb szintű természetes nyelvfeldolgozási alkalmazásokhoz.

A következőkben a természetes nyelvfeldolgozási alkalmazások egy részhalmazát szekvencia-szintű és token-szintű feladatokként általánosítjuk. A szekvencia szintjén bemutatjuk, hogyan alakítható át a szöveges bemenet BERT-reprezentációja kimeneti címkévé egyetlen szöveg osztályozásánál, illetve szövegpár osztályozásnál vagy regressziónál. A token szintjén röviden bevezetünk néhány új alkalmazást, mint például a szövegcímkézés és a kérdés-megválaszolás, és rávilágítunk arra, hogyan képes a BERT ezek bemenetét reprezentálni, majd kimeneti címkékké alakítani. A finomhangolás során a BERT által a különböző alkalmazásokhoz megkövetelt „minimális architekturális változtatások" kizárólag a plusz teljesen összekötött rétegek. Egy alsóbb szintű alkalmazás felügyelt tanulása közben a plusz rétegek paraméterei nulláról kerülnek megtanulásra, míg az előtanított BERT modell összes paraméterét finomhangoljuk.


## Egyetlen szöveg osztályozása

Az *egyetlen szöveg osztályozása* egyetlen szövegsorozatot vesz bemenetként, és annak osztályozási eredményét adja ki. A fejezetben vizsgált szentimentelemzésen túl a Corpus of Linguistic Acceptability (CoLA) szintén egy egyetlen szöveg osztályozására szolgáló adathalmaz, amely azt ítéli meg, hogy egy adott mondat grammatikailag elfogadható-e :cite:`Warstadt.Singh.Bowman.2019`. Például az „I should study." elfogadható, az „I should studying." viszont nem.

![A BERT finomhangolása egyetlen szöveg osztályozási alkalmazásokhoz, mint a szentimentelemzés és a nyelvi elfogadhatóság tesztelése. Tegyük fel, hogy a bemeneti egyetlen szöveg hat tokenből áll.](../img/bert-one-seq.svg)
:label:`fig_bert-one-seq`

A :numref:`sec_bert` szakasz leírja a BERT bemeneti reprezentációját. A BERT bemeneti sorozata egyértelműen reprezentál mind egyetlen szöveget, mind szövegpárokat: a speciális osztályozási token „&lt;cls&gt;" a sorozat osztályozására szolgál, a speciális elválasztó token „&lt;sep&gt;" pedig az egyetlen szöveg végét jelöli, vagy egy szövegpárt választ el egymástól. Ahogyan azt a :numref:`fig_bert-one-seq` ábra mutatja, az egyetlen szöveg osztályozási alkalmazásokban a speciális osztályozási token „&lt;cls&gt;" BERT-reprezentációja a teljes bemeneti szövegsorozat információját kódolja. A bemeneti egyetlen szöveg reprezentációjaként egy kis MLP-be kerül betáplálásra, amely teljesen összekötött (sűrű) rétegekből áll, és az összes diszkrét címkeérték eloszlását adja ki.


## Szövegpár osztályozása vagy regressziója

Ebben a fejezetben a természetes nyelvi inferenciát is megvizsgáltuk. Ez a *szövegpár osztályozáshoz* tartozik, amely egy szövegpárt osztályoz.

Szövegpárt bemenetként véve, de folytonos értéket kimenetre adva, a *szemantikai szöveges hasonlóság* egy népszerű *szövegpár regressziós* feladat. Ez a feladat mondatok szemantikai hasonlóságát méri. Például a Semantic Textual Similarity Benchmark adathalmazban egy mondatpár hasonlósági pontszáma egy ordinális skálán mozog, 0-tól (nincs jelentésbeli átfedés) 5-ig (azonos jelentés) :cite:`Cer.Diab.Agirre.ea.2017`. A cél ezen pontszámok előrejelzése. A Semantic Textual Similarity Benchmark adathalmazból vett példák a következők (1. mondat, 2. mondat, hasonlósági pontszám):

* „A plane is taking off.", „An air plane is taking off.", 5.000;
* „A woman is eating something.", „A woman is eating meat.", 3.000;
* „A woman is dancing.", „A man is talking.", 0.000.


![A BERT finomhangolása szövegpár osztályozási vagy regressziós alkalmazásokhoz, mint a természetes nyelvi inferencia és a szemantikai szöveges hasonlóság. Tegyük fel, hogy a bemeneti szövegpár kettő és három tokenből áll.](../img/bert-two-seqs.svg)
:label:`fig_bert-two-seqs`

A :numref:`fig_bert-one-seq` ábrán látható egyetlen szöveg osztályozással összehasonlítva, a BERT finomhangolása szövegpár osztályozáshoz a :numref:`fig_bert-two-seqs` ábrán a bemeneti reprezentációban tér el. Szövegpár regressziós feladatoknál, mint a szemantikai szöveges hasonlóság, apró változtatások alkalmazhatók, például folytonos címkeérték kiadása és a közepes négyzetes veszteség használata: ezek a regresszióban megszokottak.


## Szövegcímkézés

Vizsgáljuk most a token-szintű feladatokat, például a *szövegcímkézést*, ahol minden tokenhez egy-egy címkét rendelünk. A szövegcímkézési feladatok között a *szófaji egyértelműsítés* minden szóhoz szófaji címkét rendel (pl. melléknév vagy névelő) a szó mondatban betöltött szerepe alapján. Például a Penn Treebank II tagkészlete szerint a „John Smith 's car is new" mondatot a következőképpen kell megcímkézni: „NNP (noun, proper singular) NNP POS (possessive ending) NN (noun, singular or mass) VB (verb, base form) JJ (adjective)".

![A BERT finomhangolása szövegcímkézési alkalmazásokhoz, mint a szófaji egyértelműsítés. Tegyük fel, hogy a bemeneti egyetlen szöveg hat tokenből áll.](../img/bert-tagging.svg)
:label:`fig_bert-tagging`

A BERT finomhangolása szövegcímkézési alkalmazásokhoz a :numref:`fig_bert-tagging` ábrán látható. A :numref:`fig_bert-one-seq` ábrával összehasonlítva az egyetlen különbség abban rejlik, hogy szövegcímkézésnél a bemeneti szöveg *minden egyes tokenjének* BERT-reprezentációját ugyanazokba a plusz teljesen összekötött rétegekbe táplálják be a token címkéjének – például egy szófaji tag – meghatározásához.



## Kérdés-megválaszolás

Egy másik token-szintű alkalmazásként a *kérdés-megválaszolás* az olvasásértési képességeket tükrözi. Például a Stanford Question Answering Dataset (SQuAD v1.1) olvasási szövegrészletekből és kérdésekből áll, ahol minden kérdés válasza csupán egy szövegrészlet (szövegszakasz) a kérdéses szövegből :cite:`Rajpurkar.Zhang.Lopyrev.ea.2016`. A magyarázat kedvéért vegyük a következő szövegrészletet: „Some experts report that a mask's efficacy is inconclusive. However, mask makers insist that their products, such as N95 respirator masks, can guard against the virus." és a kérdést: „Who say that N95 respirator masks can guard against the virus?". A válasznak a szövegben szereplő „mask makers" szövegszakasznak kell lennie. Így a SQuAD v1.1-ben a cél egy kérdés–szövegrészlet pár alapján megjósolni a szövegszakasz kezdetének és végének pozícióját a szövegben.

![A BERT finomhangolása kérdés-megválaszoláshoz. Tegyük fel, hogy a bemeneti szövegpár kettő és három tokenből áll.](../img/bert-qa.svg)
:label:`fig_bert-qa`

A BERT kérdés-megválaszolásra való finomhangolásához a kérdés és a szövegrészlet az első, illetve a második szövegsorozatként kerül csomagolásra a BERT bemenetébe. A szövegszakasz kezdőpozíciójának előrejelzéséhez ugyanaz a kiegészítő teljesen összekötött réteg alakítja át a szöveg $i$ pozíciójú tokenjének BERT-reprezentációját $s_i$ skaláris pontozattá. Az összes szöveg-token ilyen pontszámait ezután a softmax művelet valószínűségi eloszlássá alakítja, így a szöveg minden $i$ tokenállásához egy $p_i$ valószínűség kerül hozzárendelve, amely annak valószínűségét fejezi ki, hogy az adott token a szövegszakasz kezdete. A szövegszakasz végének előrejelzése ugyanígy működik, azzal a különbséggel, hogy a kiegészítő teljesen összekötött réteg paraméterei függetlenek a kezdőpont előrejelzéséhez használt paraméterektől. A vég előrejelzésekor az $i$ pozíciójú szöveg-tokent ugyanaz a teljesen összekötött réteg alakítja $e_i$ skaláris pontozattá. A :numref:`fig_bert-qa` ábra a BERT kérdés-megválaszolásra való finomhangolását szemlélteti.

Kérdés-megválaszolás esetén a felügyelt tanulás tanítási célja egyszerű: maximalizálni a valós kezdő- és végpozíciók log-valószínűségét. A szakasz előrejelzésekor az $i$ pozíciótól $j$ pozícióig ($i \leq j$) terjedő érvényes szakaszra kiszámítható az $s_i + e_j$ pontszám, és a legmagasabb pontszámú szakaszt adjuk ki kimenetként.


## Összefoglalás

* A BERT minimális architekturális változtatásokat (plusz teljesen összekötött rétegeket) igényel a szekvencia-szintű és token-szintű természetes nyelvfeldolgozási alkalmazásokhoz, mint például az egyetlen szöveg osztályozása (pl. szentimentelemzés és nyelvi elfogadhatóság tesztelése), szövegpár osztályozása vagy regressziója (pl. természetes nyelvi inferencia és szemantikai szöveges hasonlóság), szövegcímkézés (pl. szófaji egyértelműsítés) és kérdés-megválaszolás.
* Egy alsóbb szintű alkalmazás felügyelt tanulása során a plusz rétegek paraméterei nulláról kerülnek megtanulásra, míg az előtanított BERT modell összes paraméterét finomhangoljuk.


## Gyakorlatok

1. Tervezzünk keresőmotor-algoritmust hírcikkekhez! Amikor a rendszer egy lekérdezést kap (pl. „oil industry during the coronavirus outbreak"), a lekérdezéshez legrelevánsabb hírcikkek rangsorolt listáját kell visszaadnia. Tegyük fel, hogy hatalmas mennyiségű hírcikkel és nagyszámú lekérdezéssel rendelkezünk. A feladat egyszerűsítése érdekében tegyük fel, hogy minden lekérdezéshez megjelöltük a legrelevánsabb cikket. Hogyan alkalmazhatjuk a negatív mintavételezést (lásd: :numref:`subsec_negative-sampling`) és a BERT-et az algoritmus tervezésekor?
1. Hogyan használhatjuk fel a BERT-et nyelvi modellek tanításához?
1. Felhasználható-e a BERT gépi fordításhoz?

[Discussions](https://discuss.d2l.ai/t/396)
