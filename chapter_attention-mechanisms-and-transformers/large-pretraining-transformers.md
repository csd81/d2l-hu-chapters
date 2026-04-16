# Nagy Méretű Előtanítás Transzformerekkel
:label:`sec_large-pretraining-transformers`

Eddig a képbesorolási és gépi fordítási kísérleteinkben
a modelleket *semmiből* tanítottuk be adathalmazokon
bemenet-kimenet példákkal konkrét feladatok elvégzésére.
Például egy Transzformert angol-francia párokkal tanítottunk be
(:numref:`sec_transformer`)
úgy, hogy ez a modell bemeneti angol szöveget franciára fordítson.
Ennek eredményeként minden modell *specifikus szakértővé* válik,
ami érzékeny még az adateloszlás enyhe változására is
(:numref:`sec_environment-and-distribution-shift`).
Jobban általánosított modellek, vagy még kompetensebb *generalisták*
érdekében, amelyek több feladatot képesek elvégezni adaptációval vagy anélkül,
a modellek *előtanítása* nagy adatmennyiségeken egyre gyakoribb.

Nagyobb előtanítási adatmennyiség esetén a Transzformer architektúra
jobban teljesít megnövelt modellmérettel és számítási kapacitással,
kimagasló *skálázási* viselkedést mutatva.
Konkrétan, a Transzformer-alapú nyelvi modellek teljesítménye
hatványtörvény szerint skálázódik a modellparaméterek,
tanító tokenek és számítási kapacitás mennyiségével :cite:`kaplan2020scaling`.
A Transzformerek skálázhatóságát az is bizonyítja,
hogy a nagyobb adaton tanított nagyobb látási Transzformerek
jelentősen jobb teljesítményt mutatnak
(a :numref:`sec_vision-transformer`-ben tárgyalva).
Frissebb sikertörténetek közé tartozik a Gato, egy *generalista* modell,
ami Atarit játszhat, képeket feliratozhat, cseveghet és robotként viselkedhet :cite:`reed2022generalist`. A Gato egyetlen Transzformer, amely jól skálázódik különböző modalitásokon
való előtanítás során, beleértve szöveget, képeket, ízületi nyomatékokat és gombnyomásokat.
Figyelemre méltó, hogy minden ilyen multimodális adat lapos token-szekvenciává
sorozatosítódik, amely feldolgozható hasonlóan szövegtokenekhez (:numref:`sec_transformer`)
vagy képfoltokhoz (:numref:`sec_vision-transformer`) Transzformerek által.

A Transzformerek multimodális adatokon való előtanításának meggyőző sikere előtt
a Transzformereket széles körben előtanították gazdag szöveges anyaggal.
Eredetileg gépi fordításra javasolták,
a :numref:`fig_transformer`-ben látható Transzformer architektúra
bemeneti szekvenciák reprezentálására szolgáló kódolóból
és cél szekvenciák generálására szolgáló dekóderből áll.
Elsődlegesen a Transzformerek három különböző módban használhatók:
*csupán kódoló*, *kódoló-dekóder*, és *csupán dekóder*.
A fejezet befejezéseként áttekintjük ezt a három módot
és elmagyarázzuk a Transzformerek előtanításának skálázhatóságát.

## Csak Kódoló

Ha csak a Transzformer kódolót használjuk,
a bemeneti tokenek sorozata
ugyanannyi reprezentációvá alakul,
ami tovább vetíthető kimenetre
(például besorolás). Egy Transzformer kódoló
önfigyelem rétegekből áll,
ahol minden bemeneti token figyel minden másikra.
Például a :numref:`fig_vit`-ben ábrázolt látási Transzformerek
csak kódolóból állnak, bemeneti képfoltok sorozatát
átalakítják egy speciális "&lt;cls&gt;" token reprezentációjává.
Mivel ez a reprezentáció minden bemeneti tokentől függ,
tovább vetítik besorolási címkékre.
Ezt a tervezetet egy korábbi, csak kódolóból álló Transzformer
inspirálta, amelyet szövegen előtanítottak: BERT (Bidirectional Encoder Representations from Transformers) :cite:`Devlin.Chang.Lee.ea.2018`.


### A BERT Előtanítása

![Bal: A BERT előtanítása maszkolt nyelvi modellezéssel. Az elmaszkolt "love" token előrejelzése a "love" előtti és utáni összes bemeneti tokentől függ. Jobb: Figyelemi minta a Transformer kódolóban. A függőleges tengely mentén lévő minden token figyel a vízszintes tengely mentén lévő összes bemeneti tokenre.](../img/bert-encoder-only.svg)
:label:`fig_bert-encoder-only`

A BERT szövegszekvenciákon van előtanítva *maszkolt nyelvi modellezéssel*:
a véletlenszerűen elmaszkolt tokeneket tartalmazó bemeneti szöveg
egy Transformer kódolóba kerül, amely megjósolja az elmaszkolt tokeneket.
Ahogy a :numref:`fig_bert-encoder-only` szemlélteti,
az "I", "love", "this", "red", "car" eredeti szövegszekvencia
elé kerül a "&lt;cls&gt;" token, a "&lt;mask&gt;" token
pedig véletlenszerűen lecseréli a "love"-t; ekkor az elmaszkolt "love" token
és annak előrejelzése közti keresztentrópia-veszteség minimalizálása
az előtanítás célja.
Megjegyzendő, hogy a Transformer kódolók figyelemi mintájában
nincs megkötés (:numref:`fig_bert-encoder-only` jobb oldala),
tehát minden token figyelhet minden másikra.
Így a "love" előrejelzése a szekvenciában előtte és utána lévő bemeneti tokenektől egyaránt függ.
Ezért nevezik a BERT-et „kétirányú kódolónak".
Manuális felcímkézés nélkül, könyvekből és Wikipédiából
származó nagy méretű szöveges adatok felhasználhatók a BERT előtanítására.


### A BERT Finomhangolása

Az előtanított BERT *finomhangolható* olyan alárendelt kódolási feladatokra, amelyek egyedi szövegeket vagy szövegpárokat érintenek. A finomhangolás során véletlenszerű paraméterekkel rendelkező további rétegek adhatók a BERT-hez: ezek a paraméterek és az előtanított BERT paraméterei *frissülnek*, hogy illeszkedjenek az alárendelt feladatok tanítási adataihoz.

![A BERT finomhangolása szentimentelemzéshez.](../img/bert-finetune-classification.svg)
:label:`fig_bert-finetune-classification`

A :numref:`fig_bert-finetune-classification` szemlélteti
a BERT finomhangolását szentimentelemzésre.
A Transformer kódoló egy előtanított BERT,
amely szövegszekvenciát vesz bemenetként,
és a "&lt;cls&gt;" reprezentációt
(a bemenet globális reprezentációját)
egy további teljesen összekötött rétegbe táplálja
a hangulat előrejelzéséhez.
A finomhangolás során az előrejelzés és a címke közti
keresztentrópia-veszteség minimalizálódik
gradiens alapú algoritmusokkal a szentimentelemzési adatokon,
ahol a további réteg nulláról tanul,
miközben a BERT előtanított paraméterei frissülnek.
A BERT többre képes, mint szentimentelemzés.
A 350 millió paraméteres BERT által
250 milliárd tanítási tokenből tanult
általános nyelvi reprezentációk
élvonalbeli eredményeket értek el természetes nyelvi feladatokon,
mint például egyedi szövegosztályozás,
szövegpár-osztályozás vagy regresszió,
szöveges jelölés és kérdés-megválaszolás.

Észrevehetjük, hogy ezek az alárendelt feladatok szövegpár-megértést is tartalmaznak.
A BERT előtanítása egy másik veszteséget is tartalmaz annak előrejelzésére,
hogy az egyik mondat közvetlenül követi-e a másikat.
Ez a veszteség azonban később kevésbé hasznosnak bizonyult
a RoBERTa előtanítása során,
amely egy azonos méretű BERT-változat, 2000 milliárd tokenre tanítva :cite:`Liu.Ott.Goyal.ea.2019`.
A BERT egyéb származékai javítottak a modellarchitektúrákon vagy az előtanítási célokon,
mint például az ALBERT (paramétermegosztás kényszerítése) :cite:`lan2019albert`,
a SpanBERT (szövegszakaszok reprezentálása és előrejelzése) :cite:`joshi2020spanbert`,
a DistilBERT (könnyűsúlyú, tudásdesztilláció révén) :cite:`sanh2019distilbert`,
és az ELECTRA (kicserélt token detektálása) :cite:`clark2019electra`.
Emellett a BERT inspirálta a Transformer előtanítást a számítógépes látásban is,
például látási Transformerekkel :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`,
Swin Transformerekkel :cite:`liu2021swin`,
és MAE-vel (maszkolt autoenkóderek) :cite:`he2022masked`.

## Kódoló–Dekóder

Mivel egy Transformer kódoló a bemeneti tokenek sorozatát
azonos számú kimeneti reprezentációvá alakítja,
a csupán kódolós mód nem képes tetszőleges hosszúságú sorozatot generálni,
mint ahogy azt a gépi fordítás megkívánja.
Ahogy eredetileg gépi fordításra javasolták,
a Transformer architektúra kiegészíthető egy dekóderrel,
amely autoregresszív módon, tokenenként, tetszőleges hosszúságú
célszekvenciát jósol meg a kódoló és a dekóder kimenetétől feltételesen:
(i) a kódoló kimenetétől való feltételezéshez a kódoló–dekóder keresztfigyelelem
(a dekóder többfejű figyelme a :numref:`fig_transformer`-ben)
lehetővé teszi, hogy a cél tokenek *összes* bemeneti tokenre figyeljenek;
(ii) a dekóder kimenetétől való feltételezés egy úgynevezett *kauzális* figyelem
(ez az elnevezés elterjedt az irodalomban, de félrevezető,
mivel kevés köze van az okozatiság tényleges tanulmányozásához)
mintával valósul meg (a dekóder maszkolt többfejű figyelme a :numref:`fig_transformer`-ben),
ahol bármely cél token csak a célszekvencia *múltbeli* és *jelenlegi* tokenjeire figyelhet.

A kódoló–dekóder Transformerek emberek által felcímkézett gépi fordítási adatokon
túli előtanításához a BART :cite:`lewis2019bart` és a T5 :cite:`raffel2020exploring`
két párhuzamosan javasolt kódoló–dekóder Transformer,
amelyeket nagy méretű szövegkorpuszokon tanítottak elő.
Mindkettő az eredeti szöveg rekonstruálására törekszik az előtanítási célokban,
miközben az előbbi a bemenet zajosítását hangsúlyozza
(pl. maszkolás, törlés, permutáció és elforgatás),
az utóbbi pedig a többfeladatos egységesítést emeli ki
átfogó ablációs vizsgálatokkal.


### A T5 Előtanítása


Az előtanított Transformer kódoló–dekóder példájaként
a T5 (Text-to-Text Transfer Transformer)
sok feladatot ugyanolyan szöveg-szöveg problémává egységesít:
bármely feladatnál a kódoló bemenete egy feladatleírás
(pl. "Summarize", ":"), amelyet feladatbemenet követ
(pl. egy cikkből származó tokensorozat),
a dekóder pedig a feladatkimenetet jósolja meg
(pl. a bemeneti cikket összefoglaló tokensorozat).
Szöveg-szöveg teljesítményhez a T5-t arra tanítják,
hogy bizonyos cél szöveget generáljon bemeneti szövegtől feltételesen.


![Bal: A T5 előtanítása egymást követő szakaszok előrejelzésével. Az eredeti mondat: "I", "love", "this", "red", "car", ahol a "love"-t egy speciális "&lt;X&gt;" token, az egymást követő "red", "car" szavakat pedig egy speciális "&lt;Y&gt;" token helyettesíti. A célszekvencia egy speciális "&lt;Z&gt;" tokennel végződik. Jobb: Figyelemi minta a Transformer kódoló–dekóderben. A kódoló önfigyelemben (alsó négyzet) az összes bemeneti token egymásra figyel; a kódoló–dekóder keresztfigyelemben (felső téglalap) minden cél token az összes bemeneti tokenre figyel; a dekóder önfigyelemben (felső háromszög) minden cél token csak a jelenlegi és múltbeli cél tokenekre figyel (kauzális).](../img/t5-encoder-decoder.svg)
:label:`fig_t5-encoder-decoder`

Hogy bármely eredeti szövegből bemenetet és kimenetet kapjon,
a T5 egymást követő szakaszok előrejelzésére van előtanítva.
Pontosabban, a szöveg tokenjeit véletlenszerűen
speciális tokenekre cserélik, ahol minden egymást követő szakaszt
ugyanaz a speciális token helyettesít.
Vegyük a :numref:`fig_t5-encoder-decoder` példáját,
ahol az eredeti szöveg: "I", "love", "this", "red", "car".
A "love", "red", "car" tokeneket véletlenszerűen speciális tokenekre cserélik.
Mivel a "red" és a "car" egymást követő szakaszt alkot,
ugyanazzal a speciális tokennel helyettesítik őket.
Ennek eredményeként a bemeneti szekvencia: "I", "&lt;X&gt;", "this", "&lt;Y&gt;",
a célszekvencia pedig:
"&lt;X&gt;", "love", "&lt;Y&gt;", "red", "car", "&lt;Z&gt;",
ahol "&lt;Z&gt;" egy másik speciális token, amely a végét jelöli.
Ahogy a :numref:`fig_t5-encoder-decoder` mutatja,
a dekóder kauzális figyelemi mintával rendelkezik, amely megakadályozza,
hogy jövőbeli tokenekre figyeljen a szekvencia-előrejelzés során.

A T5-ben az egymást követő szakaszok előrejelzése
sérült szöveg rekonstruálásának is nevezett.
Ezzel a céllal a T5-t a C4
(Colossal Clean Crawled Corpus) adataiból
1000 milliárd tokennel tanítják elő,
amely weboldalakról származó, tiszta angol szövegből áll :cite:`raffel2020exploring`.

### A T5 Finomhangolása

A BERT-hez hasonlóan a T5-t is finomhangolni kell (a T5 paramétereinek frissítésével)
feladatspecifikus tanítási adatokon az adott feladat elvégzéséhez.
A BERT finomhangolásától való fő különbségek:
(i) a T5 bemenete tartalmaz feladatleírásokat;
(ii) a T5 tetszőleges hosszúságú sorozatokat tud generálni
Transformer dekóderével;
(iii) nem szükségesek további rétegek.

![A T5 finomhangolása szövegösszefoglaláshoz. Mind a feladatleírás, mind a cikk tokenjei a Transformer kódolóba kerülnek az összefoglalás előrejelzéséhez.](../img/t5-finetune-summarization.svg)
:label:`fig_t5-finetune-summarization`

A :numref:`fig_t5-finetune-summarization`
a T5 finomhangolását magyarázza el
szövegösszefoglalást használva példaként.
Ebben az alárendelt feladatban
a "Summarize", ":" feladatleíró tokeneket,
amelyeket a cikk tokenjei követnek, a kódolóba táplálják be.

A finomhangolás után a 11 milliárd paraméteres T5 (T5-11B)
élvonalbeli eredményeket ért el több kódolási (pl. osztályozás)
és generálási (pl. összefoglalás) benchmarkon.
Megjelenése óta a T5-t széles körben alkalmazzák a további kutatásokban.
Például a switch Transformereket a T5 alapján tervezték,
hogy a paraméterek egy részhalmazát aktiválják
a jobb számítási hatékonyság érdekében :cite:`fedus2022switch`.
Az Imagen nevű szöveg-képpé alakító modellben
a szöveg egy befagyasztott T5 kódolóba (T5-XXL)
kerül 4,6 milliárd paraméterrel :cite:`saharia2022photorealistic`.
A :numref:`fig_imagen` fotorealisztikus szöveg-kép példái
azt sugallják, hogy a T5 kódoló önmagában is hatékonyan
képes szöveget reprezentálni finomhangolás nélkül is.

![Szöveg-kép példák az Imagen modelltől, amelynek szövegkódolója a T5-ből származik (ábrák forrása: :citet:`saharia2022photorealistic`).](../img/imagen.png)
:width:`700px`
:label:`fig_imagen`


## Csak Dekóder


Áttekintettük a csupán kódolós és a kódoló–dekóder Transformereket.
Alternatívaként a csupán dekóderes Transformerek
az eredeti kódoló–dekóder architektúrából
(:numref:`fig_transformer`)
eltávolítják a teljes kódolót és a kódoló–dekóder keresztfigyelemmel rendelkező dekóder alréteget.
Napjainkban a csupán dekóderes Transformerek váltak a *de facto* architektúrává
a nagy méretű nyelvi modellezésben (:numref:`sec_language-model`),
amely önfelügyelt tanulással hasznosítja a világ gazdag, felcímkézetlen szövegkorpuszait.



### GPT és GPT-2

Nyelvi modellezést alkalmazva tanítási célként,
a GPT (generatív előtanítás) modell
Transformer dekódert választ
gerincként :cite:`Radford.Narasimhan.Salimans.ea.2018`.

![Bal: A GPT előtanítása nyelvi modellezéssel. A célszekvencia egy tokennel eltolt bemeneti szekvencia. A "&lt;bos&gt;" és "&lt;eos&gt;" speciális tokenek jelölik a szekvenciák elejét és végét. Jobb: Figyelemi minta a Transformer dekóderben. A függőleges tengely mentén lévő minden token csak a vízszintes tengely mentén lévő múltbeli tokenjeire figyel (kauzális).](../img/gpt-decoder-only.svg)
:label:`fig_gpt-decoder-only`

A :numref:`subsec_partitioning-seqs`-ban leírt
autoregresszív nyelvi modell tanítást követve,
a :numref:`fig_gpt-decoder-only` szemlélteti
a GPT előtanítását Transformer kódolóval,
ahol a célszekvencia egy tokennel eltolt bemeneti szekvencia.
Megjegyzendő, hogy a Transformer dekóder figyelemi mintája
kikényszeríti, hogy minden token csak a múltbeli tokenjeire figyeljen
(a jövőbeli tokenekre nem lehet figyelni, mert azok még nem lettek kiválasztva).


A GPT 100 millió paraméterrel rendelkezik, és finomhangolásra szorul
az egyes alárendelt feladatokhoz.
Egy sokkal nagyobb Transformer-dekóderes nyelvi modell,
a GPT-2, egy évvel később jelent meg :cite:`Radford.Wu.Child.ea.2019`.
A GPT eredeti Transformer dekóderével összehasonlítva előnormalizálást
(:numref:`subsec_vit-encoder`-ben tárgyalva)
és továbbfejlesztett inicializálást és súlyméretre-skálázást alkalmaztak a GPT-2-ben.
A 40 GB szövegen előtanított, 1,5 milliárd paraméteres
GPT-2 élvonalbeli eredményeket ért el nyelvi modellezési benchmarkokon
és ígéretes eredményeket mutatott több más feladaton
*a paraméterek vagy az architektúra frissítése nélkül*.


### GPT-3 és Azon Túl

A GPT-2 bizonyította, hogy ugyanaz a nyelvi modell
felhasználható több feladathoz a modell frissítése nélkül.
Ez számítási szempontból hatékonyabb, mint a finomhangolás,
amely gradiens-számítással végzett modellfrissítéseket igényel.


![Zero-shot, one-shot, few-shot kontextusbeli tanulás nyelvi modellekkel (Transformer dekóderek). Nincs szükség paraméterfrissítésre.](../img/gpt-3-xshot.svg)
:label:`fig_gpt-3-xshot`

Mielőtt elmagyaráznánk a nyelvi modellek paraméterfrissítés nélküli,
számítási szempontból hatékonyabb felhasználását,
idézzük fel a :numref:`sec_rnn-scratch`-ből, hogy egy nyelvi modell
megtanítható szövegszekvencia generálására
valamilyen prefix szövegszekvenciától feltételesen.
Így egy előtanított nyelvi modell *paraméterfrissítés nélkül* is
képes a feladatkimenetet sorozatként generálni,
feltételezve egy bemeneti szekvenciát a feladatleírással,
feladatspecifikus bemenet–kimenet példákkal és egy prompttal (feladatbemenet).
Ezt a tanulási paradigmát *kontextusbeli tanulásnak* nevezzük :cite:`brown2020language`,
amely tovább bontható
*zero-shot*, *one-shot* és *few-shot* kategóriákra,
amikor nincs, egy vagy néhány feladatspecifikus bemenet–kimenet példa áll rendelkezésre (:numref:`fig_gpt-3-xshot`).


![A GPT-3 összesített teljesítménye mind a 42 pontossággal mért benchmarkon (felirat adaptálva, ábra forrása: :citet:`brown2020language`).](../img/gpt3-xshot-scaling.png)
:width:`400px`
:label:`fig_gpt3-xshot-scaling`

Ezt a három beállítást a GPT-3-ban tesztelték :cite:`brown2020language`,
amelynek legnagyobb változata körülbelül két nagyságrenddel nagyobb
adatot és modellméretet használ, mint a GPT-2.
A GPT-3 ugyanazt a Transformer dekóder architektúrát alkalmazza,
mint közvetlen elődje, a GPT-2,
azzal a különbséggel, hogy a figyelemi minták
(:numref:`fig_gpt-decoder-only` jobb oldala)
ritkábbak a váltakozó rétegekben.
300 milliárd tokennel előtanítva,
a GPT-3 nagyobb modellmérettel jobban teljesít,
ahol a few-shot teljesítmény növekszik a leggyorsabban (:numref:`fig_gpt3-xshot-scaling`).

A rákövetkező GPT-4 modell nem tárta fel teljes mértékben a műszaki részleteket a jelentésében :cite:`openai2023gpt4`.
Elődjeitől eltérően a GPT-4
egy nagy méretű, multimodális modell,
amely szöveget és képeket egyaránt fogad bemenetként,
és szöveges kimenetet generál.


## Skálázhatóság

A :numref:`fig_gpt3-xshot-scaling` empirikusan bizonyítja
a Transformerek skálázhatóságát a GPT-3 nyelvi modellben.
A nyelvi modellezés terén a Transformerek skálázhatóságára vonatkozó
átfogóbb empirikus vizsgálatok arra ösztönözték a kutatókat,
hogy több adattal és számítási kapacitással nagyobb Transformereket tanítsanak :cite:`kaplan2020scaling`.

![A Transformer nyelvi modellek teljesítménye egyenletesen javul a modellméret, az adathalmaz mérete és a tanításhoz felhasznált számítási mennyiség növelésével. Az optimális teljesítményhez mind a három tényezőt együttesen kell növelni. Az empirikus teljesítmény hatványtörvény-kapcsolatban áll az egyes tényezőkkel, amikor a másik kettő nem jelent szűk keresztmetszetet (felirat adaptálva, ábra forrása: :citet:`kaplan2020scaling`).](../img/scaling-power-law.png)
:width:`700px`
:label:`fig_scaling-power-law3`

Ahogy a :numref:`fig_scaling-power-law3` mutatja,
*hatványtörvény-skálázás* figyelhető meg a teljesítményben
a modellméret (paraméterek száma, beágyazási rétegek nélkül),
az adathalmaz mérete (tanítási tokenek száma)
és a tanítási számítási mennyiség (PetaFLOP/s-napok, beágyazási rétegek nélkül) tekintetében.
Általánosságban elmondható, hogy mind a három tényező együttes növelése jobb teljesítményhez vezet.
Azonban *hogyan* kell őket együttesen növelni,
ez még vita tárgya :cite:`hoffmann2022training`.

![Transformer nyelvi modellek tanítási futtatásai (ábra forrása: :citet:`kaplan2020scaling`).](../img/scaling-sample-conv.png)
:width:`700px`
:label:`fig_scaling-sample-conv`

A jobb teljesítmény mellett a nagy modellek jobb mintahatékonyságot is élveznek, mint a kis modellek. A :numref:`fig_scaling-sample-conv` megmutatja, hogy a nagy modellek kevesebb tanítási mintával (feldolgozott tokenekkel) képesek ugyanolyan szinten teljesíteni, mint a kis modellek, és a teljesítmény egyenletesen skálázódik a számítási kapacitással.



![A GPT-3 teljesítménye (keresztentrópia validációs veszteség) hatványtörvény-trendet követ a tanításhoz felhasznált számítási mennyiséggel. A :citet:`kaplan2020scaling`-ban megfigyelt hatványtörvény-viselkedés további két nagyságrenddel folytatódik, az előrejelzett görbétől csak kis eltérésekkel. A beágyazási paraméterek ki vannak zárva a számítási és paraméterszámokból (felirat adaptálva, ábra forrása: :citet:`brown2020language`).](../img/scaling-gpt3.png)
:width:`250px`
:label:`fig_scaling-gpt3`


A :citet:`kaplan2020scaling`-ban leírt empirikus skálázási viselkedéseket a rákövetkező nagy Transformer modellekben tesztelték. Például a GPT-3 két további nagyságrenddel támogatta ezt a hipotézist a :numref:`fig_scaling-gpt3`-ban.





## Nagy Nyelvi Modellek

A Transformerek skálázhatósága a GPT-sorozatban inspirálta a rákövetkező nagy nyelvi modelleket.
A GPT-2 Transformer dekódert a 270 milliárd tanítási tokennel rendelkező 530 milliárd paraméteres Megatron-Turing NLG tanítására használták :cite:`smith2022using`. A GPT-2 tervezetét követve a 300 milliárd tokennel előtanított, 280 milliárd paraméteres Gopher :cite:`rae2021scaling` versenyképesen teljesített különböző feladatokon.
A Gopher azonos architektúráját örökölve és ugyanakkora számítási keretet felhasználva a Chinchilla :cite:`hoffmann2022training` egy lényegesen kisebb (70 milliárd paraméteres) modell, amely sokkal tovább tanul (1,4 trillió tanítási token), felülmúlja a Gophert sok feladaton, és nagyobb hangsúlyt fektet a tokenek számára, mint a paraméterek számára.
A nyelvi modellezés skálázási vonalának folytatásaként
a PaLM (Pathway Language Model) :cite:`chowdhery2022palm`, egy módosított tervezetű, 780 milliárd tokennel előtanított 540 milliárd paraméteres Transformer dekóder, felülmúlta az átlagos emberi teljesítményt a BIG-Bench benchmarkon :cite:`srivastava2022beyond`. Későbbi változata, a PaLM 2 :cite:`anil2023palm`, az adatokat és a modellt nagyjából 1:1 arányban skálázta, és javította a többnyelvű és érvelési képességeket.
Más nagy nyelvi modellek, mint a Minerva :cite:`lewkowycz2022solving`, amely egy általános modellt (PaLM) tanít tovább, és a Galactica :cite:`taylor2022galactica`, amely nem általános korpuszon van tanítva, ígéretes kvantitatív és tudományos érvelési képességeket mutattak.


A nyílt forráskódú kiadások, mint az OPT (Open Pretrained Transformers) :cite:`zhang2022opt`, a BLOOM :cite:` scao2022bloom` és a FALCON :cite:`penedo2023refinedweb`,
demokratizálták a nagy nyelvi modellek kutatását és használatát.
Az inferencia idejű számítási hatékonyságra összpontosítva
a nyílt forráskódú Llama 1 :cite:`touvron2023llama` felülmúlta a sokkal nagyobb modelleket azzal, hogy a szokásosnál több tokenre tanítottak. A frissített Llama 2 :cite:`touvron2023llama2` tovább növelte az előtanítási korpuszt 40%-kal, ami versenyképes zárt forráskódú modellekéhez hasonló teljesítményű termékmodellekhez vezetett.



:citet:`wei2022emergent` tárgyalta a nagy nyelvi modellek feltörekvő képességeit, amelyek a nagyobb modellekben jelen vannak, a kisebbiekben viszont nem.
Ugyanakkor egyszerűen a modellméret növelése önmagában nem teszi a modelleket jobban követővé az emberi utasításoknak.
:citet:`wei2021finetuned,sanh2021multitask` megállapította, hogy a nagy nyelvi modellek
*utasításokon* keresztül leírt adathalmazok sorozatán való finomhangolása
javíthatja a zero-shot teljesítményt a kihagyott feladatokon.
Az *emberi visszajelzésből történő megerősítéses tanulást* alkalmazva
:citet:`ouyang2022training` finomhangolta a GPT-3-t
hogy sokféle utasítást kövessen.
Az ezt követő InstructGPT nyomán, amely
finomhangolással hangolja össze a nyelvi modelleket az emberi szándékkal :cite:`ouyang2022training`,
a [ChatGPT](https://chat.openai.com/)
emberszerű válaszokat képes generálni (pl. kód hibakeresés és kreatív írás)
az emberekkel folytatott beszélgetések alapján,
és számos természetes nyelvfeldolgozási feladatot tud elvégezni
zero-shot módban :cite:`qin2023chatgpt`.
:citet:`bai2022constitutional` az emberi bemeneteket (pl. ember által felcímkézett adatokat) modellkimenetekkel váltotta fel
az utasítások hangolási folyamatának részleges automatizálásához, amelyet *AI-visszajelzésből történő megerősítéses tanulásnak* is neveznek.


A nagy nyelvi modellek izgalmas lehetőséget kínálnak
arra, hogy szöveges bemenetek megfogalmazásával rávegyék a modelleket a kívánt feladatok elvégzésére kontextusbeli tanulás révén,
amelyet *promptolásnak* is neveznek.
Különösen a *gondolatlánc-promptolás* :cite:`wei2022chain`,
egy kontextusbeli tanulási módszer
few-shot „kérdés, közbenső érvelési lépések, válasz" bemutatókkal,
előhívja a nagy nyelvi modellek
összetett érvelési képességeit
matematikai, józan ésszel és szimbolikus érvelési feladatok megoldása érdekében.
Több érvelési útvonal mintavételezése :cite:`wang2023self`, a few-shot bemutatók változatosítása :cite:`zhang2023automatic`,
és az összetett problémák részproblémákra bontása :cite:`zhou2023least`
mind javíthatja az érvelési pontosságot. Valójában egyszerű promptokkal, mint például „Gondolkodjunk lépésről lépésre" minden válasz előtt,
a nagy nyelvi modellek akár *zero-shot*
gondolatlánc-érvelést is képesek elvégezni elfogadható pontossággal :cite:`kojima2022large`.
Még szöveget és képeket egyaránt tartalmazó multimodális bemenetek esetén is
a nyelvi modellek magasabb pontossággal képesek multimodális gondolatlánc-érvelést végezni, mint csak szöveges bemenet használatával :cite:`zhang2023multicot`.




## Összefoglalás és Megvitatás

A Transformereket csupán kódolóként (pl. BERT), kódoló–dekóderként (pl. T5) és csupán dekóderként (pl. GPT-sorozat) tanítottak elő. Az előtanított modellek adaptálhatók különböző feladatok elvégzésére modellfrissítéssel (pl. finomhangolás) vagy anélkül (pl. few-shot). A Transformerek skálázhatósága azt sugallja, hogy a jobb teljesítmény nagyobb modellekből, több tanítási adatból és több tanítási számítási kapacitásból profitál. Mivel a Transformereket eredetileg szöveges adatokra tervezték és tanítottak elő, ez a fejezet kissé a természetes nyelvfeldolgozás felé hajlik. Mindazonáltal a fent tárgyalt modellek gyakran megtalálhatók a legújabb, több modalitást lefedő modellekben. Például:
(i) a Chinchilla :cite:`hoffmann2022training` tovább bővült Flamingóvá :cite:`alayrac2022flamingo`, amely egy vizuális nyelvi modell few-shot tanuláshoz;
(ii) a GPT-2 :cite:`Radford.Wu.Child.ea.2019` és a látási Transformer szöveget és képeket kódol a CLIP-ben (Contrastive Language-Image Pre-training) :cite:`radford2021learning`, amelynek kép- és szövegembedingjeit később a DALL-E 2 szöveg-képpé alakító rendszerben alkalmazták :cite:`ramesh2022hierarchical`. Bár a Transformer skálázhatóságáról multimodális előtanítás terén még nem készültek szisztematikus vizsgálatok, egy Parti nevű, teljes egészében Transformer alapú szöveg-képpé alakító modell :cite:`yu2022scaling` skálázhatósági potenciált mutat a különböző modalitások között:
a nagyobb Parti képesebb nagy hűségű képgenerálásra és tartalomgazdag szövegmegértésre (:numref:`fig_parti`).


![Képpéldák, amelyeket a Parti modell növekvő méretű változatai (350M, 750M, 3B, 20B) generáltak ugyanabból a szövegből (példák forrása: :citet:`yu2022scaling`).](../img/parti.png)
:width:`700px`
:label:`fig_parti`




## Gyakorlatok

1. Lehetséges-e a T5 finomhangolása különböző feladatokból álló minibatch segítségével? Miért igen vagy miért nem? Mi a helyzet a GPT-2 esetében?
1. Egy hatékony nyelvi modell adottságával milyen alkalmazásokra gondolhat?
1. Tegyük fel, hogy megkérnek egy nyelvi modell finomhangolására szövegosztályozás elvégzéséhez további rétegek hozzáadásával. Hová adnád hozzá őket? Miért?
1. Vegyük figyelembe a szekvencia-szekvencia problémákat (pl. gépi fordítás), ahol a bemeneti szekvencia mindig elérhető a célszekvencia előrejelzése során. Milyen korlátai lehetnek a csupán dekóderes Transformerekkel való modellezésnek? Miért?


[Discussions](https://discuss.d2l.ai/t/9232)
