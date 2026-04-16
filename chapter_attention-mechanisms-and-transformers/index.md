# Figyelemmechanizmusok és transformerek
:label:`chap_attention-and-transformers`


A mélytanulás fellendülésének korai éveit elsősorban
a többrétegű perceptron, a konvolúciós hálózat
és a rekurrens hálózati architektúrák eredményei hajtották.
Figyelemre méltó, hogy azok a modell-architektúrák, amelyek
a mélytanulás 2010-es évekbeli áttöréseit megalapozták,
szinte alig változtak elődjeikhez képest
annak ellenére, hogy közel 30 év telt el.
Bár számos módszertani újítás
bekerült a legtöbb szakember eszköztárába – mint a ReLU
aktiváció, a maradék rétegek, a batch normalizáció, a dropout
és az adaptív tanulási ráta ütemezések –, az alapul szolgáló
architektúrák egyértelműen felismerhetőek voltak
mint klasszikus ötletek felskálázott megvalósításai.
Annak ellenére, hogy ezrével születtek alternatív ötleteket javasló cikkek,
a klasszikus konvolúciós neurális hálózatokhoz hasonló modellek (:numref:`chap_cnn`)
megőrizték a *legkorszerűbb* státuszukat a számítógépes látásban,
és a Sepp Hochreiter által tervezett eredeti LSTM rekurrens neurális hálózathoz
(:numref:`sec_lstm`) hasonló modellek uralták a természetes nyelvfeldolgozás
legtöbb alkalmazását.
Vitathatatlanul, addig a pontig a mélytanulás gyors fejlődése
elsősorban az elérhető számítási erőforrások változásának
volt betudható (a GPU-kkal való párhuzamos számítás terén elért
innovációknak köszönhetően) és a tömeges adatforrások
elérhetőségének (az olcsó tárolás és az internetes szolgáltatások jóvoltából).
Bár ezek a tényezők valóban továbbra is az elsődleges
hajtóerők lehetnek e technológia növekvő ereje mögött,
végre tanúi vagyunk az uralkodó architektúrák
tájképének gyökeres megváltozásának is.

Jelenleg szinte minden természetes nyelvfeldolgozási feladathoz
a Transformer architektúrán alapuló modellek a meghatározók.
Bármely új természetes nyelvfeldolgozási feladat esetén
az alapértelmezett első megközelítés az, hogy fogunk egy nagy,
Transformer-alapú előre tanított modellt
(pl. BERT :cite:`Devlin.Chang.Lee.ea.2018`, ELECTRA :cite:`clark2019electra`, RoBERTa :cite:`Liu.Ott.Goyal.ea.2019`, vagy Longformer :cite:`beltagy2020longformer`),
szükség szerint módosítjuk a kimeneti rétegeket,
és finomhangoljuk a modellt a rendelkezésre álló
adatokon a downstream feladathoz.
Ha figyelemmel követted az elmúlt néhány évben
az OpenAI nagy nyelvi modelljeiről szóló
lázas hírközlést, akkor nyomon követted
a GPT-2 és GPT-3 Transformer-alapú modellekről
:cite:`Radford.Wu.Child.ea.2019,brown2020language` szóló vitát.
Eközben a vision Transformer vált a különféle látási feladatok
alapértelmezett modelljévé,
beleértve a képfelismerést, az objektumdetektálást,
a szemantikai szegmentálást és a szuperfelbontást :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,liu2021swin`.
A Transformerek versenyképes megoldásként is megjelentek
a beszédfelismerésben :cite:`gulati2020conformer`,
a megerősítéses tanulásban :cite:`chen2021decision`
és a gráf neurális hálózatokban :cite:`dwivedi2020generalization`.

A Transformer modell mögötti alapötlet a *figyelemmechanizmus*,
egy olyan innováció, amelyet eredetileg
a kódoló–dekódoló RNN-ek fejlesztéseként képzeltek el,
szekvencia-szekvencia alkalmazásokhoz,
mint például a gépi fordítás :cite:`Bahdanau.Cho.Bengio.2014`.
Talán emlékszel, hogy az első szekvencia-szekvencia modellekben
a gépi fordításhoz :cite:`Sutskever.Vinyals.Le.2014`,
a teljes bemenetet a kódoló egy egyetlen rögzített hosszúságú vektorba tömörítette,
amelyet aztán a dekódolóba tápláltak.
A figyelem mögötti intuíció az, hogy a bemenet tömörítése helyett
jobb lehet, ha a dekódoló minden lépésnél visszatér a bemeneti szekvenciához.
Továbbá, ahelyett, hogy mindig ugyanazt a bemeneti reprezentációt látná,
elképzelhető, hogy a dekódolónak szelektíven kellene fókuszálnia
a bemeneti szekvencia egyes részeire az egyes dekódolási lépéseknél.
Bahdanau figyelemmechanizmusa egyszerű eszközt biztosított
arra, hogy a dekódoló dinamikusan *odafigyeljen* a bemenet
különböző részeire minden dekódolási lépésben.
A magas szintű ötlet az, hogy a kódoló olyan reprezentációt képes előállítani,
amelynek hossza megegyezik az eredeti bemeneti szekvenciával.
Ezután dekódoláskor a dekódoló (valamilyen vezérlőmechanizmuson keresztül)
bemeneteként megkap egy kontextusvektort, amely a bemeneten
szereplő reprezentációk súlyozott összegéből áll minden időlépésnél.
Intuitívan a súlyok határozzák meg,
hogy az egyes lépések kontextusa mennyire „fókuszál" az egyes bemeneti tokenekre,
és a kulcs az, hogy ezt a súly-hozzárendelési folyamatot
differenciálhatóvá tegyük,
hogy megtanítható legyen az összes többi neurális hálózati paraméterrel együtt.

Kezdetben ez az ötlet figyelemre méltóan sikeres fejlesztése volt
a rekurrens neurális hálózatoknak,
amelyek már uralták a gépi fordítási alkalmazásokat.
A modellek jobban teljesítettek, mint az eredeti
kódoló–dekódoló szekvencia-szekvencia architektúrák.
Ezenkívül a kutatók észrevették, hogy a figyelemsúlyok mintázatának vizsgálatából
néha szép minőségi betekintések nyerhetők.
Fordítási feladatokban a figyelemmodellek
gyakran magas figyelemsúlyokat rendeltek a nyelvközi szinonimákhoz,
amikor a célnyelvben a megfelelő szavakat generálták.
Például a „my feet hurt" mondat „j'ai mal au pieds"-re fordításakor
a neurális hálózat magas figyelemsúlyokat rendelhet
a „feet" reprezentációjához
a megfelelő francia szó, a „pieds" generálásakor.
Ezek a felismerések olyan állításokhoz vezettek,
hogy a figyelemmodellek „értelmezhetőséget" biztosítanak,
bár pontosan mit is jelentenek a figyelemsúlyok – azaz
hogyan és egyáltalán hogyan kell *értelmezni* őket –
homályos kutatási téma marad.

Azonban a figyelemmechanizmusok hamarosan fontosabb kérdésekként
kerültek előtérbe a kódoló–dekódoló rekurrens neurális hálózatok
fejlesztéseként való hasznosságukon és
a kiemelkedő bemenetek kiválasztásában vélt hasznosságukon túl.
:citet:`Vaswani.Shazeer.Parmar.ea.2017` javasolta
a Transformer architektúrát a gépi fordításhoz,
teljesen mellőzve a rekurrens kapcsolatokat,
és ehelyett ügyesen elrendezett figyelemmechanizmusokra támaszkodva
a bemeneti és kimeneti tokenek közötti összes kapcsolat megragadásához.
Az architektúra figyelemre méltóan jól teljesített,
és 2018-ra a Transformer megjelent
a legtöbb legkorszerűbb természetes nyelvfeldolgozási rendszerben.
Ezenkívül ugyanekkor a természetes nyelvfeldolgozás domináns
gyakorlata lett, hogy nagy léptékű modelleket előtanítsanak
hatalmas általános háttérkorpuszon
valamilyen önfelügyelt előtanítási célkitűzés optimalizálásához,
majd ezeket a modelleket finomhangolják
a rendelkezésre álló downstream adatokon.
A Transformerek és a hagyományos architektúrák közötti szakadék
különösen szélessé vált az előtanítási paradigmában alkalmazva,
így a Transformerek felemelkedése egybeesett
az ilyen nagy léptékű előtanított modellek felemelkedésével,
amelyeket most néha *alapmodelleknek* (*foundation models*) neveznek :cite:`bommasani2021opportunities`.


Ebben a fejezetben bemutatjuk a figyelemmodelleket,
a legalapvetőbb intuícióktól kezdve
és az ötlet legegyszerűbb megvalósításaitól.
Ezután fokozatosan haladunk fel a Transformer architektúráig,
a vision Transformerig és a modern
Transformer-alapú előtanított modellek világáig.

```toc
:maxdepth: 2

queries-keys-values
attention-pooling
attention-scoring-functions
bahdanau-attention
multihead-attention
self-attention-and-positional-encoding
transformer
vision-transformer
large-pretraining-transformers
```
