# Természetes nyelvfeldolgozás: előtanítás
:label:`chap_nlp_pretrain`


Az embereknek szükségük van a kommunikációra.
Ebből az alapvető emberi igényből fakadóan hatalmas mennyiségű írott szöveg keletkezik nap mint nap.
A közösségi médiában, csevegőalkalmazásokban, e-mailekben, termékértékelésekben, hírcikkekben, kutatási cikkekben és könyvekben gazdag szövegek találhatók, ezért létfontosságúvá vált, hogy a számítógépek meg tudják ezeket érteni, és emberi nyelvek alapján segítséget nyújtsanak vagy döntéseket hozzanak.

A *természetes nyelvfeldolgozás* a számítógépek és az emberek közötti, természetes nyelveken alapuló interakciókat vizsgálja.
A gyakorlatban nagyon elterjedt a természetes nyelvfeldolgozási technikák alkalmazása szöveges (emberi természetes nyelvű) adatok feldolgozására és elemzésére, például a :numref:`sec_language-model` fejezetben tárgyalt nyelvmodellek vagy a :numref:`sec_machine_translation` fejezetben bemutatott gépi fordítási modellek esetén.

A szöveg megértéséhez először érdemes megtanulni annak reprezentációit.
A nagy korpuszokból származó meglévő szövegsorozatokra támaszkodva
az *önfelügyelt tanulást* széles körben alkalmazzák szövegreprezentációk előtanítására,
például úgy, hogy a szöveg egy rejtett részét a szöveg többi részéből jósolják meg.
Ily módon a modellek *hatalmas* mennyiségű szöveges adatból tanulnak felügyelet alatt,
anélkül, hogy *költséges* annotálási munkára lenne szükség!


Ahogy ebben a fejezetben látni fogjuk,
ha minden szót vagy részszót egyedi tokenként kezelünk,
az egyes tokenek reprezentációja előtanítható
word2vec, GloVe vagy részszó-beágyazási modellekkel
nagy korpuszokon.
Az előtanítás után minden token reprezentációja egy vektor lesz,
ez azonban ugyanaz marad a kontextustól függetlenül.
Például a „bank" szó vektoros reprezentációja azonos
a „go to the bank to deposit some money"
és a „go to the bank to sit down" mondatokban is.
Ezért számos újabb előtanítási modell a kontextushoz igazítja az azonos tokenek reprezentációját.
Ezek közé tartozik a BERT, egy sokkal mélyebb, Transformer-enkóderre épülő önfelügyelt modell.
Ebben a fejezetben arra összpontosítunk, hogyan taníthatók elő ilyen szövegreprezentációk,
ahogy az :numref:`fig_nlp-map-pretrain` ábrán is látható.

![Az előtanított szövegreprezentációk különböző deep learning architektúrákba táplálhatók be, különböző downstream természetes nyelvfeldolgozási alkalmazásokhoz. Ez a fejezet az upstream szövegreprezentáció előtanítására összpontosít.](../img/nlp-map-pretrain.svg)
:label:`fig_nlp-map-pretrain`


A nagy kép áttekintéséhez
az :numref:`fig_nlp-map-pretrain` ábra megmutatja,
hogy az előtanított szövegreprezentációk számos különböző deep learning architektúrába táplálhatók be
különböző downstream természetes nyelvfeldolgozási alkalmazásokhoz.
Ezeket a :numref:`chap_nlp_app` fejezetben tárgyaljuk.

```toc
:maxdepth: 2

word2vec
approx-training
word-embedding-dataset
word2vec-pretraining
glove
subword-embedding
similarity-analogy
bert
bert-dataset
bert-pretraining

```

