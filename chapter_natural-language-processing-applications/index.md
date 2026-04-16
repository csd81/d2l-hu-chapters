# Természetes nyelvfeldolgozás: Alkalmazások
:label:`chap_nlp_app`

Láttuk, hogyan lehet szöveges sorozatokban a tokeneket reprezentálni, és hogyan lehet ezek reprezentációit betanítani a :numref:`chap_nlp_pretrain` fejezetben.
Az ilyen előtanított szöveges reprezentációkat különböző modellek bemeneteként lehet felhasználni különféle downstream természetes nyelvfeldolgozási feladatokhoz.

Valójában
a korábbi fejezetek már tárgyaltak néhány természetes nyelvfeldolgozási alkalmazást
*előtanítás nélkül*,
csupán a deep learning architektúrák szemléltetése céljából.
Például a :numref:`chap_rnn` fejezetben
RNN-eket alkalmaztunk nyelvi modellek tervezéséhez, amelyek novella-szerű szövegeket generálnak.
A :numref:`chap_modern_rnn` és a :numref:`chap_attention-and-transformers` fejezetekben
szintén terveztünk RNN-eken és figyelemmechanizmusokon alapuló modelleket gépi fordításhoz.

Azonban ez a könyv nem kívánja átfogó módon tárgyalni az összes ilyen alkalmazást.
Ehelyett
figyelmünk középpontjában az áll, hogy *hogyan alkalmazhatjuk a nyelvek (mély) reprezentációs tanulását természetes nyelvfeldolgozási problémák megoldásához*.
Az előtanított szöveges reprezentációkból kiindulva
ez a fejezet két
népszerű és reprezentatív
downstream természetes nyelvfeldolgozási feladatot vizsgál:
a szentimentelemzést és a természetes nyelvi következtetést,
amelyek egyedi szövegeket, illetve szövegpárok kapcsolatait elemzik.

![Az előtanított szöveges reprezentációk különböző deep learning architektúrák bemeneteként szolgálhatnak különféle downstream természetes nyelvfeldolgozási alkalmazásokhoz. Ez a fejezet arra összpontosít, hogyan tervezzünk modelleket különböző downstream természetes nyelvfeldolgozási alkalmazásokhoz.](../img/nlp-map-app.svg)
:label:`fig_nlp-map-app`

Ahogyan a :numref:`fig_nlp-map-app` ábrán látható,
ez a fejezet a természetes nyelvfeldolgozási modellek tervezésének alapvető ötleteinek bemutatásával foglalkozik, különböző típusú deep learning architektúrák, például MLP-k, CNN-ek, RNN-ek és figyelem alkalmazásával.
Bár az előtanított szöveges reprezentációk bármelyike kombinálható bármelyik architektúrával a :numref:`fig_nlp-map-app` ábra bármelyik alkalmazásához,
mi néhány reprezentatív kombinációt választunk ki.
Pontosabban, a szentimentelemzéshez az RNN-eken és CNN-eken alapuló népszerű architektúrákat vizsgáljuk.
A természetes nyelvi következtetéshez figyelemmechanizmusokat és MLP-ket választunk annak bemutatására, hogyan lehet szövegpárokat elemezni.
Végül bemutatjuk, hogyan lehet egy előtanított BERT modellt finomhangolni
természetes nyelvfeldolgozási alkalmazások széles köréhez,
például szekvenciaszinten (egyedi szöveg osztályozása és szövegpár osztályozása)
és tokenszinten (szövegcímkézés és kérdés-válasz feladatok).
Konkrét empirikus esetként
a BERT-et természetes nyelvi következtetésre fogjuk finomhangolni.

Ahogyan a :numref:`sec_bert` fejezetben bemutattuk,
a BERT minimális architektúraváltoztatást igényel
a természetes nyelvfeldolgozási alkalmazások széles köréhez.
Azonban ez az előny a downstream alkalmazásokhoz szükséges
nagy számú BERT paraméter finomhangolásának költségén jön.
Ha a tár- vagy számítási kapacitás korlátozott,
az MLP-kre, CNN-ekre, RNN-ekre és figyelemre épülő, gondosan tervezett modellek
megvalósíthatóbbak.
A következőkben a szentimentelemzési alkalmazással kezdünk,
és bemutatjuk az RNN-eken, illetve CNN-eken alapuló modelltervet.

```toc
:maxdepth: 2

sentiment-analysis-and-dataset
sentiment-analysis-rnn
sentiment-analysis-cnn
natural-language-inference-and-dataset
natural-language-inference-attention
finetuning-bert
natural-language-inference-bert
```
