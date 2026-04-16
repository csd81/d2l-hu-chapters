# Rekurrens Neurális Hálózatok
:label:`chap_rnn`

Eddig elsősorban rögzített hosszúságú adatokra összpontosítottunk.
Amikor a lineáris és logisztikus regressziót bevezettük
a :numref:`chap_regression` és :numref:`chap_classification` fejezetekben,
illetve a többrétegű perceptronokat a :numref:`chap_perceptrons` fejezetben,
örömmel feltételeztük, hogy minden $\mathbf{x}_i$ jellemzővektor
rögzített számú $x_1, \dots, x_d$ komponensből áll,
ahol minden $x_j$ numerikus jellemző
egy adott attribútumnak felel meg.
Ezeket az adathalmazokat néha *táblázatos* adatoknak nevezzük,
mert táblázatokba rendezhetők,
ahol minden $i$ példa saját sort kap,
és minden attribútum saját oszlopot.
Ami lényeges: táblázatos adatok esetén ritkán
feltételezünk bármilyen különleges struktúrát az oszlopok felett.

Ezt követően a :numref:`chap_cnn` fejezetben
képadatokra tértünk át, ahol a bemenetek
egy kép minden koordinátájában lévő nyers pixelértékekből állnak.
A képadatok nem illeszkedtek különösebben
a tipikus táblázatos adathalmaz képéhez.
Ott konvolúciós neurális hálózatokat (CNN-eket) kellett alkalmaznunk
a hierarchikus struktúra és az invarianciák kezelésére.
Az adataink azonban még mindig rögzített hosszúságúak voltak.
Minden Fashion-MNIST képet
$28 \times 28$ méretű pixelérték-rácsként ábrázolunk.
Sőt, célunk egy olyan modell fejlesztése volt,
amely egyetlen képet néz meg, majd
egyetlen előrejelzést ad ki.
De mit tegyünk, ha képek sorozatával szembesülünk,
mint egy videóban, vagy ha sorozatosan strukturált előrejelzést
kell előállítanunk, mint képfeliratozás esetén?

Számos tanulási feladat megköveteli a szekvenciális adatok kezelését.
A képfeliratozás, a beszédszintézis és a zenegenerálás
mind azt igényli, hogy a modellek sorozatokból álló kimeneteket állítsanak elő.
Más területeken, például idősor-előrejelzésnél,
videóelemzésnél és zenei információ-visszakeresésnél,
a modellnek sorozatokból álló bemenetekből kell tanulnia.
Ezek az igények gyakran egyidejűleg merülnek fel:
olyan feladatok, mint szövegrészletek fordítása
egyik természetes nyelvről a másikra,
párbeszéd folytatása vagy robot vezérlése,
megkövetelik, hogy a modellek mind befogadjanak,
mind kibocsássanak sorozatosan strukturált adatokat.


A rekurrens neurális hálózatok (RNN-ek) olyan mélytanulás modellek,
amelyek sorozatok dinamikáját ragadják meg
*rekurrens* kapcsolatokon keresztül, amelyek
a csomópontok hálózatában ciklusoknak tekinthetők.
Ez elsőre ellentmondásosnak tűnhet.
Végül is a neurális hálózatok előrecsatolt jellege
teszi a számítás sorrendjét egyértelművé.
A rekurrens élek azonban pontosan úgy vannak meghatározva,
hogy ilyen kétértelműség ne merülhessen fel.
A rekurrens neurális hálózatokat az időlépések (vagy sorozatlépések) mentén *kitekerve*,
minden lépésnél *ugyanazokat* az alapparamétereket alkalmazva ábrázolják.
Míg a szokásos kapcsolatok *szinkron* módon alkalmazzák
az egyes rétegek aktivációit
a következő rétegre *ugyanabban az időlépésben*,
a rekurrens kapcsolatok *dinamikusak*,
információt adva át szomszédos időlépések között.
Ahogy a :numref:`fig_unfolded-rnn` kitekerített nézete feltárja,
az RNN-ek előrecsatolt neurális hálózatoknak tekinthetők,
ahol minden réteg paraméterei (mind a szokásos, mind a rekurrens)
megosztottak az időlépések között.


![Bal oldalon a rekurrens kapcsolatok ciklikus éleken keresztül vannak ábrázolva. Jobb oldalon az RNN-t időlépések mentén tekerjük ki. Itt a rekurrens élek szomszédos időlépések között feszülnek, míg a szokásos kapcsolatok szinkron módon kerülnek kiszámításra.](../img/unfolded-rnn.svg)
:label:`fig_unfolded-rnn`


Ahogy a neurális hálózatok általában,
az RNN-eknek is hosszú, több tudományterületet átfogó történelmük van;
az agy modelljeiként indultak, amelyeket
kognitív tudósok terjesztettek el, majd
a gépi tanulás közössége is átvette őket
praktikus modellezési eszközként.
Ahogy a mélytanulás esetében általában,
ebben a könyvben a gépi tanulás perspektíváját alkalmazzuk,
az RNN-ekre mint praktikus eszközökre összpontosítva, amelyek
a 2010-es évekbeli népszerűségüket
áttörő eredményeiknek köszönhetik
olyan változatos feladatokon, mint a kézírás-felismerés :cite:`graves2008novel`,
a gépi fordítás :cite:`Sutskever.Vinyals.Le.2014`,
és az orvosi diagnózisok felismerése :cite:`Lipton.Kale.2016`.
Az olvasót, aki bővebb háttéranyag iránt érdeklődik,
egy nyilvánosan elérhető átfogó összefoglalóhoz irányítjuk :cite:`Lipton.Berkowitz.Elkan.2015`.
Azt is megjegyezzük, hogy a szekvencialitás nem csak az RNN-ekre jellemző.
Például a már bemutatott CNN-ek
is adaptálhatók változó hosszúságú adatok kezelésére,
pl. változó felbontású képek esetén.
Sőt, az RNN-ek az utóbbi időben jelentős
piaci részesedést engedtek át a Transformer modelleknek,
amelyekre a :numref:`chap_attention-and-transformers` fejezetben térünk ki.
Az RNN-ek azonban a mélytanulásban a komplex szekvenciális struktúra
kezelésének alapértelmezett modelljeiként kerültek előtérbe,
és a mai napig alap modelleknek számítanak a szekvenciális modellezésben.
Az RNN-ek és a sorozatmodellezés történetei
elválaszthatatlanul összefonódnak, és ez annyira
a sorozatmodellezési problémák alapjairól szóló fejezet,
mint amennyire az RNN-ekről szól.


Egy kulcsfontosságú felismerés egyengette az utat a sorozatmodellezés forradalma felé.
Bár a gépi tanulás számos alapvető feladatának bemenetei és céljai
nem ábrázolhatók könnyen rögzített hosszúságú vektorokként,
mégis gyakran ábrázolhatók
rögzített hosszúságú vektorok változó hosszúságú sorozataiként.
Például a dokumentumok szavak sorozataként ábrázolhatók;
az orvosi nyilvántartások gyakran eseménysorozatokként ábrázolhatók
(találkozók, gyógyszerek, beavatkozások, laboratóriumi tesztek, diagnózisok);
a videók változó hosszúságú álló képsorozatokként ábrázolhatók.


Bár a sorozatmodellek számos alkalmazási területen megjelentek,
az ezen a területen folyó alapkutatást elsősorban
a természetes nyelvfeldolgozás alapfeladatain elért fejlődés hajtja.
Ezért ebben a fejezetben
bemutatásunkat és példáinkat szöveges adatokra összpontosítjuk.
Ha sikerül megértened ezeket a példákat,
akkor a modellek más adatmodalitásokra való alkalmazása
viszonylag egyszerű lesz.
A következő néhány szakaszban bemutatjuk
a sorozatok alapvető jelöléseit és néhány értékelési mértéket
a sorozatosan strukturált modellkimenetek minőségének értékelésére.
Ezt követően tárgyaljuk a nyelvmodell alapvető fogalmait,
és ezt a tárgyalást használjuk az első RNN modelljeink motiválásához.
Végül leírjuk a gradiensek kiszámítási módszerét
az RNN-eken való visszaterjesztés során, és feltárjuk néhány kihívást,
amelyekkel az ilyen hálózatok tanítása során gyakran találkozunk,
motiválva a modern RNN architektúrákat, amelyek
a :numref:`chap_modern_rnn` fejezetben követnek majd.

```toc
:maxdepth: 2

sequence
text-sequence
language-model
rnn
rnn-scratch
rnn-concise
bptt
```

