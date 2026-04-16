# Általánosítás
:label:`sec_generalization_basics`

Képzeljünk el két főiskolai hallgatót, akik szorgalmasan
készülnek a záróvizsgájukra.
Általában ez a felkészülés abból áll,
hogy korábbi évek vizsgáit megoldva
gyakorolják és tesztelik képességeiket.
Ennek ellenére, a múltbeli vizsgákon való jó teljesítmény
nem garantálja, hogy akkor is kitűnnek, amikor igazán számít.
Képzeljük el például az egyik hallgatót, Rendkívüli Rékát,
akinek felkészülése kizárólag abból állt,
hogy memorizálta a korábbi évek
vizsgakérdéseire adott válaszokat.
Még ha Réka rendkívüli memóriával is bírna,
és így tökéletesen emlékezne
bármely *korábban látott* kérdés válaszára,
talán mégis megmerevedne,
ha egy új (*korábban nem látott*) kérdéssel szembesülne.
Ezzel szemben képzeljük el a másik hallgatót,
Induktív Irmát, aki összehasonlíthatóan gyenge
memorizálási képességekkel bír,
de van érzéke a minták felismeréséhez.
Megjegyezzük, hogy ha a vizsga valóban
előző évi újrahasznosított kérdésekből állna,
Réka könnyedén felülmúlná Irmát.
Még ha Irma következtetett mintái
90%-os pontosságú előrejelzéseket is adnának,
soha nem versenyezhetnének
Réka 100%-os visszahívásával.
Azonban még ha a vizsga kizárólag
új kérdésekből is állna,
Irma fenntarthatná 90%-os átlagát.

Gépi tanulás kutatóiként
célunk a *minták felfedezése*.
De hogyan lehetünk biztosak abban, hogy valóban
egy *általános* mintát fedeztünk fel,
és nem csupán memorizáltuk az adatainkat?
Az esetek többségében az előrejelzéseink csak akkor hasznosak,
ha a modellünk ilyen mintát fedez fel.
Nem a tegnapi részvényárakat szeretnénk megjósolni, hanem a holnapiakat.
Nincs szükségünk arra, hogy felismerjük
a már korábban diagnosztizált betegségeket
korábban látott betegeknél,
hanem inkább a korábban nem diagnosztizált
betegségeket korábban nem látott betegeknél.
Ez a probléma — hogyan fedezzük fel az *általánosítható* mintákat —
a gépi tanulás alapvető problémája,
és vitathatatlanul az egész statisztikáé is.
Ezt a problémát egyszerűen egy sokkal nagyobb kérdés
szeleteként foghatjuk fel,
amely az egész tudományt átfogja:
mikor vagyunk jogosultak
a konkrét megfigyelésektől
az általánosabb állításokig ugrani?


A való életben véges adatgyűjtemény segítségével kell illeszteni a modelleinket.
Az adatok tipikus méretrendjei
területenként nagyon eltérőek.
Sok fontos orvosi problémánál
csak néhány ezer adatponthoz férünk hozzá.
Ritka betegségek vizsgálatánál
szerencsésnek tarthatjuk magunkat, ha néhány száz adathoz férünk hozzá.
Ezzel szemben a legnagyobb nyilvános adathalmazok,
amelyek címkézett fényképekből állnak,
pl. ImageNet :cite:`Deng.Dong.Socher.ea.2009`,
millió képet tartalmaznak.
Néhány címkézetlen képgyűjtemény
mint a Flickr YFC100M adathalmaz
még ennél is nagyobb lehet, több mint
100 millió képet tartalmazva :cite:`thomee2016yfcc100m`.
Azonban még ilyen extrém méretarányban is
a rendelkezésre álló adatpontok száma
végtelenül kicsi marad
megapixeles felbontású összes lehetséges kép
terének méretéhez képest.
Valahányszor véges mintákkal dolgozunk,
szem előtt kell tartanunk azt a kockázatot,
hogy illeszthetjük a tanítási adatainkat,
csak hogy aztán rájöjjünk: nem sikerült
általánosítható mintát felfedezni.

A tanítási adatokhoz való szorosabb illeszkedés jelensége
az alap eloszláshoz képest *túlillesztés* (overfitting) névvel ismert,
és a túlillesztés elleni technikákat
gyakran *regularizációs* módszereknek nevezzük.
Bár ez nem pótol egy megfelelő bevezetést
a statisztikai tanulás elméletébe (lásd :citet:`Vapnik98,boucheron2005theory`),
elegendő intuíciót nyújtunk az induláshoz.
Az általánosításhoz a könyv sok fejezetében visszatérünk,
feltárva mind azt, ami az általánosítás alapelveiről
ismert a különböző modellekben,
mind a heurisztikus technikákat,
amelyekről (empirikusan) megállapítottuk,
hogy javított általánosítást eredményeznek
a gyakorlatilag releváns feladatoknál.



## Tanítási hiba és általánosítási hiba


A standard felügyelt tanulási beállításban
feltételezzük, hogy a tanítási adatok és a tesztelési adatok
*egymástól független*, *azonos* eloszlásokból vannak vetve.
Ezt általában *FHA feltételnek* (Független, Homogén eloszlású Adatok; angolul IID — independently and identically distributed) nevezik.
Bár ez a feltételezés erős,
érdemes megjegyezni, hogy ilyen feltételezés hiányában
teljesen elvesznek a lehetőségeink.
Miért kellene hinnünk, hogy a $P(X,Y)$ eloszlásból
vett tanítási adatok
megmondják nekünk, hogyan tegyünk előrejelzéseket
egy *eltérő $Q(X,Y)$ eloszlás* által generált tesztelési adatokon?
Az ilyen ugrások megtételéhez
erős feltételezések szükségesek arról, hogyan kapcsolódnak $P$ és $Q$ egymáshoz.
Később tárgyalunk néhány feltételezést,
amelyek lehetővé teszik az eloszlás megváltoztatását,
de először meg kell értenünk a FHA esetet,
ahol $P(\cdot) = Q(\cdot)$.

Kezdetnek különbséget kell tennünk
a *tanítási hiba* $R_\textrm{emp}$ között,
amely egy *statisztika*,
amelyet a tanítási adathalmazon számítunk,
és az *általánosítási hiba* $R$ között,
amely egy *várható érték*,
amelyet az alap eloszlásra vonatkozóan veszünk.
Az általánosítási hibát úgy képzelhetjük el,
mint amit akkor látnánk, ha a modellünket
egy végtelen folyamú, az alap adateloszlásból
vett újabb adatpéldányra alkalmaznánk.
Formálisan a tanítási hiba *összegként* fejezhető ki (ugyanolyan jelöléssel, mint a :numref:`sec_linear_regression` részben):

$$R_\textrm{emp}[\mathbf{X}, \mathbf{y}, f] = \frac{1}{n} \sum_{i=1}^n l(\mathbf{x}^{(i)}, y^{(i)}, f(\mathbf{x}^{(i)})),$$


míg az általánosítási hiba integrálként fejezhető ki:

$$R[p, f] = E_{(\mathbf{x}, y) \sim P} [l(\mathbf{x}, y, f(\mathbf{x}))] =
\int \int l(\mathbf{x}, y, f(\mathbf{x})) p(\mathbf{x}, y) \;d\mathbf{x} dy.$$

Problematikusan, soha nem tudjuk pontosan kiszámítani
az általánosítási hibát $R$.
Senki sem mondja meg nekünk a $p(\mathbf{x}, y)$ sűrűségfüggvény pontos alakját.
Ezenkívül nem tudunk végtelen adatpontot mintavételezni.
Ezért a gyakorlatban *becsülnünk* kell az általánosítási hibát
azáltal, hogy a modellünket egy független teszthalmazra alkalmazzuk,
amely a tanítóhalmazból visszatartott
$\mathbf{X}'$ példányok és $\mathbf{y}'$ címkék
véletlenszerű kiválasztásából áll.
Ez ugyanazon képlet alkalmazásából áll,
amelyet az empirikus tanítási hiba kiszámításához használtunk,
de egy $\mathbf{X}', \mathbf{y}'$ teszthalmazra.


Döntő fontosságú, hogy amikor az osztályozónkat a teszthalmazon értékeljük,
egy *rögzített* osztályozóval dolgozunk
(amely nem függ a teszthalmaz mintájától),
és így a hibájának becslése
egyszerűen az átlag becslésének problémája.
Azonban ugyanez nem mondható el
a tanítóhalmazra.
Vegyük észre, hogy az általunk kapott modell
explicit módon függ a tanítóhalmaz kiválasztásától,
és így a tanítási hiba általában
egy elfogult becslés lesz az alap populáción
valódi hibára vonatkozóan.
Az általánosítás központi kérdése tehát az,
hogy mikor várható el, hogy tanítási hibánk
közel legyen a populáció hibájához
(és ezáltal az általánosítási hibához).

### Modell-komplexitás

A klasszikus elméletben, amikor
egyszerű modelljeink és bőséges adatunk van,
a tanítási és általánosítási hibák közel esnek egymáshoz.
Azonban, amikor összetettebb modellekkel és/vagy kevesebb példánnyal dolgozunk,
elvárjuk, hogy a tanítási hiba csökkenjen,
de az általánosítási rés nőjön.
Ez nem meglepő.
Képzeljünk el egy olyan modellosztályt, amely annyira kifejező, hogy
bármely $n$ példányból álló adathalmazhoz
találhatunk egy paraméterkészletet,
amely tökéletesen illeszkedik tetszőleges címkékre,
még ha azokat véletlenszerűen rendelték is hozzá.
Ebben az esetben, még ha tökéletesen illesztjük is a tanítási adatainkat,
hogyan következtethetünk bármire is az általánosítási hibáról?
Ami azt illeti, az általánosítási hibánk
esetleg nem jobb a véletlen találgatásnál.

Általában véve, bármilyen korlátozás hiányában a modellosztályunkra,
nem következtethetünk arra, csupán a tanítási adatokhoz való illeszkedés alapján,
hogy modellünk általánosítható mintát fedezett fel :cite:`vapnik1994measuring`.
Másrészt, ha a modellosztályunk
nem volt képes tetszőleges címkékre illeszkedni,
akkor szükségszerűen felfedezett valamilyen mintát.
A modell-komplexitásról szóló tanuláselméleti elképzelések
bizonyos inspirációt merítettek
Karl Popper, a tudomány befolyásos filozófusának gondolataiból,
aki formalizálta a cáfolhatóság kritériumát.
Popper szerint egy elmélet,
amely bármely és minden megfigyelést meg tud magyarázni,
egyáltalán nem tudományos elmélet!
Végül is, mit mondott nekünk a világról,
ha egyetlen lehetőséget sem zárt ki?
Röviden, amit szeretnénk, az egy hipotézis,
amely *nem* tudna magyarázni semmilyen általunk elképzelhető megfigyelést,
és mégis kompatibilisnek bizonyul
azokkal a megfigyelésekkel, amelyeket *valójában* teszünk.

Pontosan mi alkotja a modell-komplexitás megfelelő
fogalmát, összetett kérdés.
Gyakran a több paraméterrel rendelkező modellek
képesek nagyobb számú
tetszőlegesen hozzárendelt címkét illeszteni.
Ez azonban nem feltétlenül igaz.
Például a kernel módszerek végtelen számú paraméterrel rendelkező terekben működnek,
mégis komplexitásukat
más eszközökkel szabályozzák :cite:`Scholkopf.Smola.2002`.
A komplexitás egy fogalma, amely sokszor hasznosnak bizonyul,
a paraméterek által felvehető értékek tartománya.
Ebben az esetben egy modell, amelynek paraméterei
tetszőleges értékeket vehetnek fel,
összetettebbnek tekinthető.
Ezt az ötletet a következő részben újra megvizsgáljuk,
amikor bemutatjuk a *súlycsökkentést (weight decay)*,
az első gyakorlati regularizációs technikát.
Figyelemre méltó, hogy nehéz lehet összehasonlítani
a komplexitást lényegesen eltérő modellosztályok tagjai között
(mondjuk döntési fák vs. neurális hálózatok).


Ezen a ponton hangsúlyoznunk kell egy másik fontos pontot,
amelyhez akkor térünk vissza, amikor mély neurális hálózatokat mutatunk be.
Amikor egy modell képes tetszőleges címkékre illeszkedni,
az alacsony tanítási hiba nem feltétlenül
jelent alacsony általánosítási hibát.
*Azonban nem feltétlenül jelent
magas általánosítási hibát sem!*
Mindössze annyit mondhatunk biztosan, hogy
az alacsony tanítási hiba önmagában nem elegendő
az alacsony általánosítási hiba tanúsításához.
A mély neurális hálózatok éppen ilyen modellek:
bár a gyakorlatban jól általánosítanak,
túl erősek ahhoz, hogy sokat következtessünk
a tanítási hiba alapján egyedül.
Ezekben az esetekben nagyobb mértékben kell támaszkodnunk
a visszatartott adatainkra az általánosítás utólagos igazolásához.
A visszatartott adatokon, azaz a validációs halmazon elkövetett hiba
neve *validációs hiba*.

## Alulillesztés vagy túlillesztés?

Amikor összehasonlítjuk a tanítási és validációs hibákat,
ügyelni kell két általános helyzetre.
Először is figyelni kell azokra az esetekre,
amikor mind a tanítási hibánk, mind a validációs hibánk jelentős,
de köztük csak kis rés van.
Ha a modell nem képes csökkenteni a tanítási hibát,
ez azt jelentheti, hogy modellünk túl egyszerű
(azaz nem elég kifejező)
az általunk modellezni kívánt minta megragadásához.
Ezenkívül, mivel az *általánosítási rés* ($R_\textrm{emp} - R$)
a tanítási és általánosítási hibáink között kicsi,
okunk van azt hinni, hogy megúsznánk egy összetettebb modellel.
Ezt a jelenséget *alulillesztésnek* nevezzük.

Másrészt, ahogy fentebb tárgyaltuk,
figyelni kell azokra az esetekre is,
amikor a tanítási hibánk lényegesen alacsonyabb
a validációs hibánknál, ami súlyos *túlillesztésre* utal.
Megjegyezzük, hogy a túlillesztés nem mindig rossz dolog.
Különösen a deep learningben
a legjobb prediktív modellek sokszor
sokkal jobban teljesítenek a tanítási adatokon, mint a visszatartott adatokon.
Végső soron általában az általánosítási hiba csökkentésére törekszünk,
és csak annyiban foglalkoztat a rés,
amennyiben ez akadályává válik e célnak.
Vegyük észre, hogy ha a tanítási hiba nulla,
akkor az általánosítási rés pontosan egyenlő az általánosítási hibával,
és csak a rés csökkentésével haladhatunk előre.

### Polinomiális görbeillesztés
:label:`subsec_polynomial-curve-fitting`

A túlillesztéssel és modell-komplexitással kapcsolatos
klasszikus intuíció illusztrálásához
vegyük figyelembe a következőt:
adott egy egyetlen $x$ jellemzőből és
egy megfelelő valós értékű $y$ címkéből álló tanítási adat,
megpróbálunk $d$-fokú polinomot találni:

$$\hat{y}= \sum_{i=0}^d x^i w_i$$

a $y$ címke becslésére.
Ez csupán egy lineáris regressziós probléma,
ahol a jellemzőinket az $x$ hatványai adják,
a modell súlyait $w_i$ adja,
és az eltolást $w_0$ adja, mivel $x^0 = 1$ minden $x$-re.
Mivel ez csak egy lineáris regressziós probléma,
a négyzethibát használhatjuk veszteségfüggvényként.


A magasabb rendű polinomfüggvény összetettebb
az alacsonyabb rendűnél,
mivel a magasabb rendű polinomnak több paramétere van,
és a modell függvény kiválasztási tartománya szélesebb.
A tanítási adathalmazt rögzítve,
a magasabb rendű polinomfüggvényeknek mindig
alacsonyabb (legrosszabb esetben azonos) tanítási hibát kellene elérniük
az alacsonyabb fokú polinomokhoz képest.
Valójában, valahányszor minden adatpéldánynak
különböző $x$ értéke van,
egy olyan polinomfüggvény, amelynek foka
egyenlő az adatpéldányok számával,
tökéletesen illeszkedhet a tanítóhalmazra.
Összehasonlítjuk a polinom foka (modell-komplexitás)
és az alulillesztés, illetve túlillesztés közötti kapcsolatot a :numref:`fig_capacity_vs_error` részben.

![A modell-komplexitás hatása az alulillesztésre és a túlillesztésre.](../img/capacity-vs-error.svg)
:label:`fig_capacity_vs_error`


### Adathalmaz mérete

Ahogy a fenti korlát is jelzi,
egy másik fontos tényező,
amelyet figyelembe kell venni, az adathalmaz mérete.
A modellünket rögzítve, minél kevesebb mintánk van
a tanítási adathalmazban,
annál valószínűbb (és annál súlyosabb),
hogy túlillesztéssel találkozunk.
Ahogy növeljük a tanítási adatok mennyiségét,
az általánosítási hiba általában csökken.
Ezenkívül általában véve, a több adat soha nem árt.
Rögzített feladat és adateloszlás esetén
a modell-komplexitás nem szabad gyorsabban növekedjen,
mint az adatok mennyisége.
Több adat esetén esetleg megpróbálhatunk
összetettebb modellt illeszteni.
Elegendő adat hiányában az egyszerűbb modellek
nehezebbek lehetnek legyőzni.
Sok feladatnál a deep learning
csak akkor múlja felül a lineáris modelleket,
ha sok ezernyi tanítási példány áll rendelkezésre.
Részben a deep learning jelenlegi sikerét
az Internet vállalatoktól, az olcsó tárhelyektől,
a csatlakoztatott eszközöktől és a gazdaság széleskörű digitalizációjából
eredő hatalmas adathalmazok bősége magyarázza.

## Modellválasztás
:label:`subsec_generalization-model-selection`

Általában a végső modellt csak azután választjuk,
hogy több, különböző szempontból eltérő modellt kiértékeltünk
(különböző architektúrák, tanítási célkitűzések,
kiválasztott jellemzők, adatelőfeldolgozás,
tanulási ráták stb.).
A sok modell közül való választást találóan
*modellválasztásnak* nevezzük.

Elvben nem szabad megérintenünk a teszthalmazunkat
addig, amíg az összes hiperparaméterünket ki nem választottuk.
Ha a tesztelési adatokat a modellválasztási folyamatban alkalmaznánk,
fennáll a veszélye, hogy túlillesztjük a tesztelési adatokat.
Ekkor komoly bajban lennénk.
Ha a tanítási adatainkat túlillesztjük,
mindig ott van a tesztelési adatokon való kiértékelés, amely ébren tart minket.
De ha a tesztelési adatokat illesztjük túl, honnan tudnánk?
Lásd :citet:`ong2005learning` egy példáért, hogy ez
hogyan vezethet abszurd eredményekhez, még olyan modelleknél is, ahol a komplexitás
szorosan szabályozható.

Ezért soha nem szabad a tesztelési adatokra támaszkodni a modellválasztásnál.
Mégis, kizárólag a tanítási adatokra sem támaszkodhatunk
a modellválasztásnál, mert
nem tudjuk becsülni az általánosítási hibát
azon az adaton, amelyet a modell tanítására használunk.


A gyakorlati alkalmazásokban a kép zavarosabbá válik.
Bár ideális esetben csak egyszer érintenénk a tesztelési adatokat,
a legjobb modell értékeléséhez vagy
kis számú modell összehasonlításához,
a valós tesztelési adatokat ritkán vetik el csak egy használat után.
Ritkán engedhetjük meg magunknak, hogy minden kísérleti körhöz új teszthalmazt alkalmazzunk.
Sőt, a benchmark adatok évtizedekig tartó újrahasznosítása
jelentős hatással lehet az algoritmusok fejlesztésére,
pl. [képosztályozáshoz](https://paperswithcode.com/sota/image-classification-on-imagenet)
és [optikai karakterfelismeréshez](https://paperswithcode.com/sota/image-classification-on-mnist).

Az általános megközelítés a *teszthalmazon való tanítás* problémájának kezeléséhez
az adatok háromféle felosztása,
amely magában foglalja a *validációs halmazt*
a tanítási és tesztelési adathalmazok mellett.
Az eredmény egy zavaros üzlet, ahol a határok
a validációs és tesztelési adatok között aggasztóan kétértelműek.
Hacsak másképpen nincs jelezve, a könyv kísérleteiben
valójában olyan adatokkal dolgozunk, amelyeket helyesen
tanítási adatoknak és validációs adatoknak kellene nevezni, valódi teszthalmazok nélkül.
Ezért a könyv egyes kísérleteiben közölt pontosság valójában
validációs pontosság, nem valódi teszthalmaz pontossága.

### Keresztvalidáció

Amikor a tanítási adatok kevések,
esetleg nem is engedhetjük meg magunknak
elegendő adat visszatartását,
amely megfelelő validációs halmazt alkotna.
Ennek a problémának egy népszerű megoldása
a $K$*-szoros keresztvalidáció* alkalmazása.
Ebben az esetben az eredeti tanítási adatokat $K$ nem átfedő részhalmazra osztjuk.
Ezután a modell tanítása és validálása $K$-szor hajtódik végre,
minden alkalommal $K-1$ részhalmazon tanítva és
egy másik részhalmazon validálva (amelyet az adott körben nem használtak tanításhoz).
Végül a tanítási és validációs hibákat
a $K$ kísérlet eredményeinek átlagolásával becsüljük.



## Összefoglalás

Ez a rész a gépi tanulásban az általánosítás alapjait vizsgálta.
Néhány ezek közül az ötletek közül összetett
és ellentmondásos lesz, amikor mélyebb modellekhez jutunk; ott a modellek képesek komolyan túlilleszteni az adatokat,
és a komplexitás releváns fogalmai
egyaránt lehetnek implicit és ellentmondásosak
(pl. több paraméterrel rendelkező nagyobb architektúrák
jobban általánosítanak).
Néhány ökölszabályt adunk:

1. Használj validációs halmazokat (vagy $K$*-szoros keresztvalidációt*) a modellválasztáshoz;
1. Az összetettebb modellek gyakran több adatot igényelnek;
1. A komplexitás releváns fogalmai magukban foglalják mind a paraméterek számát, mind a felvehető értékek tartományát;
1. Minden más feltétel egyenlőségét feltételezve, a több adat szinte mindig jobb általánosításhoz vezet;
1. Ez az egész általánosítási vita a FHA feltételen alapszik. Ha lazítunk ezen a feltételezésen, megengedve az eloszlások eltolódását a tanítási és tesztelési időszakok között, akkor nem mondhatunk semmit az általánosításról egy további (talán enyhébb) feltételezés nélkül.


## Feladatok

1. Mikor oldhatod meg pontosan a polinomiális regresszió problémáját?
1. Adj legalább öt példát arra, ahol a függő véletlen változók lehetetlenné teszik a probléma FHA adatként való kezelését.
1. Valaha is elvárható-e a nulla tanítási hiba? Milyen körülmények között látnánk nulla általánosítási hibát?
1. Miért drága számítani a $K$-szoros keresztvalidációt?
1. Miért elfogult a $K$-szoros keresztvalidáció hibabecslése?
1. A VC dimenzió az olyan pontok maximális számaként van definiálva, amelyeket egy bizonyos függvényosztály valamely függvénye tetszőleges $\{\pm 1\}$ címkékkel osztályozni tud. Miért nem feltétlenül jó ötlet ez a függvényosztály bonyolultságának mérésére? Tipp: vedd figyelembe a függvények nagyságát.
1. A menedzsered egy nehéz adathalmazt ad neked, amelyen a jelenlegi algoritmusod nem teljesít jól. Hogyan indokolnád meg neki, hogy több adatra van szükséged? Tipp: nem növelheted az adatokat, de csökkentheted.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/96)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/97)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/234)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17978)
:end_tab:
