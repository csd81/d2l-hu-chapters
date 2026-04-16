# Bevezetés
:label:`chap_introduction`

Egészen a közelmúltig szinte minden számítógépes program, amellyel egy
átlagos nap során kapcsolatba kerültél, merev szabályrendszer alapján
működött, amely pontosan előírta, hogyan kell viselkednie. Tegyük fel,
hogy egy e-kereskedelmi platform kezelésére szolgáló alkalmazást akarunk
írni. Miután néhány órán át egy táblánál állva gondolkodtunk a
problémán, megállapodhatnánk egy működő megoldás fő vonalaiban, például
abban, hogy (i) a felhasználók egy webböngészőben vagy mobilalkalmazásban
futó felületen keresztül lépnek kapcsolatba az alkalmazással; (ii) az
alkalmazás egy vállalati szintű adatbázismotorral kommunikál, hogy nyomon
kövesse az egyes felhasználók állapotát és megőrizze a korábbi
tranzakciók adatait; valamint (iii) az alkalmazás középpontjában az
*üzleti logika* áll, vagyis az alkalmazás *agya*, amely olyan
szabályrendszert fogalmaz meg, amely minden elképzelhető helyzetet ahhoz
a művelethez rendel, amelyet a programnak végre kell hajtania.

Az alkalmazás agyának megépítéséhez felsorolhatnánk az összes gyakori
eseményt, amelyet a programnak kezelnie kell. Például amikor egy vásárló
rákattint arra, hogy egy terméket a kosárba tegyen, a programnak fel kell
vennie egy bejegyzést a kosár adatbázistáblájába, összekapcsolva a
felhasználó azonosítóját a kért termék azonosítójával. Ezután
végigpróbálhatnánk az összes lehetséges szélsőséges esetet,
ellenőrizhetnénk, hogy a szabályaink megfelelőek-e, és elvégezhetnénk a
szükséges módosításokat. Mi történik például, ha egy felhasználó üres
kosárral kezdeményez vásárlást? Bár kevés fejlesztő talál el mindent
teljesen jól elsőre
(elképzelhető, hogy néhány tesztfuttatásra is szükség van a hibák
kisimításához), az ilyen programokat nagyrészt meg tudjuk írni, és
magabiztosan el is tudjuk indítani *még azelőtt*, hogy egyetlen valódi
ügyfél használná őket. Az a képességünk, hogy kézzel tervezzünk meg
olyan automatizált rendszereket, amelyek működő termékeket és
rendszereket hajtanak, gyakran új helyzetekben is, figyelemre méltó
kognitív teljesítmény. És ha képes vagy olyan megoldásokat kitalálni,
amelyek az esetek $100\%$-ában működnek, akkor általában nem kell a gépi
tanulással foglalkoznod.

A gépi tanulással foglalkozó, egyre növekvő közösség szerencséjére sok
olyan feladat van, amelyet szeretnénk automatizálni, de ezek nem engednek
olyan könnyen az emberi leleményességnek. Képzeld el, hogy a
legokosabb emberekkel együtt állsz a tábla előtt, de ezúttal az alábbi
problémák egyikét próbáljátok megoldani:

* Írj programot, amely földrajzi információk, műholdképek és a korábbi időjárás egy csúszó időablaka alapján megjósolja a holnapi időjárást.
* Írj programot, amely egy szabad szövegesen megfogalmazott tényszerű kérdésre helyes választ ad.
* Írj programot, amely egy kép alapján azonosítja az összes rajta látható embert, és körvonalat rajzol köréjük.
* Írj programot, amely olyan termékeket jelenít meg a felhasználóknak, amelyeket valószínűleg kedvelni fognak, de normál böngészés közben valószínűleg nem találnának meg.

Ezekhez a problémákhoz még a kiváló programozók is nehezen tudnának
nulláról megírt megoldásokat készíteni. Az okok eltérőek lehetnek.
Néha a program, amelyet keresünk, olyan mintát követ, amely idővel
változik, így nincs egyetlen fix helyes válasz. Ilyen esetekben minden
sikeres megoldásnak rugalmasan kell alkalmazkodnia a változó világhoz.
Máskor a kapcsolat
(mondjuk a pixelek és az absztrakt kategóriák között) túl bonyolult,
ezrek vagy milliók nagyságrendjébe eső számításokat igényel, és ismeretlen
elveket követ. Képfelismerés esetén a feladat elvégzéséhez szükséges
pontos lépések meghaladják a tudatos megértésünket, még akkor is, ha a
tudatalatti kognitív folyamataink könnyedén végrehajtják ugyanezt.

A *gépi tanulás* olyan algoritmusok tanulmányozása, amelyek képesek
tapasztalatból tanulni. Ahogy egy gépi tanulási algoritmus egyre több
tapasztalatot gyűjt, jellemzően megfigyelési adatok vagy környezettel
való kölcsönhatások formájában, a teljesítménye javul. Ez szemben áll a
determinista e-kereskedelmi platformunkkal, amely ugyanazt az üzleti
logikát követi, függetlenül attól, mennyi tapasztalat halmozódik fel,
egészen addig, amíg maguk a fejlesztők nem tanulnak valamit, és úgy nem
döntenek, hogy ideje frissíteni a szoftvert. Ebben a könyvben a gépi
tanulás alapjait tanítjuk meg, különösen a *deep learningre*
összpontosítva, amely olyan erőteljes technikák összessége, amelyek a
számítógépes látástól a természetesnyelv-feldolgozáson és az
egészségügyön át a genomikáig számos területen hajtanak előre
innovációkat.

## Egy motiváló példa

Mielőtt a könyv szerzői nekiláttak az írásnak, nekik is szükségük volt
némi koffeinre, ahogy a munkaerő nagy részének. Beültünk az autóba és
elindultunk. Alex egy iPhone segítségével odaszólt: "Hey Siri", ezzel
felébresztve a telefon beszédfelismerő rendszerét. Ezután Mu azt mondta:
"directions to Blue Bottle coffee shop". A telefon gyorsan megjelenítette
a parancs átiratát. Azt is felismerte, hogy útbaigazítást kérünk, és
elindította a Térkép alkalmazást a kérés teljesítéséhez. Miután az
alkalmazás elindult, több útvonalat is azonosított. Mindegyik mellett
megjelenítette a becsült utazási időt. Bár ez a történet pedagógiai
okokból kissé kitalált, jól mutatja, hogy mindennapi okostelefonos
interakcióink akár néhány másodperc alatt is több gépi tanulási modellt
is igénybe vehetnek.


Képzeld el, hogy egyszerűen csak egy olyan programot kell írnod, amely
reagál egy *ébresztőszóra*, például az "Alexa", az "OK Google" vagy a
"Hey Siri" kifejezésre. Próbáld meg egyedül, egy szobában, csupán egy
számítógéppel és egy kódszerkesztővel megírni, ahogy azt
:numref:`fig_wake_word` szemlélteti. Hogyan írnál ilyen programot teljesen
az alapoktól? Gondolj bele: ez nehéz feladat. A mikrofon minden
másodpercben nagyjából 44,000 mintát gyűjt. Minden minta a hanghullám
amplitúdójának mérése. Milyen szabály tudna megbízhatóan leképezni egy
nyers hangrészletet arra vonatkozó biztos előrejelzésekre,
$\{\textrm{igen}, \textrm{nem}\}$, hogy az adott részlet tartalmazza-e az
ébresztőszót? Ha elakadtál, ne aggódj. Mi sem tudjuk, hogyan lehetne egy
ilyen programot nulláról megírni. Ezért használunk gépi tanulást.

![Egy ébresztőszó azonosítása.](../img/wake-word.svg)
:label:`fig_wake_word`


Íme a trükk. Gyakran még akkor is képesek vagyunk mi magunk végrehajtani
egy kognitív feladatot, amikor nem tudjuk pontosan megmondani a
számítógépnek, hogyan térképezze le a bemeneteket kimenetekre. Más szóval,
még ha nem is tudod, hogyan kellene egy számítógépet úgy programozni,
hogy felismerje az "Alexa" szót, te magad mégis képes vagy rá. Ezt a
képességet kihasználva hatalmas *adathalmazt* gyűjthetünk össze
hangrészletekből és a hozzájuk tartozó címkékből, amelyek jelzik, hogy az
adott részlet tartalmazza-e az ébresztőszót. A ma uralkodó gépi tanulási
megközelítésben nem próbálunk *explicit módon* olyan rendszert tervezni,
amely felismeri az ébresztőszavakat. Ehelyett definiálunk egy rugalmas
programot, amelynek viselkedését számos *paraméter* határozza meg.
Ezután az adathalmazt használjuk arra, hogy meghatározzuk a lehető legjobb
paraméterértékeket, vagyis azokat, amelyek javítják a program teljesítményét
egy választott teljesítménymérték szerint.

A paraméterekre úgy gondolhatsz, mint tekerőgombokra, amelyeket
elforgatva befolyásolhatjuk a program viselkedését. Miután a paraméterek
rögzítve vannak, a programot *modellnek* nevezzük. Az összes olyan különböző
program (bemenet--kimenet leképezés) halmazát, amelyet pusztán a paraméterek
változtatásával elő tudunk állítani, *modellcsaládnak* hívjuk. Azt a
"meta-programot" pedig, amely az adathalmazunk segítségével kiválasztja a
paramétereket, *tanulási algoritmusnak* nevezzük.

Mielőtt ténylegesen bevethetnénk a tanulási algoritmust, pontosan meg kell
fogalmaznunk a problémát: rögzítenünk kell a bemenetek és kimenetek pontos
természetét, valamint ki kell választanunk egy megfelelő modellcsaládot.
Ebben az esetben a modellünk egy hangrészletet kap *bemenetként*, és
$\{\textrm{igen}, \textrm{nem}\}$ közül választ *kimenetként*. Ha minden
a terv szerint alakul, akkor a modell tippjei rendszerint helyesek lesznek
arra nézve, hogy a részlet tartalmazza-e az ébresztőszót.

Ha a megfelelő modellcsaládot választjuk, akkor kell léteznie a
tekerőgombok olyan beállításának, amelynél a modell minden alkalommal
"igen"-t jelez, amikor az "Alexa" szót hallja. Mivel maga az
ébresztőszó választása önkényes, valószínűleg olyan modellcsaládra lesz
szükségünk, amely elég gazdag ahhoz, hogy egy másik beállítással csak az
"Apricot" szó hallatán adjon "igen" választ. Arra számítunk, hogy ugyanaz
a modellcsalád alkalmas legyen az "Alexa" és az "Apricot" felismerésére,
mert intuitíve hasonló feladatoknak tűnnek. Ugyanakkor lehet, hogy teljesen
más modellcsaládra lenne szükségünk, ha alapvetően eltérő bemenetekkel vagy
kimenetekkel akarunk dolgozni, például képeket akarunk képaláírásokra,
vagy angol mondatokat kínai mondatokra leképezni.

Ahogy sejtheted, ha minden tekerőgombot véletlenszerűen állítunk be, nem
valószínű, hogy a modell felismeri az "Alexa", az "Apricot" vagy bármely
más angol szót. A gépi tanulásban a *tanulás* az a folyamat, amelynek
során megtaláljuk a gombok megfelelő beállítását, hogy a modell a kívánt
viselkedést produkálja. Más szóval adatokkal *betanítjuk* a modellt.
Ahogy azt :numref:`fig_ml_loop` mutatja, a tanítási folyamat általában így
néz ki:

1. Kezdünk egy véletlenszerűen inicializált modellel, amely semmi hasznosat nem tud.
1. Fogunk valamennyit az adatainkból
   (például hangrészleteket és a hozzájuk tartozó $\{\textrm{igen}, \textrm{nem}\}$ címkéket).
1. Úgy állítjuk a gombokat, hogy a modell jobban teljesítsen ezeken a példákon.
1. Ismételjük a 2. és 3. lépést, amíg a modell nagyon jó nem lesz.

![Egy tipikus tanítási folyamat.](../img/ml-loop.svg)
:label:`fig_ml_loop`

Összefoglalva: ahelyett, hogy kézzel megírnánk egy ébresztőszó-felismerőt,
olyan programot írunk, amely képes *megtanulni* az ébresztőszavak
felismerését, ha elegendően nagy, címkézett adathalmazt kap. Úgy is
gondolhatsz erre, mint arra, hogy egy program viselkedését egy
adathalmaz segítségével határozzuk meg: ez a *programozás adatokkal*.
Vagyis úgy is "programozhatunk" egy macskadetektort, hogy rengeteg
macskás és kutyás példát adunk a gépi tanulási rendszerünknek. Így a
detektor végül megtanulhat nagyon nagy pozitív számot kiadni, ha macskát
lát, nagyon nagy negatív számot, ha kutyát lát, és nullához közelebbi
értéket, ha bizonytalan. Ez még csak a felszínt karcolja abból, mire képes
a gépi tanulás. A deep learning, amelyet később részletesebben is
megmagyarázunk, csupán egy a gépi tanulási problémák megoldására használt
népszerű módszerek közül.


## Kulcselemek

Az ébresztőszavas példában hangrészletekből és bináris címkékből álló
adathalmazt írtunk le, és nagy vonalakban megmutattuk, hogyan taníthatnánk
be egy modellt arra, hogy a hangrészleteket osztályokhoz közelítő módon
leképezze. Az ilyen típusú problémát, amikor ismert inputok alapján
megpróbálunk megjósolni egy kijelölt, ismeretlen címkét egy olyan
adathalmazból kiindulva, amelyben a példák címkéi ismertek,
*felügyelt tanulásnak* nevezzük. Ez csak egy a gépi tanulási problémák sok
fajtája közül. Mielőtt más változatokat is megvizsgálnánk, szeretnénk
nagyobb fényt vetni néhány olyan alapvető elemre, amelyek minden gépi
tanulási feladatban velünk maradnak:

1. Az *adat*, amelyből tanulhatunk.
1. Egy *modell*, amely megmutatja, hogyan alakítsuk át az adatot.
1. Egy *célfüggvény*, amely számszerűsíti, mennyire jól
   (vagy rosszul) teljesít a modell.
1. Egy *algoritmus*, amely a célfüggvény optimalizálásához igazítja a modell paramétereit.

### Adat

Szinte magától értetődik, hogy adatok nélkül nem lehet adattudományt
csinálni. Arról, hogy pontosan mi is az *adat*, akár több száz oldalon át
is elmélkedhetnénk, de most inkább az általunk vizsgált adathalmazok
fontos tulajdonságaira koncentrálunk. Általában példák egy gyűjteményével
dolgozunk. Ahhoz, hogy az adatokkal hasznosan dolgozhassunk, rendszerint
megfelelő numerikus reprezentációt kell találnunk. Minden *példa*
(vagy *adattétel*, *adatpéldány*, *minta*) tipikusan attribútumok egy
halmazából áll, ezeket *jellemzőknek* nevezzük
(néha *kovariánsoknak* vagy *bemeneteknek* is hívják őket), és a modell
ezek alapján készíti el az előrejelzéseit. Felügyelt tanulási problémákban
az a célunk, hogy megjósoljunk egy speciális attribútumot, az úgynevezett
*címkét* (vagy *célt*), amely nem része a modell bemenetének.

Ha képadatokkal dolgozunk, minden példa állhat egy egyedi fényképből
(ezek a jellemzők) és egy számból, amely megadja, hogy a fénykép melyik
kategóriába tartozik (ez a címke). A fényképet numerikusan három,
számértékekből álló rácsként ábrázolhatjuk, amelyek minden pixelhelyen a
vörös, a zöld és a kék fény intenzitását reprezentálják. Például egy
$200\times 200$ pixeles színes fénykép
$200\times200\times3=120000$ numerikus értékből állna.

Másik lehetőségként dolgozhatunk elektronikus egészségügyi
nyilvántartások adataival, és azt a feladatot tűzhetjük ki, hogy
megjósoljuk, egy adott beteg mekkora valószínűséggel éli túl a következő
30 napot. Ilyenkor a jellemzők olyan könnyen hozzáférhető attribútumokból
és gyakran rögzített mérésekből állhatnak, mint az életkor, az
életjelek, a társbetegségek, a jelenlegi gyógyszerek és a közelmúltbeli
beavatkozások. A tanításhoz rendelkezésre álló címke bináris érték lenne,
amely jelzi, hogy a történeti adatokban szereplő beteg túlélte-e a
30 napos időablakot.

Ilyen esetekben, amikor minden példát ugyanannyi numerikus jellemző
jellemez, azt mondjuk, hogy a bemenetek rögzített hosszúságú vektorok, és
a vektorok (állandó) hosszát az adatok *dimenziószámának* nevezzük.
Ahogy sejtheted, a rögzített hosszúságú bemenetek kényelmesek lehetnek,
mert eggyel kevesebb bonyodalommal kell foglalkoznunk. Ugyanakkor nem
minden adat reprezentálható könnyen *rögzített hosszúságú* vektorként.
Bár a mikroszkópképekről feltételezhetjük, hogy szabványos
berendezésekből származnak, az internetről gyűjtött képekről nem várhatjuk
el, hogy mind azonos felbontásúak vagy alakúak legyenek. Képeknél
megfontolhatjuk a szabványos méretre vágást, de ez csak bizonyos határig
működik, hiszen elveszíthetjük a levágott részek információit. A
szöveges adatok pedig még makacsabban állnak ellen a rögzített hosszúságú
reprezentációknak. Gondolj az olyan e-kereskedelmi oldalak vásárlói
véleményeire, mint az Amazon, az IMDb vagy a TripAdvisor. Némelyik rövid:
"borzalmas!". Mások oldalakon át hömpölyögnek. A deep learning egyik
nagy előnye a hagyományos módszerekkel szemben, hogy a modern modellek
viszonylag elegánsan tudnak *változó hosszúságú* adatokat kezelni.

Általánosságban minél több adatunk van, annál könnyebb a dolgunk. Több
adat mellett erősebb modelleket tudunk tanítani, és kevésbé kell
előfeltevésekre támaszkodnunk. A
(viszonylag) kis adatmennyiségről a nagy adatmennyiségre való átállás a
modern deep learning sikerének egyik fő mozgatórugója. Ezt jól mutatja,
hogy a deep learning legizgalmasabb modelljei közül sok egyszerűen nem
működik nagy adathalmazok nélkül. Más modellek kis adatmennyiségnél is
használhatók lehetnek, de ilyenkor gyakran nem jobbak a hagyományos
megközelítéseknél.

Végül nem elég, ha sok adatunk van, és ügyesen dolgozzuk fel őket. A
*megfelelő* adatokra van szükségünk. Ha az adatok tele vannak hibával,
vagy a választott jellemzők nem jeleznek előre a minket érdeklő célváltozóra,
a tanulás kudarcot vall. Jól leírja ezt a közhely:
*szemét be, szemét ki*. Ráadásul a gyenge prediktív teljesítmény nem az
egyetlen lehetséges következmény. A gépi tanulás érzékeny alkalmazásaiban,
például prediktív rendészetben, önéletrajzok szűrésében vagy hitelezési
kockázati modellekben különösen oda kell figyelnünk a rossz minőségű
adatok következményeire. Gyakori hibamód, amikor bizonyos embercsoportok
egyáltalán nem szerepelnek a tanítóadatokban. Képzeld el, hogy egy
bőrrákfelismerő rendszert alkalmazunk, amely korábban soha nem látott
fekete bőrt. Kudarchoz az is vezethet, ha az adatok nemcsak
alulreprezentálnak bizonyos csoportokat, hanem társadalmi előítéleteket
is tükröznek. Ha például korábbi felvételi döntések alapján tanítunk be
egy prediktív modellt, amelyet később önéletrajzok szűrésére használunk,
akkor a gépi tanulási modellek akaratlanul is átvehetik és automatizálhatják
a történeti igazságtalanságokat. Fontos észrevenni, hogy mindez úgy is
megtörténhet, hogy az adattudós nem sző semmiféle összeesküvést, sőt
akár tudatában sincs a problémának.


### Modellek

A legtöbb gépi tanulási feladat valamilyen értelemben az adatok
átalakításával jár. Lehet, hogy olyan rendszert szeretnénk építeni,
amely fényképeket kap bemenetként, és azt jósolja meg, mennyire mosolygós
az arc rajtuk. Máskor érzékelőmérések egy halmazát kapjuk, és azt
szeretnénk előre jelezni, mennyire normális vagy rendellenes az adott
mérés. *Modell* alatt azt a számítási gépezetet értjük, amely egy adott
típusú adatot bemenetként fogad, és egy esetleg más típusú előrejelzést
ad ki. Különösen azok a *statisztikai modellek* érdekelnek bennünket,
amelyeket adatokból lehet becsülni. Bár az egyszerű modellek tökéletesen
alkalmasak megfelelően egyszerű problémák megoldására, a könyvben
tárgyalt feladatok már a klasszikus módszerek határait feszegetik. A deep
learninget elsősorban az különbözteti meg a klasszikus megközelítésektől,
hogy milyen erőteljes modellekre támaszkodik. Ezek a modellek az adatok
sok egymás utáni transzformációjából állnak, amelyeket alulról felfelé
láncolunk össze, innen ered a *deep learning* elnevezés is. A mély
modellek tárgyalása felé haladva néhány hagyományosabb módszert is meg
fogunk beszélni.

### Célfüggvények

Korábban úgy vezettük be a gépi tanulást, mint tapasztalatból való
tanulást. A *tanulás* alatt itt azt értjük, hogy idővel jobbak leszünk
egy adott feladatban. De ki dönti el, mi számít javulásnak? El tudod
képzelni, hogy javaslunk egy modellfrissítést, és egyesek vitatják, hogy
ez valóban javulásnak tekinthető-e.

Ahhoz, hogy formális matematikai rendszert alkossunk tanuló gépekhez,
formális mérőszámokra van szükségünk arra, mennyire jók
(vagy rosszak) a modelljeink. A gépi tanulásban, és általánosabban az
optimalizálásban ezeket *célfüggvényeknek* nevezzük. Szokás szerint úgy
definiáljuk őket, hogy a kisebb érték legyen jobb. Ez csupán konvenció.
Bármely olyan függvényt, amelynél a nagyobb érték jobb, átfordíthatunk
előjellel olyan új függvénnyé, amely minőségileg ugyanaz, csak éppen a
kisebb érték a kedvezőbb. Mivel úgy döntünk, hogy a kisebb jobb, ezeket
a függvényeket néha *veszteségfüggvényeknek* is nevezzük.

Numerikus értékek előrejelzésekor a leggyakoribb veszteségfüggvény a
*négyzetes hiba*, vagyis az előrejelzés és a valós célérték különbségének
négyzete. Osztályozásnál a leggyakoribb cél a
hibaarány minimalizálása, vagyis azon példák arányának csökkentése,
amelyeken az előrejelzéseink eltérnek a valós címkéktől. Egyes célok
(például a négyzetes hiba) könnyen optimalizálhatók, míg mások
(például a hibaarány) közvetlenül nehezen optimalizálhatók nem
differenciálhatóság vagy egyéb komplikációk miatt. Ilyenkor gyakran egy
*helyettesítő célfüggvényt* optimalizálunk.

Optimalizálás közben a veszteségre úgy gondolunk, mint a modell
paramétereinek függvényére, miközben a tanítóadathalmazt konstansnak
tekintjük. A modell paramétereinek legjobb értékeit úgy tanuljuk meg,
hogy minimalizáljuk a tanításhoz összegyűjtött példákon elszenvedett
veszteséget. Ugyanakkor az, hogy jól teljesítünk a tanítóadaton, nem
garantálja, hogy jól fogunk teljesíteni ismeretlen adatokon is. Ezért az
elérhető adatokat rendszerint két részre osztjuk:
a *tanítóadathalmazra* (vagy *tanítóhalmazra*), amelyből a modell
paramétereit tanuljuk; és a *teszthalmazra* (vagy *teszthalmazra*),
amelyet félreteszünk az értékelésre. Végül általában mindkét részen
jelentjük a modellek teljesítményét. Úgy is gondolhatsz a tanítási
teljesítményre, mint arra, ahogyan egy diák a valódi záróvizsgára való
felkészülés során a gyakorlóteszteken teljesít. Még ha az eredmények
bátorítóak is, ez nem garantál sikert a valódi vizsgán. A tanulás során a
diák elkezdheti bemagolni a gyakorlókérdéseket, és úgy tűnhet, mintha
elsajátította volna a témát, de elbizonytalanodik, amikor a valódi
vizsgán korábban nem látott kérdésekkel találkozik. Amikor egy modell jól
teljesít a tanítóhalmazon, de nem tud általánosítani ismeretlen adatokra,
azt mondjuk, hogy *túlilleszkedik* a tanítóadatra.


### Optimalizálási algoritmusok

Miután van adatforrásunk és reprezentációnk, modellünk és jól definiált
célfüggvényünk, szükségünk van egy algoritmusra, amely képes megtalálni a
veszteségfüggvény minimalizálásához szükséges lehető legjobb
paramétereket. A deep learning népszerű optimalizálási algoritmusai a
*gradienscsökkenés* nevű megközelítésre épülnek. Röviden: minden
lépésben ez a módszer megvizsgálja, hogy az egyes paraméterekre nézve
hogyan változna a tanítóhalmaz vesztesége, ha azt a paramétert egy kicsit
megváltoztatnánk. Ezután a paramétert abba az irányba frissíti, amely
csökkenti a veszteséget.


## A gépi tanulási problémák fajtái

Az ébresztőszavas probléma motiváló példánkban csak egy a sok közül,
amellyel a gépi tanulás foglalkozhat. Hogy tovább motiváljuk az olvasót,
és kialakítsunk egy közös nyelvet, amely végigkíséri a könyvet, most
széles körű áttekintést adunk a gépi tanulási problémák világáról.

### Felügyelt tanulás

A felügyelt tanulás olyan feladatokat ír le, amelyeknél olyan
adathalmazt kapunk, amely jellemzőket és címkéket egyaránt tartalmaz, és
arra kérnek bennünket, hogy készítsünk egy modellt, amely a címkéket
jósolja meg a bemeneti jellemzők alapján. Minden jellemző--címke párt
egy példának nevezünk. Néha, ha a szövegkörnyezet egyértelmű, a
*példák* kifejezést a bemenetek gyűjteményére is használjuk, még akkor is, ha a
hozzájuk tartozó címkék ismeretlenek. A felügyelet ott jelenik meg, hogy
a paraméterek kiválasztásához mi
(a felügyelők) címkézett példákból álló adathalmazt adunk a modellnek.
Valószínűségi értelemben rendszerint azt szeretnénk becsülni, hogy egy
címke milyen feltételes valószínűséggel következik a bemeneti jellemzőkből.
Bár a több paradigma közül csak az egyik, a felügyelt tanulás adja az
iparban sikeres gépi tanulási alkalmazások többségét. Részben azért,
mert sok fontos feladat jól leírható úgy, mint valami ismeretlen
valószínűségének becslése egy adott, rendelkezésre álló adathalmaz alapján:

* Döntsük el, rákos-e vagy sem, egy számítógépes tomográfiás kép alapján.
* Jósoljuk meg egy angol mondat helyes francia fordítását.
* Jósoljuk meg egy részvény jövő havi árát az aktuális hónap pénzügyi jelentési adatai alapján.

Bár minden felügyelt tanulási probléma leírható úgy, hogy
"jósoljuk meg a címkéket a bemeneti jellemzők alapján", maga a
felügyelt tanulás sokféle formát ölthet, és rengeteg modellezési döntést
igényel attól függően
(sok más tényező mellett), hogy milyen típusúak, méretűek és számosságúak
a bemenetek és a kimenetek. Például más modelleket használunk tetszőleges
hosszúságú sorozatok és rögzített hosszúságú vektorreprezentációk
feldolgozására. A könyv során sok ilyen problémát részletesen is
vizsgálni fogunk.

Informálisan a tanulási folyamat valahogy így néz ki. Először veszünk
egy nagy gyűjteményt olyan példákból, amelyeknél a jellemzők ismertek, és
ebből kiválasztunk egy véletlen részhalmazt, amelyhez megszerezzük a
valódi címkéket. Néha ezek a címkék már eleve rendelkezésre állnak,
korábban összegyűjtött adatok formájában
(például meghalt-e a beteg a következő egy évben?), máskor pedig emberi
annotátorokra van szükségünk, hogy címkézzék az adatot
(például képek kategóriákhoz rendelésével). Ezek a bemenetek és a
hozzájuk tartozó címkék együtt alkotják a tanítóhalmazt. A
tanítóadathalmazt betápláljuk egy felügyelt tanulási algoritmusba,
amely bemenetként adathalmazt kap, és egy másik függvényt ad vissza:
a megtanult modellt. Végül korábban nem látott bemeneteket is beadhatunk
a modellnek, és a kimeneteit a hozzájuk tartozó címkék előrejelzéseként
használhatjuk. A teljes folyamatot :numref:`fig_supervised_learning`
ábrázolja.

![Felügyelt tanulás.](../img/supervised-learning.svg)
:label:`fig_supervised_learning`

#### Regresszió

Talán a legkönnyebben átlátható felügyelt tanulási feladat a
*regresszió*. Vegyünk például egy lakáseladási adatbázisból származó
adathalmazt. Összeállíthatunk egy táblázatot, amelyben minden sor egy
különböző háznak felel meg, és minden oszlop valamilyen releváns
tulajdonságot jelöl, például az alapterületet, a hálószobák számát, a
fürdőszobák számát vagy azt, hány perc sétára van a városközponttól.
Ebben az adathalmazban minden példa egy konkrét ház lenne, a hozzá
tartozó jellemzővektor pedig a táblázat egyik sora. Ha New Yorkban vagy
San Franciscóban élsz, és nem te vagy az Amazon, a Google, a Microsoft
vagy a Facebook vezérigazgatója, akkor az otthonod
(alapterület, hálószobák száma, fürdőszobák száma, gyalogos távolság)
jellemzővektora nagyjából így nézhet ki: $[600, 1, 1, 60]$.
Pittsburghben viszont inkább ilyen lehet: $[3000, 4, 3, 10]$.
Az ilyen rögzített hosszúságú jellemzővektorok a legtöbb klasszikus gépi
tanulási algoritmus számára alapvetőek.

Az tesz egy problémát regresszióvá, hogy milyen alakú a célváltozó.
Tegyük fel, hogy új otthont keresel. Előfordulhat, hogy egy ház
tisztességes piaci értékét szeretnéd megbecsülni a fentihez hasonló
jellemzők alapján. Az adatok itt történeti ingatlanhirdetésekből
állhatnak, a címkék pedig a tényleges eladási árak lehetnek. Amikor a
címkék tetszőleges numerikus értékeket vehetnek fel
(még ha csak egy bizonyos intervallumon belül is), ezt *regressziós*
problémának nevezzük. A cél olyan modell létrehozása, amelynek
előrejelzései szorosan közelítik a valódi címkeértékeket.


Sok gyakorlati probléma könnyen leírható regresszióként. Annak
előrejelzése, hogy egy felhasználó milyen értékelést ad majd egy filmnek,
regressziós problémának tekinthető, és ha 2009-ben remek algoritmust
terveztél volna erre a feladatra, akár meg is nyerhetted volna az
[egymillió dolláros Netflix-díjat](https://en.wikipedia.org/wiki/Netflix_Prize).
A kórházi tartózkodás hosszának előrejelzése szintén regressziós
probléma. Jó ökölszabály, hogy minden *mennyit?* vagy *hányat?* kérdés
valószínűleg regresszió. Például:

* Hány óráig fog tartani ez a műtét?
* Mennyi csapadék hullik ebben a városban a következő hat órában?


Még ha korábban soha nem is foglalkoztál gépi tanulással, valószínűleg
már oldottál meg informálisan regressziós problémát. Képzeld el például,
hogy kitisztíttattad a lefolyóidat, és a szakember 3 órát töltött a
szennyvízcsövek tisztításával. Ezután 350 dollárról szóló számlát küldött.
Most képzeld el, hogy a barátod ugyanezt a szakembert 2 órára fogadta fel,
és 250 dolláros számlát kapott. Ha ezután valaki megkérdezné, mire számítson
a saját közelgő csőtisztítási számláján, valószínűleg ésszerű
feltételezésekkel élnél, például hogy több ledolgozott óra több pénzbe
kerül. Azt is feltételezhetnéd, hogy van valamilyen alapdíj, és utána
óradíjat számolnak fel. Ha ezek a feltételezések igazak, akkor már e két
adatpélda alapján azonosíthatnád is a szakember árszabását:
100 dollár óránként, plusz 50 dollár kiszállási díj. Ha eddig követted a
gondolatmenetet, akkor a *lineáris* regresszió alapötletét már
megértetted.

Ebben az esetben olyan paramétereket tudnánk előállítani, amelyek pontosan
illeszkednek a szakember áraihoz. Néha ez nem lehetséges, például ha a
változékonyság egy része olyan tényezőkből fakad, amelyek a két
jellemződön kívül esnek. Ilyenkor olyan modelleket próbálunk tanulni,
amelyek minimalizálják az előrejelzéseink és a megfigyelt értékek közötti
távolságot. A fejezetek többségében a négyzetes hibaveszteség
minimalizálására összpontosítunk. Ahogy később látni fogjuk, ez a
veszteség megfelel annak a feltevésnek, hogy adatainkat Gauss-zaj
torzítja.

#### Osztályozás

Míg a regressziós modellek nagyszerűek a *mennyit?* típusú kérdések
megválaszolására, sok probléma nem illeszkedik kényelmesen ebbe a
sablonba. Gondoljunk például egy bankra, amely csekkbeolvasó funkciót
szeretne fejleszteni a mobilalkalmazásához. Ideális esetben az ügyfél
egyszerűen lefotózza a csekket, az alkalmazás pedig automatikusan
felismeri a képen lévő szöveget. Ha feltételezzük, hogy képesek vagyunk
kivágni az egyes kézírt karakterekhez tartozó képrészleteket, akkor a
fő megmaradó feladat az, hogy eldöntsük, melyik ismert karakter szerepel
az egyes képrészleteken. Az ilyen *melyik?* típusú problémákat
*osztályozásnak* nevezzük, és a regressziótól eltérő eszközkészletet
igényelnek, még ha sok technika át is vihető egyikből a másikba.

Az *osztályozásban* azt szeretnénk, hogy a modell a jellemzők alapján,
például egy kép pixelértékei alapján megjósolja, hogy egy példa melyik
*kategóriába*
(más néven *osztályba*) tartozik egy diszkrét lehetőségkészletből.
Kézzel írt számjegyek esetén például tíz osztályunk lehet, a 0-tól 9-ig
terjedő számjegyek. Az osztályozás legegyszerűbb formája az, amikor csak
két osztály van; ezt *bináris osztályozásnak* nevezzük. Például
adathalmazunk állhat állatokról készült képekből, címkéink pedig a
$\textrm{\{macska, kutya\}}$ osztályok lehetnek. Míg regresszióban olyan
regresszort kerestünk, amely numerikus értéket ad ki, osztályozásban
osztályozót keresünk, amelynek kimenete a jósolt osztályhozzárendelés.

Azokból az okokból, amelyekről akkor beszélünk majd, amikor a könyv
technikaibbá válik, nehéz lehet olyan modellt optimalizálni, amely csak
egy *határozott* kategóriacímkét tud kiadni, például vagy "macska", vagy
"kutya". Ilyenkor általában sokkal egyszerűbb a modellt a
valószínűségek nyelvén kifejezni. Egy példa jellemzői alapján a modell
minden lehetséges osztályhoz valószínűséget rendel. Visszatérve az
állatos osztályozási példára, ahol az osztályok
$\textrm{\{macska, kutya\}}$, egy osztályozó megnézhet egy képet, és
0.9-es valószínűséget adhat arra, hogy a képen macska van. Ezt úgy
értelmezhetjük, hogy az osztályozó 90\%-ig biztos benne, hogy a kép
macskát ábrázol. A jósolt osztályhoz tartozó valószínűség nagysága
bizonytalanságot is közvetít. Ez nem az egyetlen lehetséges módja a
bizonytalanság kifejezésének, és a haladóbb témáknál más módszerekről is
beszélünk majd.

Ha kettőnél több lehetséges osztály van, a problémát
*többosztályos osztályozásnak* nevezzük. Gyakori példa erre a kézzel írt
karakterek felismerése
$\textrm{\{0, 1, 2, ... 9, a, b, c, ...\}}$.
Míg a regressziós problémákat a négyzetes hibaveszteség minimalizálásával
támadtuk meg, az osztályozási problémák szokásos veszteségfüggvénye a
*keresztentrópia*, amelynek nevét a későbbi fejezetekben, az
információelmélet bevezetésekor tisztázzuk.

Fontos megjegyezni, hogy a legvalószínűbb osztály nem feltétlenül az,
amelyet a döntésedben ténylegesen felhasználsz. Tegyük fel, hogy a kerted
végében találsz egy gyönyörű gombát, mint a :numref:`fig_death_cap`
ábrán.

![Gyilkos galóca --- ne edd meg!](../img/death-cap.jpg)
:width:`200px`
:label:`fig_death_cap`

Most tegyük fel, hogy építettél egy osztályozót, és betanítottad arra,
hogy egy fénykép alapján megjósolja, mérgező-e egy gomba. Legyen a
mérgezőgomba-detektorod kimenete az, hogy annak a valószínűsége, hogy a
:numref:`fig_death_cap` képen gyilkos galóca látható, 0.2. Más szóval az
osztályozó 80\%-ban biztos abban, hogy a gomba nem gyilkos galóca.
Mégis bolond lennél, ha megennéd. Ennek az az oka, hogy egy finom vacsora
biztos haszna egyszerűen nem ér fel a 20\%-os halálozási kockázattal.
Másképp fogalmazva: a bizonytalan kockázat hatása messze felülmúlja a
várható előnyt. Ahhoz tehát, hogy eldöntsük, megegyük-e a gombát, minden
lehetséges cselekvéshez ki kell számolnunk a várható kárt, amely függ
mind a valószínű kimenetektől, mind az egyes kimenetekhez társuló
hasznoktól és károktól. Ebben az esetben a gomba elfogyasztásának kára
lehet például
$0.2 \times \infty + 0.8 \times 0 = \infty$,
míg a kidobás vesztesége
$0.2 \times 0 + 0.8 \times 1 = 0.8$.
Az óvatosságunk indokolt volt: ahogy bármely mikológus megmondaná, a
:numref:`fig_death_cap` ábrán valóban gyilkos galóca látható.

Az osztályozás jóval bonyolultabb is lehet, mint a bináris vagy
többosztályos osztályozás. Léteznek például olyan változatai, amelyek
hierarchikusan szervezett osztályokkal dolgoznak. Ilyenkor nem minden
hiba egyforma: ha már tévedünk, inkább egy rokon osztályba szeretnénk
félreosztályozni a példát, mint egy távoli osztályba. Ezt általában
*hierarchikus osztályozásnak* nevezzük. Inspirációként gondolhatsz
[Linnére](https://en.wikipedia.org/wiki/Carl_Linnaeus), aki hierarchiába
rendezte az élővilágot.

Állatok osztályozásánál talán nem akkora baj, ha egy uszkárt
schnauzernek nézünk, de a modellünk súlyos árat fizetne, ha egy uszkárt
dinoszaurusszal keverne össze. Hogy melyik hierarchia releváns, az attól
is függhet, mire akarod használni a modellt. Például a csörgőkígyók és a
harisnyakígyók közel lehetnek egymáshoz a törzsfejlődési fán, de egy
csörgőkígyó összetévesztése egy ártalmatlan fajjal végzetes következménnyel
járhat.

#### Címkézés

Néhány osztályozási probléma szépen beleillik a bináris vagy a többosztályos
osztályozási keretbe. Például betaníthatunk egy egyszerű bináris
osztályozót arra, hogy megkülönböztesse a macskákat a kutyáktól. A
számítógépes látás jelenlegi állása mellett ezt könnyen meg tudjuk tenni
kész eszközökkel. Ennek ellenére, bármilyen pontos is lesz a modellünk,
gondba kerülhetünk, amikor az osztályozó a *Brémai muzsikusok* című,
négy állatot felvonultató népszerű német meséből származó képpel
találkozik (:numref:`fig_stackedanimals`).

![Egy szamár, egy kutya, egy macska és egy kakas.](../img/stackedanimals.png)
:width:`300px`
:label:`fig_stackedanimals`

Ahogy látható, a képen macska, kakas, kutya és szamár is szerepel, néhány
fával a háttérben. Ha számítunk az ilyen képek előfordulására, a
többosztályos osztályozás nem biztos, hogy a megfelelő
problémaformalizálás. Ehelyett inkább azt szeretnénk, hogy a modellnek
lehetősége legyen azt mondani: a kép macskát, kutyát, szamarat *és*
kakast ábrázol.

Azokat a feladatokat, ahol nem egymást kizáró osztályokat kell
előrejelezni, *többcímkés osztályozásnak* nevezzük. Az automatikus címkézési
problémákat tipikusan így a legjobb leírni. Gondolj azokra a címkékre,
amelyeket az emberek egy technológiai blog bejegyzéseire tehetnek,
például "machine learning", "technology", "gadgets",
"programming languages", "Linux", "cloud computing", "AWS". Egy tipikus
cikkre 5--10 címke is kerülhet. Általában a címkék között valamilyen
korrelációs szerkezet is megfigyelhető. A "cloud computing" témájú
bejegyzések valószínűleg megemlítik az "AWS"-t, a "machine learning"
témájúak pedig valószínűleg a "GPU"-kat.

Néha az ilyen címkézési problémák óriási címkehalmazokkal dolgoznak. Az
amerikai National Library of Medicine sok hivatásos annotátort alkalmaz,
akik minden PubMedben indexelendő cikket összekapcsolnak a
Medical Subject Headings (MeSH) ontológiából származó címkék halmazával;
ez nagyjából 28,000 címkéből álló gyűjtemény. A cikkek helyes címkézése
fontos, mert lehetővé teszi a kutatók számára, hogy kimerítő
irodalomáttekintéseket készítsenek. Ez időigényes folyamat, és az
archiválás és a címkézés között rendszerint körülbelül egy év telik el. A
gépi tanulás ideiglenes címkéket adhat, amíg minden cikk meg nem kapja a
megfelelő kézi ellenőrzést. Nem véletlen, hogy a BioASQ szervezet már
több éve [versenyeket is rendez](http://bioasq.org/) ehhez a feladathoz.

#### Keresés

Az információkeresés területén gyakran rangsoroljuk az elemek halmazait.
Vegyük például a webes keresést. Itt a cél kevésbé az eldöntése, *hogy*
egy adott oldal releváns-e a lekérdezésre, sokkal inkább az, hogy a
releváns találatok közül melyiket kell a legelőkelőbb helyen megjeleníteni
egy adott felhasználó számára. Ennek egyik módja lehet, hogy először
minden elemhez pontszámot rendelünk, majd a legmagasabb pontszámú elemeket
hozzuk vissza. A
[PageRank](https://en.wikipedia.org/wiki/PageRank), amely eredetileg a
Google keresőjének titkos fegyvere volt, korai példája egy ilyen
pontozási rendszernek. Furcsa módon a PageRank által adott pontszám nem
függött a konkrét lekérdezéstől. Ehelyett egy egyszerű relevanciaszűrőt
használtak a releváns jelöltek azonosítására, majd a PageRankkel
rangsorolták a hitelesebb oldalakat. Napjainkban a keresőmotorok gépi
tanulást és viselkedési modelleket használnak ahhoz, hogy lekérdezésfüggő
relevanciapontszámokat kapjanak. Egész tudományos konferenciákat
szentelnek ennek a témának.

#### Ajánlórendszerek
:label:`subsec_recommender_systems`

Az ajánlórendszerek egy másik olyan problémakört jelentenek, amely szorosan
kapcsolódik a kereséshez és a rangsoroláshoz. A problémák annyiban
hasonlók, hogy itt is a felhasználó számára releváns elemek halmazát
kell megjeleníteni. A fő különbség az, hogy az ajánlórendszerekben
nagy hangsúly kerül az egyes felhasználókra szabott
*személyre szabásra*. Például filmajánlásnál egy sci-fi-rajongó és Peter
Sellers vígjátékainak ínyence egészen eltérő találati oldalt kaphat.
Hasonló problémák más ajánlási helyzetekben is megjelennek, például
kiskereskedelmi termékek, zene vagy hírek ajánlásánál.

Bizonyos esetekben a felhasználók explicit visszajelzést adnak arról,
mennyire tetszett nekik egy adott termék
(például értékelések és szöveges vélemények formájában az Amazonon, az
IMDb-n vagy a Goodreads-en). Más esetekben implicit visszajelzést
kapunk, például amikor valaki átugor egy számot egy lejátszási listán;
ez utalhat elégedetlenségre, de arra is, hogy a dal egyszerűen nem illett
a helyzethez. A legegyszerűbb megfogalmazásokban ezeket a rendszereket
arra tanítjuk, hogy valamilyen pontszámot becsüljenek, például egy
várható csillagértékelést vagy annak valószínűségét, hogy egy adott
felhasználó megvásárol egy adott terméket.

Egy ilyen modell birtokában bármely felhasználóhoz lekérhetjük a
legnagyobb pontszámú objektumokat, és ezeket ajánlhatjuk neki. A
valós üzemi rendszerek ennél jóval fejlettebbek, és az ilyen pontszámok
kiszámításakor részletesen figyelembe veszik a felhasználói aktivitást és
az elemek tulajdonságait is. :numref:`fig_deeplearning_amazon` azokat a
deep learning könyveket mutatja, amelyeket az Amazon Aston preferenciáira
hangolt személyre szabási algoritmusok alapján ajánlott.

![Az Amazon által ajánlott deep learning könyvek.](../img/deeplearning-amazon.jpg)
:label:`fig_deeplearning_amazon`

Óriási gazdasági értékük ellenére a prediktív modellekre naivan épített
ajánlórendszereknek komoly fogalmi hibáik vannak. Először is csak
*cenzúrázott visszajelzést* figyelünk meg: a felhasználók hajlamosak
azokat a filmeket értékelni, amelyekkel kapcsolatban erős érzéseik
vannak. Például egy ötfokozatú skálán észreveheted, hogy sok egy- és
ötcsillagos értékelés születik, de feltűnően kevés a háromcsillagos.
Ráadásul a jelenlegi vásárlási szokások gyakran a már működésben lévő
ajánlóalgoritmus következményei, de a tanulóalgoritmusok ezt a részletet
nem mindig veszik figyelembe. Így könnyen kialakulhatnak
visszacsatolási hurkok, amikor az ajánlórendszer előnyben részesít egy
terméket, amelyet aztán a nagyobb vásárlásszám miatt még jobbnak hiszünk,
és ennek következtében még gyakrabban ajánljuk. Sok ilyen probléma --- a
cenzúrázás, az ösztönzők és a visszacsatolási hurkok kezelése --- fontos,
nyitott kutatási kérdés.

#### Sorozattanulás

Eddig olyan problémákat vizsgáltunk, ahol adott számú bemenetből adott
számú kimenetet állítunk elő. Például házárakat jósoltunk egy rögzített
jellemzőhalmaz alapján: alapterület, hálószobák száma, fürdőszobák száma
és a belvárosba jutás ideje. Beszéltünk arról is, hogyan képezzünk le egy
(rögzített méretű) képet annak valószínűségeire, hogy egy adott, fix
számú osztály valamelyikébe tartozik, és hogyan jósoljunk vásárlásokhoz
tartozó csillagértékeket pusztán a felhasználó azonosítója és a termék
azonosítója alapján. Ezekben az esetekben, amint a modellünk betanult,
minden tesztpéldát, miután átment a modellen, azonnal el is felejtünk.
Azt feltételeztük, hogy az egymást követő megfigyelések függetlenek,
ezért nem volt szükség a kontextus megőrzésére.

De hogyan kezeljük a videórészleteket? Itt minden részlet eltérő számú
képkockából állhat. És az egyes képkockák értelmezésére vonatkozó
becslésünk sokkal jobb lehet, ha figyelembe vesszük az előző vagy a
következő képkockákat is. Ugyanez igaz a nyelvre is. Az egyik népszerű
deep learning feladat például a gépi fordítás: olyan mondatok fogadása,
amelyek egy forrásnyelven íródtak, és ezek fordításának előrejelzése egy
másik nyelven.

Ilyen problémák az orvostudományban is előfordulnak. Előfordulhat, hogy
olyan modellt szeretnénk, amely az intenzív osztályon figyeli a
betegeket, és riasztást küld, amikor a következő 24 órában bekövetkező
halálozás kockázata egy küszöb fölé emelkedik. Ilyenkor nem dobhatjuk ki
minden órában mindazt, amit a beteg előzményeiről tudunk, mert nem
akarunk kizárólag a legfrissebb mérések alapján előrejelzést készíteni.

Az ehhez hasonló kérdések a gépi tanulás legizgalmasabb alkalmazásai
közé tartoznak, és mind a *sorozattanulás* példái. Ezek olyan modelleket
igényelnek, amelyek vagy bemeneti sorozatokat fogadnak, vagy kimeneti
sorozatokat állítanak elő
(vagy mindkettőt). Konkrétabban a *sequence-to-sequence learning* olyan
problémákat vizsgál, ahol mind a bemenetek, mind a kimenetek változó
hosszúságú sorozatokból állnak. Ilyen például a gépi fordítás vagy a
beszéd--szöveg átírás. Bár lehetetlen a sorozattranszformációk minden
fajtáját áttekinteni, a következő speciális eseteket érdemes megemlíteni.

**Címkézés és elemzés**.
Itt egy szövegsorozatot látunk el attribútumokkal. A bemenetek és a
kimenetek *illeszkednek* egymáshoz, vagyis ugyanannyi elemük van, és
azonos sorrendben követik egymást. Például *szófaji címkézésben*
(PoS-tagging) a mondat minden szavához hozzárendeljük a megfelelő
szófajt, például "főnév" vagy "tárgy". Másik lehetőségként azt akarhatjuk
megtudni, hogy egymást követő szavak mely csoportjai utalnak tulajdonnévi
entitásokra, például *személyekre*, *helyekre* vagy *szervezetekre*. Az
alábbi játékosan egyszerű példában csak azt szeretnénk jelezni, hogy a
mondat egyes szavai részei-e valamilyen névvel megjelölt entitásnak
("Ent" címkével).

```text
Tom has dinner in Washington with Sally
Ent  -    -    -     Ent      -    Ent
```


**Automatikus beszédfelismerés**.
Beszédfelismerésnél a bemeneti sorozat egy beszélő hangfelvétele
(:numref:`fig_speech`), a kimenet pedig annak átirata, amit a beszélő
mondott. A kihívás az, hogy sokkal több hangkeret van
(a hangot jellemzően 8kHz vagy 16kHz frekvencián mintavételezik), mint
szöveges egység, vagyis nincs 1:1 megfelelés a hang és a szöveg között,
mivel több ezer minta is tartozhat egyetlen kimondott szóhoz. Ezek tehát
olyan sequence-to-sequence tanulási problémák, ahol a kimenet jóval
rövidebb, mint a bemenet. Bár az emberek elképesztően jól ismerik fel a
beszédet még rossz minőségű hangfelvételekből is, a számítógépekkel
ugyanezt elérni komoly kihívás.

![`-D-e-e-p- L-ea-r-ni-ng-` egy hangfelvételben.](../img/speech.png)
:width:`700px`
:label:`fig_speech`

**Szöveg felolvasása beszéddé**.
Ez az automatikus beszédfelismerés inverze. Itt a bemenet szöveg, a
kimenet pedig egy hangfájl. Ebben az esetben a kimenet jóval hosszabb,
mint a bemenet.

**Gépi fordítás**.
A beszédfelismeréssel ellentétben, ahol a megfelelő bemenetek és
kimenetek ugyanabban a sorrendben jelennek meg, a gépi fordításban az
illesztetlen adatok új kihívást jelentenek. Itt a bemeneti és kimeneti
sorozatok hossza eltérhet, és a megfelelő részek más sorrendben is
megjelenhetnek. Tekintsük például a német nyelv arra hajlamos szerkezetét,
hogy az igéket a mondat végére helyezi:

```text
German:           Haben Sie sich schon dieses grossartige Lehrwerk angeschaut?
English:          Have you already looked at this excellent textbook?
Wrong alignment:  Have you yourself already this excellent textbook looked at?
```


Sok kapcsolódó probléma más tanulási feladatokban is felbukkan. Például
annak meghatározása, hogy a felhasználó milyen sorrendben olvas el egy
weboldalt, kétdimenziós elrendezéselemzési probléma. A párbeszédes
feladatok mindenféle további bonyodalmat hordoznak, ahol annak
eldöntéséhez, hogy mit kellene mondani legközelebb, figyelembe kell venni
a valós világról szóló tudást és a beszélgetés korábbi állapotát is, akár
hosszú időbeli távolságokon keresztül. Ezek ma is aktív kutatási
területek.

### Felügyelet nélküli és önfelügyelt tanulás

Az előző példák a felügyelt tanulásra összpontosítottak, ahol egy olyan
hatalmas adathalmazt adunk a modellnek, amely tartalmazza mind a
jellemzőket, mind a hozzájuk tartozó címkéket. Úgy is gondolhatsz a
felügyelt tanulóra, mint valakire, akinek rendkívül specializált munkája
és rendkívül diktatórikus főnöke van. A főnök a válla fölött áll, és
minden helyzetben pontosan megmondja, mit kell tennie, egészen addig,
amíg meg nem tanulja, hogyan képezze le a helyzeteket cselekvésekre. Egy
ilyen főnöknek dolgozni elég kellemetlennek hangzik. Másfelől viszont
egy ilyen főnököt könnyű kielégíteni: csak minél gyorsabban fel kell
ismerned a mintát, és utánoznod kell a főnök viselkedését.

Az ellenkező helyzet viszont frusztráló lehet: amikor a főnöknek fogalma
sincs, mit akar, hogy csinálj. Ha azonban adattudós szeretnél lenni,
jobb, ha hozzászoksz ehhez. Lehet, hogy a főnök csak letesz eléd egy
hatalmas adag adatot, és annyit mond: *csinálj vele valami adattudományt!*
Ez homályosan hangzik, mert valóban az. Ezt a problémakört
*felügyelet nélküli tanulásnak* nevezzük, és az, hogy milyen kérdéseket
tehetünk fel, valamint hogy ezekből mennyit, valójában csak a
képzelőerőnkön múlik. A felügyelet nélküli tanulási technikákkal későbbi
fejezetekben foglalkozunk részletesebben. Az érdeklődés felkeltésére most
néhány olyan kérdést sorolunk fel, amelyet ilyenkor érdemes lehet
feltenni.

* Találhatunk-e kevés számú prototípust, amelyek pontosan összefoglalják
  az adatot? Ha van egy képhalmazunk, csoportosíthatjuk-e tájképekre,
  kutyákról, csecsemőkről, macskákról és hegycsúcsokról készült képekre?
  Ugyanígy: ha felhasználók böngészési aktivitásának gyűjteménye áll
  rendelkezésünkre, csoportosíthatjuk-e őket hasonló viselkedésű
  felhasználók szerint? Ezt a problémát tipikusan *klaszterezésnek*
  nevezzük.
* Találhatunk-e kevés számú paramétert, amelyek pontosan megragadják az
  adatok releváns tulajdonságait? Egy labda mozgását például jól leírhatja
  a sebessége, átmérője és tömege. A szabók is kialakítottak néhány olyan
  paramétert, amelyek viszonylag pontosan leírják az emberi testalakot a
  ruhák illesztése céljából. Az ilyen problémákat *altérbecslésnek*
  nevezzük. Ha a kapcsolat lineáris, akkor *főkomponens-analízisről*
  beszélünk.
* Létezik-e tetszőlegesen strukturált objektumoknak olyan reprezentációja
  az euklideszi térben, amelyben a szimbolikus tulajdonságok jól
  megfeleltethetők? Ez használható entitások és kapcsolataik leírására,
  például "Róma" $-$ "Olaszország" $+$ "Franciaország" $=$ "Párizs".
* Létezik-e leírása annak, hogy a megfigyelt adatok nagy részének mik a
  gyökérokai? Ha például vannak demográfiai adataink a lakásárakról, a
  szennyezésről, a bűnözésről, az elhelyezkedésről, az oktatásról és a
  fizetésekről, felfedezhetjük-e, hogyan kapcsolódnak egymáshoz pusztán
  empirikus adatok alapján? Az ilyen kérdésekkel az *okság* és a
  *valószínűségi grafikus modellek* területe foglalkozik.
* A felügyelet nélküli tanulás egy másik fontos és izgalmas új fejleménye
  a *mély generatív modellek* megjelenése. Ezek a modellek az adatok
  sűrűségét becsülik, vagy explicit módon, vagy *implicit* módon.
  Betanítás után egy generatív modellt használhatunk arra, hogy a példákat
  pontozzuk aszerint, mennyire valószínűek, vagy arra, hogy a megtanult
  eloszlásból szintetikus példákat mintavételezzünk. A generatív modellezés
  korai deep learning áttörései a *variációs autokódolóek*
  :cite:`Kingma.Welling.2014,rezende2014stochastic` feltalálásával
  kezdődtek, majd a *generatív versengő hálózatok*
  :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014` kifejlesztésével
  folytatódtak. Az újabb fejlesztések közé tartoznak a normalizing flow-k
  :cite:`dinh2014nice,dinh2017density` és a diffúziós modellek
  :cite:`sohl2015deep,song2019generative,ho2020denoising,song2021score`.



A felügyelet nélküli tanulás további fejlődési iránya az
*önfelügyelt tanulás* felemelkedése volt, vagyis olyan technikáké,
amelyek a címkézetlen adatok valamely tulajdonságát használják fel arra,
hogy felügyeletet biztosítsanak. Szövegnél például megtaníthatunk egy
modellt arra, hogy "kitöltse az üres helyeket" úgy, hogy véletlenszerűen
kitakart szavakat jósol meg a környező szavak
(kontextusok) alapján nagy korpuszokból, mindenféle címkézési erőfeszítés
nélkül :cite:`Devlin.Chang.Lee.ea.2018`.
Képeknél betaníthatunk modelleket arra, hogy megmondják két, ugyanabból a
képből kivágott részlet relatív helyzetét
:cite:`Doersch.Gupta.Efros.2015`, előrejelezzék egy kép kitakart részét a
maradék részek alapján, vagy eldöntsék, hogy két példa ugyanannak az
alapul szolgáló képnek a megzavart változata-e. Az önfelügyelt modellek
gyakran olyan reprezentációkat tanulnak, amelyeket aztán valamilyen
számunkra érdekes későbbi feladaton finomhangolva hasznosítunk.


### Kölcsönhatás a környezettel

Eddig nem beszéltünk arról, valójában honnan származnak az adatok, illetve
mi történik pontosan akkor, amikor egy gépi tanulási modell kimenetet
állít elő. Ennek az az oka, hogy a felügyelt és a felügyelet nélküli
tanulás ezeket a kérdéseket nem kezeli különösebben kifinomult módon.
Mindkét esetben előre összegyűjtünk egy nagy adag adatot, majd beindítjuk
a mintafelismerő gépeinket anélkül, hogy később újra kapcsolatba lépnénk
a környezettel. Mivel a teljes tanulás azután zajlik le, hogy az
algoritmus már elszakadt a környezettől, ezt néha *offline tanulásnak*
nevezzük. A felügyelt tanulás például a :numref:`fig_data_collection`
ábrán látható egyszerű interakciós mintát feltételezi.

![Adatgyűjtés felügyelt tanuláshoz egy környezetből.](../img/data-collection.svg)
:label:`fig_data_collection`

Az offline tanulás egyszerűsége kétségtelenül vonzó. Előnye, hogy a
mintafelismeréssel elszigetelten foglalkozhatunk, anélkül hogy a
dinamikus környezettel való kölcsönhatásból adódó bonyodalmak miatt
kellene aggódnunk. Ez a problémaformalizálás azonban korlátozó. Ha
Asimov robotregényein nőttél fel, valószínűleg olyan mesterségesen
intelligens ágenseket képzelsz el, amelyek nemcsak előrejelzéseket
készítenek, hanem cselekedni is tudnak a világban. Mi intelligens
*ágensekről* akarunk gondolkodni, nem pusztán prediktív modellekről. Ez
azt jelenti, hogy *cselekvések* kiválasztásáról kell gondolkodnunk, nem
csak előrejelzések készítéséről. Az egyszerű előrejelzésekkel ellentétben
a cselekvések ténylegesen befolyásolják a környezetet. Ha intelligens
ágenset akarunk tanítani, figyelembe kell vennünk, hogy a cselekvései
miként hathatnak az ágens jövőbeli megfigyeléseire, ezért az offline
tanulás itt nem megfelelő.

Ha komolyan vesszük a környezettel való kölcsönhatást, egy egész sor új
modellezési kérdés nyílik meg előttünk. Az alábbiak csak ízelítők:

* Emlékszik a környezet arra, mit tettünk korábban?
* Segíteni akar nekünk a környezet, például egy felhasználó, aki szöveget diktál egy beszédfelismerőnek?
* Le akar győzni minket a környezet, például a spammerek, akik a leveleiket a spam-szűrők kijátszásához igazítják?
* Változó dinamikájú a környezet? Például a jövőbeli adatok mindig hasonlítanak majd a múltra, vagy a minták idővel megváltoznak, akár természetes módon, akár az automatizált eszközeink hatására?

Ezek a kérdések felvetik az *eloszláseltolódás* problémáját, amikor a
tanító- és tesztadatok különböznek egymástól. Sokunk számára ismerős
példa erre az, amikor a vizsgát az oktató írja, a házi feladatokat
viszont a gyakorlatvezetők állítják össze. Ezután röviden bemutatjuk a
megerősítéses tanulást, amely gazdag keretrendszert ad olyan tanulási
problémák megfogalmazására, ahol egy ágens kölcsönhatásba lép a
környezetével.


### Megerősítéses tanulás

Ha gépi tanulással szeretnél olyan ágenst fejleszteni, amely
kölcsönhatásba lép a környezetével és cselekszik is benne, akkor nagy
valószínűséggel a *megerősítéses tanulásra* fogsz összpontosítani. Ez
magában foglalhat robotikai alkalmazásokat, párbeszédrendszereket, sőt
akár videójátékokhoz készült mesterséges intelligenciát is. A
*mély megerősítéses tanulás*, amely a deep learninget alkalmazza
megerősítéses tanulási problémákra, rendkívüli népszerűségre tett szert.
A vizuális bemenetre támaszkodó, embereket legyőző Atari-játékos deep
Q-network :cite:`mnih2015human`, valamint a Go világbajnokát letaszító
AlphaGo program :cite:`Silver.Huang.Maddison.ea.2016` két kiemelkedő
példa erre.

A megerősítéses tanulás nagyon általánosan fogalmaz meg egy olyan
problémát, amelyben egy ágens időlépések sorozatán keresztül lép
kölcsönhatásba a környezetével. Minden időlépésben az ágens valamilyen
*megfigyelést* kap a környezettől, és választania kell egy *cselekvést*,
amelyet aztán valamilyen mechanizmuson
(néha *végrehajtószervnek* nevezik) keresztül visszajuttatunk a
környezetnek, majd minden kör után az ágens jutalmat kap a környezettől.
Ezt a folyamatot :numref:`fig_rl-environment` szemlélteti. Az ágens ezután
újabb megfigyelést kap, újabb cselekvést választ, és így tovább. Egy
megerősítéses tanulási ágens viselkedését egy *policy* szabályozza.
Röviden, a *policy* egyszerűen egy függvény, amely a környezet
megfigyeléseit cselekvésekre képezi le. A megerősítéses tanulás célja jó
policy-k létrehozása.

![A megerősítéses tanulás és a környezet közötti kölcsönhatás.](../img/rl-environment.svg)
:label:`fig_rl-environment`

Nehéz túlbecsülni a megerősítéses tanulás keretrendszerének általánosságát.
Például a felügyelt tanulás is újrafogalmazható megerősítéses tanulásként.
Tegyük fel, hogy osztályozási problémánk van. Létrehozhatunk egy
megerősítéses tanuló ágenst, amelynek minden osztályhoz tartozik egy
lehetséges cselekvése. Ezután létrehozhatunk egy környezetet, amelynek
jutalma pontosan megegyezik az eredeti felügyelt tanulási probléma
veszteségfüggvényével.

Továbbá a megerősítéses tanulás sok olyan problémát is kezelni tud,
amelyet a felügyelt tanulás nem. Felügyelt tanulásban például mindig azt
feltételezzük, hogy a tanítási bemenethez tartozik a helyes címke.
A megerősítéses tanulásban viszont nem feltételezzük, hogy a környezet
minden megfigyeléshez megmondja az optimális cselekvést. Általában csak
valamilyen jutalmat kapunk. Ráadásul a környezet még azt sem feltétlenül
árulja el, mely cselekvések vezettek ehhez a jutalomhoz.

Gondolj a sakkra. Az egyetlen valódi jutalomjel a játék végén érkezik,
amikor vagy nyerünk, mondjuk $1$ jutalomért, vagy veszítünk, mondjuk
$-1$ jutalomért. Ezért a megerősítéses tanulóknak meg kell küzdeniük a
*jóváírás kiosztásának* problémájával: annak eldöntésével, hogy mely
cselekvésekért jár dicséret vagy hibáztatás az adott kimenet alapján.
Ugyanez igaz egy alkalmazottra is, akit október 11-én előléptetnek.
Ez az előléptetés valószínűleg az előző évben hozott sok jó döntés
eredménye. Ahhoz, hogy a jövőben is előléptessék, ki kell derítenie,
hogy útközben mely cselekvések vezettek a korábbi sikerhez.

A megerősítéses tanulóknak a részleges megfigyelhetőség problémájával is
szembe kell nézniük. Vagyis az aktuális megfigyelés nem feltétlenül mond el
mindent a jelenlegi állapotról. Képzeld el, hogy a takarítórobotod a házad
sok egyforma gardróbszekrénye közül az egyikben rekedt. A robot
kimentéséhez ki kell következtetni a pontos helyét, amihez lehet, hogy
figyelembe kell venni a szekrénybe jutás előtti korábbi megfigyeléseit is.

Végül egy adott időpontban a megerősítéses tanuló ismerhet egy jó
policy-t, de létezhet sok még jobb is, amelyet az ágens soha nem próbált
ki. A megerősítéses tanulónak folyamatosan döntenie kell arról, hogy a
jelenleg ismert legjobb stratégiát *kiaknázza-e* mint policy-t, vagy
inkább *feltérképezi* a stratégiák terét, esetleg lemondva némi rövid távú
jutalomról a tudás megszerzéséért cserébe.

Az általános megerősítéses tanulási probléma nagyon tág keretet ad.
A cselekvések hatással vannak a későbbi megfigyelésekre. Jutalmat csak
akkor figyelünk meg, amikor az megfelel a kiválasztott cselekvéseknek.
A környezet lehet teljesen vagy csak részben megfigyelhető. Mindezt az
összetettséget egyszerre kezelni sokszor túl nagy feladat lenne.
Ráadásul nem minden gyakorlati probléma hordozza magában ezt az összes
bonyolultságot. Emiatt a kutatók a megerősítéses tanulási problémák számos
speciális esetét tanulmányozták.

Ha a környezet teljesen megfigyelhető, a megerősítéses tanulási
problémát *Markov-döntési folyamatnak* nevezzük. Ha az állapot nem függ a
korábbi cselekvésektől, *kontextuális többkarú rabló* problémáról
beszélünk. Ha nincs állapot, csak elérhető cselekvések halmaza kezdetben
ismeretlen jutalmakkal, akkor a klasszikus *multi-armed bandit* problémát
kapjuk.

## Gyökerek

Most csupán a gépi tanulás által kezelhető problémák kis részét
tekintettük át. A gépi tanulási feladatok sokféle körében a deep learning
erőteljes eszközöket kínál a megoldásukhoz. Bár számos deep learning
módszer viszonylag új találmány, az adatokból való tanulás alapötleteit
évszázadok óta vizsgálják. Valójában az emberek régóta vágynak arra, hogy
adatokat elemezzenek és a jövőbeli kimeneteket előre jelezzék, és ez a
vágy a természettudományok és a matematika jelentős részének gyökerénél
ott található. Két példa erre a
[Jacob Bernoulli (1655--1705)](https://en.wikipedia.org/wiki/Jacob_Bernoulli)
nevét viselő Bernoulli-eloszlás, illetve a
[Carl Friedrich Gauss (1777--1855)](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss)
által felfedezett Gauss-eloszlás. Gauss például feltalálta a legkisebb
négyzetek algoritmusát, amelyet ma is sokféle problémában használnak a
biztosítási számításoktól az orvosi diagnosztikáig. Az ilyen eszközök
megerősítették a természettudományok kísérleti megközelítését --- például
Ohm törvénye, amely az ellenálláson átfolyó áram és a feszültség
kapcsolatát írja le, tökéletesen ábrázolható lineáris modellel.

Már a középkorban is erős intuíciójuk volt a matematikusoknak a
becslésekről. Például
[Jacob Köbel (1460--1533)](https://www.maa.org/press/periodicals/convergence/mathematical-treasures-jacob-kobels-geometry)
geometriai könyve azt mutatja be, hogyan lehet 16 felnőtt férfi
lábhosszának átlagolásával megbecsülni a népesség tipikus lábhosszát
(:numref:`fig_koebel`).

![A lábhossz becslése.](../img/koebel.jpg)
:width:`500px`
:label:`fig_koebel`


Amikor egy csoport ember kilépett a templomból, 16 felnőtt férfit arra
kértek, hogy álljanak sorba, és mérjék meg a lábukat. A mérések összegét
ezután elosztották 16-tal, hogy becslést kapjanak arra, amit ma egy
lábnak nevezünk. Ezt az "algoritmust" később továbbfejlesztették a
torz lábak kezelésére; a két legrövidebb és leghosszabb lábú férfit
kivették a mintából, és csak a maradék átlagát számolták. Ez a csonkolt
átlag becslésének egyik legkorábbi példája.

A statisztika igazán akkor kapott lendületet, amikor az adatok elérhetővé
váltak és gyűjteni kezdték őket. Egyik úttörője,
[Ronald Fisher (1890--1962)](https://en.wikipedia.org/wiki/Ronald_Fisher),
jelentős hozzájárulást tett mind az elmélethez, mind a genetikai
alkalmazásokhoz. Számos algoritmusa
(például a lineáris diszkriminanciaanalízis) és fogalma
(például a Fisher-információs mátrix) ma is hangsúlyos helyet foglal el a
modern statisztika alapjai között. Még az általa közzétett adathalmazoknak
is tartós hatásuk volt. A Fisher által 1936-ban publikált Iris-adathalmazt
ma is időnként gépi tanulási algoritmusok bemutatására használják.
Fisher az eugenika híve is volt, ami emlékeztet bennünket arra, hogy az
adattudomány erkölcsileg kétes felhasználásának legalább olyan hosszú és
tartós története van, mint ipari és természettudományos, hasznos
alkalmazásának.


A gépi tanulásra további nagy hatást gyakorolt
[Claude Shannon (1916--2001)](https://en.wikipedia.org/wiki/Claude_Shannon)
információelmélete és
[Alan Turing (1912--1954)](https://en.wikipedia.org/wiki/Alan_Turing)
számításelmélete. Turing híres *Computing Machinery and Intelligence*
című cikkében :cite:`Turing.1950` feltette a kérdést:
"képesek-e a gépek gondolkodni?" A ma Turing-tesztként ismert gondolatot
leírva azt javasolta, hogy egy gépet akkor tekinthetünk
*intelligensnek*, ha egy emberi értékelő pusztán szöveges interakciók
alapján nehezen tud különbséget tenni a gép és az ember válaszai között.

További fontos inspirációk érkeztek az idegtudományból és a
pszichológiából. Végül is az emberek nyilvánvalóan intelligens
viselkedést mutatnak. Sok kutató tette fel a kérdést, vajon meg lehet-e
magyarázni, és talán vissza is lehet-e fejteni ezt a képességet. Az egyik
első, biológia által inspirált algoritmust
[Donald Hebb (1904--1985)](https://en.wikipedia.org/wiki/Donald_O._Hebb)
fogalmazta meg. Úttörő jelentőségű könyvében,
*The Organization of Behavior* :cite:`Hebb.1949`, azt állította, hogy a
neuronok pozitív megerősítéssel tanulnak. Ez vált ismertté Hebb-féle
tanulási szabályként. Ezek az ötletek inspirálták a későbbi munkákat,
például Rosenblatt perceptron tanulási algoritmusát, és megalapozták sok
olyan sztochasztikus gradienscsökkenéses algoritmus működését, amely ma a
deep learning alapját adja: erősítsük a kívánatos viselkedést, és
gyengítsük a nem kívánatost, hogy jó paraméterbeállításokat találjunk egy
neurális hálózat számára.

A biológiai inspiráció adta a *neurális hálózatok* nevét is. Több mint
egy évszázada
(egészen Alexander Bain 1873-as és James Sherrington 1890-es modelljeiig
visszamenően) a kutatók olyan számítási áramköröket próbálnak
összeállítani, amelyek hasonlítanak az egymással kölcsönhatásban álló
neuronhálózatokra. Az idők során a biológiai értelmezés egyre kevésbé lett
szó szerinti, a név azonban megmaradt. A legtöbb ma használatos hálózat
szívében néhány kulcselv található:

* Lineáris és nemlineáris feldolgozóegységek váltakozása, amelyeket gyakran *rétegeknek* nevezünk.
* A láncszabály
  (más néven *visszaterjesztés*, visszaterjesztés) használata arra, hogy a
  teljes hálózat paramétereit egyszerre igazítsuk.

Az első gyors előrehaladás után a neurális hálózatok kutatása nagyjából
1995 és 2005 között megrekedt. Ennek főként két oka volt. Először is, egy
hálózat betanítása számításilag nagyon költséges. Bár a múlt század végére
a véletlen hozzáférésű memória bőséges lett, a számítási teljesítmény
szűkös maradt. Másodszor, az adathalmazok viszonylag kicsik voltak.
Valójában Fisher 1936-os Iris-adathalmaza továbbra is népszerű eszköz volt
az algoritmusok hatékonyságának tesztelésére. A 60,000 kézzel írt
számjegyet tartalmazó MNIST-adathalmaz hatalmasnak számított.

Az adatok és a számítási kapacitás szűkössége mellett az erős statisztikai
eszközök, például a kernelmódszerek, döntési fák és grafikus modellek sok
alkalmazásban empirikusan jobbnak bizonyultak. Ráadásul a neurális
hálózatokkal ellentétben nem kellett őket hetekig tanítani, és erős
elméleti garanciák mellett kiszámítható eredményeket adtak.


## Az út a deep learningig

Mindez nagyrészt akkor változott meg, amikor óriási mennyiségű adat vált
elérhetővé a világhálónak, a több százmillió felhasználót kiszolgáló
vállalatok megjelenésének, az olcsó, jó minőségű érzékelők elterjedésének,
az olcsó adattárolásnak
(Kryder-törvény) és a kedvező árú számítási teljesítménynek
(Moore-törvény) köszönhetően. Különösen a deep learning számítási
világát forradalmasították azok a GPU-fejlesztések, amelyeket eredetileg
számítógépes játékokhoz terveztek. Hirtelen olyan algoritmusok és modellek,
amelyek korábban számításilag megvalósíthatatlannak tűntek, elérhető közelségbe
kerültek. Ezt jól szemlélteti :numref:`tab_intro_decade`.

:Adathalmazok, számítógépmemória és számítási teljesítmény összevetése
:label:`tab_intro_decade`

|Évtized|Adathalmaz|Memória|Lebegőpontos műveletek másodpercenként|
|:--|:-|:-|:-|
|1970|100 (Iris)|1 KB|100 KF (Intel 8080)|
|1980|1 K (bostoni lakásárak)|100 KB|1 MF (Intel 80186)|
|1990|10 K (optikai karakterfelismerés)|10 MB|10 MF (Intel 80486)|
|2000|10 M (weboldalak)|100 MB|1 GF (Intel Core)|
|2010|10 G (hirdetések)|1 GB|1 TF (NVIDIA C2050)|
|2020|1 T (közösségi hálózat)|100 GB|1 PF (NVIDIA DGX-2)|


Figyeld meg, hogy a véletlen hozzáférésű memória növekedése nem tartott
lépést az adatmennyiség növekedésével. Ugyanakkor a számítási teljesítmény
gyorsabban nőtt, mint az adathalmazok mérete. Ez azt jelenti, hogy a
statisztikai modelleknek memóriahatékonyabbá kell válniuk, viszont a
megnövekedett számítási keret miatt több gépidőt fordíthatnak a
paraméterek optimalizálására. Ennek következtében a gépi tanulás és a
statisztika "édes pontja" az
(általánosított) lineáris modellek és kernelmódszerek felől a mély
neurális hálózatok felé tolódott. Ez az egyik oka annak is, hogy a deep
learning sok alappillére, mint a többrétegű perceptronok
:cite:`McCulloch.Pitts.1943`, a konvolúciós neurális hálózatok
:cite:`LeCun.Bottou.Bengio.ea.1998`, a hosszú rövid távú memória
:cite:`Hochreiter.Schmidhuber.1997` és a Q-learning
:cite:`Watkins.Dayan.1992`, lényegében "újrafelfedeződtek" az elmúlt
évtizedben, miután hosszú ideig viszonylag tétlenül hevertek.

A statisztikai modellek, alkalmazások és algoritmusok közelmúltbeli
fejlődését néha a kambriumi robbanáshoz hasonlítják: a fajok fejlődésének
egy rendkívül gyors szakaszához. A mai csúcsteljesítmény valóban nem
pusztán annak következménye, hogy több erőforrást adtunk évtizedek óta
ismert algoritmusok alá. Az alábbi ötletlista csupán a felszínt karcolja
abból, mi minden segítette a kutatókat abban, hogy az elmúlt évtizedben
ilyen óriási előrelépést érjenek el.


* Az új kapacitásszabályozási módszerek, például a *dropout*
  :cite:`Srivastava.Hinton.Krizhevsky.ea.2014`,
  segítettek csökkenteni a túlillesztést. Ilyenkor a tanítás során zajt
  injektálunk :cite:`Bishop.1995` a neurális hálózat különböző pontjaira.
* A *figyelemmechanizmusok* megoldottak egy másik problémát is, amely több
  mint egy évszázada gyötörte a statisztikát: hogyan lehet növelni egy
  rendszer memóriáját és összetettségét anélkül, hogy növelnénk a tanulható
  paraméterek számát. A kutatók elegáns megoldást találtak egy
  *tanulható mutatószerkezet* :cite:`Bahdanau.Cho.Bengio.2014`
  alkalmazásával. Például gépi fordításnál, ahelyett hogy egy teljes
  szövegsorozatot kellett volna megjegyezni egy rögzített dimenziós
  reprezentációban, elegendő volt egy mutatót tárolni a fordítási folyamat
  köztes állapotára. Ez jelentősen növelte a hosszú sorozatok pontosságát,
  mivel a modellnek többé nem kellett a teljes sorozatot megjegyeznie,
  mielőtt új sorozatot generálhatott volna.
* A kizárólag figyelemmechanizmusokra épülő *Transformer* architektúra
  :cite:`Vaswani.Shazeer.Parmar.ea.2017` kiváló *skálázódási* viselkedést
  mutatott: jobban teljesít az adathalmaz méretének, a modell méretének és
  a tanítási számítási kapacitás növelésével
  :cite:`kaplan2020scaling`. Ez az architektúra lenyűgöző sikereket ért el
  a természetesnyelv-feldolgozásban
  :cite:`Devlin.Chang.Lee.ea.2018,brown2020language`, a számítógépes
  látásban :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,liu2021swin`, a
  beszédfelismerésben :cite:`gulati2020conformer`, a megerősítéses
  tanulásban :cite:`chen2021decision` és a gráfneurális hálózatokban
  :cite:`dwivedi2020generalization`. Például egyetlen, szövegen, képeken,
  csuklómozgásokon és gombnyomásokon előtanított Transformer képes lehet
  Atarit játszani, képeket feliratozni, csevegni és robotot vezérelni
  :cite:`reed2022generalist`.
* A *nyelvi modellek* a szövegsorozatok valószínűségeit modellezve képesek
  szöveget jósolni más szövegek alapján. Az adatok, a modellméret és a
  számítási kapacitás skálázása egyre több képességet nyitott meg a nyelvi
  modellek számára, hogy emberhez hasonló szöveggenerálással hajtsanak
  végre feladatokat a bemeneti szöveg alapján
  :cite:`brown2020language,rae2021scaling,hoffmann2022training,chowdhery2022palm,openai2023gpt4,anil2023palm,touvron2023llama,touvron2023llama2`.
  Például a nyelvi modellek emberi szándékhoz igazítása
  :cite:`ouyang2022training` révén az OpenAI
  [ChatGPT](https://chat.openai.com/) rendszere beszélgetéses módon
  képes a felhasználókkal együttműködni olyan feladatokban, mint a
  hibakeresés a kódban vagy a kreatív írás.
* A többlépcsős tervek, például a memóriarendszerek
  :cite:`Sukhbaatar.Weston.Fergus.ea.2015`
  és a neurális programozó-értelmező
  :cite:`Reed.De-Freitas.2015` lehetővé tették, hogy a statisztikai
  modellezők iteratív következtetési megközelítéseket írjanak le. Ezek az
  eszközök lehetővé teszik, hogy a mély neurális hálózat belső állapota
  ismételten módosuljon, így a rendszer egymást követő lépéseket hajtson
  végre egy gondolatmenet láncában, ahogyan egy processzor is módosítja a
  memóriát egy számítás során.
* A *mély generatív modellezés* egyik kulcsfontosságú fejleménye a
  *generatív versengő hálózatok*
  :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`.
  Hagyományosan a sűrűségbecslésre és a generatív modellekre szolgáló
  statisztikai módszerek a megfelelő valószínűségi eloszlások és az azokból
  való mintavételhez szükséges
  (gyakran közelítő) algoritmusok keresésére összpontosítottak. Emiatt ezek
  az algoritmusok nagyrészt a statisztikai modellek korlátozott rugalmassága
  miatt voltak behatárolva. A GAN-ok kulcsfontosságú újítása az volt, hogy
  a mintavevőt egy tetszőleges, differenciálható paraméterekkel rendelkező
  algoritmussal helyettesítették. Ezeket aztán úgy igazítják, hogy a
  diszkriminátor
  (gyakorlatilag egy kétmintás teszt) ne tudja megkülönböztetni a hamis
  adatot a valós adatoktól. Az, hogy tetszőleges algoritmusokkal lehetett
  adatot generálni, a sűrűségbecslést a technikák széles választéka felé
  nyitotta meg. A vágtató zebrák :cite:`Zhu.Park.Isola.ea.2017` és a hamis
  celebportrék :cite:`Karras.Aila.Laine.ea.2017` egyaránt ennek a
  fejlődésnek a bizonyítékai. Ma már még amatőr firkálók is képesek
  fotórealisztikus képeket előállítani csupán a jelenet elrendezését
  leíró vázlatok alapján :cite:`Park.Liu.Wang.ea.2019`.
* Továbbá, miközben a diffúziós folyamat fokozatosan véletlen zajt ad az
  adatmintaelemekhez, a *diffúziós modellek*
  :cite:`sohl2015deep,ho2020denoising` megtanulják a zajtalanítás
  folyamatát, és így fokozatosan véletlen zajból állítanak elő
  adatmintákat, megfordítva a diffúziós folyamatot. Az újabb mély
  generatív modellekben elkezdték leváltani a generatív versengő
  hálózatokat, például a DALL-E 2-ben :cite:`ramesh2022hierarchical` és az
  Imagenben :cite:`saharia2022photorealistic`, amelyek kreatív művészetre
  és szöveges leírások alapján történő képgenerálásra használhatók.
* Sok esetben egyetlen GPU nem elegendő a tanításhoz rendelkezésre álló
  nagy adatmennyiségek feldolgozására. Az elmúlt évtizedben jelentősen
  javult a párhuzamos és elosztott tanítóalgoritmusok felépítésének
  képessége. A skálázható algoritmusok egyik kulcsproblémája, hogy a deep
  learning optimalizálásának igáslova, a sztochasztikus
  gradienscsökkenés, viszonylag kis mini-batch-ek feldolgozására támaszkodik.
  Ugyanakkor a kis batch-ek korlátozzák a GPU-k hatékonyságát. Ezért az
  1,024 GPU-n történő tanítás, mondjuk 32 képes mini-batch-ekkel, nagyjából
  32,000 képből álló aggregált mini-batch-nek felel meg. Az előbb
  :citet:`Li.2017`, majd :citet:`You.Gitman.Ginsburg.2017` és
  :citet:`Jia.Song.He.ea.2018` által végzett munkák 64,000 megfigyelésig
  tolták fel ezt a méretet, és az ImageNet adathalmazon a ResNet-50
  tanítási idejét 7 perc alá szorították. Összehasonlításképpen: a
  tanítási idők kezdetben napokban voltak mérhetők.
* A számítás párhuzamosításának képessége a *megerősítéses tanulásban* is
  hozzájárult a fejlődéshez. Ez jelentős előrelépést hozott abban, hogy a
  számítógépek emberfeletti teljesítményt érjenek el olyan feladatokban,
  mint a Go, az Atari-játékok, a StarCraft vagy fizikai szimulációk
  (például MuJoCo használatával), ahol környezetszimulátorok állnak
  rendelkezésre. Lásd például :citet:`Silver.Huang.Maddison.ea.2016`
  munkáját az AlphaGo eredményeiről. Röviden: a megerősítéses tanulás
  akkor működik a legjobban, ha rengeteg
  (állapot, cselekvés, jutalom) hármas áll rendelkezésre. A szimuláció
  ezt lehetővé teszi.
* A deep learning keretrendszerek kulcsszerepet játszottak az ötletek
  terjesztésében. A neurális hálózatok modellezésére szolgáló, nyílt
  forráskódú keretrendszerek első generációjához tartozott a
  [Caffe](https://github.com/BVLC/caffe),
  [Torch](https://github.com/torch) és
  [Theano](https://github.com/Theano/Theano).
  Számos alapművet ezekkel az eszközökkel írtak. Ezeket mára felváltotta a
  [TensorFlow](https://github.com/tensorflow/tensorflow)
  (gyakran magas szintű API-ján, a
  [Kerasen](https://github.com/keras-team/keras) keresztül használva), a
  [CNTK](https://github.com/Microsoft/CNTK), a
  [Caffe 2](https://github.com/caffe2/caffe2) és az
  [Apache MXNet](https://github.com/apache/incubator-mxnet). A
  keretrendszerek harmadik generációját az úgynevezett *imperatív*
  deep learning eszközök alkotják; ezt a trendet sokak szerint a
  [Chainer](https://github.com/chainer/chainer) indította be, amely a
  modellek leírására a Python NumPy-hoz hasonló szintaxist használt. Ezt
  az ötletet átvette a [PyTorch](https://github.com/pytorch/pytorch), az
  MXNet [Gluon API-ja](https://github.com/apache/incubator-mxnet) és a
  [JAX](https://github.com/google/jax) is.


A jobb eszközöket fejlesztő rendszerkutatók és a jobb neurális
hálózatokat építő statisztikai modellezők közötti munkamegosztás sokat
egyszerűsített a dolgokon. Például egy lineáris logisztikus regressziós
modell betanítása 2014-ben még nem triviális házi feladatnak számított a
Carnegie Mellon University új gépi tanulásos PhD-hallgatói számára. Ma
ezt a feladatot már 10 sornál rövidebb kóddal is meg lehet oldani, így
szinte bármely programozó számára elérhetővé vált.


## Sikertörténetek

A mesterséges intelligencia hosszú múltra tekint vissza olyan
eredmények terén, amelyeket másképp nehéz lett volna elérni. Például az
optikai karakterfelismerést használó levélválogató rendszereket már az
1990-es évek óta alkalmazzák. Végső soron innen származik a híres,
kézzel írt számjegyekből álló MNIST-adathalmaz is. Ugyanez igaz a banki
csekkek olvasására és a hitelképesség értékelésére. A pénzügyi
tranzakciókat automatikusan ellenőrzik csalás szempontjából. Ez sok
e-kereskedelmi fizetési rendszer gerincét adja, például a PayPal, a
Stripe, az AliPay, a WeChat, az Apple, a Visa és a MasterCard esetében.
A sakkprogramok évtizedek óta versenyképesek. A gépi tanulás hajtja a
keresést, az ajánlást, a személyre szabást és a rangsorolást az
interneten. Más szóval a gépi tanulás mindenütt jelen van, még ha gyakran
észrevétlen is.

Csak a közelmúltban került az AI a reflektorfénybe, főként olyan
problémák megoldásai miatt, amelyeket korábban kezelhetetlennek tartottak,
és amelyek közvetlenül érintik a felhasználókat. Sok ilyen előrelépést a
deep learningnek tulajdonítanak.

* Az intelligens asszisztensek, mint az Apple Siri-je, az Amazon Alexa-ja
  és a Google asszisztense, képesek elfogadható pontossággal reagálni a
  kimondott kérésekre. Ez magában foglal egyszerűbb feladatokat, például a
  lámpák felkapcsolását, és összetettebbeket is, mint a fodrászidőpont
  szervezése vagy telefonos ügyfélszolgálati párbeszéd lebonyolítása. Ez
  talán a legszembetűnőbb jele annak, hogy az AI már most is hatással van
  az életünkre.
* A digitális asszisztensek egyik kulcseleme a pontos beszédfelismerés.
  Az ilyen rendszerek pontossága fokozatosan odáig nőtt, hogy bizonyos
  alkalmazásokban elérte az emberi szintet
  :cite:`Xiong.Wu.Alleva.ea.2018`.
* A tárgyfelismerés is óriási utat tett meg. Egy képen szereplő tárgy
  azonosítása 2010-ben még meglehetősen nehéz feladatnak számított. Az
  ImageNet benchmarkon az NEC Labs és az University of Illinois at
  Urbana-Champaign kutatói 28%-os top-5 hibaarányt értek el
  :cite:`Lin.Lv.Zhu.ea.2010`. 2017-re ez az arány 2.25%-ra csökkent
  :cite:`Hu.Shen.Sun.2018`. Hasonlóan lenyűgöző eredmények születtek a
  madárhangok azonosításában és a bőrrák diagnosztizálásában is.
* A játékokban nyújtott teljesítmény sokáig az emberi képességek
  mércéjének számított. A TD-Gammon óta, amely időbeli különbség alapú
  megerősítéses tanulással játszott backgammont, az algoritmikus és
  számítási fejlődés a legkülönfélébb alkalmazásokhoz vezetett. A
  backgammonhoz képest a sakk jóval összetettebb állapottérrel és
  cselekvéstérrel rendelkezik. A DeepBlue hatalmas párhuzamosítással,
  speciális hardverrel és a játékfa hatékony keresésével győzte le Garry
  Kaszparovot :cite:`Campbell.Hoane-Jr.Hsu.2002`. A Go még nehezebb a
  hatalmas állapottér miatt. Az AlphaGo 2015-ben érte el az emberi szintet
  a deep learning és a Monte Carlo faalapú mintavételezés kombinációjával
  :cite:`Silver.Huang.Maddison.ea.2016`. A pókerben az jelentette a
  kihívást, hogy az állapottér nagy és csak részben megfigyelhető
  (nem ismerjük az ellenfelek lapjait). A Libratus strukturált stratégiák
  segítségével felülmúlta az emberi teljesítményt pókerben
  :cite:`Brown.Sandholm.2017`.
* Az AI fejlődésének másik jele az önvezető járművek megjelenése. Bár a
  teljes autonómia még nincs kézzelfogható közelségben, kiváló előrelépés
  történt ezen a területen, és olyan vállalatok, mint a Tesla, az NVIDIA
  vagy a Waymo már részleges autonómiát nyújtó termékeket szállítanak. A
  teljes autonómiát az teszi olyan nehézzé, hogy a helyes vezetéshez
  érzékelni, következtetni és szabályokat kell a rendszerbe építeni.
  Jelenleg a deep learninget elsősorban ezeknek a problémáknak a vizuális
  aspektusában használják. A többit nagyrészt mérnökök hangolják finomra.

Ez még mindig csak a felszínt karcolja a gépi tanulás jelentős
alkalmazásait illetően. A robotika, a logisztika, a számítási biológia,
a részecskefizika és a csillagászat például leglenyűgözőbb közelmúltbeli
előrelépéseik legalább egy részét a gépi tanulásnak köszönhetik, amely
így egyre általánosabb eszközzé válik a mérnökök és kutatók kezében.

Nem műszaki cikkekben gyakran felmerül az AI-apokalipszis és az úgynevezett
*szingularitás* lehetősége. A félelem az, hogy a gépi tanulási rendszerek
valahogy öntudatra ébrednek, és programozóiktól függetlenül olyan
döntéseket hoznak majd, amelyek közvetlenül befolyásolják az emberek
életét. Bizonyos értelemben az AI már most is közvetlenül hat az emberek
megélhetésére: a hitelképességet automatikusan értékelik, az autopilot
rendszerek nagyrészt járműveket vezetnek, és az óvadékról szóló döntésekhez
is statisztikai adatokat használnak fel. Könnyedebb példa, hogy
megkérhetjük Alexát, hogy kapcsolja be a kávéfőzőt.

Szerencsére nagyon messze vagyunk egy olyan öntudatos AI-rendszertől,
amely tudatosan manipulálhatná emberi alkotóit. Először is az
AI-rendszereket konkrét, célorientált módon tervezik, tanítják és
telepítik. Bár viselkedésük keltheti az általános intelligencia
látszatát, a háttérben szabályok, heurisztikák és statisztikai modellek
kombinációja áll. Másodszor jelenleg egyszerűen nincsenek olyan
*általános mesterséges intelligenciára* alkalmas eszközök, amelyek
képesek lennének önmagukat fejleszteni, önmagukról gondolkodni, és saját
architektúrájukat módosítani, bővíteni és javítani általános feladatok
megoldása közben.

Sokkal sürgetőbb kérdés, hogyan használjuk az AI-t a mindennapi életben.
Valószínű, hogy sok rutinfeladat, amelyet jelenleg emberek végeznek,
automatizálható és automatizálni is fogják. A mezőgazdasági robotok
valószínűleg csökkentik majd a biogazdák költségeit, ugyanakkor a
betakarítást is automatizálják. Az ipari forradalom ezen szakasza mély
következményekkel járhat a társadalom nagy rétegeire, hiszen a betanított
munkák sok országban rengeteg embernek adnak megélhetést. Ráadásul a
statisztikai modellek, ha kellő körültekintés nélkül alkalmazzuk őket,
faji, nemi vagy életkori torzításokhoz vezethetnek, és jogos aggályokat
vethetnek fel az eljárási igazságossággal kapcsolatban, ha súlyos
döntések automatizálására használják őket. Fontos biztosítani, hogy ezeket
az algoritmusokat gondosan alkalmazzuk. A mai tudásunk alapján ez sokkal
sürgetőbb aggodalomnak tűnik, mint annak a lehetősége, hogy egy
rosszindulatú szuperintelligencia elpusztítja az emberiséget.


## A deep learning lényege

Eddig tág értelemben beszéltünk a gépi tanulásról. A deep learning a
gépi tanulásnak az a részhalmaza, amely sokrétegű neurális hálózatokon
alapuló modellekkel foglalkozik. Annyiban *mély*, hogy modelljei sok
*rétegnyi* transzformációt tanulnak meg. Bár ez elsőre szűknek hangozhat,
a deep learning elképesztően sokféle modellt, technikát,
problémaformalizálást és alkalmazást hívott életre. Sok intuíció
született a mélység előnyeinek magyarázatára. Tulajdonképpen minden gépi
tanulás sok rétegnyi számításból áll, amelyek közül az első a jellemzők
feldolgozását végzi. A deep learninget az különbözteti meg, hogy a sok
reprezentációs rétegben megtanult műveleteket közösen, adatokból tanulja.

Az eddig tárgyalt problémák, például a nyers hangjelből való tanulás, a
képek nyers pixelértékeiből való tanulás, vagy a tetszőleges hosszúságú
mondatok és idegen nyelvi megfelelőik közötti leképezés, tipikusan olyan
területek, ahol a deep learning kiemelkedik, a hagyományos módszerek
pedig elbuknak. Kiderült, hogy ezek a sokrétegű modellek úgy tudják
kezelni az alacsony szintű érzékelési adatokat, ahogyan a korábbi
eszközök nem voltak erre képesek. A deep learning módszerek talán
legfontosabb közös jellemzője az *end-to-end tanítás*. Vagyis ahelyett,
hogy külön-külön hangolt komponensekből raknánk össze a rendszert, inkább
felépítjük az egészet, majd együttesen hangoljuk a teljesítményét.
Például a számítógépes látásban a kutatók korábban elkülönítették a
*jellemzőtervezés* folyamatát a gépi tanulási modellek építésétől. A
Canny-éldetektor :cite:`Canny.1987` és Lowe SIFT-jellemzőkivonója
:cite:`Lowe.2004` több mint egy évtizeden át uralta a képekből
jellemzővektorokat előállító algoritmusokat. A régebbi időkben a gépi
tanulás alkalmazásának lényeges része az volt, hogy kézzel tervezett
transzformációkat találjunk, amelyek az adatokat sekély modellek számára
alkalmassá teszik. Sajnos az emberi leleményesség csak bizonyos pontig
jut el ahhoz a következetes értékeléshez képest, amelyet egy algoritmus
automatikusan végez el több millió választáson keresztül. Amikor a deep
learning átvette a szerepet, ezeket a jellemzőkivonókat automatikusan
hangolt szűrők váltották fel, amelyek jobb pontosságot értek el.

Így a deep learning egyik kulcsfontosságú előnye, hogy nemcsak a
hagyományos tanulási pipeline-ok végén álló sekély modelleket váltja le,
hanem a jellemzőtervezés munkaigényes folyamatát is. Ráadásul azzal, hogy
felváltja a területspecifikus előfeldolgozás nagy részét, a deep learning
ledöntötte azokat a határokat, amelyek korábban elválasztották egymástól
a számítógépes látást, a beszédfelismerést, a
természetesnyelv-feldolgozást, az orvosi informatikát és más alkalmazási
területeket, így egységes eszközkészletet kínál sokféle probléma
megoldására.

Az end-to-end tanításon túl azt is látjuk, hogy a parametrikus
statisztikai leírásokról fokozatosan átállunk a teljesen nemparametrikus
modellekre. Ha kevés az adat, hasznos modellekhez egyszerűsítő
feltevésekre kell támaszkodnunk a valóságról. Ha viszont bőségesen áll
rendelkezésre adat, ezeket felválthatják olyan nemparametrikus modellek,
amelyek jobban illeszkednek az adatokhoz. Ez bizonyos mértékig emlékeztet
arra a fejlődésre, amelyet a fizika a múlt század közepén a számítógépek
megjelenésével átélt. Az elektronok viselkedésének kézi, parametrikus
közelítése helyett ma már az ehhez tartozó parciális differenciálegyenletek
numerikus szimulációjára támaszkodhatunk. Ez sokkal pontosabb modellekhez
vezetett, noha gyakran az értelmezhetőség rovására.

Egy másik különbség a korábbi munkákhoz képest a szuboptimális
megoldások elfogadása, a nem konvex nemlineáris optimalizálási problémákkal
való együttélés, valamint az a hajlandóság, hogy kipróbáljunk dolgokat,
mielőtt bizonyítanánk őket. Ez az újfajta empirizmus a statisztikai
problémák kezelésében, együtt a tehetséges kutatók gyors beáramlásával,
gyors előrelépéshez vezetett a gyakorlati algoritmusok fejlesztésében,
még ha sokszor azzal az árral is, hogy évtizedek óta létező eszközöket
módosítunk vagy újrafeltalálunk.

Végső soron a deep learning közösség büszke arra, hogy az akadémiai és
vállalati határokon átívelve osztja meg az eszközöket, és számos kiváló
könyvtárat, statisztikai modellt és betanított hálózatot tesz közzé nyílt
forráskódként. E szellemiség jegyében a könyvet alkotó notebookok is
szabadon terjeszthetők és használhatók. Sok munkát fektettünk abba, hogy
csökkentsük a deep learning megismerésének belépési korlátait, és
reméljük, hogy olvasóink profitálni fognak ebből.


## Összefoglalás

A gépi tanulás azt vizsgálja, hogyan tudnak a számítógépes rendszerek a
tapasztalatból
(gyakran adatokból) tanulni, hogy javítsák teljesítményüket konkrét
feladatokban. Ötvözi a statisztika, az adatbányászat és az optimalizálás
ötleteit. Gyakran az AI-megoldások megvalósításának eszközeként
használjuk. A gépi tanulás egyik osztályaként a reprezentációtanulás
arra összpontosít, hogyan lehet automatikusan megtalálni az adatok
megfelelő reprezentációját. Többszintű reprezentációtanulásként
felfogva, amely sok egymásra épülő transzformációs réteget tanul meg, a
deep learning nemcsak a hagyományos gépi tanulási pipeline-ok végén álló
sekély modelleket váltja le, hanem a jellemzőtervezés munkaigényes
folyamatát is. A deep learning közelmúltbeli fejlődésének nagy részét az
olcsó érzékelőkből és az internetes méretű alkalmazásokból származó
adattömeg, valamint a számítási kapacitás jelentős fejlődése, főként a
GPU-kon keresztül, indította el. Emellett a hatékony deep learning
keretrendszerek elérhetősége jelentősen megkönnyítette a teljes rendszer
optimalizálásának tervezését és megvalósítását, ami a magas teljesítmény
egyik kulcstényezője.

## Feladatok

1. Az általad jelenleg írt kód mely részei lehetnének "tanulhatók",
   vagyis javíthatók tanulással és a kódban meghozott tervezési döntések
   automatikus meghatározásával? Tartalmaz-e a kódod heurisztikus
   tervezési döntéseket? Milyen adatokra lenne szükséged a kívánt
   viselkedés megtanulásához?
1. Azok közül a problémák közül, amelyekkel találkozol, melyekre létezik
   sok megoldási példa, mégsem ismert egyértelmű automatizálási módszer?
   Ezek kiváló jelöltek lehetnek a deep learning alkalmazására.
1. Írd le az algoritmusok, az adatok és a számítás kapcsolatát. Hogyan
   befolyásolják az adatok jellemzői és az aktuálisan rendelkezésre álló
   számítási erőforrások azt, hogy mely algoritmusok tekinthetők
   megfelelőnek?
1. Nevezz meg néhány olyan helyzetet, ahol az end-to-end tanítás jelenleg
   nem alapértelmezett megközelítés, de mégis hasznos lehetne.

[Beszélgetések](https://discuss.d2l.ai/t/22)
