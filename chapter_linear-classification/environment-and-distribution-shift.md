# Környezet és eloszláseltolódás
:label:`sec_environment-and-distribution-shift`

Az előző szakaszokban számos gyakorlati gépi tanulási alkalmazáson dolgoztunk,
modelleket illesztve különböző adathalmazokhoz.
Mégsem álltunk meg egy pillanatra sem elgondolkozni azon,
honnan származnak az adatok,
vagy mit tervezünk végső soron tenni
modelljeink kimeneteivel.
Túl gyakran előfordul, hogy az adatokkal rendelkező gépi tanulás fejlesztők
rohannak modelleket fejleszteni,
anélkül hogy megállnának e fundamentális kérdések megfontolásánál.

Számos sikertelen gépi tanulási telepítés
visszavezethető erre a mulasztásra.
Néha a modellek csodálatosan teljesítenek
a teszthalmaz pontossága alapján mérve,
de katasztrofálisan vallanak kudarcot a valós alkalmazásban,
amikor az adatok eloszlása hirtelen megváltozik.
Még alattomosabb esetekben maga a modell telepítése
lehet a katalizátor, amely megzavarja az adateloszlást.
Tegyük fel például, hogy tanítottunk egy modellt
annak előrejelzésére, ki fizeti vissza a hitelt és ki nem,
és azt találtuk, hogy a pályázó lábbeliválasztása
összefügg a nemfizetés kockázatával
(a félcipő visszafizetést, az edzőcipő nemfizetést jelez).
Ezután hajlamosak lennénk hitelt adni
minden félcipőt viselő pályázónak,
és elutasítani az edzőcipőt viselőket.

Ebben az esetben a mintafelismerésből döntéshozatalba való
elhamarkodott ugrásunk
és a környezet kritikai megfontolásának elmulasztása
katasztrofális következményekkel járhat.
Kezdetnek: amint a cipőviselet alapján kezdtünk döntéseket hozni,
az ügyfelek észre vennék és megváltoztatnák viselkedésüket.
Nem sokkal később minden pályázó félcipőt viselne,
hitelképességük tényleges javulása nélkül.
Állj meg egy percre, és emésztd meg ezt, mert hasonló problémák tömege
kísér sok gépi tanulási alkalmazást:
a modell alapú döntések bevezetésével a környezetbe
tönkre tehetjük a modellt.

Bár egy szakaszban nem nyújthatunk teljes körű tárgyalást e témákról,
célunk néhány általános aggodalmat feltárni,
és ösztönözni azt a kritikai gondolkodást,
amely szükséges az ilyen helyzetek korai felismeréséhez,
a kár enyhítéséhez és a gépi tanulás felelős alkalmazásához.
Néhány megoldás egyszerű
(kérd a „megfelelő" adatokat),
néhány technikailag nehéz
(valósíts meg egy megerősítéses tanulási rendszert),
mások pedig megkövetelik, hogy teljesen kilépjünk
a statisztikai előrejelzés területéről,
és nehéz filozófiai kérdésekkel küzdjünk
meg az algoritmusok etikus alkalmazásával kapcsolatban.

## Az eloszláseltolódás típusai

Kezdjük a passzív jóslási beállítással,
megvizsgálva, milyen módokon változhat az adatok eloszlása,
és mit lehet tenni a modell teljesítményének megőrzéséért.
Egy klasszikus beállításban feltételezzük, hogy a tanítási adatainkat
valamely $p_S(\mathbf{x},y)$ eloszlásból vettük,
de a tesztadataink
valamilyen más $p_T(\mathbf{x},y)$ eloszlásból vett
osztályozatlan példányokból állnak.
Már most szembe kell néznünk egy józanító valósággal:
$p_S$ és $p_T$ közötti kapcsolatra vonatkozó feltételezések nélkül
robusztus osztályozó tanítása lehetetlen.

Vegyünk egy bináris osztályozási feladatot,
ahol macskákat és kutyákat szeretnénk megkülönböztetni.
Ha az eloszlás tetszőleges módokon változhat,
akkor megengedett a patologikus eset,
amikor a bemenetek eloszlása változatlan marad:
$p_S(\mathbf{x}) = p_T(\mathbf{x})$,
de minden felirat felcserélődik:
$p_S(y \mid \mathbf{x}) = 1 - p_T(y \mid \mathbf{x})$.
Más szóval: ha Isten hirtelen eldönthetné,
hogy a jövőben minden „macska" kutyává válik,
és amit korábban „kutya"-nak hívtunk, most macska — bármiféle változás nélkül
a bemenetek $p(\mathbf{x})$ eloszlásában —
akkor semmiképpen sem tudnánk megkülönböztetni ezt a helyzetet
attól, amelyben az eloszlás egyáltalán nem változott.

Szerencsére néhány korlátozott feltételezés alatt
az adataink jövőbeli változásainak módjaira vonatkozóan
az elvhű algoritmusok képesek érzékelni az eltolódást,
és néha röptében alkalmazkodni is,
javítva az eredeti osztályozó pontosságán.

### Kovariáns eltolódás

Az eloszláseltolódás kategóriái közül
a kovariáns eltolódás talán a legszélesebb körben tanulmányozott.
Itt feltételezzük, hogy bár a bemenetek eloszlása változhat idővel,
a feliratozó függvény,
vagyis a feltételes eloszlás
$P(y \mid \mathbf{x})$ nem változik.
A statisztikusok ezt *kovariáns eltolódásnak* nevezik,
mert a probléma a kovariánsok (jellemzők) eloszlásának eltolódásából ered.
Bár néha kauzalitás feltételezése nélkül is gondolkodhatunk az eloszláseltolódásról,
a kovariáns eltolódás a természetes feltételezés azokban a beállításokban,
ahol úgy gondoljuk, hogy $\mathbf{x}$ okozza $y$-t.

Gondoljunk a macskák és kutyák megkülönböztetésének kihívására.
Tanítási adataink a :numref:`fig_cat-dog-train` ábrán látható típusú képekből állhatnak.

![Tanítási adatok macskák és kutyák megkülönböztetéséhez (illusztrációk: Lafeez Hossain / 500px / Getty Images; ilkermetinkursova / iStock / Getty Images Plus; GlobalP / iStock / Getty Images Plus; Musthafa Aboobakuru / 500px / Getty Images).](../img/cat-dog-train.png)
:label:`fig_cat-dog-train`


A tesztelési fázisban a :numref:`fig_cat-dog-test` ábrán szereplő képeket kell osztályozni.

![Tesztadatok macskák és kutyák megkülönböztetéséhez (illusztrációk: SIBAS_minich / iStock / Getty Images Plus; Ghrzuzudu / iStock / Getty Images Plus; id-work / DigitalVision Vectors / Getty Images; Yime / iStock / Getty Images Plus).](../img/cat-dog-test.png)
:label:`fig_cat-dog-test`

A tanítóhalmaz fényképekből áll,
míg a teszthalmaz kizárólag rajzfilmszerű képeket tartalmaz.
Ha egy lényegesen eltérő jellemzőkkel rendelkező adathalmazon tanítunk,
mint a teszthalmaz, gondok adódhatnak,
ha nincs koherens tervünk az új tartományhoz való alkalmazkodásra.

### Felirateltolódás

A *felirateltolódás* a fordított problémát írja le.
Itt feltételezzük, hogy a felirat marginális eloszlása $P(y)$
változhat,
de az osztály-feltételes eloszlás
$P(\mathbf{x} \mid y)$ tartományok között változatlan marad.
A felirateltolódás ésszerű feltételezés akkor,
ha úgy gondoljuk, hogy $y$ okozza $\mathbf{x}$-et.
Például orvosi diagnózisokat akarhatunk megjósolni
tüneteik (vagy egyéb megnyilvánulásaik) alapján,
miközben a diagnózisok relatív prevalenciája
idővel változik.
A felirateltolódás itt a megfelelő feltételezés,
mert a betegségek okozzák a tüneteket.
Néhány degenerált esetben a felirateltolódási
és kovariáns eltolódási feltételezések egyidejűleg teljesülhetnek.
Például ha a felirat determinisztikus,
a kovariáns eltolódás feltételezése teljesül,
még akkor is, ha $y$ okozza $\mathbf{x}$-et.
Érdekes módon ezekben az esetekben
sokszor előnyös a felirateltolódás feltételezéséből eredő
módszerekkel dolgozni.
Ez azért van, mert ezek a módszerek általában
feliratszerű objektumok manipulálásával járnak (amelyek sokszor alacsony dimenziósak),
szemben a bemeneti objektumokkal,
amelyek a mély tanulásban általában nagy dimenziósak.

### Fogalomeltolódás

Találkozhatunk a kapcsolódó *fogalomeltolódás* problémájával is,
amely akkor merül fel, amikor maguk a feliratdefiníciók változnak.
Ez furcsán hangzik — egy *macska* az egy *macska*, nem?
Más kategóriák azonban az idő múlásával változnak.
A mentális betegségek diagnosztikai kritériumai,
ami divatosnak számít, és a munkaköri megnevezések
mind jelentős mértékű fogalomeltolódásnak vannak kitéve.
Ha az Egyesült Államokon belül utazunk,
és adatforrásunkat földrajzilag változtatjuk,
jelentős fogalomeltolódást fogunk találni
a *szénsavas üdítők* elnevezéseinek eloszlásában,
ahogy a :numref:`fig_popvssoda` ábrán látható.

![Fogalomeltolódás az üdítők elnevezéseiben az Egyesült Államokban (CC-BY: Alan McConchie, PopVsSoda.com).](../img/popvssoda.png)
:width:`400px`
:label:`fig_popvssoda`

Ha gépi fordítórendszert építenénk,
a $P(y \mid \mathbf{x})$ eloszlás különböző lehet
a tartózkodási helyünktől függően.
Ez a probléma nehéz lehet észrevenni.
Remélhetőleg kihasználhatjuk azt a tudást,
hogy az eltolódás csak fokozatosan következik be
időbeli vagy földrajzi értelemben.

## Az eloszláseltolódás példái

Mielőtt belemerülnénk a formalizmusba és az algoritmusokba,
megbeszélhetünk néhány konkrét helyzetet,
ahol a kovariáns eltolódás vagy fogalomeltolódás esetleg nem nyilvánvaló.


### Orvosi diagnosztika

Képzeld el, hogy rákérzékelő algoritmust tervezel.
Adatokat gyűjtesz egészséges és beteg emberektől,
és megtanítod az algoritmusodat.
Jól működik, magas pontosságot ad,
és arra a következtetésre jutsz, hogy készen állsz
egy sikeres karrierre az orvosi diagnosztikában.
*Ne siess ennyire.*

Az adatok tanítóhalmazát létrehozó eloszlások
és azok, amelyekkel a valóságban találkozol, jelentősen eltérhetnek.
Ez történt egy szerencsétlen startuppal,
amellyel a szerzők néhányan együtt dolgoztak évekkel ezelőtt.
Vérvizsgálatot fejlesztettek egy betegségre,
amely főként idősebb férfiakat érint,
és vérminták felhasználásával kívánták tanulmányozni,
amelyeket már a rendszerben lévő betegektől gyűjtöttek.
Azonban egészséges férfiaktól vérmintát gyűjteni
lényegesen nehezebb, mint a már a rendszerben lévő beteg betegektől.
A kompenzáció érdekében a startup véradásra kért fel
egy egyetem kampuszán lévő diákokat,
hogy egészséges kontrollként szolgáljanak a teszt fejlesztésekor.
Majd megkérdezték, hogy tudunk-e segíteni
a betegség kimutatásának osztályozójának felépítésében.

Ahogy elmagyaráztuk nekik,
valóban könnyű lenne megkülönböztetni
az egészséges és beteg csoportokat
közel tökéletes pontossággal.
Ez azonban azért lehetséges, mert a vizsgált alanyok
különböztek korban, hormonszintekben,
fizikai aktivitásban, étrendben, alkoholfogyasztásban
és számos egyéb, a betegséggel nem összefüggő tényezőben.
Ez valószínűleg nem lett volna igaz valódi betegekre.
A mintavételi eljárásukból adódóan
szélsőséges kovariáns eltolódásra számíthattunk.
Ráadásul ez az eset aligha volt
javítható hagyományos módszerekkel.
Röviden: jelentős összeget pazaroltak el.



### Önvezető autók

Tegyük fel, hogy egy vállalat gépi tanulást szeretne felhasználni
önvezető autók fejlesztéséhez.
Az egyik kulcsfontosságú komponens itt egy útszéli detektor.
Mivel a valós annotált adatok drágák,
a vállalat azt az (okos, de kérdéses) ötletet fogta fel,
hogy szintetikus adatokat használ egy játék renderelőmotorból
kiegészítő tanítási adatként.
Ez nagyon jól működött a renderelőmotorból vett „tesztadatokon".
Sajnos valódi autóban katasztrófa volt.
Kiderült, hogy az útszélt
nagyon egyszerűsített textúrával renderelték.
Ráadásul *minden* útszélt *ugyanolyan* textúrával rendereltek,
és az útszéli detektor nagyon gyorsan megtanulta ezt a „jellemzőt".

Hasonló dolog történt az USA hadseregével,
amikor először próbáltak harckocsikat felderíteni az erdőben.
Légi felvételeket készítettek az erdőről tankok nélkül,
majd bevezényelték a tankokat az erdőbe
és egy újabb sorozat képet készítettek.
Az osztályozó úgy tűnt, *tökéletesen* működik.
Sajnálatos módon csupán annyit tanult meg,
hogyan kell megkülönböztetni az árnyékos fákat
az árnyék nélküliektől — az első sorozat képeket
kora reggel vették fel,
a másodikakat délben.

### Nem stacionárius eloszlások

Jóval finomabb helyzet adódik akkor,
ha az eloszlás lassan változik
(más néven *nem stacionárius eloszlás*),
és a modellt nem frissítik megfelelően.
Néhány tipikus eset:

* Tanítunk egy számítógépes hirdetési modellt, majd nem frissítjük megfelelő gyakorisággal (például elfelejtjük belefoglalni, hogy most jelent meg egy iPadnek nevezett alig ismert új eszköz).
* Spam-szűrőt építünk. Jól működik az összes eddig látott spam szűrésénél. De aztán az ​​​​​​​​​​​​​​​​levélszemét-küldők okosodnak, és új üzeneteket alkotnak, amelyek semmire sem hasonlítanak, amit korábban láttunk.
* Termékajánló rendszert építünk. Egész télen jól működik, de aztán karácsony után is folyamatosan Mikulás-sapkákat ajánl.

### További anekdoták

* Arcdetektort építünk. Jól működik minden referenciahalmazon. Sajnos megbukik a tesztadatokon — az ütköző példányok közelképek, ahol az arc az egész képet kitölti (ilyen adatok nem voltak a tanítóhalmazban).
* Webes keresőmotort építünk az USA piacára, majd az Egyesült Királyságban szeretnénk telepíteni.
* Képosztályozót tanítunk egy nagy adathalmaz összeállításával, ahol a nagy osztályok mindegyike egyenlő arányban szerepel, mondjuk 1000 kategória, egyenként 1000 képpel. Majd a valós világban telepítjük a rendszert, ahol a fényképek tényleges felirat eloszlása határozottan nem egyenletes.


## Az eloszláseltolódás korrekciója

Ahogy tárgyaltuk, számos eset van,
ahol a tanítási és tesztelési eloszlások
$P(\mathbf{x}, y)$ különböznek.
Néha szerencsénk van, és a modellek működnek
a kovariáns eltolódás, felirateltolódás vagy fogalomeltolódás ellenére.
Más esetekben elvhű stratégiák alkalmazásával
kezelhetjük az eltolódást.
E szakasz többi része lényegesen technikaibbá válik.
A türelmetlen olvasó folytathatja a következő szakasszal,
mivel ez az anyag nem előfeltétele a következő fogalmaknak.

### Empirikus kockázat és kockázat
:label:`subsec_empirical-risk-and-risk`

Először gondolkozzunk el pontosan azon,
mi történik a modell tanítása során:
a tanítási adatok jellemzőin és hozzájuk tartozó feliratain iterálunk
$\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$,
és minden mini-batch után frissítjük az $f$ modell paramétereit.
Az egyszerűség kedvéért nem vesszük figyelembe a regularizációt,
ezért nagyjából minimalizáljuk a tanítás veszteségét:

$$\mathop{\mathrm{minimize}}_f \frac{1}{n} \sum_{i=1}^n l(f(\mathbf{x}_i), y_i),$$
:eqlabel:`eq_empirical-risk-min`

ahol $l$ a veszteségfüggvény,
amely azt méri, „mennyire rossz" a $f(\mathbf{x}_i)$ jóslat az $y_i$ felirathoz képest.
A statisztikusok az :eqref:`eq_empirical-risk-min` kifejezést *empirikus kockázatnak* nevezik.
Az *empirikus kockázat* a tanítási adatokon számolt átlagos veszteség,
amely a *kockázatot* közelíti,
vagyis a veszteség várható értékét az összes adaton
a valódi $p(\mathbf{x},y)$ eloszlás szerint:

$$E_{p(\mathbf{x}, y)} [l(f(\mathbf{x}), y)] = \int\int l(f(\mathbf{x}), y) p(\mathbf{x}, y) \;d\mathbf{x}dy.$$
:eqlabel:`eq_true-risk`

A gyakorlatban azonban általában nem férünk hozzá az összes populációs adathoz.
Ezért az *empirikus kockázat minimalizálása*,
vagyis az :eqref:`eq_empirical-risk-min`-beli empirikus kockázat minimalizálása
egy praktikus stratégia a gépi tanulásban,
abban a reményben, hogy közelítőleg
minimalizálja a kockázatot.



### Kovariáns eltolódás korrekciója
:label:`subsec_covariate-shift-correction`

Tegyük fel, hogy egy $P(y \mid \mathbf{x})$ függőséget akarunk becsülni,
amelyhez osztályozott $(\mathbf{x}_i, y_i)$ adataink vannak.
Sajnos az $\mathbf{x}_i$ megfigyelések
valamely *forrás eloszlásból* $q(\mathbf{x})$
kerültek kihúzásra a *cél eloszlás* $p(\mathbf{x})$ helyett.
Szerencsére
a függőségi feltételezés azt jelenti,
hogy a feltételes eloszlás nem változik: $p(y \mid \mathbf{x}) = q(y \mid \mathbf{x})$.
Ha a forrás eloszlás $q(\mathbf{x})$ „rossz",
kiigazíthatjuk a következő egyszerű azonossággal a kockázatban:

$$
\begin{aligned}
\int\int l(f(\mathbf{x}), y) p(y \mid \mathbf{x})p(\mathbf{x}) \;d\mathbf{x}dy =
\int\int l(f(\mathbf{x}), y) q(y \mid \mathbf{x})q(\mathbf{x})\frac{p(\mathbf{x})}{q(\mathbf{x})} \;d\mathbf{x}dy.
\end{aligned}
$$

Más szóval minden adatpéldányt újra kell súlyoznunk
a helyes eloszlásból való kihúzás valószínűségének
és a helytelen eloszlásból való kihúzás valószínűségének arányával:

$$\beta_i \stackrel{\textrm{def}}{=} \frac{p(\mathbf{x}_i)}{q(\mathbf{x}_i)}.$$

A $\beta_i$ súly behelyettesítésével
minden $(\mathbf{x}_i, y_i)$ adatpéldányhoz
*súlyozott empirikus kockázat minimalizálásával* taníthatjuk a modellünket:

$$\mathop{\mathrm{minimize}}_f \frac{1}{n} \sum_{i=1}^n \beta_i l(f(\mathbf{x}_i), y_i).$$
:eqlabel:`eq_weighted-empirical-risk-min`



Sajnos nem ismerjük ezt az arányt,
tehát mielőtt bármit hasznosat tehetnénk, meg kell becsülnünk.
Számos módszer áll rendelkezésre,
beleértve néhány szép operátorelméleti megközelítést,
amelyek közvetlenül kalibrálják a várható értéket
minimum-norma vagy maximum-entrópia elvvel.
Megjegyezzük, hogy bármely ilyen megközelítéshez mintákra van szükségünk
mindkét eloszlásból — a „valódi" $p$-ből,
például tesztadatokhoz való hozzáféréssel,
és a $q$-ból, amelyet a tanítóhalmaz generálásához használtunk
(az utóbbi triviálisan elérhető).
Azonban figyelem: csak $\mathbf{x} \sim p(\mathbf{x})$ jellemzőkre van szükségünk;
$y \sim p(y)$ feliratokhoz nem kell hozzáférnünk.

Ebben az esetben egy nagyon hatékony megközelítés létezik,
amely szinte olyan jó eredményeket ad, mint az eredeti: nevezetesen a logisztikus regresszió,
amely a softmax regresszió egy speciális esete (lásd :numref:`sec_softmax`)
bináris osztályozásnál.
Ez minden, amire szükség van a becsült valószínűségi arányok kiszámításához.
Egy osztályozót tanítunk meg
a $p(\mathbf{x})$-ből vett adatok és a $q(\mathbf{x})$-ből vett adatok megkülönböztetésére.
Ha lehetetlen megkülönböztetni a két eloszlást,
ez azt jelenti, hogy az érintett példányok
egyformán valószínűek bármelyik eloszlásból.
Ezzel szemben bármely jól megkülönböztethető példányt
ennek megfelelően jelentősen túl- vagy alulsúlyozni kell.

Az egyszerűség kedvéért feltételezzük, hogy egyenlő számú példányunk van
a $p(\mathbf{x})$ és $q(\mathbf{x})$ eloszlásokból.
Most jelölje $z$ azokat a feliratokat, amelyek $1$ értékűek
$p$-ből vett adatokra és $-1$ értékűek $q$-ból vett adatokra.
Ekkor a vegyes adathalmazban a valószínűség:

$$P(z=1 \mid \mathbf{x}) = \frac{p(\mathbf{x})}{p(\mathbf{x})+q(\mathbf{x})} \textrm{ és ezért } \frac{P(z=1 \mid \mathbf{x})}{P(z=-1 \mid \mathbf{x})} = \frac{p(\mathbf{x})}{q(\mathbf{x})}.$$

Tehát ha logisztikus regressziós megközelítést alkalmazunk,
ahol $P(z=1 \mid \mathbf{x})=\frac{1}{1+\exp(-h(\mathbf{x}))}$ ($h$ egy parametrizált függvény),
akkor következik, hogy

$$
\beta_i = \frac{1/(1 + \exp(-h(\mathbf{x}_i)))}{\exp(-h(\mathbf{x}_i))/(1 + \exp(-h(\mathbf{x}_i)))} = \exp(h(\mathbf{x}_i)).
$$

Ennek eredményeképpen két problémát kell megoldanunk:
az első a két eloszlásból vett adatok megkülönböztetése,
majd egy súlyozott empirikus kockázat minimalizálási feladat
az :eqref:`eq_weighted-empirical-risk-min` alapján,
ahol a tagokat $\beta_i$-vel súlyozzuk.

Most már leírhatunk egy korrekciós algoritmust.
Tegyük fel, hogy rendelkezünk egy $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ tanítóhalmazzal és egy $\{\mathbf{u}_1, \ldots, \mathbf{u}_m\}$ osztályozatlan teszthalmazra.
Kovariáns eltolódás esetén
feltételezzük, hogy az $\mathbf{x}_i$ értékek minden $1 \leq i \leq n$-re valamely forrás eloszlásból,
és az $\mathbf{u}_i$ értékek minden $1 \leq i \leq m$-re a cél eloszlásból kerültek kihúzásra.
Íme egy prototipikus algoritmus a kovariáns eltolódás korrekciójára:

1. Hozzunk létre egy bináris osztályozási tanítóhalmazt: $\{(\mathbf{x}_1, -1), \ldots, (\mathbf{x}_n, -1), (\mathbf{u}_1, 1), \ldots, (\mathbf{u}_m, 1)\}$.
1. Tanítsunk egy bináris osztályozót logisztikus regresszióval a $h$ függvény megkapásához.
1. Súlyozzuk a tanítási adatokat $\beta_i = \exp(h(\mathbf{x}_i))$, vagy még inkább $\beta_i = \min(\exp(h(\mathbf{x}_i)), c)$ segítségével valamely $c$ állandóval.
1. Használjuk a $\beta_i$ súlyokat a $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ halmazon való tanításhoz az :eqref:`eq_weighted-empirical-risk-min` alapján.

Megjegyezzük, hogy a fenti algoritmus egy kulcsfontosságú feltételezésen alapul.
Ahhoz, hogy ez a megközelítés működjön, szükséges, hogy a cél (pl. tesztelési) eloszlás
minden adatpéldányának pozitív valószínűsége legyen a tanítás idején is.
Ha találunk egy pontot, ahol $p(\mathbf{x}) > 0$, de $q(\mathbf{x}) = 0$,
akkor a megfelelő fontossági súly végtelennek kellene lennie.


### Felirateltolódás korrekciója

Tegyük fel, hogy $k$ kategóriás osztályozási feladattal van dolgunk.
A :numref:`subsec_covariate-shift-correction` szakasz jelöléseivel,
$q$ és $p$ rendre a forrás (pl. tanítási) és cél (pl. tesztelési) eloszlás.
Tegyük fel, hogy a felirat eloszlás idővel változik:
$q(y) \neq p(y)$, de az osztály-feltételes eloszlás
változatlan marad: $q(\mathbf{x} \mid y)=p(\mathbf{x} \mid y)$.
Ha a forrás $q(y)$ eloszlás „rossz",
kiigazíthatjuk az :eqref:`eq_true-risk`-ban definiált kockázatban
az alábbi azonossággal:

$$
\begin{aligned}
\int\int l(f(\mathbf{x}), y) p(\mathbf{x} \mid y)p(y) \;d\mathbf{x}dy =
\int\int l(f(\mathbf{x}), y) q(\mathbf{x} \mid y)q(y)\frac{p(y)}{q(y)} \;d\mathbf{x}dy.
\end{aligned}
$$



A fontossági súlyok a felirat-valószínűségi arányoknak felelnek meg:

$$\beta_i \stackrel{\textrm{def}}{=} \frac{p(y_i)}{q(y_i)}.$$

A felirateltolódás szép tulajdonsága,
hogy ha ésszerűen jó modellünk van a forrás eloszlásra,
akkor e súlyok konzisztens becsléseit kaphatjuk
anélkül, hogy valaha is a környezeti dimenzióval kellene foglalkoznunk.
A mély tanulásban a bemenetek általában nagy dimenziós objektumok, mint képek,
míg a feliratok sokszor egyszerűbb objektumok, mint kategóriák.

A cél felirat eloszlás becsléséhez
először vesszük az ésszerűen jó, készen kapható osztályozónkat
(jellemzően a tanítási adatokon tanítva),
és kiszámítjuk a „tévesítési" mátrixát a validációs halmaz segítségével
(szintén a tanítási eloszlásból).
A *tévesítési mátrix* $\mathbf{C}$ egy egyszerű $k \times k$-as mátrix,
ahol minden oszlop a felirat kategóriának (igazság) felel meg,
és minden sor a modellünk jósolt kategóriájának.
Minden $c_{ij}$ cellaelemé a validációs halmazban
az összes jóslat azon aránya,
ahol a valódi felirat $j$ volt, és modellünk $i$-t jósolt.

Most nem számíthatjuk ki a tévesítési mátrixot
közvetlenül a cél adatokon,
mert nem látjuk a valódi adatokon lévő példányok feliratait,
hacsak nem fektetünk be egy összetett valós idejű annotációs csővezetékbe.
Amit tehetünk azonban, az az, hogy a tesztidőn átlagoljuk
az összes modell jóslatunkat, adva a $\mu(\hat{\mathbf{y}}) \in \mathbb{R}^k$ átlagos modellkimenetet,
ahol az $i$-edik $\mu(\hat{y}_i)$ elem
a teszthalmazon az összes jóslat azon aránya,
ahol modellünk $i$-t jósolt.

Kiderül, hogy néhány enyhe feltétel alatt — ha
az osztályozónk eleve ésszerűen pontos volt,
és ha a cél adatok csak olyan kategóriákat tartalmaznak,
amelyeket korábban is láttunk,
és ha a felirateltolódás feltételezése egyáltalán teljesül
(itt ez a legerősebb feltételezés) — a teszthalmaz felirat eloszlását
megbecsülhetjük egy egyszerű lineáris rendszer megoldásával:

$$\mathbf{C} p(\mathbf{y}) = \mu(\hat{\mathbf{y}}),$$

mert becslésként minden $1 \leq i \leq k$-ra fennáll $\sum_{j=1}^k c_{ij} p(y_j) = \mu(\hat{y}_i)$,
ahol $p(y_j)$ a $k$-dimenziós $p(\mathbf{y})$ felirat eloszlás vektor $j$-edik eleme.
Ha az osztályozónk eleve kellően pontos,
akkor a $\mathbf{C}$ tévesítési mátrix invertálható,
és $p(\mathbf{y}) = \mathbf{C}^{-1} \mu(\hat{\mathbf{y}})$ megoldást kapunk.

Mivel a forrás adatokon látjuk a feliratokat,
könnyen megbecsülhetjük a $q(y)$ eloszlást.
Ekkor minden $i$-edik tanítási példányhoz az $y_i$ felirattal
felvehetjük a becsült $p(y_i)/q(y_i)$ arány alapján a $\beta_i$ súlyt,
és behelyettesíthetjük az :eqref:`eq_weighted-empirical-risk-min`-beli
súlyozott empirikus kockázat minimalizálásba.


### Fogalomeltolódás korrekciója

A fogalomeltolódást sokkal nehezebb elvhű módon javítani.
Például olyan helyzetben, ahol a feladat hirtelen megváltozik
a macskák és kutyák megkülönböztetéséről
a fehér és fekete állatok megkülönböztetésére,
ésszerűtlen lenne feltételezni,
hogy jobbat tehetünk, mint új feliratokat gyűjtünk
és nulláról tanítjuk.
Szerencsére a gyakorlatban ilyen szélső eltolódások ritkák.
Ehelyett általában az történik, hogy a feladat lassan változik.
Hogy konkrétabbá tegyük, néhány példa:

* A számítógépes hirdetésben új termékeket dobnak piacra,
a régiek pedig kevésbé lesznek népszerűek. Ez azt jelenti, hogy a hirdetések és azok népszerűsége feletti eloszlás lassan változik, és bármely átkattintási arány előrejelzőnek ezzel együtt kell lassan változnia.
* A forgalmi kamerák lencséi fokozatosan degradálódnak a környezeti kopás miatt, fokozatosan befolyásolva a képminőséget.
* A hírtartalom fokozatosan változik (azaz a hírek nagy része változatlan marad, de új történetek jelennek meg).

Ilyen esetekben ugyanazt a megközelítést alkalmazhatjuk, amelyet a hálózatok tanításához használtunk, hogy alkalmazkodjanak az adatok változásához. Más szóval: a meglévő hálózati súlyokat felhasználjuk, és csupán néhány frissítési lépést hajtunk végre az új adatokkal, ahelyett hogy nulláról tanítanánk.


## A tanulási problémák taxonómiája

Felfegyverkezve az eloszlásváltozásokkal való megküzdés ismeretével, most megvizsgálhatunk néhány egyéb szempontot a gépi tanulási problémaformulálásban.


### Batch tanulás

A *batch tanulásban* hozzáférünk a tanítási jellemzőkhöz és feliratokhoz $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$, amelyeket egy $f(\mathbf{x})$ modell tanítására használunk. Ezután telepítjük ezt a modellt az ugyanolyan eloszlásból vett új $(\mathbf{x}, y)$ adatok pontszámozásához. Ez az alapértelmezett feltételezés az itt tárgyalt bármely problémára. Például egy macskadetektor taníthatunk sok macska és kutya kép alapján. Miután megtanítottuk, egy intelligens macskaajtó számítógépes látórendszer részeként szállítjuk, amely csak macskákat enged be. Ezt aztán egy ügyfél otthonában telepítik, és soha többé nem frissítik (kivételes körülményektől eltekintve).


### Online tanulás

Most képzeljük el, hogy az $(\mathbf{x}_i, y_i)$ adatok egyszerre egy mintánként érkeznek. Pontosabban: feltételezzük, hogy először megfigyeljük az $\mathbf{x}_i$-t, majd el kell készítenünk az $f(\mathbf{x}_i)$ becslést. Csak ezután figyeljük meg $y_i$-t, és jutalmakat kapunk vagy veszteséget szenvedünk a döntésünk alapján.
Sok valós probléma ebbe a kategóriába esik. Például holnap részvényárat kell megjósolnunk, ami lehetővé teszi a kereskedést a becslés alapján, és a nap végén megtudjuk, hogy becslésünk nyereséget hozott-e. Más szóval az *online tanulásban* az alábbi ciklus zajlik, ahol folyamatosan javítjuk a modellünket az új megfigyelések alapján:

$$\begin{aligned}&\textrm{model } f_t \longrightarrow \textrm{data }  \mathbf{x}_t \longrightarrow \textrm{estimate } f_t(\mathbf{x}_t) \longrightarrow\\ \textrm{obs}&\textrm{ervation } y_t \longrightarrow \textrm{loss } l(y_t, f_t(\mathbf{x}_t)) \longrightarrow \textrm{model } f_{t+1}\end{aligned}$$

### Banditek (Bandits)

A *banditek* a fenti probléma egy speciális esete. Míg a legtöbb tanulási problémában folyamatosan parametrizált $f$ függvényünk van, amelynek paramétereit meg akarjuk tanulni (pl. egy mély hálózat), egy *bandit* problémában csak véges számú karunk van, amelyet húzhatunk, vagyis véges számú cselekvés közül választhatunk. Nem meglepő, hogy erre az egyszerűbb problémára erősebb elméleti garanciák szerezhetők az optimalitás tekintetében. Főleg azért soroljuk fel, mert ezt a problémát sokszor (tévesen) különálló tanulási beállításként kezelik.


### Irányítás (Control)

Sok esetben a környezet emlékszik arra, amit tettünk. Nem feltétlenül ellenséges módon, de emlékezni fog, és a válasz attól fog függeni, mi történt korábban. Például egy kávéfőző vezérlő különböző hőmérsékleteket fog mérni attól függően, hogy korábban melegítette-e a kazánt. A PID (proporcionális-integrális-derivatív) vezérlő algoritmusok ott népszerű választás.
Hasonlóan, egy felhasználó viselkedése egy híroldalon attól függhet, mit mutattunk nekik korábban (pl. a legtöbb hírt csak egyszer olvassák el). Sok ilyen algoritmus modellt alkot a környezetről, amelyben cselekszenek, hogy döntéseik kevésbé véletlenszerűnek tűnjenek.
A közelmúltban
az irányításelméletet (pl. PID variánsok) is alkalmazták
hiperparaméterek automatikus hangolásához
a jobb szétválasztás és rekonstrukciós minőség elérése érdekében,
és a generált szöveg változatosságának javítására és a generált képek rekonstrukciós minőségére :cite:`Shao.Yao.Sun.ea.2020`.


### Megerősítéses tanulás (Reinforcement Learning)

A memóriával rendelkező környezet általánosabb esetében találkozhatunk olyan helyzetekkel, ahol a környezet együttműködni próbál velünk (kooperatív játékok, különösen nemnull-összegű játékokra), vagy ahol a környezet megpróbál nyerni. A sakk, a Go, a backgammon vagy a StarCraft a *megerősítéses tanulás* esetei. Hasonlóan jó vezérlőt akarhatunk építeni önvezető autókhoz. Más autók valószínűleg nem triviális módon reagálnak az önvezető autó vezetési stílusára, pl. megpróbálják elkerülni, megpróbálnak balesetet okozni vagy együttműködni akarnak vele.

### A környezet figyelembe vétele

A fenti különböző helyzetek közötti egyik kulcsfontosságú különbség az, hogy egy stacionárius környezetben végig működő stratégia nem feltétlenül működik végig egy alkalmazkodni képes környezetben. Például egy kereskedő által felfedezett arbitrázs lehetőség valószínűleg eltűnik, amint kihasználják. A környezet változásának sebessége és módja nagymértékben meghatározza az alkalmazható algoritmusok típusát. Például ha tudjuk, hogy a dolgok csak lassan változhatnak, kényszert tehetünk arra, hogy bármely becslés is csak lassan változzon. Ha tudjuk, hogy a környezet pillanatszerűen, de csak nagyon ritkán változhat, engedményt tehetünk erre. Az ilyen típusú ismeretek döntőek a törekvő adattudós számára a fogalomeltolódás kezelésekor, vagyis amikor a megoldandó probléma idővel változhat.




## Méltányosság, elszámoltathatóság és átláthatóság a gépi tanulásban

Végül fontos emlékezni arra,
hogy gépi tanulási rendszerek telepítésekor
nem csupán egy prediktív modellt optimalizálunk —
általában olyan eszközt biztosítunk,
amely a döntések (részleges vagy teljes) automatizálására szolgál.
Ezek a technikai rendszerek befolyásolhatják
az érintett döntések alá eső egyének életét.
A jóslásokból döntéshozatalba való ugrás
nemcsak új technikai kérdéseket vet fel,
hanem számos etikai kérdést is,
amelyeket gondosan kell megfontolni.
Ha orvosi diagnosztikai rendszert telepítünk,
tudnunk kell, melyik populációkra működhet
és melyekre esetleg nem.
Az alpopulációk jólétének előre látható kockázatainak figyelmen kívül hagyása
alacsonyabb minőségű ellátáshoz vezethet.
Ráadásul, ha döntéshozatali rendszereket vizsgálunk,
vissza kell lépnünk és újra kell gondolnunk, hogyan értékeljük a technológiánkat.
E hatókörváltozás egyéb következményei mellett
azt fogjuk találni, hogy a *pontosság* ritkán a megfelelő mérőszám.
Például a jóslatok cselekvésekre fordításánál
sokszor figyelembe kell vennünk
a különféle módokon elkövetett hibák potenciális költségérzékenységét.
Ha egy kép egyik osztályba való téves osztályozása
faji sértésnek tekinthető,
míg egy másik kategóriába való téves osztályozás
ártalmatlan lenne, akkor esetleg
el kell állítanunk a küszöbértékeinket,
figyelembe véve a társadalmi értékeket
a döntéshozatali protokoll tervezésekor.
Arra is ügyelni kell,
hogy az előrejelző rendszerek hogyan vezethetnek visszacsatolási hurkokhoz.
Vegyük például az előrejelző rendészeti rendszereket,
amelyek járőrtiszteket allokálnak
a magas előrejelzett bűnözési rátájú területekre.
Könnyen látható, hogyan alakulhat ki egy aggasztó minta:

 1. A több bűncselekménnyel rendelkező negyedek több járőrt kapnak.
 1. Következésképpen ezekben a negyedekben több bűncselekményt fedeznek fel, amelyek bekerülnek a jövőbeli iterációkhoz rendelkezésre álló tanítási adatokba.
 1. Több pozitívnak kitéve a modell még több bűncselekményt jósol ezekben a negyedekben.
 1. A következő iterációban a frissített modell még nagyobb mértékben célozza meg ugyanazt a negyedet, ami még több felfedezett bűncselekményhez vezet, stb.

Sokszor a különféle mechanizmusok, amelyek révén
egy modell jóslatai összekapcsolódnak a tanítási adatokkal,
nincsenek figyelembe véve a modellezési folyamatban.
Ez vezethet ahhoz, amit a kutatók *elszabadult visszacsatolási hurkoknak* neveznek.
Ezen felül ügyelni kell arra is,
hogy egyáltalán a helyes problémát célozzuk-e meg.
Az előrejelző algoritmusok mára hatalmas szerepet játszanak
az információterjesztés közvetítésében.
Az egyén által látott híreket
a Facebook-oldalak halmaza határozza-e meg, amelyeket *lájkoltak*?
Ezek csupán néhány a számos sürgős etikai dilemmák közül,
amellyel a gépi tanulásban való karriered során találkozhatsz.


## Összefoglalás

Sok esetben a tanítási és teszthalmazok nem ugyanabból az eloszlásból kerülnek ki. Ezt eloszláseltolódásnak nevezzük.
A kockázat a veszteség várható értéke az összes, valódi eloszlásból vett adaton. Ez az összes populáció azonban általában nem elérhető. Az empirikus kockázat a tanítási adatokon számolt átlagos veszteség a kockázat közelítésére. A gyakorlatban empirikus kockázat minimalizálást hajtunk végre.
A megfelelő feltételezések alatt a kovariáns és felirateltolódás tesztelési időben érzékelhető és korrigálható. E torzítás figyelmen kívül hagyása problémás lehet tesztelési időben.
Egyes esetekben a környezet emlékezhet az automatizált cselekvésekre, és meglepő módon reagálhat. Ezt a lehetőséget figyelembe kell venni a modellek építésekor, és folytatni kell az élő rendszerek monitorozását, nyitva a lehetőségre, hogy modelljeink és a környezet váratlan módokon fonódhatnak össze.

## Feladatok

1. Mi történhet, ha megváltoztatjuk egy keresőmotor viselkedését? Mit tehetnének a felhasználók? Mit tennének a hirdetők?
1. Valósíts meg egy kovariáns eltolódás detektort! Tipp: építs egy osztályozót!
1. Valósíts meg egy kovariáns eltolódás korrekciót!
1. Az eloszláseltolódáson kívül mi más befolyásolhatja azt, hogy az empirikus kockázat hogyan közelíti a kockázatot?


[Megbeszélések](https://discuss.d2l.ai/t/105)
