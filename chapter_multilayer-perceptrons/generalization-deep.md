# Általánosítás a mélytanulásban


A :numref:`chap_regression` és a :numref:`chap_classification` fejezetekben
regressziós és osztályozási problémákat oldottunk meg
lineáris modellek tanítási adatokhoz való illesztésével.
Mindkét esetben gyakorlati algoritmusokat adtunk meg
olyan paraméterek megtalálásához, amelyek maximalizálják
a megfigyelt tanítási címkék valószínűségét.
Majd az egyes fejezetek végén
felidéztük, hogy a tanítási adatokhoz való illesztés
csupán közbülső cél volt.
Valódi törekvésünk mindig az volt, hogy *általános mintákat* fedezzünk fel,
amelyek alapján pontos előrejelzéseket tehetünk
még az ugyanabból az alapsokaságból vett új példányokon is.
A gépi tanulás kutatói az optimalizálási algoritmusok *felhasználói*.
Néha még új optimalizálási algoritmusokat is kell fejlesztenünk.
De a lényeg az, hogy az optimalizálás csupán eszköz a célhoz.
Alapvetően a gépi tanulás statisztikai tudományág,
és a tanítási veszteséget csak annyiban kívánjuk optimalizálni,
amennyiben valamely statisztikai elv (ismert vagy ismeretlen)
arra vezeti az eredményes modelleket, hogy általánosítsanak a tanítóhalmazon túlra.


A pozitív oldal az, hogy a sztochasztikus gradienscsökkenéssel tanított
mély neurális hálózatok meglepően jól általánosítanak
számos előrejelzési problémában, beleértve a számítógépes látást,
a természetes nyelvfeldolgozást, az idősor-adatokat, az ajánlórendszereket,
az elektronikus egészségügyi nyilvántartásokat, a fehérjehajtogatást,
az értékfüggvény közelítését videójátékokban és táblajátékokban,
és számos más területet.
A negatív oldalon viszont, ha valaki egyszerű magyarázatot keresne
akár az optimalizálás kérdéséről
(miért tudjuk illeszteni a tanítási adatokra)
akár az általánosítás kérdéséről
(miért általánosítanak a kapott modellek nem látott példányokra),
az jobb, ha tölt magának egy italt.
Míg a lineáris modellek optimalizálásának eljárásai
és a megoldások statisztikai tulajdonságai
mindkettő jól leírható az elmélet átfogó testületével,
addig a mélytanulás megértése
mindkét fronton még mindig hasonlít a vad nyugathoz.

A mélytanulás elmélete és gyakorlata is
gyorsan fejlődik,
ahol az elméleti kutatók új stratégiákat dolgoznak ki
a történések magyarázatára,
miközben a szakemberek folytatják az innovációt égető iramban,
heurisztikák arzenálját fejlesztik ki a mély hálózatok tanításához
és olyan intuíciók és népi tudás gyűjteményét,
amelyek iránymutatást adnak arról,
hogy melyik technikákat alkalmazzák milyen helyzetekben.

A jelen pillanat összefoglalása az, hogy a mélytanulás elmélete
ígéretes támadási vonalakat produkált és elszórt lenyűgöző eredményeket,
de még mindig messze tűnik egy átfogó magyarázattól
mind (i) arról, hogy miért vagyunk képesek neurális hálózatokat optimalizálni,
mind (ii) arról, hogy a gradienscsökkenés által megtanult modellek
hogyan általánosítanak annyira jól, még nagy dimenziós feladatokon is.
A gyakorlatban azonban az (i) ritkán jelent problémát
(mindig találhatunk paramétereket, amelyek illeszkednek az összes tanítási adatunkra),
és így az általánosítás megértése messze a nagyobb probléma.
Másrészt, még egy koherens tudományos elmélet vigasza nélkül is,
a szakemberek nagy gyűjteményt fejlesztettek ki olyan technikákból,
amelyek segíthetnek a gyakorlatban jól általánosító modellek készítésében.
Bár semmilyen tömör összefoglalás nem tehet teljes igazságot
a mélytanulásban való általánosítás hatalmas témájának,
és bár a kutatás összességének állapota messze van a megoldástól,
reméljük, hogy ebben a részben széleskörű áttekintést nyújtunk
a kutatás és a gyakorlat jelenlegi állapotáról.


## A túlillesztés és a regularizáció újragondolása

:citet:`wolpert1995no` „nincsen ingyenes ebéd" tétele szerint
bármely tanulási algoritmus jobban általánosít bizonyos eloszlású adatokon, és rosszabbul más eloszlásokon.
Így egy véges tanítóhalmaz esetén
egy modell bizonyos feltételezésekre támaszkodik:
az emberi szintű teljesítmény eléréséhez
hasznos lehet azonosítani az *induktív torzításokat*,
amelyek tükrözik, hogy az emberek hogyan gondolkodnak a világról.
Az ilyen induktív torzítások preferenciákat mutatnak
bizonyos tulajdonságokkal rendelkező megoldások iránt.
Például egy mély MLP induktív torzítással rendelkezik
bonyolult függvények felépítése felé egyszerűbb függvények kompozícióján keresztül.

A gépi tanulási modellek induktív torzításokat kódolva,
a tanítási megközelítésünk
általában két fázisból áll: (i) a tanítási adatok illesztése;
és (ii) az *általánosítási hiba* becslése
(az alapsokaságon vett valódi hiba)
a modell holdout adatokon való értékelésével.
A tanítási adatokon való illeszkedésünk
és a teszt adatokon való illeszkedésünk közötti különbséget *általánosítási résnek* nevezzük,
és ha ez nagy, azt mondjuk, hogy a modell *túlilleszt* a tanítási adatokra.
A túlillesztés szélsőséges esetein
tökéletesen illeszkedhetünk a tanítási adatokra,
még akkor is, ha a teszt hiba jelentős marad.
A klasszikus nézet szerint
az értelmezés az, hogy a modelljeink túl bonyolultak,
ami miatt csökkentenünk kell a jellemzők számát,
a megtanult nemnulla paraméterek számát,
vagy a paraméterek méretét.
Idézzük fel a modellek összetettségét és a veszteséget összehasonlító ábrát
(:numref:`fig_capacity_vs_error`)
a :numref:`sec_generalization_basics` részből.


A mélytanulás azonban ellentmondásos módon bonyolítja ezt a képet.
Először is, az osztályozási problémák esetén
modelljeink általában elég kifejezők ahhoz,
hogy minden tanítási példányt tökéletesen illesszünk,
még milliós méretű adathalmazokban is
:cite:`zhang2021understanding`.
A klasszikus képen azt gondolhatnánk,
hogy ez a beállítás a modell bonyolultsági tengely szélső jobb oldalán helyezkedik el,
és hogy az általánosítási hiba bármely javulása
a regularizáció útján kell, hogy jöjjön,
akár a modellosztály összetettségének csökkentésével,
akár büntetés alkalmazásával, amely erősen korlátozza
a paraméterek által felvehető értékek körét.
De itt kezdenek furcsává válni a dolgok.

Furcsán, sok mélytanulási feladatnál
(pl. képfelismerés és szövegbesorolás)
általában olyan modellarchitektúrák közül választunk,
amelyek mindegyike tetszőlegesen alacsony tanítási veszteséget érhet el
(és nulla tanítási hibát).
Mivel a vizsgált modellek mindegyike nulla tanítási hibát ér el,
*az egyetlen lehetőség további javulásra a túlillesztés csökkentése*.
Még különösebb, hogy az eset, amikor a tanítási adatokat tökéletesen illesztjük,
valóban tovább tudjuk *csökkenteni az általánosítási hibát*
azzal, hogy a modellt *még kifejezőbbé* tesszük,
pl. rétegek, csomópontok hozzáadásával, vagy nagyobb számú epochig tanítva.
Még különösebb, hogy az általánosítási rés és a modell *bonyolultsága* közötti minta
(ahogyan azt pl. a hálózatok mélysége vagy szélessége ragadja meg)
nem-monoton lehet,
ahol a nagyobb bonyolultság kezdetben árt,
de utána segít egy úgynevezett „kettős ereszkedés" mintában
:cite:`nakkiran2021deep`.
Így a mélytanulás szakemberének van egy trükkökből álló eszköztára,
amelyek közül néhány látszólag valamilyen módon korlátozza a modellt,
mások látszólag még kifejezőbbé teszik,
és mindet egyfajta értelemben a túlillesztés enyhítésére alkalmazzák.

A dolgokat még tovább bonyolítja,
hogy míg a klasszikus tanulási elmélet által adott garanciák
még a klasszikus modellek esetén is konzervatívak lehetnek,
teljesen tehetetlennek tűnnek annak magyarázatában,
hogy a mély neurális hálózatok miért általánosítanak egyáltalán.
Mivel a mély neurális hálózatok képesek illeszkedni
tetszőleges címkékhez nagy adathalmazok esetén is,
és annak ellenére, hogy ismerős módszereket, mint például az $\ell_2$ regularizáció, alkalmazunk,
a hagyományos komplexitásalapú általánosítási korlátok,
pl. a VC-dimenzión vagy egy hipotézis osztály Rademacher komplexitásán alapuló korlátok,
nem tudják megmagyarázni, miért általánosítanak a neurális hálózatok.

## Ihletet merítve a nemparametrikus módszerekből

Aki először találkozik a mélytanulással,
hajlamos parametrikus modelleknek tekinteni azokat.
Végül is a modellek *valóban* millió paramétert tartalmaznak.
Amikor frissítjük a modelleket, frissítjük a paramétereiket.
Amikor elmentjük a modelleket, a paramétereiket lemezre írjuk.
A matematika és az informatika azonban tele van
ellentmondásos perspektívaváltásokkal,
és meglepő izomorfizmusokkal látszólag különböző problémák között.
Bár a neurális hálózatoknak *vannak* paraméterei,
bizonyos szempontból termékenyen lehet rájuk gondolni
úgy, mintha nemparametrikus modellekként viselkednének.
Tehát mi tesz egy modellt pontosan nemparametrikussá?
Bár a kifejezés számos különböző megközelítést fed le,
egy közös jellemző az, hogy a nemparametrikus módszerek
összetettsége általában növekszik
a rendelkezésre álló adatok mennyiségének növekedésével.

Talán a legegyszerűbb nemparametrikus modell példája
a $k$-legközelebbi szomszéd algoritmus (több nemparametrikus modellt fogunk tárgyalni később, pl. a :numref:`sec_attention-pooling` részben).
Ebben az esetben tanítás idején
a tanuló egyszerűen memorizálja az adathalmazt.
Majd előrejelzés idején,
amikor egy új $\mathbf{x}$ ponttal szembesül,
a tanuló megkeresi a $k$ legközelebbi szomszédot
(a $k$ darab $\mathbf{x}_i'$ pontot, amelyek minimalizálják
valamely $d(\mathbf{x}, \mathbf{x}_i')$ távolságot).
Amikor $k=1$, ezt az algoritmust $1$-legközelebbi szomszédnak nevezzük,
és az algoritmus mindig nulla tanítási hibát ér el.
Ez azonban nem jelenti azt, hogy az algoritmus nem általánosít.
Valójában kiderül, hogy enyhe feltételek mellett
az 1-legközelebbi szomszéd algoritmus konzisztens
(végül az optimális prediktorhoz konvergál).


Vegyük észre, hogy az $1$-legközelebbi szomszéd megköveteli, hogy megadjunk
valamely $d$ távolságfüggvényt, vagy egyenértékűen,
hogy megadjunk valamely vektoros $\phi(\mathbf{x})$ bázisfüggvényt
az adataink jellemzővé alakításához.
A távolságmérték bármely megválasztása esetén
nulla tanítási hibát érünk el
és végül egy optimális prediktorhoz jutunk,
de a különböző $d$ távolságmértékek
különböző induktív torzításokat kódolnak,
és véges mennyiségű rendelkezésre álló adattal
különböző prediktorokat eredményeznek.
A $d$ távolságmérték különböző megválasztásai
különböző feltételezéseket reprezentálnak az alapmintákról,
és a különböző prediktorok teljesítménye
attól függ, mennyire összeegyeztethetők a feltételezések
a megfigyelt adatokkal.

Bizonyos értelemben, mivel a neurális hálózatok túlparametrizáltak,
sokkal több paraméterük van, mint amennyi a tanítási adatok illesztéséhez szükséges,
hajlamosak *interpolálni* a tanítási adatokat (tökéletesen illeszkedni hozzájuk),
és így bizonyos szempontból inkább nemparametrikus modellekként viselkednek.
A legújabb elméleti kutatás mélyreható kapcsolatot mutatott ki
nagy neurális hálózatok és nemparametrikus módszerek,
különösen kernel-módszerek között.
Konkrétan :citet:`Jacot.Grabriel.Hongler.2018`
megmutatta, hogy a határon, amint a véletlenszerűen inicializált
súlyú többrétegű perceptronok végtelen szélessé válnak,
egyenértékűvé válnak a (nemparametrikus) kernel-módszerekkel
egy specifikus kernel-függvény (lényegében egy távolságfüggvény) megválasztásánál,
amelyet neurális tangenskernelnek neveznek.
Bár a jelenlegi neurális tangenskernel-modellek nem feltétlenül magyarázzák teljesen
a modern mély hálózatok viselkedését,
az analitikus eszközként való sikerük
kiemeli a nemparametrikus modellezés hasznosságát
a túlparametrizált mély hálózatok viselkedésének megértésében.


## Korai megállás

Miközben a mély neurális hálózatok képesek illeszkedni tetszőleges címkékhez,
még akkor is, amikor a címkéket helytelenül vagy véletlenszerűen rendelik hozzá
:cite:`zhang2021understanding`,
ez a képesség csak a tanítás sok iterációja után jelenik meg.
Egy új kutatási irány :cite:`Rolnick.Veit.Belongie.Shavit.2017`
feltárta, hogy címke-zaj esetén
a neurális hálózatok hajlamosak először a tisztán címkézett adatokra illeszkedni,
és csak utána az rosszul címkézett adatok interpolálására.
Ráadásul megállapítást nyert, hogy ez a jelenség
közvetlenül átfordítható az általánosítás garanciájává:
amikor egy modell illeszkedett a tisztán címkézett adatokra,
de nem a tanítóhalmazban szereplő véletlenszerűen megcímkézett példányokra,
valóban általánosított :cite:`Garg.Balakrishnan.Kolter.Lipton.2021`.

Ezek az eredmények együttesen motiválják a *korai megállást*,
a mély neurális hálózatok regularizálásának klasszikus technikáját.
Ebben az esetben, ahelyett hogy közvetlenül korlátozná a súlyok értékeit,
a tanítási epochok számát korlátozzuk.
A leállítási kritérium meghatározásának legáltalánosabb módja
az, hogy figyelemmel kísérjük a validációs hibát a tanítás során
(általában egy epochonként egyszer ellenőrzünk),
és leállítjuk a tanítást, amikor a validációs hiba
nem csökkent $\epsilon$-nál kisebb értékkel
valamely számú epochon keresztül.
Ezt néha *türelem kritériumnak* nevezik.
A zajos címkék esetén való jobb általánosítás lehetőségén kívül
a korai megállás másik előnye az időmegtakarítás.
Amint a türelem kritérium teljesül, a tanítást le lehet állítani.
Olyan nagy modelleknél, amelyek több napos tanítást igényelnek
egyidejűleg nyolc vagy több GPU-n,
a jól hangolt korai megállás napokat takaríthat meg a kutatóknak,
és munkáltatóiknak sok ezer dollárt spórolhat meg.

Érdemes megjegyezni, hogy amikor nincs címke-zaj, és az adathalmazok *realizálhatók*
(az osztályok valóban szeparálhatók, pl. macskák megkülönböztetése kutyáktól),
a korai megállás általában nem vezet jelentős javuláshoz az általánosításban.
Másrészt, amikor van címka-zaj,
vagy belső változékonyság a címkében
(pl. betegek halálozásának előrejelzése),
a korai megállás kritikus fontosságú.
A modellek tanítása addig, amíg azok zajos adatokat interpolálnak, általában rossz ötlet.


## Klasszikus regularizálási módszerek mély hálózatokhoz

A :numref:`chap_regression` fejezetben leírtunk
számos klasszikus regularizálási technikát
a modellek bonyolultságának korlátozásához.
Különösen a :numref:`sec_weight_decay` rész
bevezette a súlybomlás módszert,
amely abból áll, hogy regularizálási tagot adunk a veszteségfüggvényhez
a nagy súlyértékek büntetésére.
Attól függően, hogy melyik súlynormát büntetjük,
ezt a technikát ridge regularizációnak (az $\ell_2$ büntetés esetén)
vagy lasso regularizációnak (az $\ell_1$ büntetés esetén) nevezzük.
Ezeknek a regularizálóknak a klasszikus elemzésében
úgy tekintik őket, mint amelyek elég korlátozóak
a súlyok által felvehető értékekre,
hogy megakadályozzák a modellt tetszőleges címkék illesztésétől.

A mélytanulási implementációkban
a súlybomlás népszerű eszköz maradt.
Azonban a kutatók megjegyezték,
hogy az $\ell_2$ regularizáció tipikus erőssége
nem elegendő ahhoz, hogy megakadályozza a hálózatokat
az adatok interpolálásától :cite:`zhang2021understanding`,
és így az előnyök, ha regularizációként értelmezik,
csak a korai megállás kritériummal kombinálva nyernek értelmet.
A korai megállás hiányában lehetséges,
hogy akárcsak a rétegek száma
vagy csomópontok száma (a mélytanulásban)
vagy a távolságmérték (az 1-legközelebbi szomszédban),
ezek a módszerek jobb általánosításhoz vezethetnek
nem azért, mert érdemben korlátozzák
a neurális hálózat erejét,
hanem mert valahogyan olyan induktív torzításokat kódolnak,
amelyek jobban összeegyeztethetők az érdeklődési körbe eső
adathalmazokban talált mintákkal.
Így a klasszikus regularizálók népszerűek maradnak
a mélytanulási implementációkban,
még ha a hatékonyságuk elméleti indoklása
radikálisan eltérő is lehet.

Érdemes megjegyezni, hogy a mélytanulás kutatói
a klasszikus regularizálási kontextusokban először népszerűsített technikákra is építenek,
mint például a modell bemeneteibe zajt adni.
A következő részben bemutatjuk
a híres dropout technikát
(:citet:`Srivastava.Hinton.Krizhevsky.ea.2014` találmánya),
amely a mélytanulás alappillérévé vált,
még ha a hatékonyságának elméleti alapjai
hasonlóan rejtélyesek is maradnak.


## Összefoglalás

A klasszikus lineáris modellektől eltérően,
amelyeknek általában kevesebb paramétere van, mint a példányoknak,
a mély hálózatok általában túlparametrizáltak,
és a legtöbb feladatnál képesek
a tanítóhalmazt tökéletesen illeszteni.
Ez az *interpolációs rezsim* kihívást jelent
sok erősen tartott intuícióval szemben.
Funkcionálisan a neurális hálózatok parametrikus modelleknek tűnnek.
De ha nemparametrikus modellként gondolunk rájuk,
néha megbízhatóbb forrása az intuíciónak.
Mivel általában az összes vizsgált mély hálózat képes
az összes tanítási címkét illeszteni,
majdnem az összes nyereségnek a túlillesztés enyhítéséből kell jönnie
(az *általánosítási rés* csökkentéséből).
Paradox módon az általánosítási rést csökkentő beavatkozások
néha úgy tűnik, hogy növelik a modell bonyolultságát,
más esetekben pedig csökkentik azt.
Azonban ezek a módszerek ritkán csökkentik a bonyolultságot
annyira, hogy a klasszikus elmélet
megmagyarázza a mély hálózatok általánosítását,
és *miért vezet bizonyos döntések jobb általánosításhoz*
nagyrészt egy hatalmas nyitott kérdés marad
annak ellenére, hogy sok kiváló kutató összehangolt erőfeszítéseket tett.


## Feladatok

1. Milyen értelemben nem sikerül a hagyományos, bonyolultságon alapuló mértékeknek magyarázni a mély neurális hálózatok általánosítását?
1. Miért tekinthető a *korai megállás* regularizálási technikának?
1. Hogyan határozzák meg általában a kutatók a leállítási kritériumot?
1. Milyen fontos tényező látszik megkülönböztetni azokat az eseteket, amikor a korai megállás nagy javuláshoz vezet az általánosításban?
1. Az általánosításon túl, írj le egy másik előnyt a korai megállásnak.

[Discussions](https://discuss.d2l.ai/t/7473)
