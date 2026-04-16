# Általánosítás az osztályozásban

:label:`chap_classification_generalization`



Eddig arra összpontosítottunk, hogyan kezeljük a többosztályos osztályozási feladatokat
több kimenetű (lineáris) neurális hálózatok és softmax függvények segítségével.
Modellünk kimeneteit valószínűségi jóslatokként értelmezve
levezettük és indokoltuk a keresztentrópia-veszteség függvényt,
amely a negatív log-valószínűséget számítja ki,
amelyet modellünk (rögzített paraméterhalmazra) a tényleges címkékhez rendel.
Végül ezeket az eszközöket a gyakorlatba is átültettük
a modell tanítóhalmazra való illesztésével.
Azonban — ahogy mindig — célunk *általános mintákat* megtanulni,
amelyeket empirikusan, korábban nem látott adatokon (teszthalmaz) értékelünk.
A tanítóhalmazon elért magas pontosság semmit sem jelent.
Valahányszor minden bemeneti adat egyedi
(ami a legtöbb nagy dimenziójú adathalmazban igaz),
az első tanítási epokon az adathalmaz egyszerű memorizálásával
tökéletes pontosságot érhetünk el a tanítóhalmazon,
majd a megfelelő címkét visszakereshetjük, ha új képet látunk.
Azonban a pontos tanítási példákhoz tartozó pontos címkék memorizálása
nem mondja meg, hogyan osztályozzuk az új példákat.
Útmutatás hiányában előfordulhat, hogy új példák esetén
véletlenszerű találgatásra szorulunk.

Számos égető kérdés azonnali figyelmet igényel:

1. Hány tesztpéldányra van szükségünk ahhoz, hogy jó becslést adjunk az osztályozóink pontosságáról az alapul szolgáló populáción?
1. Mi történik, ha ismételten modelleket értékelünk ki ugyanazon a teszthalmazon?
1. Miért várhatjuk, hogy a lineáris modellek tanítóhalmazra való illesztése jobb eredményt ad, mint a naiv memorizálási megközelítés?


Míg a :numref:`sec_generalization_basics` szakasz bevezette
a túlillesztés és az általánosítás alapjait
a lineáris regresszió kontextusában,
ez a fejezet kicsit mélyebbre megy,
bemutatva a statisztikus tanulási elmélet néhány alapgondolatát.
Kiderül, hogy az általánosítást *a priori* sokszor garantálni tudjuk:
sok modell esetén,
és bármely kívánt felső korláthoz
az általánosítási résen $\epsilon$-on,
meg tudunk határozni egy szükséges mintaszámot $n$,
úgy, hogy ha tanítóhalmazunk legalább $n$ mintát tartalmaz,
az empirikus hibánk $\epsilon$-on belül lesz a valódi hibától,
*bármely adatgeneráló eloszlásra*.
Sajnos az is kiderül,
hogy bár e garantálások mély intellektuális alapköveket nyújtanak,
a mélytanulás gyakorlói számára korlátozott a gyakorlati hasznosságuk.
Röviden: e garanciák arra utalnak, hogy a mély neurális hálózatok *a priori* általánosításának biztosítása
abszurd számú példát igényel
(esetleg billiókat vagy többet),
még ha azt tapasztaljuk is, hogy a számunkra fontos feladatokon
a mély neurális hálózatok jellemzően
meglepően jól általánosítanak jóval kevesebb (ezres) példával.
Ezért a mélytanulás gyakorlói sokszor lemondanak az *a priori* garanciákról,
és inkább olyan módszereket alkalmaznak,
amelyek hasonló problémákon korábban jól általánosítottak,
és az általánosítást empirikus kiértékeléssel *post hoc* igazolják.
Amikor a :numref:`chap_perceptrons` fejezethez érünk,
visszatérünk az általánosítás kérdésére
és könnyű bevezetést nyújtunk
abba a hatalmas tudományos irodalomba,
amely annak magyarázatával keletkezett,
miért általánosítanak a mély neurális hálózatok a gyakorlatban.

## A teszthalmaz

Mivel már a teszthalmazra támaszkodtunk
mint az általánosítási hiba becslésének arany standardjára,
kezdjük az ilyen hibabecslések tulajdonságainak tárgyalásával.
Összpontosítsunk egy rögzített $f$ osztályozóra,
anélkül hogy aggódnánk, hogyan kaptuk.
Tegyük fel továbbá, hogy rendelkezünk
egy $\mathcal{D} = {(\mathbf{x}^{(i)},y^{(i)})}_{i=1}^n$ *friss* adathalmazzal,
amelyet nem használtak az $f$ osztályozó tanításához.
Az $f$ osztályozó $\mathcal{D}$-n mért *empirikus hibája*
egyszerűen azon példányok aránya,
amelyeknél az $f(\mathbf{x}^{(i)})$ jóslat
eltér a tényleges $y^{(i)}$ címkétől,
és az alábbi kifejezéssel adható meg:

$$\epsilon_\mathcal{D}(f) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(f(\mathbf{x}^{(i)}) \neq y^{(i)}).$$

Ezzel szemben a *populációs hiba*
az alapul szolgáló populáció
(valamilyen $P(X,Y)$ eloszlás, amelyet a $p(\mathbf{x},y)$ valószínűségi sűrűségfüggvény jellemez)
azon példányainak *várható* aránya,
amelyeknél az osztályozónk eltér a valódi címkétől:

$$\epsilon(f) =  E_{(\mathbf{x}, y) \sim P} \mathbf{1}(f(\mathbf{x}) \neq y) =
\int\int \mathbf{1}(f(\mathbf{x}) \neq y) p(\mathbf{x}, y) \;d\mathbf{x} dy.$$

Bár $\epsilon(f)$ az a mennyiség, amelyre valójában kíváncsiak vagyunk,
közvetlenül nem figyelhetjük meg,
ahogy közvetlenül sem mérhetjük
a nagy populáció átlagos magasságát
minden egyes személy megmérése nélkül.
Ezt a mennyiséget csak mintákon alapuló becslésekkel közelíthetjük.
Mivel a teszthalmazunk $\mathcal{D}$
statisztikailag reprezentálja az alapul szolgáló populációt,
az $\epsilon_\mathcal{D}(f)$-t tekinthetjük a $\epsilon(f)$ populációs hiba
statisztikai becslőjének.
Ráadásul mivel $\epsilon(f)$ egy várható érték
(a $\mathbf{1}(f(X) \neq Y)$ véletlen változó várható értéke),
a megfelelő $\epsilon_\mathcal{D}(f)$ becslő pedig a mintaátlag,
a populációs hiba becslése egyszerűen
az átlagbecslés klasszikus problémája,
amelyre visszaemlékezhetsz a :numref:`sec_prob` szakaszból.

A valószínűségszámítás egy fontos klasszikus eredménye,
a *centrális határeloszlás tétel* garantálja,
hogy valahányszor rendelkezünk $n$ véletlen mintával $a_1, ..., a_n$,
amelyeket bármely $\mu$ várható értékű és $\sigma$ szórású eloszlásból veszünk,
a mintaszám $n$ végtelenhez közeledtével
a mintaátlag $\hat{\mu}$ közelítőleg
egy normális eloszláshoz tart,
amelynek középpontja a valódi átlag,
szórása pedig $\sigma/\sqrt{n}$.
Ez már valami fontosat árul el:
ahogy a példák száma nő,
a teszt hibánk $\epsilon_\mathcal{D}(f)$
$\mathcal{O}(1/\sqrt{n})$ ütemben közelíti meg a valódi hibát $\epsilon(f)$.
Tehát a teszthiba kétszer pontosabb becsléséhez
négyszer akkora teszthalmazt kell gyűjteni.
A teszthiba százszorosnyi csökkentéséhez
tízezerszer akkora teszthalmazt kell gyűjteni.
Általában a statisztikában az $\mathcal{O}(1/\sqrt{n})$ sebesség
sokszor a legjobb, amire remélhetünk.

Most, hogy tudjuk, milyen aszimptotikus ütemben konvergál
$\epsilon_\mathcal{D}(f)$ teszthiba a valódi $\epsilon(f)$ hibához,
ráközelíthetünk néhány fontos részletre.
Emlékeztetőül: az érdekes
$\mathbf{1}(f(X) \neq Y)$ véletlen változó
csak $0$ és $1$ értéket vehet fel,
ezért ez egy Bernoulli véletlen változó,
amelyet egy paraméter jellemez,
amely megmutatja, milyen valószínűséggel vesz fel $1$ értéket.
Itt $1$ azt jelenti, hogy az osztályozónak hibája volt,
tehát a véletlen változó paramétere
valójában a valódi hibaarány $\epsilon(f)$.
Egy Bernoulli $\sigma^2$ varianciája
a paraméterétől (itt $\epsilon(f)$-től) függ
az $\epsilon(f)(1-\epsilon(f))$ kifejezés szerint.
Bár $\epsilon(f)$ kezdetben ismeretlen,
tudjuk, hogy nem lehet nagyobb $1$-nél.
E függvény vizsgálata feltárja, hogy a varianciánk akkor a legnagyobb,
ha a valódi hibaarány $0.5$ közelében van,
és sokkal kisebb lehet, ha $0$ vagy $1$ közelében van.
Ez azt jelenti, hogy a $\epsilon_\mathcal{D}(f)$ becslő aszimptotikus szórása
(az $n$ tesztminta megválasztása felett)
nem lehet nagyobb, mint $\sqrt{0.25/n}$.

Ha figyelmen kívül hagyjuk, hogy ez az ütem
végtelen teszthalmazméret esetén jellemzi a viselkedést,
nem véges mintaszám esetén,
ez azt jelenti, hogy ha azt akarjuk, hogy $\epsilon_\mathcal{D}(f)$ teszthibánk
közelítse a $\epsilon(f)$ populációs hibát
úgy, hogy egy szórásnyi intervallum $\pm 0.01$-nek felel meg,
akkor körülbelül 2500 mintát kell gyűjtenünk.
Ha két szórásnyi pontossággal akarunk rendelkezni
és így 95%-osan biztosak akarunk lenni abban, hogy
$\epsilon_\mathcal{D}(f) \in \epsilon(f) \pm 0.01$,
akkor 10 000 mintára lesz szükségünk!

Ez a mérete a gépi tanulás számos népszerű referenciahalmazának teszthalmazainak.
Meglepődhetsz, hogy évente több ezer alkalmazott mélytanulási cikket publikálnak,
amelyek $0.01$ vagy kisebb hibaarány-javítást ünnepelnek.
Természetesen ha a hibaarányok sokkal közelebb vannak $0$-hoz,
akkor a $0.01$ javulás valóban nagy ügy lehet.


Az eddigi elemzésünk bosszantó sajátossága,
hogy valójában csak az aszimptotikáról mond valamit,
vagyis arról, hogyan alakul $\epsilon_\mathcal{D}$ és $\epsilon$ kapcsolata,
ahogy a mintaméret végtelenhez tart.
Szerencsére, mivel a véletlen változónk korlátos,
érvényes véges mintabeli korlátokat kaphatunk
a Hoeffding (1963) egyenlőtlenségének alkalmazásával:

$$P(\epsilon_\mathcal{D}(f) - \epsilon(f) \geq t) < \exp\left( - 2n t^2 \right).$$

Az azon legkisebb adathalmazméret megkeresése,
amely 95%-os bizonyossággal lehetővé teszi annak levonását,
hogy a $\epsilon_\mathcal{D}(f)$ becslésünk
és a valódi $\epsilon(f)$ hibaarány távolsága $t$
nem haladja meg a $0.01$-et,
körülbelül 15 000 példányt igényel,
szemben a fenti aszimptotikus elemzés által javasolt 10 000 példánnyal.
Ha mélyebbre merülsz a statisztikában,
látni fogod, hogy ez a tendencia általánosan érvényes.
Még véges mintákra is érvényes garanciák
általában kissé konzervatívabbak.
Megjegyezzük, hogy ezek a számok
nem állnak nagyon messze egymástól,
ami tükrözi az aszimptotikus elemzés általános hasznosságát
abban, hogy hozzávetőleges értékeket ad
akkor is, ha ezek nem bírósági szintű garanciák.

## A teszthalmaz újrafelhasználása

Bizonyos értelemben most már készen állsz
az empirikus gépi tanulási kutatásban való sikerre.
Szinte minden gyakorlati modellt
teszthalmazos teljesítmény alapján fejlesztenek és validálnak,
és most már te is ismered a teszthalmaz kezelésének módját.
Bármely rögzített $f$ osztályozóhoz
tudod, hogyan értékeld a $\epsilon_\mathcal{D}(f)$ teszthiba,
és pontosan tudod, mit lehet (és mit nem lehet)
mondani a $\epsilon(f)$ populációs hibáról.

Tegyük fel tehát, hogy felhasználod ezt a tudást
és felkészülsz az első $f_1$ modelled tanítására.
Pontosan tudva, mennyire kell megbíznod
az osztályozód hibaarányának teljesítményében,
a fenti elemzés alapján meghatározod
a teszthalmaz méretét.
Tegyük fel továbbá, hogy megszívlelted
a :numref:`sec_generalization_basics` szakasz tanításait,
és biztosítottad a teszthalmaz sérthetetlenségét
azáltal, hogy az összes előzetes elemzést,
hiperparaméter-hangolást, sőt több versengő modellarchitektúra közötti választást
is egy validációs halmazon végezted.
Végül kiértékeled az $f_1$ modellt a teszthalmazon,
és közölsz egy torzítatlan becslést a populációs hibára
a hozzá tartozó konfidencia-intervallummal.

Eddig minden rendben látszik.
Azonban azon az éjjelen hajnali 3-kor felébredsz
egy zseniális ötlettel az új modellezési megközelítésre.
Másnap leprogramozod az új modellt,
a validációs halmazon hangolod a hiperparamétereit,
és nemcsak azt éred el, hogy az új $f_2$ modell működjön,
hanem úgy tűnik, a hibaaránya sokkal alacsonyabb, mint $f_1$-é.
Azonban a felfedezés izgalma hirtelen elmúlik,
ahogy felkészülsz a végső kiértékelésre.
Nincs teszthalmaz!

Bár az eredeti $\mathcal{D}$ teszthalmaz
még mindig ott van a szerveredre,
most két komoly problémával találod magad szemben.
Először: amikor összegyűjtötted a teszthalmazt,
a szükséges pontossági szintet
azon feltételezés alapján határoztad meg,
hogy egyetlen $f$ osztályozót értékelsz ki.
Ha azonban több $f_1, ..., f_k$ osztályozó
ugyanazon teszthalmazon való kiértékelésével kezdesz foglalkozni,
figyelembe kell venned a téves felfedezés problémáját.
Korábban 95%-ra lehettél biztos abban,
hogy $\epsilon_\mathcal{D}(f) \in \epsilon(f) \pm 0.01$
egyetlen $f$ osztályozóra,
és így a félrevezető eredmény valószínűsége mindössze 5% volt.
$k$ osztályozó esetén
nehéz garantálni,
hogy egyikük sem kap félrevezető teszthalmazos teljesítményt.
20 osztályozó figyelembe vételénél
esetleg semmi sem kizárható,
hogy legalább egyikük
félrevezető pontszámot kapott.
Ez a probléma a többszörös hipotézisteszteléshez kapcsolódik,
amely a statisztika hatalmas irodalma ellenére
továbbra is kísértő probléma a tudományos kutatásban.


Ha ez még nem elég aggasztó,
különös oka van arra is, hogy kételkedj
a következő kiértékeléseken kapott eredményekben.
Emlékeztetőül: a teszthalmaz-teljesítmény elemzésünk
azon feltételezésen alapult, hogy az osztályozót
a teszthalmazzal való kapcsolat nélkül választottuk,
és így a teszthalmaz tekinthető az alapul szolgáló populációból
véletlenszerűen kihúzottnak.
Itt azonban nemcsak több függvényt teszteltünk,
hanem a következő $f_2$ függvényt
az $f_1$ teszthalmaz-teljesítményének megfigyelése után választottuk.
Ha a teszthalmaz információja egyszer kiszivárgott a modellezőhöz,
a legszigorúbb értelemben soha többé nem lehet igazi teszthalmaz.
Ezt a problémát *adaptív túlillesztésnek* nevezik, és
a közelmúltban a tanuláselméleti kutatók és statisztikusok körében
intenzív érdeklődés tárgyává vált :cite:`dwork2015preserving`.
Szerencsére, bár lehetséges
az összes információt kiszivárogtatni egy holdout halmazból,
és az elméleti legrosszabb forgatókönyvek riasztók,
ezek az elemzések túl konzervatívak lehetnek.
A gyakorlatban ügyelj arra, hogy valódi teszthalmazokat hozz létre,
a lehető legritkábban konzultálj velük,
a konfidencia-intervallumok megadásakor vedd figyelembe
a többszörös hipotézistesztelést,
és éberségedet fokozd agresszívebben,
ha a tétek magasak és az adathalmaz kicsi.
Referenciaversenysorozat futtatásakor
sokszor jó gyakorlat több teszthalmazt fenntartani,
hogy minden forduló után
a régi teszthalmaz validációs halmazra léptethető le.

## Statisztikus tanulási elmélet

Egyszerűen fogalmazva: *valójában csak teszthalmazaink vannak*,
és mégis ez a tény furcsán kielégítetlennek tűnik.
Először is ritkán rendelkezünk *valódi teszthalmazsal* — hacsak
mi magunk nem hozzuk létre az adathalmazt,
valaki más valószínűleg már kiértékelte saját osztályozóját
a látszólagos „teszthalmazunkon".
Még ha elsők is vagyunk,
hamarosan frusztráltak leszünk, és azt kívánjuk,
hogy értékelhessük a következő modellezési kísérleteinket
anélkül, hogy az a gyötrelmes érzés kísért minket,
hogy nem bízhatunk a számokban.
Ráadásul még egy valódi teszthalmaz is csak *post hoc* mondhatja meg,
hogy egy osztályozó valóban általánosított-e a populációra,
nem azt, hogy van-e bármilyen okunk *a priori* elvárni,
hogy általánosítani fog.

Ezekkel a kételyekkel vértezve
talán most eléggé felkészültél arra,
hogy értékelni tudd a *statisztikus tanulási elmélet* vonzerejét,
a gépi tanulás azon matematikai területét,
amelynek művelői célja feltárni
azokat az alapelvi elveket, amelyek megmagyarázzák,
miért/mikor képesek az empirikus adatokon tanított modellek
nem látott adatokra is általánosítani.
A statisztikus tanulási kutatás egyik fő célja
az általánosítási rés korlátainak meghatározása,
a modelltulajdonságok és az adathalmaz mintaszáma közötti kapcsolat feltárásával.

A tanuláselméleti kutatók célja korlátot adni
a tanult $f_\mathcal{S}$ osztályozó
$\epsilon_\mathcal{S}(f_\mathcal{S})$ *empirikus hibája*
(amelyet a $\mathcal{S}$ tanítóhalmazon tanítottak és értékeltek ki)
és ugyanazon osztályozó
$\epsilon(f_\mathcal{S})$ valódi hibája közötti különbségre
az alapul szolgáló populáción.
Ez hasonlónak tűnhet az imént tárgyalt kiértékelési problémához,
de van egy lényeges különbség.
Korábban az $f$ osztályozó rögzített volt,
és csak kiértékelési célra volt szükségünk adathalmazra.
Valóban, bármely rögzített osztályozó általánosít:
hibája egy (korábban nem látott) adathalmazon
torzítatlan becslése a populációs hibának.
De mit mondhatunk, ha egy osztályozót
ugyanazon az adathalmazon tanítanak és értékelnek ki?
Lehetünk-e valaha biztosak abban,
hogy a tanítási hiba közel lesz a tesztelési hibához?


Tegyük fel, hogy a tanult $f_\mathcal{S}$ osztályozót
valamely előre meghatározott $\mathcal{F}$ függvénykészletből kell kiválasztani.
Emlékeztetőül a teszthalmazokról szóló vitából:
bár egyetlen osztályozó hibájának becslése egyszerű,
a dolgok bonyolódnak, ha osztályozók gyűjteményét vizsgáljuk.
Még ha bármely (rögzített) osztályozó empirikus hibája
nagy valószínűséggel közel lesz a valódi hibájához,
ha osztályozók gyűjteményét vizsgáljuk,
aggódni kell amiatt, hogy *legalább egyiküknek*
rosszul becsülik meg a hibáját.
Fennáll a veszély, hogy éppen ezt az osztályozót választjuk,
és ezzel nagymértékben alábecsüljük a populációs hibát.
Ráadásul még a lineáris modellek esetén is,
mivel paramétereik folyamatosan értékeltek,
általában végtelen számú függvényosztályból választunk ($|\mathcal{F}| = \infty$).

A probléma egyik nagyravágyó megoldása
analitikus eszközök kifejlesztése
az egyenletes konvergencia bizonyítására, azaz
annak megmutatására, hogy nagy valószínűséggel
az $f\in\mathcal{F}$ osztályban lévő összes osztályozó empirikus hibaaránya
*egyidejűleg* konvergál a valódi hibaarányhoz.
Más szóval egy olyan elméleti elvet keresünk,
amely lehetővé tenné annak kimondását, hogy
legalább $1-\delta$ valószínűséggel
(valamely kis $\delta$-ra)
egyetlen osztályozó hibaaránya $\epsilon(f)$ sem
(az $\mathcal{F}$ osztályban lévő összes osztályozó közül)
lesz $\alpha$ kisebb összegnél jobban alulbecsülve.
Nyilvánvaló, hogy ilyen állításokat
nem tehetünk minden $\mathcal{F}$ modelosztályra.
Emlékeztetőül a memorizáló gépek osztályára,
amelyek mindig elérnek $0$ empirikus hibát,
de soha nem teljesítenek jobban a véletlenszerű találgatásnál
az alapul szolgáló populáción.

Bizonyos értelemben a memorizálók osztálya túl rugalmas.
Ilyen egyenletes konvergencia eredménye nem állhat fenn.
Másrészt a rögzített osztályozó haszontalan —
tökéletesen általánosít, de nem illeszkedik
sem a tanítási, sem a tesztadatokhoz.
A tanulás alapkérdését ezért történelmileg
egy kompromisszumként fogalmazták meg:
rugalmasabb (nagyobb varianciájú) modelosztályok
jobban illeszkednek a tanítási adatokhoz, de kockáztatják a túlillesztést,
szemben a merevebb (nagyobb torzítású) modelosztályokkal,
amelyek jól általánosítanak, de kockáztatják az alulillesztést.
A tanuláselméleti kutatás egyik alapkérdése
a megfelelő matematikai elemzés kifejlesztése volt
annak kvantifikálásához,
hol helyezkedik el egy modell e spektrumon,
és a kapcsolódó garanciák megadásához.

Úttörő cikkek sorozatában
Vapnik és Chervonenkis kiterjesztette
a relatív frekvenciák konvergenciájára vonatkozó elméletet
általánosabb függvényosztályokra
:cite:`VapChe64,VapChe68,VapChe71,VapChe74b,VapChe81,VapChe91`.
E munkavonal egyik kulcsfontosságú eredménye
a Vapnik–Chervonenkis (VC) dimenzió,
amely (a komplexitás egy fogalmát mérve)
egy modelosztály komplexitását (rugalmasságát) méri.
Emellett egyik kulcsfontosságú eredményük korlátot ad
az empirikus és a populációs hiba különbségére
a VC dimenzió és a mintaszám függvényeként:

$$P\left(R[p, f] - R_\textrm{emp}[\mathbf{X}, \mathbf{Y}, f] < \alpha\right) \geq 1-\delta
\ \textrm{ for }\ \alpha \geq c \sqrt{(\textrm{VC} - \log \delta)/n}.$$

Itt $\delta > 0$ annak valószínűsége, hogy a korlátot megszegik,
$\alpha$ az általánosítási rés felső korlátja,
és $n$ az adathalmaz mérete.
Végül $c > 0$ egy állandó, amely csak
a felmerülő veszteség skálájától függ.
A korlát egyik felhasználása az lehet, hogy behelyettesítjük
a kívánt $\delta$ és $\alpha$ értékeket
a szükséges mintaszám meghatározásához.
A VC dimenzió azt a legnagyobb adatpontszámot kvantifikálja,
amelyre bármely (bináris) osztályozást hozzárendelhetünk,
és mindegyikre találunk valamilyen $f$ modellt az osztályban,
amely egyezik azzal az osztályozással.
Például a $d$-dimenziós bemenetű lineáris modellek VC dimenziója $d+1$.
Könnyen belátható, hogy egy egyenes bármilyen osztályozást hozzárendelhet
három kétdimenziós ponthoz, de négyhez nem.
Sajnos az elmélet hajlamos
túlságosan pesszimistának lenni a bonyolultabb modellek esetén,
és e garancia megszerzése jellemzően
sokkal több példányt igényel, mint amennyire valójában szükség van
a kívánt hibaarány eléréséhez.
Megjegyezzük továbbá, hogy a modelosztályt és $\delta$-t rögzítve,
a hibaarányunk ismét a megszokott $\mathcal{O}(1/\sqrt{n})$ ütemben csökken.
Valószínűtlen, hogy $n$ tekintetében jobbat tehetnénk.
Azonban ahogy változtatjuk a modelosztályt,
a VC dimenzió pesszimista képet festhet
az általánosítási résről.



## Összefoglalás

A modell kiértékelésének legkézenfekvőbb módja
egy korábban nem látott adatokat tartalmazó teszthalmaz.
A teszthalmaz-kiértékelések torzítatlan becslést adnak a valódi hibáról,
és a kívánt $\mathcal{O}(1/\sqrt{n})$ ütemben konvergálnak, ahogy a teszthalmaz nő.
Hozzávetőleges konfidencia-intervallumokat adhatunk
pontos aszimptotikus eloszlások alapján,
vagy érvényes véges mintás konfidencia-intervallumokat
(konzervatívabb) véges mintás garanciák alapján.
A teszthalmaz-kiértékelés valóban az alapja
a modern gépi tanulási kutatásnak.
Azonban a teszthalmazok ritkán valódi teszthalmazok
(amelyeket több kutató újra és újra felhasznál).
Ha ugyanazt a teszthalmazt több modell kiértékelésére alkalmazzák,
a téves felfedezés ellenőrzése nehéz lehet.
Ez elméletileg hatalmas problémákhoz vezethet.
A gyakorlatban a probléma jelentősége
a szóban forgó holdout halmazok méretétől függ,
és attól, hogy csupán hiperparaméterek kiválasztására használják-e őket,
vagy közvetlenebbül szivárogtatnak-e ki információt.
Mindazonáltal jó gyakorlat valódi (vagy több) teszthalmazt összeállítani,
és a lehető legkonzervatívabban kezelni felhasználásukat.


A kielégítőbb megoldás reményében
a statisztikus tanuláselméleti kutatók
módszereket dolgoztak ki az egyenletes konvergencia garantálására
egy modelosztályon.
Ha valóban minden modell empirikus hibája egyszerre
konvergál a valódi hibájához,
akkor szabadon kiválaszthatjuk a legjobban teljesítő modellt,
minimalizálva a tanítási hibát,
tudva, hogy a holdout adatokon is hasonlóan jól fog teljesíteni.
Alapvetően bármely ilyen eredménynek
a modelosztály valamely tulajdonságától kell függenie.
Vladimir Vapnik és Alexey Chervonenkis
bevezette a VC dimenziót,
egyenletes konvergencia eredményeket bemutatva,
amelyek érvényesek a VC osztály összes modelljére.
Az osztályban lévő összes modell tanítási hibái
(egyszerre) garantáltan közel vannak a valódi hibájukhoz,
és garantáltan $\mathcal{O}(1/\sqrt{n})$ ütemben egyre közelebb kerülnek.
A VC dimenzió forradalmi felfedezése után
számos alternatív komplexitásmértéket javasoltak,
mindegyik analóg általánosítási garanciát nyújt.
A függvénykomplexitás mérésének számos fejlett módjáról
részletes tárgyalásért lásd :citet:`boucheron2005theory`.
Sajnálatos módon, bár e komplexitásmértékek
a statisztikai elméletben széleskörűen hasznos eszközökké váltak,
tehetetlennek bizonyulnak
(egyszerűen alkalmazva)
a mély neurális hálózatok általánosításának magyarázatában.
A mély neurális hálózatoknak sokszor millió (vagy több) paraméterük van,
és könnyen véletlenszerű feliratokat rendelhetnek nagy ponthalmazokhoz.
Mégis jól általánosítanak a gyakorlati problémákon,
és meglepő módon sokszor jobban általánosítanak,
ha nagyobbak és mélyebbek,
annak ellenére, hogy magasabb VC dimenzióval járnak.
A következő fejezetben visszatérünk az általánosítás kérdésére
a mélytanulás kontextusában.

## Feladatok

1. Ha egy rögzített $f$ modell hibáját $0.0001$-en belülre akarjuk becsülni
   99.9%-os valószínűséggel,
   hány mintára van szükségünk?
1. Tegyük fel, hogy valaki más rendelkezik egy $\mathcal{D}$ osztályozott teszthalmazsal,
   és csak a nem osztályozott bemeneteket (jellemzőket) teszi elérhetővé.
   Most tegyük fel, hogy a teszthalmaz felirataihoz
   csak egy $f$ modell futtatásával (bármely modelosztályra vonatkozó megszorítás nélkül)
   a nem osztályozott bemeneteken
   és a megfelelő $\epsilon_\mathcal{D}(f)$ hiba megkapásával férhetsz hozzá.
   Hány modellt kellene kiértékelned,
   mielőtt kiszivárogtatod az egész teszthalmazt,
   és így $0$ hibával látszhatnál rendelkezni,
   függetlenül a valódi hibádtól?
1. Mi az ötödfokú polinomok osztályának VC dimenziója?
1. Mi a tengelyek irányával párhuzamos téglalapok VC dimenziója kétdimenziós adatokon?

[Megbeszélések](https://discuss.d2l.ai/t/6829)
