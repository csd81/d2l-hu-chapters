# A teljesen összekötött rétegektől a konvolúciókig
:label:`sec_why-conv`

A mai napig az eddig tárgyalt modellek megfelelő választási lehetőségek maradnak, amikor táblázatos adatokkal dolgozunk. Táblázatos alatt azt értjük, hogy az adatok példáknak megfelelő sorokból és jellemzőknek megfelelő oszlopokból állnak. Táblázatos adatok esetén arra számíthatunk, hogy a keresett minták a jellemzők közötti kölcsönhatásokat is tartalmazhatnak, de *a priori* nem feltételezünk semmilyen struktúrát a jellemzők kölcsönhatásával kapcsolatban.

Néha valóban hiányzik az a tudásunk, amely lehetővé tenné kifinomultabb architektúrák megalkotását. Ilyen esetekben az MLP lehet a legjobb megoldás. Azonban nagy dimenziós perceptuális adatok esetén az ilyen struktúra nélküli hálózatok kezelhetetlenné válhatnak.

Például térjünk vissza a macskák és kutyák megkülönböztetésére vonatkozó folyó példánkhoz. Tegyük fel, hogy alapos munkát végzünk az adatgyűjtésben, és egy egymegapixeles fénykép annotált adathalmazát gyűjtjük össze. Ez azt jelenti, hogy a hálózat minden bemenete egymillió dimenzióval rendelkezik. Még az ezer rejtett dimenzióra való agresszív csökkentés is egy $10^6 \times 10^3 = 10^9$ paraméterrel jellemzett teljesen összekötött réteget igényelne. Hacsak nincs sok GPU-nk, tehetségünk az elosztott optimalizáláshoz és rendkívüli türelmünk, akkor ennek a hálózatnak a paramétereinek megtanulása kivitelezhetetlen lehet.

Az éles szemű olvasó azzal ellenérezhet, hogy az egymegapixeles felbontás talán nem szükséges. Azonban bár kijöhetnénk százezer pixellel is, az 1000-es méretű rejtett rétegünk durván alábecsüli azoknak a rejtett egységeknek a számát, amelyek szükségesek a képek jó reprezentációinak megtanulásához, így egy gyakorlati rendszernek még mindig milliárdnyi paraméterre lesz szüksége. Ráadásul egy osztályozó megtanulása ennyi paraméter illesztésével hatalmas adathalmazok összegyűjtését igényelheti. Mégis ma mind az emberek, mind a számítógépek meglehetősen jól meg tudják különböztetni a macskákat a kutyáktól, ami látszólag ellentmond ezeknek az intuícióknak. Ennek oka, hogy a képek gazdag struktúrát mutatnak, amelyet mind az emberek, mind a gépi tanulási modellek kihasználhatnak. A konvolúciós neurális hálózatok (CNN-ek) az egyik kreatív módszer, amelyet a gépi tanulás alkalmazott a természetes képekben lévő ismert struktúra kiaknázásához.


## Invariancia

Képzeljük el, hogy egy objektumot szeretnénk észlelni egy képen. Ésszerűnek tűnik, hogy bármilyen módszert is használunk az objektumok felismerésére, annak nem kellene túlzottan törődnie az objektum pontos helyzetével a képen. Ideális esetben rendszerünk kihasználná ezt a tudást. A disznók általában nem repülnek, a repülők általában nem úsznak. Mindazonáltal fel kellene ismernünk egy disznót, ha a kép tetején jelenne meg. Inspirációt meríthetünk itt a "Hol van Waldo?" gyerekjátékból (amely maga is sok valós imitációt inspirált, mint például a :numref:`img_waldo`-ban látható). A játék számos kaotikus, tevékenységekkel teli jelenetet tartalmaz. Waldo valahol megjelenik mindegyikben, általában valamilyen valószínűtlen helyen lapul. Az olvasó feladata, hogy megtalálja őt. Jellegzetes öltözéke ellenére ez meglepően nehéz lehet, a sok elvonó elem miatt. Azonban *hogyan néz ki Waldo* nem függ attól, *hol van Waldo*. Végigpásztázhatnánk a képet egy Waldo-detektorral, amely pontszámot adhatna minden egyes foltnak, jelezve annak valószínűségét, hogy a folt tartalmazza Waldót. Valójában sok objektumdetektálási és szegmentálási algoritmus ezen a megközelítésen alapul :cite:`Long.Shelhamer.Darrell.2015`. A CNN-ek szisztematizálják a *térbeli invariancia* gondolatát, és kihasználják azt kevesebb paraméterrel hasznos reprezentációk megtanulásához.

![Meg tudod találni Waldót (a kép William Murphy (Infomatique) jóvoltából)?](../img/waldo-football.jpg)
:width:`400px`
:label:`img_waldo`

Most konkrétabbá tehetjük ezeket az intuíciókat azzal, hogy felsorolunk néhány kívánalmat egy számítógépes látásra alkalmas neurális hálózati architektúra tervezéséhez:

1. A korai rétegekben hálózatunknak hasonlóan kellene reagálnia ugyanarra a foltra, függetlenül attól, hol jelenik meg a képen. Ezt az elvet *eltolás-invarianciának* (vagy *eltolás-egyenértékűségnek*) nevezzük.
1. A hálózat korai rétegeinek lokális régiókra kellene összpontosítaniuk, a kép távolabbi területein lévő tartalmak figyelembe vétele nélkül. Ez a *lokalitás* elve. Végül ezeket a lokális reprezentációkat össze lehet gyűjteni az egész képre vonatkozó előrejelzések készítéséhez.
1. Ahogy haladunk előre, a mélyebb rétegeknek képesnek kellene lenniük a kép hosszabb hatótávolságú jellemzőinek rögzítésére, a természetes magasabb szintű látáshoz hasonló módon.

Lássuk, hogyan fordítható ez le matematikára.


## Az MLP korlátok közé szorítása

Kezdésképpen figyelembe vehetünk egy MLP-t kétdimenziós $\mathbf{X}$ képekkel mint bemenetekkel és azok közvetlen $\mathbf{H}$ rejtett reprezentációival, amelyeket hasonlóan mátrixokként ábrázolunk (kódban kétdimenziós tenzorok), ahol mind $\mathbf{X}$, mind $\mathbf{H}$ azonos alakú. Gondoljuk meg ezt. Most elképzeljük, hogy nemcsak a bemenetek, hanem a rejtett reprezentációk is térbeli struktúrával rendelkeznek.

Legyen $[\mathbf{X}]_{i, j}$ és $[\mathbf{H}]_{i, j}$ a bemeneti kép és rejtett reprezentáció $(i,j)$ helyzetű pixele. Következésképpen, ha minden rejtett egység bemenetet kap minden bemeneti pixeltől, súlymátrixok használatáról (ahogyan korábban az MLP-kben tettük) áttérnénk paramétereink negyedrendű $\mathsf{W}$ súlytenzorként való ábrázolásához. Tegyük fel, hogy $\mathbf{U}$ tartalmazza az eltolásokat, akkor formálisan kifejezhetjük a teljesen összekötött réteget:

$$\begin{aligned} \left[\mathbf{H}\right]_{i, j} &= [\mathbf{U}]_{i, j} + \sum_k \sum_l[\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}\\ &=  [\mathbf{U}]_{i, j} +
\sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}.\end{aligned}$$

Az átváltás $\mathsf{W}$-ről $\mathsf{V}$-re egyelőre pusztán kozmetikai, mivel egyértelmű megfelelés van a két negyedrendű tenzor együtthatói között. Egyszerűen újraindexeljük az $(k, l)$ alsó indexeket úgy, hogy $k = i+a$ és $l = j+b$. Más szóval, beállítjuk $[\mathsf{V}]_{i, j, a, b} = [\mathsf{W}]_{i, j, i+a, j+b}$. Az $a$ és $b$ indexek pozitív és negatív eltolásokra egyaránt futnak, lefedve az egész képet. A rejtett reprezentáció $[\mathbf{H}]_{i, j}$ bármely adott $(i, j)$ helyzetéhez értékét úgy számítjuk, hogy összegezzük az $x$ pixeleit $(i, j)$ körül, $[\mathsf{V}]_{i, j, a, b}$ által súlyozva. Mielőtt folytatnánk, vegyük figyelembe az ebben a parametrizációban szükséges paraméterek teljes számát *egyetlen* réteghez: egy $1000 \times 1000$-es kép (1 megapixel) egy $1000 \times 1000$-es rejtett reprezentációra van leképezve. Ez $10^{12}$ paramétert igényel, messze meghaladja a számítógépek jelenlegi kapacitását.

### Eltolás-invariancia

Most alkalmazzuk a fent megállapított első elvet: az eltolás-invarianciát :cite:`Zhang.ea.1988`. Ez azt jelenti, hogy a $\mathbf{X}$ bemeneti eltolásának egyszerűen a $\mathbf{H}$ rejtett reprezentáció eltolásához kell vezetnie. Ez csak akkor lehetséges, ha $\mathsf{V}$ és $\mathbf{U}$ valójában nem függnek $(i, j)$-től. Mint ilyen, $[\mathsf{V}]_{i, j, a, b} = [\mathbf{V}]_{a, b}$ és $\mathbf{U}$ konstans, mondjuk $u$. Ennek eredményeként egyszerűsíthetjük $\mathbf{H}$ definícióját:

$$[\mathbf{H}]_{i, j} = u + \sum_a\sum_b [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$


Ez egy *konvolúció*! Hatékonyan súlyozzuk az $(i+a, j+b)$ pixeleket az $(i, j)$ hely közelében $[\mathbf{V}]_{a, b}$ együtthatókkal, hogy megkapjuk a $[\mathbf{H}]_{i, j}$ értékét. Vegyük figyelembe, hogy $[\mathbf{V}]_{a, b}$-nek sokkal kevesebb együtthatóra van szüksége, mint $[\mathsf{V}]_{i, j, a, b}$-nek, mivel az már nem függ a képen belüli helyzettől. Következésképpen a szükséges paraméterek száma már nem $10^{12}$, hanem sokkal elfogadhatóbb $4 \times 10^6$: még mindig megmarad a függőség $a, b \in (-1000, 1000)$-re. Röviden, jelentős előrelépést tettünk. Az időkésleltetéses neurális hálózatok (TDNN-ek) az első példák egyike ennek a gondolatnak a kihasználására :cite:`Waibel.Hanazawa.Hinton.ea.1989`.

### Lokalitás

Most alkalmazzuk a második elvet: a lokalitást. Ahogyan a fentiekben motiváltuk, úgy gondoljuk, hogy nem kellene messzire tekintenünk az $(i, j)$ helytől, hogy releváns információt gyűjtsünk $[\mathbf{H}]_{i, j}$ értékeléséhez. Ez azt jelenti, hogy valamilyen $|a|> \Delta$ vagy $|b| > \Delta$ tartományon kívül be kellene állítanunk $[\mathbf{V}]_{a, b} = 0$-t. Ekvivalensen átírhatjuk $[\mathbf{H}]_{i, j}$-t:

$$[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$
:eqlabel:`eq_conv-layer`

Ez $4 \times 10^6$-ról $4 \Delta^2$-re csökkenti a paraméterek számát, ahol $\Delta$ általában kisebb, mint $10$. Így a paraméterek számát még négy nagyságrenddel csökkentettük. Vegyük figyelembe, hogy az :eqref:`eq_conv-layer` az, amit dióhéjban *konvolúciós rétegnek* nevezünk. A *konvolúciós neurális hálózatok* (CNN-ek) egy olyan speciális neurális hálózati család, amely konvolúciós rétegeket tartalmaz. A deep learning kutatóközösségben $\mathbf{V}$-t *konvolúciós kernelnek*, *szűrőnek* vagy egyszerűen a réteg tanulható paramétereinek nevezzük, azaz *súlyoknak*.

Míg korábban milliárdnyi paraméterre lett volna szükségünk egyetlen réteg megjelenítéséhez egy képfeldolgozó hálózatban, ma általában csak néhány százra van szükségünk, anélkül, hogy megváltoztatnánk a bemenetek vagy rejtett reprezentációk dimenzióit. Ennek a drasztikus paramétercsökkentésnek az ára az, hogy jellemzőink most eltolás-invariánsak, és hogy rétegünk csak lokális információkat képes beépíteni az egyes rejtett aktivációk értékének meghatározásakor. Minden tanulás az induktív elfogultság megkövetelésén alapul. Amikor ez az elfogultság megegyezik a valósággal, mintahatékony modelleket kapunk, amelyek jól általánosítanak látatlan adatokra. De természetesen ha ezek az elfogultságok nem egyeznek a valósággal, például ha a képek nem lennének eltolás-invariánsak, modelljeink küzdhetnének akár a tanítási adatok illesztésével is.

Ez a drámai paramétercsökkentés elvezet bennünket az utolsó kívánalmunkhoz, azaz hogy a mélyebb rétegeknek egy kép nagyobb és összetettebb aspektusait kell megjeleníteniük. Ezt nemlinearitások és konvolúciós rétegek ismételt váltogatásával lehet elérni.

## Konvolúciók

Röviden tekintsük át, miért nevezzük az :eqref:`eq_conv-layer`-t konvolúciónak. A matematikában két $f, g: \mathbb{R}^d \to \mathbb{R}$ függvény :cite:`Rudin.1973` *konvolúcióját* a következőképpen definiálják:

$$(f * g)(\mathbf{x}) = \int f(\mathbf{z}) g(\mathbf{x}-\mathbf{z}) d\mathbf{z}.$$

Vagyis mérjük az $f$ és $g$ közötti átfedést, amikor az egyik függvény "tükrözött" és $\mathbf{x}$ által eltolt. Amikor diszkrét objektumaink vannak, az integrál összeggé alakul. Például a $\mathbb{Z}$ indexen futó, négyzetösszegzett végtelen dimenziós vektorok halmazából vett vektorok esetén a következő definíciót kapjuk:

$$(f * g)(i) = \sum_a f(a) g(i-a).$$

Kétdimenziós tenzorok esetén egy megfelelő összeget kapunk $(a, b)$ indexekkel $f$-hez és $(i-a, j-b)$-vel $g$-hez:

$$(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b).$$
:eqlabel:`eq_2d-conv-discrete`

Ez hasonlít az :eqref:`eq_conv-layer`-re, egy lényeges különbséggel. Az $(i+a, j+b)$ helyett a különbséget használjuk. Megjegyezzük azonban, hogy ez a különbség többnyire kozmetikai, mivel mindig össze tudjuk egyeztetni az :eqref:`eq_conv-layer` és az :eqref:`eq_2d-conv-discrete` jelölését. Az :eqref:`eq_conv-layer`-ben szereplő eredeti definíciónk pontosabban egy *keresztkorrelációt* ír le. Erre a következő részben visszatérünk.


## Csatornák
:label:`subsec_why-conv-channels`

Visszatérve a Waldo-detektorunkhoz, lássuk, hogyan néz ki ez. A konvolúciós réteg adott méretű ablakokat vesz fel, és az intenzitásokat a $\mathsf{V}$ szűrő szerint súlyozza, ahogyan az a :numref:`fig_waldo_mask`-ban látható. Arra törekedhetünk, hogy olyan modellt tanuljunk, hogy ahol a "waldóság" a legmagasabb, ott csúcsot kell találnunk a rejtett réteg reprezentációiban.

![Waldo detektálása (a kép William Murphy (Infomatique) jóvoltából).](../img/waldo-mask.jpg)
:width:`400px`
:label:`fig_waldo_mask`

Csak egy probléma van ezzel a megközelítéssel. Eddig boldogan figyelmen kívül hagytuk, hogy a képek három csatornából állnak: piros, zöld és kék. Összefoglalva, a képek nem kétdimenziós objektumok, hanem harmadrendű tenzorok, amelyeket magasság, szélesség és csatorna jellemez, például $1024 \times 1024 \times 3$ pixeles alakkal. Míg az első két tengely térbeli kapcsolatokkal foglalkozik, a harmadikat úgy lehet tekinteni, mint amely többdimenziós reprezentációt rendel minden pixel-helyzethez. Így $\mathsf{X}$-et $[\mathsf{X}]_{i, j, k}$-ként indexeljük. A konvolúciós szűrőnek ennek megfelelően kell alkalmazkodnia. $[\mathbf{V}]_{a,b}$ helyett most $[\mathsf{V}]_{a,b,c}$-t kapunk.

Ráadásul, ahogyan bemenetünk harmadrendű tenzorból áll, jó ötletnek bizonyul rejtett reprezentációinkat hasonlóan harmadrendű $\mathsf{H}$ tenzorokként megfogalmazni. Más szóval, ahelyett, hogy csak egyetlen rejtett reprezentációnk lenne minden térbeli helyzethez, az egész rejtett reprezentáció-vektort szeretnénk minden térbeli helyzethez. A rejtett reprezentációkat úgy képzelhetjük el, mint egy sor egymásra rakott kétdimenziós rácsot. Ahogy a bemeneteknél, ezeket néha *csatornáknak* is nevezik. Néha *jellemzőtérképeknek* is hívják őket, mivel mindegyik egy térbeli, tanult jellemzőkészletet biztosít a következő réteg számára. Intuitívan elképzelhető, hogy az alacsonyabb, bemenetekhez közelebb lévő rétegekben egyes csatornák specializálódhatnak az élek felismerésére, míg mások textúrák felismerésére.

A bemenetek ($\mathsf{X}$) és rejtett reprezentációk ($\mathsf{H}$) több csatornájának támogatásához negyedik koordinátát adhatunk $\mathsf{V}$-hez: $[\mathsf{V}]_{a, b, c, d}$. Mindezt összerakva megkapjuk:

$$[\mathsf{H}]_{i,j,d} = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c [\mathsf{V}]_{a, b, c, d} [\mathsf{X}]_{i+a, j+b, c},$$
:eqlabel:`eq_conv-layer-channels`

ahol $d$ indexeli a $\mathsf{H}$ rejtett reprezentációk kimeneti csatornáit. A következő konvolúciós réteg ezután egy harmadrendű $\mathsf{H}$ tenzort vesz fel bemenetként. Az :eqref:`eq_conv-layer-channels`-t általánossága miatt vesszük több csatornát tartalmazó konvolúciós réteg definíciójaként, ahol $\mathsf{V}$ a réteg kernele vagy szűrője.

Még sok műveletet kell megvizsgálnunk. Például ki kell találnunk, hogyan kombináljuk az összes rejtett reprezentációt egyetlen kimenetté, például hogy van-e Waldo *bárhol* a képen. Azt is el kell döntenünk, hogyan számítsuk ki a dolgokat hatékonyan, hogyan kombináljuk a több réteget, milyen aktivációs függvényeket alkalmazzunk, és hogyan hozzunk ésszerű tervezési döntéseket a gyakorlatban hatékony hálózatok létrehozásához. Ezekkel a kérdésekkel a fejezet hátralévő részében foglalkozunk.

## Összefoglalás és vita

Ebben a részben az első elvekből vezettük le a konvolúciós neurális hálózatok struktúráját. Bár nem biztos, hogy ez volt a CNN-ek feltalálásához vezető út, megnyugtató tudni, hogy a képfeldolgozási és számítógépes látási algoritmusok működésére vonatkozó ésszerű elvek alkalmazásakor ezek a *helyes* választás, legalábbis alacsonyabb szinteken. Különösen a képek eltolás-invarianciája azt jelenti, hogy egy kép összes foltját azonos módon kezelik. A lokalitás azt jelenti, hogy csak a pixelek kis szomszédságát fogják felhasználni a megfelelő rejtett reprezentációk kiszámításához. A CNN-ekre való legkorábbi hivatkozások némelyike a Neocognitron formájában jelenik meg :cite:`Fukushima.1982`.

A második elv, amellyel okoskodásunkban találkoztunk, az, hogyan csökkentsük egy függvényosztály paramétereinek számát anélkül, hogy korlátoznánk kifejező erejét, legalábbis amikor a modellre vonatkozó bizonyos feltételezések teljesülnek. Ennek a korlátozásnak eredményeként a komplexitás drámai csökkenését láttuk, amelynek révén számítási és statisztikai szempontból megvalósíthatatlan problémák kezelhető modellekké alakultak.

A csatornák hozzáadása lehetővé tette, hogy visszahozzunk egy kis összetettséget, amely elveszett a lokalitás és eltolás-invariancia által a konvolúciós kernelre rótt korlátozások miatt. Megjegyezzük, hogy teljesen természetes más csatornákat hozzáadni, nem csak a pirosat, zöldet és kéket. Sok műholdkép, különösen a mezőgazdaság és a meteorológia területén, tíztől százig terjedő csatornával rendelkezik, hiperspektrális képeket generálva. Számos különböző hullámhosszon rögzítenek adatokat. A következőkben látni fogjuk, hogyan lehet hatékonyan felhasználni a konvolúciókat a rájuk ható képek dimenzionalitásának kezelésére, hogyan lehet átmenni a helyzet-alapú és csatorna-alapú reprezentációk között, és hogyan lehet hatékonyan kezelni a nagyszámú kategóriát.

## Feladatok

1. Tegyük fel, hogy a konvolúciós kernel mérete $\Delta = 0$. Mutassuk meg, hogy ebben az esetben a konvolúciós kernel minden csatornakészlethez függetlenül implementál egy MLP-t. Ez a Network in Network architektúrákhoz vezet :cite:`Lin.Chen.Yan.2013`.
1. Az audioadata gyakran egydimenziós szekvenciaként van ábrázolva.
    1. Mikor szeretnénk lokalitást és eltolás-invarianciát alkalmazni az audiohoz?
    1. Vezessük le az audio konvolúciós műveleteit.
    1. Kezelhető az audio ugyanolyan eszközökkel, mint a számítógépes látás? Tipp: használjuk a spektrogramot.
1. Miért lehet, hogy az eltolás-invariancia végül sem jó ötlet? Adjunk egy példát.
1. Gondolod, hogy a konvolúciós rétegek szöveges adatokra is alkalmazhatók lennének? Milyen problémákkal találkozhatunk a nyelv esetén?
1. Mi történik a konvolúciókkal, ha egy objektum egy kép határán van?
1. Bizonyítsuk be, hogy a konvolúció szimmetrikus, azaz $f * g = g * f$.

[Discussions](https://discuss.d2l.ai/t/64)
