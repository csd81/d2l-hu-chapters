# Bevezetés a Gauss-folyamatokba

A gépi tanulás sok esetben az adatokból való paraméterbecsléssel egyenértékű. Ezek a paraméterek általában nagyszámúak és viszonylag nehezen értelmezhetők — például egy neurális hálózat súlyai. A Gauss-folyamatok ezzel szemben mechanizmust biztosítanak arra, hogy közvetlenül érvelni lehessen azoknak a függvényeknek a magas szintű tulajdonságairól, amelyek illeszkedhetnek az adatainkra. Például érzékünk lehet arról, hogy ezek a függvények gyorsan változnak, periodikusak, feltételes függetlenségeket tartalmaznak, vagy eltolás-invarianciát mutatnak. A Gauss-folyamatok lehetővé teszik, hogy ezeket a tulajdonságokat könnyen beépítsük a modellünkbe, azáltal, hogy közvetlenül megadunk egy Gauss-eloszlást azokra a függvényértékekre, amelyek illeszkedhetnek az adatainkra.

Ismerkedjünk meg a Gauss-folyamatok működésével néhány példán keresztül.

Tegyük fel, hogy a következő adathalmazt figyeljük meg: regressziós célobjektumok (kimenetek), $y$, amelyeket bemenetek, $x$ indexelnek. Például a célváltozók lehetnek szén-dioxid-koncentrációk változásai, a bemenetek pedig az időpontok, amikor ezeket rögzítettük. Melyek az adatok jellemzői? Milyen gyorsan változnak látszólag? Rendszeres időközönként gyűjtöttük-e az adatpontokat, vagy vannak hiányzó bemenetek? Hogyan képzeled el a hiányzó területek kitöltését, vagy az előrejelzést $x=25$-ig?

![Megfigyelt adatok.](../img/gp-observed-data.svg)

Az adatok Gauss-folyamattal való illesztéséhez először meg kell adnunk egy prior eloszlást arról, milyen típusú függvényeket tartunk ésszerűnek. Itt megjelenítünk néhány mintafüggvényt egy Gauss-folyamatból. Ésszerűnek tűnik ez a prior? Megjegyezzük, hogy itt nem olyan függvényeket keresünk, amelyek illeszkednek az adathalmazunkra, hanem az ésszerű magas szintű tulajdonságokat definiáljuk, például azt, hogy milyen gyorsan változnak a bemenetek függvényében. Megjegyezzük, hogy az összes ábra reprodukálásához szükséges kódot a priort és az inferenciát tárgyaló következő notebookokban fogjuk látni.

![A modellel reprezentálni kívánt mintafüggvények a priorból.](../img/gp-sample-prior-functions.svg)

Miután kondicionálunk az adatokon, ezt a priort felhasználhatjuk egy poszterior eloszlás következtetéséhez azon függvények felett, amelyek illeszkedhetnek az adatokra. Itt megmutatjuk a minta poszterior függvényeket.

![Minta poszterior függvények az adatok megfigyelése után.](../img/gp-sample-posterior-functions.svg)

Látjuk, hogy ezek a függvények teljesen konzisztensek az adatainkkal, és tökéletesen átmennek minden megfigyelésen. Ahhoz, hogy ezeket a poszterior mintákat előrejelzések készítéséhez használjuk, átlagolni tudjuk a poszteriorból származó összes lehetséges mintafüggvény értékeit, hogy megkapjuk az alábbi vastag kék görbét. Megjegyezzük, hogy a várható értéket nem kell végtelen számú mintával kiszámolnunk; ahogyan később látjuk, ezt zárt formában kiszámíthatjuk.

![Poszterior minták a poszterior átlaggal együtt, amelyet pontbecsléshez lehet használni, kékkel.](../img/gp-posterior-samples.svg)

Szükségünk lehet a bizonytalanság reprezentációjára is, hogy tudjuk, mennyire bízhatunk az előrejelzéseinkben. Intuitívan nagyobb bizonytalanságunk kell, hogy legyen ott, ahol nagyobb a variabilitás a minta poszterior függvények között, mivel ez azt jelzi, hogy az igaz függvénynek sok lehetséges értéke lehet. Ezt a fajta bizonytalanságot _episztemikus bizonytalanságnak_ nevezzük, amely az _información hiányához_ kapcsolódó _csökkenthető bizonytalanság_. Ahogy több adatot gyűjtünk, ez a fajta bizonytalanság eltűnik, mivel egyre kevesebb megoldás lesz konzisztens azzal, amit megfigyelünk. Akárcsak a poszterior átlag esetén, a poszterior varianciát (ezen függvények variabilitását a poszteriorban) zárt formában kiszámíthatjuk. Árnyékolással az átlag mindkét oldalán a poszterior szórás kétszeresét mutatjuk, létrehozva egy _megbízhatósági tartományt_, amelynek 95%-os valószínűsége van, hogy tartalmazza a függvény igaz értékét bármely $x$ bemenet esetén.

![Poszterior minták, 95%-os megbízhatósági tartománnyal.](../img/gp-posterior-samples-95.svg)

Az ábra valamivel tisztábbnak néz ki, ha eltávolítjuk a poszterior mintákat, és egyszerűen az adatokat, a poszterior átlagot és a 95%-os megbízhatósági tartományt jelenítjük meg. Figyeljük meg, hogyan növekszik a bizonytalanság az adatoktól távolabb, ami az episztemikus bizonytalanság tulajdonsága.

![Pontbecslések és megbízhatósági tartomány.](../img/gp-point-predictions.svg)

Az adatillesztéshez használt Gauss-folyamat tulajdonságait erősen meghatározza az úgynevezett _kovariancia-függvény_, más néven _kernel_. Az általunk használt kovariancia-függvényt _RBF (Radial Basis Function) kernelnek_ nevezzük, amelynek alakja:
$$ k_{\textrm{RBF}}(x,x') = \textrm{Cov}(f(x),f(x')) = a^2 \exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right) $$

Ennek a kernelnek a _hiperparaméterei_ értelmezhetők. Az _amplitúdó_ paraméter $a$ szabályozza a függvény vertikális változásának skáláját, a _hosszskála_ paraméter $\ell$ pedig a függvény változási sebességét (fodrosságát). Nagyobb $a$ értékek nagyobb függvényértékeket, nagyobb $\ell$ értékek lassabban változó függvényeket jelentenek. Nézzük meg, mi történik a minta prior és poszterior függvényeinkkel, ahogy változtatjuk $a$-t és $\ell$-t.

A _hosszskálának_ különösen markáns hatása van egy Gauss-folyamat előrejelzéseire és bizonytalanságára. $||x-x'|| = \ell$ esetén két függvényérték között a kovariancia $a^2\exp(-0.5)$. $\ell$-nél nagyobb távolságoknál a függvényértékek szinte korreleálatlanná válnak. Ez azt jelenti, hogy ha előrejelzést szeretnénk készíteni az $x_*$ pontban, akkor az olyan $x$ bemenetek melletti függvényértékek, amelyekre $||x-x'||>\ell$, nem lesznek erős hatással az előrejelzéseinkre.

Nézzük meg, hogyan befolyásolja a hosszskála megváltoztatása a minta prior és poszterior függvényeket, valamint a megbízhatósági tartományokat. A fenti illesztések $2$ hosszskálát használnak. Most vizsgáljuk meg az $\ell = 0.1, 0.5, 2, 5, 10$ eseteket. Az $\ell = 0.1$ hosszskála nagyon kicsi az általunk vizsgált bemeneti tartományhoz, $25$-höz képest. Például az $x=5$ és $x=10$ pontbeli függvényértékek lényegében nulla korrelációval rendelkeznek ilyen hosszskálánál. Másrészt, $\ell = 10$ hosszskálánál e bemenetek függvényértékei erősen korreláltak lesznek. Megjegyezzük, hogy a következő ábrákban a vertikális skála változik.


![priorpoint1](../img/gp-priorpoint1.svg)
![postpoint1](../img/gp-postpoint1.svg)

![priorpoint5](../img/gp-priorpoint5.svg)
![postpoint5](../img/gp-postpoint5.svg)

![prior2](../img/gp-prior2.svg)
![post2](../img/gp-post2.svg)

![prior5](../img/gp-prior5.svg)
![post5](../img/gp-post5.svg)

Figyeljük meg, ahogy a hosszskála nő, a függvények „fodrossága" csökken, és bizonytalanságunk is csökken. Ha a hosszskála kicsi, a bizonytalanság gyorsan növekszik, ahogy eltávolodunk az adatoktól, mivel az adatpontok kevésbé informatívak lesznek a függvényértékekre vonatkozóan.

Most változtassuk meg az amplitúdó paramétert, rögzített $2$ hosszskálánál. Megjegyezzük, hogy a vertikális skála rögzített a prior mintáknál, és változik a poszterior mintáknál, hogy jól látható legyen mind a függvény skálájának növekedése, mind az adatillesztés.


![priorap1](../img/gp-priorap1.svg)
![postapoint1](../img/gp-postapoint1.svg)

![priora2](../img/gp-priora2.svg)
![posta2](../img/gp-posta2.svg)

![priora8](../img/gp-priora8.svg)
![posta8](../img/gp-posta8.svg)

Látjuk, hogy az amplitúdó paraméter a függvény skáláját befolyásolja, de nem a változás sebességét. Ezen a ponton az is érzékelhetővé válik, hogy az eljárásunk általánosítási teljesítménye függ ezeknek a hiperparamétereknek az ésszerű értékeitől. Az $\ell=2$ és $a=1$ értékek ésszerű illesztéseket adtak, míg néhány más érték nem. Szerencsére van egy robusztus és automatikus módszer a hiperparaméterek megadására, az úgynevezett _marginális valószínűség_ segítségével, amelyre visszatérünk az inferenciát tárgyaló notebookban.

Tehát mi is valójában egy Gauss-folyamat? Amint elkezdtük, egy Gauss-folyamat egyszerűen azt mondja, hogy bármely függvényértékek gyűjteménye $f(x_1),\dots,f(x_n)$, amelyeket bemenetek bármely gyűjteménye $x_1,\dots,x_n$ indexel, együttes multivariát Gauss-eloszlással rendelkezik. Ennek az eloszlásnak a $\mu$ átlagvektora egy _átlagfüggvény_ által adott, amelyet általában konstansnak vagy nullának vesznek. Az eloszlás kovariancia-mátrixát az összes bemeneti pár $x$-en kiértékelt _kernel_ adja.

$$\begin{bmatrix}f(x) \\f(x_1) \\ \vdots \\ f(x_n) \end{bmatrix}\sim \mathcal{N}\left(\mu, \begin{bmatrix}k(x,x) & k(x, x_1) & \dots & k(x,x_n) \\ k(x_1,x) & k(x_1,x_1) & \dots & k(x_1,x_n) \\ \vdots & \vdots & \ddots & \vdots \\ k(x_n, x) & k(x_n, x_1) & \dots & k(x_n,x_n) \end{bmatrix}\right)$$
:eqlabel:`eq_gp_prior`

A :eqref:`eq_gp_prior` egyenlet egy Gauss-folyamat priort határoz meg. Kiszámíthatjuk az $f(x)$ feltételes eloszlását bármely $x$-re, adott a megfigyelt $f(x_1), \dots, f(x_n)$ függvényértékek. Ezt a feltételes eloszlást _poszteriornak_ nevezzük, és ezt használjuk az előrejelzések készítéséhez.

Különösen,

$$f(x) | f(x_1), \dots, f(x_n) \sim \mathcal{N}(m,s^2)$$ 

ahol

$$m = k(x,x_{1:n}) k(x_{1:n},x_{1:n})^{-1} f(x_{1:n})$$ 

$$s^2 = k(x,x) - k(x,x_{1:n})k(x_{1:n},x_{1:n})^{-1}k(x,x_{1:n})$$ 

ahol $k(x,x_{1:n})$ egy $1 \times n$ vektor, amelyet $k(x,x_{i})$ kiértékelésével kapunk $i=1,\dots,n$ esetén, és $k(x_{1:n},x_{1:n})$ egy $n \times n$ mátrix, amelyet $k(x_i,x_j)$ kiértékelésével kapunk $i,j = 1,\dots,n$ esetén. $m$ az, amit pontbecslőként használhatunk bármely $x$-re, és $s^2$ az, amit bizonytalanságként használunk: ha egy 95%-os valószínűségű intervallumot szeretnénk létrehozni, amelybe $f(x)$ esik, a $m \pm 2s$ értéket használjuk. A fenti összes ábra prediktív átlagait és bizonytalanságait ezekkel az egyenletekkel állítottuk elő. A megfigyelt adatpontokat $f(x_1), \dots, f(x_n)$ adta, és finoman meghatározott $x$ pontok halmazát választottuk az előrejelzéshez.

Tegyük fel, hogy egyetlen adatpontot figyelünk meg, $f(x_1)$-t, és meg szeretnénk határozni $f(x)$ értékét valamely $x$-nél. Mivel $f(x)$-et Gauss-folyamat írja le, tudjuk, hogy az $(f(x), f(x_1))$ feletti együttes eloszlás Gauss:

$$
\begin{bmatrix}
f(x) \\ 
f(x_1) \\
\end{bmatrix}
\sim
\mathcal{N}\left(\mu, 
\begin{bmatrix}
k(x,x) & k(x, x_1) \\
k(x_1,x) & k(x_1,x_1)
\end{bmatrix}
\right)
$$

Az átlón kívüli $k(x,x_1) = k(x_1,x)$ kifejezés azt mutatja meg, milyen erősen korrelálnak a függvényértékek — mennyire határozza meg $f(x_1)$ az $f(x)$ értékét. Már láttuk, hogy ha nagy hosszskálát használunk $x$ és $x_1$ távolságához, $||x-x_1||$-hez képest, akkor a függvényértékek erősen korreláltak lesznek. Vizualizálhatjuk az $f(x)$ $f(x_1)$-ből való meghatározásának folyamatát mind a függvények terében, mind az $f(x_1), f(x)$ feletti együttes eloszlásban. Kezdetben tekintsük azt az $x$-et, amelyre $k(x,x_1) = 0.9$ és $k(x,x)=1$, vagyis $f(x)$ értéke mérsékelten korrelált $f(x_1)$ értékével. Az együttes eloszlásban az állandó valószínűség kontúrjai viszonylag keskeny ellipszisek lesznek.

Tegyük fel, hogy $f(x_1) = 1.2$-t figyelünk meg. Erre az $f(x_1)$ értékre való kondicionáláshoz vízszintes vonalat húzhatunk $1.2$-nél a sűrűség ábrájánál, és látjuk, hogy $f(x)$ értéke főként $[0.64,1.52]$ tartományra korlátozódik. Ezt az ábrát a függvénytérben is bemutattuk, az $f(x_1)$ megfigyelt pontot narancssárgával, és a Gauss-folyamat prediktív eloszlásának $f(x)$-re vonatkozó 1 szórását kékkel mutatva az $1.08$ átlagérték körül.

![Egy bivariát Gauss-sűrűség állandó valószínűségű kontúrjai $f(x_1)$ és $f(x)$ felett, $k(x,x_1) = 0.9$ esetén.](https://user-images.githubusercontent.com/6753639/206867364-b4707db5-0c2e-4ae4-a412-8292bca4d08d.svg)
![Gauss-folyamat prediktív eloszlás a függvénytérben $f(x)$-nél, $k(x,x_1) = 0.9$ esetén.](https://user-images.githubusercontent.com/6753639/206867367-3815720c-93c8-4b4b-80e7-296db1d3553b.svg)

Tegyük fel most, hogy erősebb a korreláció: $k(x,x_1) = 0.95$. Az ellipszisek most tovább szűkültek, és $f(x)$ értékét még jobban meghatározza $f(x_1)$. Vízszintes vonalat húzva $1.2$-nél, a $f(x)$-re vonatkozó kontúrok főként $[0.83, 1.45]$ értékeket támogatnak. Ismét megmutatjuk a függvénytérben is az ábrát, $1.14$ átlagos prediktív érték körüli egy szórással.

![Egy bivariát Gauss-sűrűség állandó valószínűségű kontúrjai $f(x_1)$ és $f(x)$ felett, $k(x,x_1) = 0.95$ esetén.](https://user-images.githubusercontent.com/6753639/206867797-20e42783-31de-4c50-8103-e9441ba6d0a9.svg)
![Gauss-folyamat prediktív eloszlás a függvénytérben $f(x)$-nél, $k(x,x_1)$ = 0.95 esetén.](https://user-images.githubusercontent.com/6753639/206867800-d9fc7add-649d-492c-8848-cab07c8fb83e.svg)

Látjuk, hogy Gauss-folyamatunk poszterior átlagbecslője közelebb van $1.2$-hez, mivel most erősebb a korreláció. Azt is látjuk, hogy bizonytalanságunk (a hibasávok) valamivel csökkent. Az erős korreláció ellenére bizonytalanságunk még mindig joggal meglehetősen nagy, mivel csak egyetlen adatpontot figyeltünk meg!

Ez az eljárás $f(x)$ posteriort adhat bármely $x$-re, bármennyi megfigyelt ponthoz. Tegyük fel, hogy $f(x_1), f(x_2)$-t figyeljük meg. Most vizualizáljuk $f(x)$ posteriort egy adott $x=x'$ pontban a függvénytérben. $f(x)$ pontos eloszlását a fenti egyenletek adják. $f(x)$ Gauss-eloszlású, átlaggal

$$m = k(x,x_{1:3}) k(x_{1:3},x_{1:3})^{-1} f(x_{1:3})$$

és varianciával

$$s^2 = k(x,x) - k(x,x_{1:3})k(x_{1:3},x_{1:3})^{-1}k(x,x_{1:3})$$

Ebben a bevezető notebookban _zajmentes_ megfigyeléseket vettünk figyelembe. Ahogyan látni fogjuk, könnyű megfigyelési zajt belefoglalni. Ha feltételezzük, hogy az adatok egy látens zajmentes $f(x)$ függvényből plusz iid Gauss-zaj $\epsilon(x) \sim \mathcal{N}(0,\sigma^2)$ és $\sigma^2$ varianciájú, akkor a kovariancia-függvényünk egyszerűen $k(x_i,x_j) \to k(x_i,x_j) + \delta_{ij}\sigma^2$ lesz, ahol $\delta_{ij} = 1$, ha $i=j$, és $0$ egyébként.

Már kezdünk némi intuíciót szerezni arról, hogyan lehet Gauss-folyamatot alkalmazni a megoldások priort és posteriort megadásához, és hogyan befolyásolja a kernel függvény ezen megoldások tulajdonságait. A következő notebookban pontosan bemutatjuk, hogyan kell megadni egy Gauss-folyamat priort, bevezetünk és levezetünk különböző kernel függvényeket, majd végigmegyünk azon, hogyan taníthatók automatikusan a kernel hiperparaméterei, és hogyan alakítható ki egy Gauss-folyamat posterior az előrejelzések készítéséhez. Bár időbe és gyakorlásba telik megszokni az olyan fogalmakat, mint a „függvények feletti eloszlások", a Gauss-folyamat prediktív egyenleteihez vezető tényleges mechanika valójában egészen egyszerű — ami megkönnyíti a gyakorlást e fogalmak intuitív megértéséhez.

## Összefoglalás

A tipikus gépi tanulásban szabad paraméterekkel rendelkező függvényt (például neurális hálózatot és annak súlyait) adunk meg, és az esetleg nem értelmezhető paraméterek becslésére összpontosítunk. Gauss-folyamattal ehelyett közvetlenül a függvények feletti eloszlásokról érvelünk, ami lehetővé teszi, hogy a megoldások magas szintű tulajdonságairól gondolkodjunk. Ezeket a tulajdonságokat a kovariancia-függvény (kernel) szabályozza, amelynek általában néhány, jól értelmezhető hiperparamétere van. Ezek a hiperparaméterek tartalmazzák a _hosszskálát_, amely szabályozza a függvények változásának sebességét (fodrosságát). Egy másik hiperparaméter az amplitúdó, amely a függvényeink vertikális változásának skáláját szabályozza.
Az adatokhoz illeszthető különböző függvények reprezentálása és ezek együttes kombinálása prediktív eloszlásba a Bayesi módszerek jellegzetes vonása. Mivel az adatoktól távolabb nagyobb a variabilitás a lehetséges megoldások között, bizonytalanságunk intuitívan növekszik, ahogy eltávolodunk az adatoktól.


A Gauss-folyamat az összes lehetséges függvényértékre vonatkozó multivariát normális (Gauss) eloszlás megadásával reprezentál eloszlást a függvények felett. Lehetséges könnyen manipulálni a Gauss-eloszlásokat, hogy megtaláljuk az egyik függvényérték eloszlását más értékek halmaza alapján. Más szóval, ha megfigyeljük a pontok egy halmazát, kondicionálhatunk ezekre a pontokra, és következtethetünk az eloszlásra arról, milyen lehet a függvény értéke bármely más bemenetnél. Azt, hogy hogyan modellezzük a korrelációkat ezen pontok között, a kovariancia-függvény határozza meg, és ez határozza meg a Gauss-folyamat általánosítási tulajdonságait. Bár időbe telik megszokni a Gauss-folyamatokat, könnyen kezelhetők, számos alkalmazásuk van, és segítenek megérteni és fejleszteni más modellosztályokat, például a neurális hálózatokat.

## Feladatok

1. Mi a különbség az episztemikus bizonytalanság és a megfigyelési bizonytalanság között?
2. A változási sebesség és az amplitúdó mellett milyen más tulajdonságait érdemes figyelembe venni a függvényeknek, és mi lenne a valós világi példa az ilyen tulajdonságokkal rendelkező függvényekre?
3. Az általunk vizsgált RBF kovariancia-függvény szerint a megfigyelések közötti kovarianciák (és korrelációk) csökkennek a bemeneti tér távolságával (időpontok, térbeli helyek stb.). Ez ésszerű feltételezés? Miért igen vagy miért nem?
4. Két Gauss-változó összege Gauss-eloszlású? Két Gauss-változó szorzata Gauss-eloszlású? Ha $(a,b)$ együttes Gauss-eloszlással rendelkezik, akkor $a|b$ (a adott b) Gauss-eloszlású? Gauss-eloszlású-e $a$?
5. Ismételd meg azt a feladatot, amelyben adatpontot figyelünk meg $f(x_1) = 1.2$-nél, de most tegyük fel, hogy $f(x_2) = 1.4$-et is megfigyelünk. Legyen $k(x,x_1) = 0.9$ és $k(x,x_2) = 0.8$. Biztosabbak leszünk-e $f(x)$ értékéről, mint amikor csak $f(x_1)$-t figyeltük meg? Mi az átlag és a 95\%-os megbízhatósági tartomány az $f(x)$ értékünkre most?
6. Szerinted a megfigyelési zaj becslésének növelése növeli vagy csökkenti az alapul szolgáló függvény hosszskálájának becslését?
7. Ahogy eltávolodunk az adatoktól, tegyük fel, hogy a prediktív eloszlásunk bizonytalansága egy pontig növekszik, majd nem növekszik tovább. Miért történhet ez?

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12115)
:end_tab:
