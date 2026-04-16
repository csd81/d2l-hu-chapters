# Visszaterjesztés időben
:label:`sec_bptt`

Ha elvégezted a :numref:`sec_rnn-scratch` fejezetbeli feladatokat,
láthattad, hogy a gradiens vágás elengedhetetlen
az esetleges hatalmas gradiensek megakadályozásához,
amelyek destabilizálhatják a tanítást.
Utaltunk arra, hogy a robbanó gradiensek
a hosszú sorozatokon keresztüli visszaterjesztésből erednek.
Mielőtt bemutatnánk a modern RNN architektúrák sorát,
nézzük meg közelebbről, hogyan működik a *visszaterjesztés*
a sorozatmodellekben matematikailag részletesen.
Remélhetőleg ez a tárgyalás némi pontosságot hoz
az *eltűnő* és *robbanó* gradiensek fogalmába.
Ha visszaemlékszünk a számítási gráfokon keresztüli előre és visszafele terjedés tárgyalására,
amelyet az MLP-k bevezetésekor ismertettük a :numref:`sec_backprop` fejezetben,
akkor az RNN-ekben való előre terjedés
viszonylag egyértelmű kell legyen.
A visszaterjesztés alkalmazása az RNN-ekben
*visszaterjesztés időben* :cite:`Werbos.1990` néven ismert.
Ez az eljárás megköveteli, hogy az RNN számítási gráfját
egy időlépésenként kibontsuk (kitekerítsük).
A kitekert RNN lényegében
egy előrecsatolt neurális hálózat,
amelynek az a különleges tulajdonsága,
hogy ugyanazok a paraméterek
ismétlődnek a kitekert hálózaton keresztül,
minden időlépésnél megjelennek.
Majd, mint bármely előrecsatolt neurális hálózatban,
alkalmazhatjuk a láncolási szabályt,
visszaterjesztve a gradienseket a kitekert hálózaton.
Az egyes paraméterekre vonatkozó gradienst
össze kell adni a kitekert hálózat összes azon helyén,
ahol a paraméter előfordul.
Az ilyen súlymegosztás kezelése ismerős lehet
a konvolúciós neurális hálózatokról szóló fejezetekből.


A bonyodalmak abból erednek, hogy a sorozatok
meglehetősen hosszúak lehetnek.
Nem ritka, hogy ezernél több tokenből álló szöveges sorozatokkal dolgozunk.
Megjegyezzük, hogy ez problémákat okoz mind
számítási (túl sok memória),
mind optimalizálási (numerikus instabilitás)
szempontból.
Az első lépés bemenete
több mint 1000 mátrixszorzaton halad át, mielőtt megérkezik a kimenetre,
és további 1000 mátrixszorzat szükséges a gradiens kiszámításához.
Most elemezzük, mi mehet rosszul,
és hogyan kezeljük ezt a gyakorlatban.


## Gradiensek elemzése az RNN-ekben
:label:`subsec_bptt_analysis`

Egy egyszerűsített modellel kezdjük, amely leírja, hogyan működik egy RNN.
Ez a modell figyelmen kívül hagyja a rejtett állapot
részleteit és frissítési módját.
Az itt használt matematikai jelölés
nem különbözteti meg explicit módon
a skalárisokat, vektorokat és mátrixokat.
Csak némi intuíciót kívánunk kialakítani.
Ebben az egyszerűsített modellben
$h_t$-vel jelöljük a rejtett állapotot,
$x_t$-vel a bemenetet, és $o_t$-vel a kimenetet
a $t$ időlépésnél.
Emlékezünk a :numref:`subsec_rnn_w_hidden_states` fejezetbeli tárgyalásunkra,
amelyben a bemenet és a rejtett állapot
összefűzhető, mielőtt megszoroznák
a rejtett réteg egy súlyváltozójával.
Ezért $w_\textrm{h}$-t és $w_\textrm{o}$-t használunk a
rejtett réteg és a kimeneti réteg súlyainak jelölésére.
Ennek eredményeként a rejtett állapotok és kimenetek
minden időlépésnél:

$$\begin{aligned}h_t &= f(x_t, h_{t-1}, w_\textrm{h}),\\o_t &= g(h_t, w_\textrm{o}),\end{aligned}$$
:eqlabel:`eq_bptt_ht_ot`

ahol $f$ és $g$ rendre a rejtett réteg és a kimeneti réteg transzformációi.
Tehát egy rekurrens számítás által egymástól
függő értéklánccal rendelkezünk:
$\{\ldots, (x_{t-1}, h_{t-1}, o_{t-1}), (x_{t}, h_{t}, o_t), \ldots\}$.
Az előre terjedés meglehetősen egyértelmű.
Mindössze annyit kell tenni, hogy végighurkoljuk az $(x_t, h_t, o_t)$ tripleteket egy időlépésenként.
Az $o_t$ kimenet és az $y_t$ kívánt cél közötti eltérést
egy célfüggvény értékeli ki
az összes $T$ időlépésen:

$$L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_\textrm{h}, w_\textrm{o}) = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).$$



A visszaterjesztésnél a dolgok egy kicsit bonyolultabbak,
különösen amikor a gradienseket a $L$ célfüggvény $w_\textrm{h}$ paramétereire vonatkozóan számítjuk ki.
Konkrétan, a láncolási szabály alapján:

$$\begin{aligned}\frac{\partial L}{\partial w_\textrm{h}}  & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_\textrm{h}}  \\& = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_\textrm{o})}{\partial h_t}  \frac{\partial h_t}{\partial w_\textrm{h}}.\end{aligned}$$
:eqlabel:`eq_bptt_partial_L_wh`

A :eqref:`eq_bptt_partial_L_wh`-ban lévő szorzat
első és második tényezője könnyen kiszámítható.
A harmadik tényező $\partial h_t/\partial w_\textrm{h}$ az, ahol a dolgok bonyolulttá válnak,
mivel rekurrensen kell kiszámítani a $w_\textrm{h}$ paraméter $h_t$-re gyakorolt hatását.
A :eqref:`eq_bptt_ht_ot`-beli rekurrens számítás szerint
$h_t$ mind $h_{t-1}$-től, mind $w_\textrm{h}$-tól függ,
ahol $h_{t-1}$ kiszámítása
szintén függ $w_\textrm{h}$-tól.
Ezért a $h_t$ teljes deriváltjának kiszámítása
$w_\textrm{h}$-ra vonatkozóan a láncolási szabállyal:

$$\frac{\partial h_t}{\partial w_\textrm{h}}= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}} +\frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_\textrm{h}}.$$
:eqlabel:`eq_bptt_partial_ht_wh_recur`


A fenti gradiens levezetéséhez tegyük fel, hogy van
három $\{a_{t}\},\{b_{t}\},\{c_{t}\}$ sorozatunk,
amelyek kielégítik az $a_{0}=0$ és $a_{t}=b_{t}+c_{t}a_{t-1}$ feltételt, $t=1, 2,\ldots$-re.
Ekkor $t\geq 1$-re könnyen belátható:

$$a_{t}=b_{t}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t}c_{j}\right)b_{i}.$$
:eqlabel:`eq_bptt_at`

Az $a_t$, $b_t$ és $c_t$ helyettesítésével a következők szerint:

$$\begin{aligned}a_t &= \frac{\partial h_t}{\partial w_\textrm{h}},\\
b_t &= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}}, \\
c_t &= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}},\end{aligned}$$

a :eqref:`eq_bptt_partial_ht_wh_recur`-beli gradiens számítás kielégíti
az $a_{t}=b_{t}+c_{t}a_{t-1}$ feltételt.
Ezért a :eqref:`eq_bptt_at` szerint
eltávolíthatjuk a rekurrens számítást
a :eqref:`eq_bptt_partial_ht_wh_recur`-ból:

$$\frac{\partial h_t}{\partial w_\textrm{h}}=\frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_\textrm{h})}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_\textrm{h})}{\partial w_\textrm{h}}.$$
:eqlabel:`eq_bptt_partial_ht_wh_gen`

Bár a láncolási szabályt használhatjuk $\partial h_t/\partial w_\textrm{h}$ rekurzív kiszámítására,
ez a lánc nagyon hosszúvá válhat, ha $t$ nagy.
Tárgyaljunk néhány stratégiát ennek a problémának a kezelésére.

### Teljes számítás

Az egyik ötlet az lenne, hogy kiszámítjuk a teljes összeget a :eqref:`eq_bptt_partial_ht_wh_gen`-ben.
Ez azonban nagyon lassú, és a gradiensek felrobbanhatnak,
mivel a kezdeti feltételek apró változásai
potenciálisan nagy mértékben befolyásolhatják az eredményt.
Vagyis olyan jelenségeket láthatnánk, mint a pillangóeffektus,
ahol a kezdeti feltételek minimális változásai
aránytalanul nagy változásokhoz vezetnek az eredményben.
Ez általában nem kívánatos.
Végül is robusztus becslőket keresünk, amelyek jól általánosítanak.
Ezért ezt a stratégiát szinte soha nem alkalmazzák a gyakorlatban.

### Időlépések csonkítása

Alternatívaként
csonkíthatjuk az összeget a :eqref:`eq_bptt_partial_ht_wh_gen`-ben
$\tau$ lépés után.
Ezt tárgyaltuk eddig.
Ez az igaz gradiens *közelítéséhez* vezet,
egyszerűen a $\partial h_{t-\tau}/\partial w_\textrm{h}$-nál megszakítva az összeget.
A gyakorlatban ez meglehetősen jól működik.
Ezt nevezzük általában csonkított
visszaterjesztésnek időben :cite:`Jaeger.2002`.
Ennek egyik következménye az, hogy a modell
elsősorban a rövid távú hatásokra összpontosít,
nem pedig a hosszú távú következményekre.
Ez valójában *kívánatos*, mivel a becslést
egyszerűbb és stabilabb modellek felé torzítja.


### Véletlenszerű csonkítás

Végül a $\partial h_t/\partial w_\textrm{h}$-t
lecserélhetjük egy véletlenszerű változóra, amely várható értékben helyes,
de csonkítja a sorozatot.
Ezt előre meghatározott $0 \leq \pi_t \leq 1$ értékű $\xi_t$ sorozat alkalmazásával érhetjük el,
ahol $P(\xi_t = 0) = 1-\pi_t$ és
$P(\xi_t = \pi_t^{-1}) = \pi_t$, tehát $E[\xi_t] = 1$.
Ezt felhasználjuk a gradiens
$\partial h_t/\partial w_\textrm{h}$
cseréjéhez a :eqref:`eq_bptt_partial_ht_wh_recur`-ban:

$$z_t= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_\textrm{h}}.$$


A $\xi_t$ definíciójából következik,
hogy $E[z_t] = \partial h_t/\partial w_\textrm{h}$.
Valahányszor $\xi_t = 0$, a rekurrens számítás
leáll ennél a $t$ időlépésnél.
Ez különböző hosszúságú sorozatok súlyozott összegéhez vezet,
ahol a hosszú sorozatok ritkák, de megfelelően túlsúlyozottak.
Ezt az ötletet :citet:`Tallec.Ollivier.2017` javasolta.

### Stratégiák összehasonlítása

![Stratégiák összehasonlítása az RNN-ekben való gradiensszámításhoz. Felülről lefelé: véletlenszerű csonkítás, rendszeres csonkítás és teljes számítás.](../img/truncated-bptt.svg)
:label:`fig_truncated_bptt`


A :numref:`fig_truncated_bptt` szemlélteti a három stratégiát
az *Az időgép* első néhány karakterének elemzésekor
az RNN-ek időbeli visszaterjesztésével:

* Az első sor a véletlenszerű csonkítás, amely változó hosszúságú szegmensekre bontja a szöveget.
* A második sor a rendszeres csonkítás, amely azonos hosszúságú részsorozatokra bontja a szöveget. Ezt csináltuk az RNN kísérletekben.
* A harmadik sor a teljes visszaterjesztés időben, amely számítási szempontból megvalósíthatatlan kifejezéshez vezet.


Sajnos, bár elméletileg vonzó,
a véletlenszerű csonkítás nem működik
lényegesen jobban, mint a rendszeres csonkítás,
valószínűleg számos tényező miatt.
Először, egy megfigyelés hatása
néhány visszaterjesztési lépés után
a múltba meglehetősen elegendő
a függőségek megragadásához a gyakorlatban.
Másodszor, a megnövekedett variancia ellensúlyozza azt a tényt,
hogy a gradiens pontosabb több lépéssel.
Harmadszor, valójában *akarjuk*, hogy a modellek
csak rövid hatástávolságú interakciókkal rendelkezzenek.
Ezért a rendszeresen csonkított visszaterjesztés időben
enyhe regularizáló hatással rendelkezik, ami kívánatos lehet.

## Visszaterjesztés időben részletesen

Az általános elvek tárgyalása után
nézzük meg részletesebben a visszaterjesztést időben.
A :numref:`subsec_bptt_analysis`-beli elemzéssel ellentétben
az alábbiakban megmutatjuk, hogyan kell kiszámítani
a célfüggvény gradienseit
az összes felbontott modellparaméterre vonatkozóan.
Az egyszerűség kedvéért egy bias paraméterek nélküli RNN-t vizsgálunk,
amelynek rejtett rétegbeli aktivációs függvénye
az identitásleképezést ($\phi(x)=x$) használja.
A $t$ időlépésnél legyen az egyetlen példa bemenete
és a cél rendre $\mathbf{x}_t \in \mathbb{R}^d$ és $y_t$.
A $\mathbf{h}_t \in \mathbb{R}^h$ rejtett állapot
és a $\mathbf{o}_t \in \mathbb{R}^q$ kimenet
a következőképpen számítható:

$$\begin{aligned}\mathbf{h}_t &= \mathbf{W}_\textrm{hx} \mathbf{x}_t + \mathbf{W}_\textrm{hh} \mathbf{h}_{t-1},\\
\mathbf{o}_t &= \mathbf{W}_\textrm{qh} \mathbf{h}_{t},\end{aligned}$$

ahol $\mathbf{W}_\textrm{hx} \in \mathbb{R}^{h \times d}$, $\mathbf{W}_\textrm{hh} \in \mathbb{R}^{h \times h}$ és
$\mathbf{W}_\textrm{qh} \in \mathbb{R}^{q \times h}$
a súlyparaméterek.
Jelöljük $l(\mathbf{o}_t, y_t)$-vel
a $t$ időlépésnél a veszteséget.
A célfüggvényünk,
a sorozat elejétől számított $T$ időlépésen át tartó veszteség:

$$L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).$$


Az RNN számítása során a
modell változói és paraméterei közötti függőségek vizualizálásához
megrajzolhatjuk a modell számítási gráfját,
ahogy a :numref:`fig_rnn_bptt` mutatja.
Például a 3. időlépés rejtett állapotának kiszámítása,
$\mathbf{h}_3$, függ a $\mathbf{W}_\textrm{hx}$ és $\mathbf{W}_\textrm{hh}$ modellparaméterektől,
az előző időlépés $\mathbf{h}_2$ rejtett állapotától
és az aktuális időlépés $\mathbf{x}_3$ bemenetétől.

![Számítási gráf, amely három időlépéses RNN modell függőségeit mutatja. A négyzetek változókat (nem árnyékolt) vagy paramétereket (árnyékolt) jelölnek, a körök operátorokat jelölnek.](../img/rnn-bptt.svg)
:label:`fig_rnn_bptt`

Ahogy éppen megemlítettük, a :numref:`fig_rnn_bptt`-beli modell paraméterei
$\mathbf{W}_\textrm{hx}$, $\mathbf{W}_\textrm{hh}$ és $\mathbf{W}_\textrm{qh}$.
Általánosan, ennek a modellnek a tanítása megköveteli
a gradiensek kiszámítását ezen paraméterekre vonatkozóan:
$\partial L/\partial \mathbf{W}_\textrm{hx}$, $\partial L/\partial \mathbf{W}_\textrm{hh}$ és $\partial L/\partial \mathbf{W}_\textrm{qh}$.
A :numref:`fig_rnn_bptt`-beli függőségek szerint
ellentétes irányban haladhatunk a nyilakhoz képest,
hogy sorban kiszámítsuk és tároljuk a gradienseket.
Különböző alakú mátrixok, vektorok és skalárisok szorzatának
rugalmas kifejezéséhez a láncolási szabályban
továbbra is a $\textrm{prod}$ operátort alkalmazzuk,
ahogy a :numref:`sec_backprop` fejezetben leírtuk.


Mindenekelőtt, a célfüggvény differenciálása
a modell kimenete tekintetében bármelyik $t$ időlépésnél
meglehetősen egyszerű:

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.$$
:eqlabel:`eq_bptt_partial_L_ot`

Most kiszámíthatjuk a cél gradiensét
a $\mathbf{W}_\textrm{qh}$ kimeneti rétegbeli paraméterrel kapcsolatban:
$\partial L/\partial \mathbf{W}_\textrm{qh} \in \mathbb{R}^{q \times h}$.
A :numref:`fig_rnn_bptt` alapján
a $L$ cél $\mathbf{W}_\textrm{qh}$-tól függ
$\mathbf{o}_1, \ldots, \mathbf{o}_T$-n keresztül.
A láncolási szabály alkalmazása:

$$
\frac{\partial L}{\partial \mathbf{W}_\textrm{qh}}
= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_\textrm{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top,
$$

ahol $\partial L/\partial \mathbf{o}_t$-t
a :eqref:`eq_bptt_partial_L_ot` adja meg.

Ezután, ahogy a :numref:`fig_rnn_bptt` mutatja,
az utolsó $T$ időlépésnél
a $L$ célfüggvény
a $\mathbf{h}_T$ rejtett állapottól
csak $\mathbf{o}_T$-n keresztül függ.
Ezért könnyen megtalálhatjuk a
$\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$ gradienst
a láncolási szabállyal:

$$\frac{\partial L}{\partial \mathbf{h}_T} = \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.$$
:eqlabel:`eq_bptt_partial_L_hT_final_step`

Bonyolultabb a helyzet bármely $t < T$ időlépésnél,
ahol a $L$ célfüggvény $\mathbf{h}_{t+1}$-en és $\mathbf{o}_t$-n
keresztül függ $\mathbf{h}_t$-től.
A láncolási szabály szerint
a rejtett állapot $\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$ gradiense
bármely $t < T$ időlépésnél rekurrensen kiszámítható:


$$\frac{\partial L}{\partial \mathbf{h}_t} = \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) + \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right) = \mathbf{W}_\textrm{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.$$
:eqlabel:`eq_bptt_partial_L_ht_recur`

Az elemzéshez, kiterjesztve a rekurrens számítást
bármely $1 \leq t \leq T$ időlépésre:

$$\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_\textrm{hh}^\top\right)}^{T-i} \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.$$
:eqlabel:`eq_bptt_partial_L_ht`

Láthatjuk a :eqref:`eq_bptt_partial_L_ht`-ből,
hogy ez az egyszerű lineáris példa már
néhány kulcsfontosságú problémát mutat a hosszú sorozatmodelleknél:
$\mathbf{W}_\textrm{hh}^\top$ potenciálisan nagyon nagy hatványait tartalmazza.
Benne az 1-nél kisebb sajátértékek eltűnnek,
és az 1-nél nagyobb sajátértékek divergálnak.
Ez numerikusan instabil,
ami az eltűnő és robbanó gradiensek formájában nyilvánul meg.
Ennek kezelésének egyik módja az időlépések csonkítása
egy számítási szempontból kényelmes méretre,
ahogy a :numref:`subsec_bptt_analysis` fejezetben tárgyaltuk.
A gyakorlatban ez a csonkítás megvalósítható
a gradiens leválasztásával adott számú időlépés után.
Később meglátjuk, hogyan lehet
a fejlettebb sorozatmodellek, mint a long short-term memory,
tovább enyhíteni ezt a problémát.

Végül, a :numref:`fig_rnn_bptt` mutatja,
hogy a $L$ célfüggvény
a rejtett rétegbeli $\mathbf{W}_\textrm{hx}$ és $\mathbf{W}_\textrm{hh}$ modellparaméterektől
a $\mathbf{h}_1, \ldots, \mathbf{h}_T$ rejtett állapotokon keresztül függ.
Az ilyen paraméterekre vonatkozó gradiensek kiszámításához,
$\partial L / \partial \mathbf{W}_\textrm{hx} \in \mathbb{R}^{h \times d}$ és $\partial L / \partial \mathbf{W}_\textrm{hh} \in \mathbb{R}^{h \times h}$,
alkalmazzuk a láncolási szabályt:

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_\textrm{hx}}
&= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_\textrm{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_\textrm{hh}}
&= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_\textrm{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top,
\end{aligned}
$$

ahol a :eqref:`eq_bptt_partial_L_hT_final_step`
és a :eqref:`eq_bptt_partial_L_ht_recur` által rekurrensen kiszámított
$\partial L/\partial \mathbf{h}_t$
a numerikus stabilitást befolyásoló kulcsfontosságú mennyiség.



Mivel a visszaterjesztés időben a visszaterjesztés alkalmazása az RNN-ekben,
ahogy a :numref:`sec_backprop` fejezetben kifejtettük,
az RNN-ek tanítása felváltva alkalmazza az előre terjedést
a visszaterjesztéssel időben.
Sőt, a visszaterjesztés időben
sorban kiszámítja és tárolja a fenti gradienseket.
Konkrétan, a tárolt köztes értékeket
újrafelhasználják az ismételt számítások elkerülésére,
például $\partial L/\partial \mathbf{h}_t$ tárolása
mind $\partial L / \partial \mathbf{W}_\textrm{hx}$,
mind $\partial L / \partial \mathbf{W}_\textrm{hh}$ kiszámításához.


## Összefoglalás

A visszaterjesztés időben csupán a visszaterjesztés alkalmazása rejtett állapotú sorozatmodellekre.
Csonkítás, mint a rendszeres vagy véletlenszerű, szükséges a számítási kényelemhez és numerikus stabilitáshoz.
A mátrixok nagy hatványai divergáló vagy eltűnő sajátértékekhez vezetnek. Ez a robbanó vagy eltűnő gradiensek formájában nyilvánul meg.
A hatékony számítás érdekében a köztes értékeket gyorsítótárazzák a visszaterjesztés időben folyamán.



## Feladatok

1. Tegyük fel, hogy van egy szimmetrikus $\mathbf{M} \in \mathbb{R}^{n \times n}$ mátrixunk $\lambda_i$ sajátértékekkel, amelyeknek megfelelő sajátvektorai $\mathbf{v}_i$ ($i = 1, \ldots, n$). Általánosság megszorítása nélkül feltételezzük, hogy $|\lambda_i| \geq |\lambda_{i+1}|$ sorrendben vannak.
   1. Mutasd meg, hogy $\mathbf{M}^k$-nak $\lambda_i^k$ sajátértékei vannak!
   1. Bizonyítsd be, hogy egy véletlenszerű $\mathbf{x} \in \mathbb{R}^n$ vektorra nagy valószínűséggel $\mathbf{M}^k \mathbf{x}$ nagymértékben igazodik a $\mathbf{M}$ $\mathbf{v}_1$ sajátvektorához! Formalizáld ezt az állítást!
   1. Mit jelent a fenti eredmény az RNN-ek gradiensei szempontjából?
1. A gradiens vágáson kívül tudnál más módszereket is javasolni a gradiens robbanás kezelésére rekurrens neurális hálózatokban?

[Discussions](https://discuss.d2l.ai/t/334)
