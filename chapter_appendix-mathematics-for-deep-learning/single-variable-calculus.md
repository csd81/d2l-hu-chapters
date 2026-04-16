# Egyváltozós analízis
:label:`sec_single_variable_calculus`

A :numref:`sec_calculus` fejezetben megismertük a differenciálszámítás alapvető elemeit. Ez a szakasz mélyebbre merül az analízis alapjaiba, és megvizsgálja, hogyan érthetjük meg és alkalmazhatjuk azokat a gépi tanulás kontextusában.

## Differenciálszámítás
A differenciálszámítás alapvetően azt tanulmányozza, hogyan viselkednek a függvények kis változások hatására. Hogy megértsük, miért ennyire alapvető ez a mélytanulás szempontjából, nézzünk egy példát.

Tegyük fel, hogy van egy mély neurális hálózatunk, amelynek súlyait az egyszerűség kedvéért egyetlen vektorba fűzzük össze: $\mathbf{w} = (w_1, \ldots, w_n)$. Adott egy tanítási adathalmaz, amelyen a neurális hálózatunk veszteségét $\mathcal{L}(\mathbf{w})$-vel jelöljük.

Ez a függvény rendkívül összetett – kódolja az adott architektúra összes lehetséges modelljének teljesítményét az adathalmazon –, ezért szinte lehetetlen megmondani, hogy a súlyok melyik $\mathbf{w}$ értéke minimalizálja a veszteséget. Ezért a gyakorlatban általában *véletlenszerűen* inicializáljuk a súlyokat, majd iteratívan kis lépéseket teszünk abba az irányba, amely a lehető leggyorsabban csökkenti a veszteséget.

A kérdés ekkor olyasmivé válik, ami felszínesen nem tűnik egyszerűbbnek: hogyan találjuk meg azt az irányt, amely a lehető leggyorsabban csökkenti a veszteséget? Ennek megvizsgálásához először tekintsük az egyetlen súlyból álló esetet: $L(\mathbf{w}) = L(x)$ egy egyetlen valós értékre, $x$-re.

Vegyük $x$-et, és próbáljuk megérteni, mi történik, ha egy kis $\epsilon$ értékkel megváltoztatjuk $x + \epsilon$-ra. Ha konkrétabb szemléltetést szeretnénk, gondoljunk például $\epsilon = 0{,}0000001$-re. A vizualizáció segítésére rajzoljuk fel az $f(x) = \sin(x^x)$ példafüggvényt a $[0, 3]$ intervallumon.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# Ábrázoljuk a függvényt normál tartományban
x_big = np.arange(0.01, 3.01, 0.01)
ys = np.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # Definiáljuk a pí-t a torch-ban

# Ábrázoljuk a függvényt normál tartományban
x_big = torch.arange(0.01, 3.01, 0.01)
ys = torch.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)).numpy() * 2  # Definiáljuk a pí-t a TensorFlow-ban

# Ábrázoljuk a függvényt normál tartományban
x_big = tf.range(0.01, 3.01, 0.01)
ys = tf.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

Ezen a nagy skálán a függvény viselkedése nem egyszerű. Ha azonban a tartományt valami kisebbre szűkítjük, például $[1{,}75; 2{,}25]$-re, azt látjuk, hogy a grafikon sokkal egyszerűbbé válik.

```{.python .input}
#@tab mxnet
# Ábrázoljuk ugyanazt a függvényt egy szűk tartományban
x_med = np.arange(1.75, 2.25, 0.001)
ys = np.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Ábrázoljuk ugyanazt a függvényt egy szűk tartományban
x_med = torch.arange(1.75, 2.25, 0.001)
ys = torch.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# Ábrázoljuk ugyanazt a függvényt egy szűk tartományban
x_med = tf.range(1.75, 2.25, 0.001)
ys = tf.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

Ha ezt a végletekig visszük, és egy apró szakaszra nagyítunk rá, a viselkedés sokkal egyszerűbbé válik: csupán egy egyenest látunk.

```{.python .input}
#@tab mxnet
# Ábrázoljuk ugyanazt a függvényt egy szűk tartományban
x_small = np.arange(2.0, 2.01, 0.0001)
ys = np.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Ábrázoljuk ugyanazt a függvényt egy szűk tartományban
x_small = torch.arange(2.0, 2.01, 0.0001)
ys = torch.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# Ábrázoljuk ugyanazt a függvényt egy szűk tartományban
x_small = tf.range(2.0, 2.01, 0.0001)
ys = tf.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

Ez az egyváltozós analízis kulcsmegfigyelése: az ismert függvények viselkedése elég kis tartományon egyenessel modellezhető. Ez azt jelenti, hogy a legtöbb függvénynél ésszerű elvárni, hogy ha a függvény $x$ értékét kicsit megváltoztatjuk, akkor az $f(x)$ kimenet is kicsit megváltozik. Az egyetlen kérdés, amelyre választ kell adnunk: „Mekkora a kimenet változása a bemenet változásához képest? Felekkora? Kétakkora?"

Így a függvény kimenetének változása és a bemenet kis megváltozása közötti arányt vizsgálhatjuk. Ezt formálisan felírhatjuk:

$$
\frac{L(x+\epsilon) - L(x)}{(x+\epsilon) - x} = \frac{L(x+\epsilon) - L(x)}{\epsilon}.
$$

Ez már elegendő ahhoz, hogy kódban is kipróbáljuk. Tegyük fel például, hogy tudjuk, $L(x) = x^{2} + 1701(x-4)^3$, ekkor megvizsgálhatjuk, mekkora ez az érték az $x = 4$ pontban.

```{.python .input}
#@tab all
# Definiáljuk a függvényünket
def L(x):
    return x**2 + 1701*(x-4)**3

# Print the difference divided by epsilon for several epsilon
for epsilon in [0.1, 0.001, 0.0001, 0.00001]:
    print(f'epsilon = {epsilon:.5f} -> {(L(4+epsilon) - L(4)) / epsilon:.5f}')
```

Ha figyelmesek vagyunk, észrevesszük, hogy a kapott szám gyanúsan közel van $8$-hoz. Ha $\epsilon$-t csökkentjük, az érték egyre közelebb kerül $8$-hoz. Így helyesen arra következtethetünk, hogy a keresett érték (azt mérő szám, mennyire változtatja meg a bemenet változása a kimenetet) az $x=4$ pontban $8$ kell legyen. A matematikus ezt a tényt a következőképpen foglalja össze:

$$
\lim_{\epsilon \rightarrow 0}\frac{L(4+\epsilon) - L(4)}{\epsilon} = 8.
$$

Egy kis történelmi kitérőként: a neurális hálózat-kutatás első néhány évtizedében a tudósok ezt az algoritmust (a *véges differenciák módszerét*) használták annak értékelésére, hogyan változik egy veszteségfüggvény kis perturbáció hatására: egyszerűen megváltoztatták a súlyokat, és megnézték, hogyan változik a veszteség. Ez számítástechnikailag nem hatékony, hiszen egyetlen változó egyetlen megváltozásának veszteségre gyakorolt hatásához a veszteségfüggvény két kiértékelése szükséges. Ha ezt akár csak néhány ezer paraméterrel próbálnánk elvégezni, a hálózatot az egész adathalmazon több ezer alkalommal kellene kiértékelni! Csak 1986-ban oldódott meg ez a probléma, amikor a :citet:`Rumelhart.Hinton.Williams.ea.1988` által bevezetett *visszaterjesztési algoritmus* lehetővé tette annak kiszámítását, hogy a súlyok *bármely* együttes változása hogyan befolyásolja a veszteséget – ugyanannyi számítási idővel, mint amennyi a hálózat egyetlen adathalmazon való előrejelzéséhez szükséges.

Visszatérve a példánkhoz: ez a $8$ érték különböző $x$ értékekre különböző, ezért ésszerű $x$ függvényeként definiálni. Formálisabban, ezt az értékfüggő változási sebességet *deriváltnak* nevezzük, amelyet így írunk fel:

$$\frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon}.$$
:eqlabel:`eq_der_def`

Különböző szövegek különböző jelölést alkalmaznak a deriváltra. Az alábbi jelölések mindegyike ugyanazt fejezi ki:

$$
\frac{df}{dx} = \frac{d}{dx}f = f' = \nabla_xf = D_xf = f_x.
$$

A legtöbb szerző egyetlen jelölést választ, és ahhoz ragaszkodik, bár ez sem garantált. Ajánlatos az összeset ismerni. Ebben a könyvben a $\frac{df}{dx}$ jelölést használjuk, kivéve ha összetett kifejezés deriváltját akarjuk venni, amelynek esetén a $\frac{d}{dx}f$ jelölést alkalmazzuk, például:
$$
\frac{d}{dx}\left[x^4+\cos\left(\frac{x^2+1}{2x-1}\right)\right].
$$

Sokszor szemléletesen hasznos visszafejteni a derivált :eqref:`eq_der_def` definícióját, és megnézni, hogyan változik a függvény, ha $x$-et kicsit megváltoztatjuk:

$$\begin{aligned} \frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon} & \implies \frac{df}{dx}(x) \approx \frac{f(x+\epsilon) - f(x)}{\epsilon} \\ & \implies \epsilon \frac{df}{dx}(x) \approx f(x+\epsilon) - f(x) \\ & \implies f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x). \end{aligned}$$
:eqlabel:`eq_small_change`

Az utolsó egyenlet különös figyelmet érdemel. Azt mondja el, hogy ha bármelyik függvénynél a bemenetet egy kis értékkel megváltoztatjuk, a kimenet egy, a derivált által skálázott kis értékkel változik meg.

Ily módon a derivált értelmezhető olyan skálázási tényezőként, amely megmutatja, mekkora kimeneti változást kapunk egy bemeneti változásra.

## Az analízis szabályai
:label:`sec_derivative_table`

Most rátérünk arra a feladatra, hogyan számíthatjuk ki egy explicit függvény deriváltját. Az analízis teljes formális tárgyalása mindent az alapelvekből vezetne le. Ezt a kísértést nem engedjük meg magunknak, hanem a leggyakoribb szabályok megértését nyújtjuk.

### Alapvető deriváltak
Ahogyan a :numref:`sec_calculus` fejezetben láttuk, a deriváltak kiszámításakor általában szabályok sorozatával lehet a számítást néhány alapfüggvényre visszavezetni. Az egyszerű visszakeresés érdekében itt megismételjük ezeket.

* **Konstans deriváltja.** $\frac{d}{dx}c = 0$.
* **Lineáris függvény deriváltja.** $\frac{d}{dx}(ax) = a$.
* **Hatványszabály.** $\frac{d}{dx}x^n = nx^{n-1}$.
* **Exponenciális függvény deriváltja.** $\frac{d}{dx}e^x = e^x$.
* **Logaritmus deriváltja.** $\frac{d}{dx}\log(x) = \frac{1}{x}$.

### Deriválási szabályok
Ha minden deriváltat külön kellene kiszámítani és táblázatban tárolni, a differenciálszámítás szinte lehetetlen lenne. A matematika ajándéka, hogy a fenti deriváltakat általánosíthatjuk, és összetettebb deriváltakat is kiszámíthatunk, például megtalálhatjuk az $f(x) = \log\left(1+(x-1)^{10}\right)$ függvény deriváltját. Ahogy a :numref:`sec_calculus` fejezetben is szó volt róla, ennek kulcsa az, hogy szabályba foglaljuk, mi történik, amikor függvényeket különféle módokon kombinálunk – legfontosabban: összeadás, szorzás és kompozíció esetén.

* **Összeg szabálya.** $\frac{d}{dx}\left(g(x) + h(x)\right) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$.
* **Szorzat szabálya.** $\frac{d}{dx}\left(g(x)\cdot h(x)\right) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$.
* **Láncszabály.** $\frac{d}{dx}g(h(x)) = \frac{dg}{dh}(h(x))\cdot \frac{dh}{dx}(x)$.

Nézzük meg, hogyan alkalmazhatjuk a :eqref:`eq_small_change` egyenletet ezen szabályok megértésére. Az összeg szabályához vegyük a következő gondolatmenetet:

$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon) + h(x+\epsilon) \\
& \approx g(x) + \epsilon \frac{dg}{dx}(x) + h(x) + \epsilon \frac{dh}{dx}(x) \\
& = g(x) + h(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right) \\
& = f(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right).
\end{aligned}
$$

Ha ezt az eredményt összevetjük azzal a ténnyel, hogy $f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x)$, látjuk, hogy $\frac{df}{dx}(x) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$, ami éppen az, amit kerestünk. Az intuíció a következő: amikor megváltoztatjuk a bemeneti $x$-et, $g$ és $h$ együttesen járulnak hozzá a kimenet $\frac{dg}{dx}(x)$ és $\frac{dh}{dx}(x)$ mértékű megváltozásához.


A szorzat esetében finomabb a helyzet, és szükség van egy új megfigyelésre a kifejezések kezeléséhez. Ugyanúgy kezdjük, mint korábban, a :eqref:`eq_small_change` egyenletet alkalmazva:

$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon)\cdot h(x+\epsilon) \\
& \approx \left(g(x) + \epsilon \frac{dg}{dx}(x)\right)\cdot\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& = g(x)\cdot h(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x) \\
& = f(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x). \\
\end{aligned}
$$


Ez hasonlít a fenti számításhoz, és valóban látjuk a válaszunkat ($\frac{df}{dx}(x) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$) az $\epsilon$ mellett, de ott van az $\epsilon^{2}$ méretű tag is. Ezt *magasabb rendű tagnak* nevezzük, mivel az $\epsilon^2$ hatványa nagyobb az $\epsilon^1$ hatványánál. Egy későbbi szakaszban látni fogjuk, hogy néha nyomon akarjuk követni ezeket, de most vegyük észre, hogy ha $\epsilon = 0{,}0000001$, akkor $\epsilon^{2}= 0{,}0000000000001$, ami óriási mértékben kisebb. Ahogy $\epsilon \rightarrow 0$ határhoz közelítünk, a magasabb rendű tagokat nyugodtan figyelmen kívül hagyhatjuk. E függelék általános szokásrendszereként a „$\approx$" jelet arra használjuk, hogy a két kifejezés a magasabb rendű tagokig egyenlő. Ha azonban formálisabbak akarunk lenni, megvizsgálhatjuk a különbséghányadost:

$$
\frac{f(x+\epsilon) - f(x)}{\epsilon} = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x) + \epsilon \frac{dg}{dx}(x)\frac{dh}{dx}(x),
$$

és látjuk, hogy ahogy $\epsilon \rightarrow 0$ felé tartunk, a jobb oldal utolsó tagja is nullához tart.

Végül a láncszabálynál ismét haladhatunk a korábbihoz hasonlóan, a :eqref:`eq_small_change` egyenletet alkalmazva:

$$
\begin{aligned}
f(x+\epsilon) & = g(h(x+\epsilon)) \\
& \approx g\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& \approx g(h(x)) + \epsilon \frac{dh}{dx}(x) \frac{dg}{dh}(h(x))\\
& = f(x) + \epsilon \frac{dg}{dh}(h(x))\frac{dh}{dx}(x),
\end{aligned}
$$

ahol a második sorban a $g$ függvény bemenetét ($h(x)$) $\epsilon \frac{dh}{dx}(x)$ apró értékkel eltoltnak tekintjük.

Ezek a szabályok rugalmas eszközkészletet adnak szinte bármilyen kívánt kifejezés kiszámításához. Például:

$$
\begin{aligned}
\frac{d}{dx}\left[\log\left(1+(x-1)^{10}\right)\right] & = \left(1+(x-1)^{10}\right)^{-1}\frac{d}{dx}\left[1+(x-1)^{10}\right]\\
& = \left(1+(x-1)^{10}\right)^{-1}\left(\frac{d}{dx}[1] + \frac{d}{dx}[(x-1)^{10}]\right) \\
& = \left(1+(x-1)^{10}\right)^{-1}\left(0 + 10(x-1)^9\frac{d}{dx}[x-1]\right) \\
& = 10\left(1+(x-1)^{10}\right)^{-1}(x-1)^9 \\
& = \frac{10(x-1)^9}{1+(x-1)^{10}}.
\end{aligned}
$$

Ahol minden sor a következő szabályokat alkalmazta:

1. A láncszabály és a logaritmus deriváltja.
2. Az összeg szabálya.
3. A konstans deriváltja, a láncszabály és a hatványszabály.
4. Az összeg szabálya, a lineáris függvény deriváltja, a konstans deriváltja.

E példa elvégzése után két dolognak kell világossá válnia:

1. Bármely függvény, amelyet összeadások, szorzatok, konstansok, hatványok, exponenciálisok és logaritmusok segítségével le tudunk írni, mechanikusan deriválható ezen szabályok alkalmazásával.
2. Ha egy ember követi ezeket a szabályokat, az fáradságos és hibalehetőségekkel teli lehet!

Szerencsére ez a két tény együttesen egy előre mutató utat sejtet: ez tökéletes jelölt a gépesítésre! A visszaterjesztés, amelyet ebben a szakaszban később ismét tárgyalunk, pontosan ezt valósítja meg.

### Lineáris közelítés
A deriváltak használatakor hasznos a fentebb alkalmazott közelítést geometriailag is értelmezni. Különösen vegyük észre, hogy az

$$
f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x)
$$

egyenlet az $f$ értékét egy olyan egyenessel közelíti, amely átmegy az $(x, f(x))$ ponton, és $\frac{df}{dx}(x)$ meredekségű. Ily módon azt mondjuk, hogy a derivált lineáris közelítést ad az $f$ függvényre, ahogyan az alábbiakban látható:

```{.python .input}
#@tab mxnet
# Számítsuk ki a szinuszt
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Számoljunk néhány lineáris közelítést. Használjuk, hogy d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Számítsuk ki a szinuszt
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Számoljunk néhány lineáris közelítést. Használjuk, hogy d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) *
                 torch.cos(torch.tensor(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# Számítsuk ki a szinuszt
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]

# Számoljunk néhány lineáris közelítést. Használjuk, hogy d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) *
                 tf.cos(tf.constant(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

### Magasabb rendű deriváltak

Tegyük most azt, ami felszínesen furcsának tűnhet. Vegyük az $f$ függvényt, és számítsuk ki a $\frac{df}{dx}$ deriváltját. Ez megadja az $f$ változási sebességét minden pontban.

A $\frac{df}{dx}$ derivált azonban maga is tekinthető függvénynek, így semmi nem akadályoz meg minket abban, hogy kiszámítsuk a $\frac{df}{dx}$ deriváltját, és megkapjuk a $\frac{d^2f}{dx^2} = \frac{df}{dx}\left(\frac{df}{dx}\right)$ értéket. Ezt $f$ második deriváltjának nevezzük. Ez a függvény az $f$ változási sebességének változási sebessége, más szóval azt fejezi ki, hogyan változik maga a változási sebesség. A deriváltat tetszőleges számú alkalommal alkalmazhatjuk, hogy az úgynevezett $n$-edik deriváltat kapjuk meg. A jelölés egyszerűsítése érdekében az $n$-edik deriváltat a következőképpen jelöljük:

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$

Próbáljuk megérteni, *miért* hasznos ez a fogalom. Az alábbiakban vizualizáljuk az $f^{(2)}(x)$, $f^{(1)}(x)$ és $f(x)$ függvényeket.

Először vegyük azt az esetet, amikor a második derivált $f^{(2)}(x)$ pozitív konstans. Ez azt jelenti, hogy az első derivált meredeksége pozitív. Emiatt az első derivált $f^{(1)}(x)$ kezdetben negatív lehet, egy ponton nullává válik, majd pozitív lesz. Ez azt mondja meg az eredeti $f$ függvény meredekségéről, és ebből következőleg maga az $f$ függvény is csökken, ellaposodik, majd növekszik. Más szóval az $f$ függvény felfelé görbül, és egyetlen minimuma van, ahogyan a :numref:`fig_positive-second` ábrán látható.

![Ha feltételezzük, hogy a második derivált pozitív konstans, akkor az első derivált növekvő, ami azt jelenti, hogy magának a függvénynek minimuma van.](../img/posSecDer.svg)
:label:`fig_positive-second`


Másodszor, ha a második derivált negatív konstans, az azt jelenti, hogy az első derivált csökken. Ebből az következik, hogy az első derivált kezdetben pozitív lehet, egy ponton nullává válik, majd negatív lesz. Tehát maga az $f$ függvény növekszik, ellaposodik, majd csökken. Más szóval az $f$ függvény lefelé görbül, és egyetlen maximuma van, ahogyan a :numref:`fig_negative-second` ábrán látható.

![Ha feltételezzük, hogy a második derivált negatív konstans, akkor az első derivált csökkenő, ami azt jelenti, hogy magának a függvénynek maximuma van.](../img/negSecDer.svg)
:label:`fig_negative-second`


Harmadszor, ha a második derivált mindig nulla, akkor az első derivált soha nem változik – állandó! Ez azt jelenti, hogy $f$ állandó ütemben növekszik (vagy csökken), és $f$ maga egy egyenes, ahogyan a :numref:`fig_zero-second` ábrán látható.

![Ha feltételezzük, hogy a második derivált nulla, akkor az első derivált konstans, ami azt jelenti, hogy maga a függvény egy egyenes.](../img/zeroSecDer.svg)
:label:`fig_zero-second`

Összefoglalva, a második derivált értelmezhető úgy, mint amely leírja, hogyan görbül az $f$ függvény. Egy pozitív második derivált felfelé görbülést eredményez, a negatív második derivált azt jelenti, hogy $f$ lefelé görbül, a nulla második derivált pedig azt, hogy $f$ egyáltalán nem görbül.

Menjünk egy lépéssel tovább. Tekintsük a $g(x) = ax^{2}+ bx + c$ függvényt. Ekkor kiszámíthatjuk:

$$
\begin{aligned}
\frac{dg}{dx}(x) & = 2ax + b \\
\frac{d^2g}{dx^2}(x) & = 2a.
\end{aligned}
$$

Ha adott egy $f(x)$ eredeti függvényünk, kiszámíthatjuk az első két deriváltat, és megtalálhatjuk az $a, b$ és $c$ értékeket, amelyek illeszkednek erre a számításra. Hasonlóan az előző szakaszhoz, ahol láttuk, hogy az első derivált adja a legjobb egyenessel való közelítést, ez a konstrukció a legjobb másodfokú közelítést adja. Vizualizáljuk ezt az $f(x) = \sin(x)$ függvényre.

```{.python .input}
#@tab mxnet
# Számítsuk ki a szinuszt
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Számoljunk néhány másodfokú közelítést. Használjuk, hogy d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0) -
                              (xs - x0)**2 * np.sin(x0) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Számítsuk ki a szinuszt
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Számoljunk néhány másodfokú közelítést. Használjuk, hogy d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) *
                 torch.cos(torch.tensor(x0)) - (xs - x0)**2 *
                 torch.sin(torch.tensor(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# Számítsuk ki a szinuszt
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]

# Számoljunk néhány másodfokú közelítést. Használjuk, hogy d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) *
                 tf.cos(tf.constant(x0)) - (xs - x0)**2 *
                 tf.sin(tf.constant(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

Ezt az ötletet a következő szakaszban a *Taylor-sor* fogalmára terjesztjük ki.

### Taylor-sor


A *Taylor-sor* módszert ad az $f(x)$ függvény közelítésére, ha ismerjük az első $n$ derivált értékét egy $x_0$ pontban, azaz a $\left\{ f(x_0), f^{(1)}(x_0), f^{(2)}(x_0), \ldots, f^{(n)}(x_0) \right\}$ értékhalmazt. Az ötlet az, hogy megtaláljuk azt az $n$-edfokú polinomot, amely egyezik az összes adott deriválttal $x_0$-ban.

Az $n=2$ esetet az előző szakaszban láttuk, és egy kis algebrával megmutatható, hogy ez:

$$
f(x) \approx \frac{1}{2}\frac{d^2f}{dx^2}(x_0)(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

Ahogy fentebb látható, a $2$-es nevező azért van ott, hogy kiejtsük azt a $2$-t, amelyet akkor kapunk, ha $x^2$-t kétszer deriválunk, míg a többi tag mind nulla. Ugyanez a logika érvényes az első deriváltra és magára az értékre is.

Ha a logikát tovább visszük $n=3$-ig, arra a következtetésre jutunk, hogy

$$
f(x) \approx \frac{\frac{d^3f}{dx^3}(x_0)}{6}(x-x_0)^3 + \frac{\frac{d^2f}{dx^2}(x_0)}{2}(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

ahol a $6 = 3 \times 2 = 3!$ abból a konstansból ered, amelyet az $x^3$ háromszoros deriválásakor kapunk.


Ráadásul $n$-edfokú polinomot kaphatunk a következőképpen:

$$
P_n(x) = \sum_{i = 0}^{n} \frac{f^{(i)}(x_0)}{i!}(x-x_0)^{i}.
$$

ahol az $n$-edik derivált jelölése:

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$


Valóban, $P_n(x)$ tekinthető az $f(x)$ függvényünk legjobb $n$-edfokú polinomiális közelítésének.

Bár nem merülünk el teljesen a fenti közelítések hibájában, érdemes megemlíteni a végtelen határesetet. Ebben az esetben jól viselkedő függvényekre (az úgynevezett valós analitikus függvényekre), mint például $\cos(x)$ vagy $e^{x}$, felírhatjuk a végtelen sok tagot, és pontosan ugyanazt a függvényt közelíthetjük meg:

$$
f(x) = \sum_{n = 0}^\infty \frac{f^{(n)}(x_0)}{n!}(x-x_0)^{n}.
$$

Vegyük az $f(x) = e^{x}$ függvényt példaként. Mivel $e^{x}$ saját maga deriváltja, tudjuk, hogy $f^{(n)}(x) = e^{x}$. Tehát $e^{x}$ rekonstruálható az $x_0 = 0$ pontban vett Taylor-sorral:

$$
e^{x} = \sum_{n = 0}^\infty \frac{x^{n}}{n!} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots.
$$

Nézzük meg, hogyan működik ez kódban, és figyeljük meg, hogyan közelít a Taylor-közelítés fokának növelése az $e^x$ kívánt függvényhez.

```{.python .input}
#@tab mxnet
# Számítsuk ki az exponenciális függvényt
xs = np.arange(0, 3, 0.01)
ys = np.exp(xs)

# Számítsunk ki néhány Taylor-sor közelítést
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponenciális függvény", "1. fokú Taylor-sor", "2. fokú Taylor-sor",
    "5. fokú Taylor-sor"])
```

```{.python .input}
#@tab pytorch
# Számítsuk ki az exponenciális függvényt
xs = torch.arange(0, 3, 0.01)
ys = torch.exp(xs)

# Számítsunk ki néhány Taylor-sor közelítést
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponenciális függvény", "1. fokú Taylor-sor", "2. fokú Taylor-sor",
    "5. fokú Taylor-sor"])
```

```{.python .input}
#@tab tensorflow
# Számítsuk ki az exponenciális függvényt
xs = tf.range(0, 3, 0.01)
ys = tf.exp(xs)

# Számítsunk ki néhány Taylor-sor közelítést
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponenciális függvény", "1. fokú Taylor-sor", "2. fokú Taylor-sor",
    "5. fokú Taylor-sor"])
```

A Taylor-soroknak két fő alkalmazása van:

1. *Elméleti alkalmazások*: Amikor egy túlságosan összetett függvényt próbálunk megérteni, a Taylor-sor segítségével olyan polinommá alakíthatjuk, amellyel közvetlenül dolgozhatunk.

2. *Numerikus alkalmazások*: Egyes függvények, mint $e^{x}$ vagy $\cos(x)$, nehezen számíthatók ki géppel. Tárolhatók rögzített pontosságú értéktáblázatok (és ezt gyakran meg is teszik), de ez még nyitva hagy olyan kérdéseket, mint „Mi $\cos(1)$ ezredik jegye?" A Taylor-sorok sokszor hasznosak az ilyen kérdések megválaszolásában.


## Összefoglalás

* A deriváltak segítségével kifejezhető, hogyan változnak a függvények, ha a bemenetet egy kis értékkel megváltoztatjuk.
* Az elemi deriváltak deriválási szabályok segítségével kombinálhatók tetszőlegesen összetett deriváltak előállításához.
* A deriváltak iterálhatók, hogy második vagy magasabb rendű deriváltakat kapjunk. Minden rendnövekedés finomabb információt ad a függvény viselkedéséről.
* Egyetlen adatpélda deriváltjainak információit felhasználva a jól viselkedő függvényeket közelíthetjük a Taylor-sorból nyert polinomokkal.


## Feladatok

1. Mi az $x^3-4x+1$ deriváltja?
2. Mi a $\log(\frac{1}{x})$ deriváltja?
3. Igaz vagy hamis: Ha $f'(x) = 0$, akkor $f$-nek maximuma vagy minimuma van $x$-ben?
4. Hol van az $f(x) = x\log(x)$ minimuma $x\ge0$ esetén (ahol feltesszük, hogy $f$ határértéke $f(0)$-ban $0$)?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/412)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1088)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1089)
:end_tab:
