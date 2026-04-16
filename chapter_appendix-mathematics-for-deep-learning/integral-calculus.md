# Integrálszámítás
:label:`sec_integral_calculus`

A differenciálszámítás csupán egyik felét teszi ki a hagyományos analízisoktatásnak. A másik pillér, az integrálszámítás, látszólag egészen más kérdéssel indul: „Mekkora a terület ezen görbe alatt?" Bár elsőre kapcsolatlannak tűnik, az integrálás szorosan összefonódik a differenciálással a *differenciálszámítás alaptételén* keresztül.

A könyvben tárgyalt gépi tanulás szintjén nem lesz szükségünk az integrálás mélyreható ismeretére. Mégis adunk egy rövid bevezetést, hogy megalapozzuk a később előforduló alkalmazásokat.

## Geometriai értelmezés
Tegyük fel, hogy adott az $f(x)$ függvény. Az egyszerűség kedvéért legyen $f(x)$ nemnegatív (soha ne vegyen fel nullánál kisebb értéket). Azt szeretnénk megérteni: mekkora a terület $f(x)$ és az $x$-tengely között?

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()

x = np.arange(-2, 2, 0.01)
f = np.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist(), f.tolist())
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from mpl_toolkits import mplot3d
import torch

x = torch.arange(-2, 2, 0.01)
f = torch.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist(), f.tolist())
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from mpl_toolkits import mplot3d
import tensorflow as tf

x = tf.range(-2, 2, 0.01)
f = tf.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.numpy(), f.numpy())
d2l.plt.show()
```

A legtöbb esetben ez a terület végtelen vagy értelmezetlen (gondoljunk az $f(x) = x^{2}$ alatti területre), ezért általában egy $a$ és $b$ végpontpár közötti területről szokás beszélni.

```{.python .input}
#@tab mxnet
x = np.arange(-2, 2, 0.01)
f = np.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist()[50:250], f.tolist()[50:250])
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
x = torch.arange(-2, 2, 0.01)
f = torch.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist()[50:250], f.tolist()[50:250])
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
x = tf.range(-2, 2, 0.01)
f = tf.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.numpy()[50:250], f.numpy()[50:250])
d2l.plt.show()
```

Ezt a területet az alábbi integráljelöléssel jelöljük:

$$
\textrm{Area}(\mathcal{A}) = \int_a^b f(x) \;dx.
$$

A belső változó egy néma változó, hasonlóan a $\sum$ összegzés indexéhez, ezért ezt bármilyen belső értékkel egyenértékűen írhatjuk:

$$
\int_a^b f(x) \;dx = \int_a^b f(z) \;dz.
$$

Hagyományos módszer az ilyen integrálok közelítésére: képzeljük el, hogy az $a$ és $b$ közötti tartományt $N$ függőleges szeletre osztjuk. Ha $N$ elég nagy, minden egyes szelet területét közelíthetjük egy téglalappal, amelynek magassága $f(x)$ értéke, alapja pedig $\epsilon$ szélességű. Ezután a téglalapok területeit összeadjuk, hogy megkapjuk a görbe alatti teljes területet. Nézzünk meg egy kódpéldát! A pontos értéket egy későbbi szakaszban fogjuk meghatározni.

```{.python .input}
#@tab mxnet
epsilon = 0.05
a = 0
b = 2

x = np.arange(a, b, epsilon)
f = x / (1 + x**2)

approx = np.sum(epsilon*f)
true = np.log(2) / 2

d2l.set_figsize()
d2l.plt.bar(x.asnumpy(), f.asnumpy(), width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'közelítés: {approx}, igaz érték: {true}'
```

```{.python .input}
#@tab pytorch
epsilon = 0.05
a = 0
b = 2

x = torch.arange(a, b, epsilon)
f = x / (1 + x**2)

approx = torch.sum(epsilon*f)
true = torch.log(torch.tensor([5.])) / 2

d2l.set_figsize()
d2l.plt.bar(x, f, width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'közelítés: {approx}, igaz érték: {true}'
```

```{.python .input}
#@tab tensorflow
epsilon = 0.05
a = 0
b = 2

x = tf.range(a, b, epsilon)
f = x / (1 + x**2)

approx = tf.reduce_sum(epsilon*f)
true = tf.math.log(tf.constant([5.])) / 2

d2l.set_figsize()
d2l.plt.bar(x, f, width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'közelítés: {approx}, igaz érték: {true}'
```

A gond az, hogy bár numerikusan elvégezhető, analitikusan csak a legegyszerűbb függvényekre alkalmazható ez a módszer, mint például

$$
\int_a^b x \;dx.
$$

Bármi valamivel összetettebb, mint a fenti kódpéldában szereplő

$$
\int_a^b \frac{x}{1+x^{2}} \;dx,
$$

már meghaladja azt, amit ilyen közvetlen módszerrel meg tudunk oldani.

Ehelyett más megközelítést választunk. Intuitív módon dolgozunk a terület fogalmával, és megtanuljuk az integrálok kiszámításához használt fő eszközt: a *differenciálszámítás alaptételét*. Ez lesz az integrálás tanulmányozásának alapja.

## A differenciálszámítás alaptétele

Az integráció elmélyítéséhez vezessük be az alábbi függvényt:

$$
F(x) = \int_0^x f(y) dy.
$$

Ez a függvény a $0$ és $x$ közötti területet méri attól függően, hogyan változtatjuk $x$-et. Figyeljük meg, hogy erre van szükségünk, hiszen

$$
\int_a^b f(x) \;dx = F(b) - F(a).
$$

Ez matematikailag azt fejezi ki, hogy a területet a távolabbi végpontig mérjük, majd kivonjuk a közelebbi végpontig mért területet, ahogy azt :numref:`fig_area-subtract` szemlélteti.

![Annak szemléltetése, hogy a görbe alatti terület kiszámítását két pont között visszavezethetjük egy pont baloldalán lévő terület kiszámítására.](../img/sub-area.svg)
:label:`fig_area-subtract`

Tehát bármely intervallum feletti integrált megkaphatjuk, ha meghatározzuk, mi az $F(x)$.

Ehhez gondolatkísérletet végzünk. Ahogy az analízisben megszokott, képzeljük el, mi történik, ha az értéket egy apró $\epsilon$-nal eltoljuk. A fentiek alapján tudjuk, hogy

$$
F(x+\epsilon) - F(x) = \int_x^{x+\epsilon} f(y) \; dy.
$$

Ez azt mondja meg, hogy a függvény értéke egy vékony függvényszelet alatti területtel változik.

Ezen a ponton közelítést alkalmazunk. Ha egy ilyen vékony területszeletet vizsgálunk, az közel áll egy olyan téglalap területéhez, amelynek magassága $f(x)$ értéke, alapszélessége pedig $\epsilon$. Belátható, hogy $\epsilon \rightarrow 0$ esetén ez a közelítés egyre pontosabb lesz. Tehát levonhatjuk a következtetést:

$$
F(x+\epsilon) - F(x) \approx \epsilon f(x).
$$

Most észrevehetjük: ez pontosan az a minta, amelyet $F$ deriváltjának kiszámításakor várnánk! Így a következő meglepő tényre jutunk:

$$
\frac{dF}{dx}(x) = f(x).
$$

Ez a *differenciálszámítás alaptétele*. Kifejtett formában írva:
$$\frac{d}{dx}\int_0^x  f(y) \; dy = f(x).$$
:eqlabel:`eq_ftc`

A terület meghatározásának fogalmát (ami *a priori* meglehetősen nehéz) visszavezeti egy deriváltakra vonatkozó állításra (amelyet sokkal jobban értünk). Egy utolsó megjegyzés: ez nem mondja meg pontosan, mi az $F(x)$. Valóban, $F(x) + C$ bármely $C$-re ugyanolyan deriválttal rendelkezik. Ez az integráció elméletének velejárója. Szerencsére, határozott integrálokkal dolgozva a konstansok kiesnek, és így irrelevánssá válnak:

$$
\int_a^b f(x) \; dx = (F(b) + C) - (F(a) + C) = F(b) - F(a).
$$

Ez talán elvontnak tűnhet, de érdemes értékelni, hogy teljesen új nézőpontot ad az integrálok kiszámításához. Célunk többé nem egyfajta felosztásos-összegzéses eljárás végrehajtása a terület visszakapásához, hanem csupán olyan függvényt kell találnunk, amelynek deriváltja az integrálandó függvény! Ez rendkívüli lehetőség, hiszen így számos nehéz integrált meg tudunk határozni, egyszerűen a :numref:`sec_derivative_table` táblázatát visszafelé olvasva. Például tudjuk, hogy $x^{n}$ deriváltja $nx^{n-1}$. Ebből az alaptétel :eqref:`eq_ftc` felhasználásával következik, hogy

$$
\int_0^{x} ny^{n-1} \; dy = x^n - 0^n = x^n.
$$

Hasonlóan, tudjuk, hogy $e^{x}$ deriváltja önmaga, tehát

$$
\int_0^{x} e^{x} \; dx = e^{x} - e^{0} = e^x - 1.
$$

Ily módon az integrálás teljes elméletét a differenciálszámítás fogalmaira támaszkodva fejleszthetjük ki. Minden integrálási szabály erre az egy tényre vezethető vissza.

## Változócsere
:label:`subsec_integral_example`

Csakúgy, mint a differenciálásnál, az integrálok kiszámítását megkönnyítő szabályok sora létezik. Valójában a differenciálszámítás minden szabályának (szorzatszabály, összegszabály, láncszabály) megfelel egy integrálszámítási szabály (részleges integráció, az integrálás linearitása, illetve a változócsere-képlet). Ebben a szakaszban a lista talán legfontosabbját tárgyaljuk: a változócsere-képletet.

Először tegyük fel, hogy adott egy függvény, amely maga is integrál:

$$
F(x) = \int_0^x f(y) \; dy.
$$

Tegyük fel, hogy tudni akarjuk, hogyan néz ki ez a függvény, ha egy másik függvénnyel összetéve kapjuk $F(u(x))$-t. A láncszabály alapján tudjuk, hogy

$$
\frac{d}{dx}F(u(x)) = \frac{dF}{du}(u(x))\cdot \frac{du}{dx}.
$$

Ezt az alaptétel :eqref:`eq_ftc` segítségével integrálásra vonatkozó állítássá alakíthatjuk:

$$
F(u(x)) - F(u(0)) = \int_0^x \frac{dF}{du}(u(y))\cdot \frac{du}{dy} \;dy.
$$

Felidézve, hogy $F$ maga is integrál, a bal oldal átírható:

$$
\int_{u(0)}^{u(x)} f(y) \; dy = \int_0^x \frac{dF}{du}(u(y))\cdot \frac{du}{dy} \;dy.
$$

Hasonlóan, mivel $F$ integrál, az alaptétel :eqref:`eq_ftc` alapján $\frac{dF}{dx} = f$, ezért

$$\int_{u(0)}^{u(x)} f(y) \; dy = \int_0^x f(u(y))\cdot \frac{du}{dy} \;dy.$$
:eqlabel:`eq_change_var`

Ez a *változócsere-képlet*.

Szemléletes levezetésként gondoljuk meg, mi történik, ha $f(u(x))$ integrálját vesszük $x$ és $x+\epsilon$ között. Kis $\epsilon$-ra ez az integrál közelítőleg $\epsilon f(u(x))$, a kapcsolódó téglalap területe. Hasonlítsuk össze ezt $f(y)$ integrálával $u(x)$-től $u(x+\epsilon)$-ig. Tudjuk, hogy $u(x+\epsilon) \approx u(x) + \epsilon \frac{du}{dx}(x)$, így ennek a téglalapnak a területe közelítőleg $\epsilon \frac{du}{dx}(x)f(u(x))$. Ahhoz tehát, hogy a két téglalap területe egyező legyen, az elsőt $\frac{du}{dx}(x)$-szel kell megszoroznunk, ahogy azt :numref:`fig_rect-transform` szemlélteti.

![Egyetlen vékony téglalap változócsere alatti transzformációjának szemléltetése.](../img/rect-trans.svg)
:label:`fig_rect-transform`

Ez azt jelenti, hogy

$$
\int_x^{x+\epsilon} f(u(y))\frac{du}{dy}(y)\;dy = \int_{u(x)}^{u(x+\epsilon)} f(y) \; dy.
$$

Ez a változócsere-képlet egyetlen kis téglalapra kifejezve.

Ha $u(x)$ és $f(x)$ megfelelően van megválasztva, rendkívül bonyolult integrálok számíthatók ki. Például ha $f(y) = 1$ és $u(x) = e^{-x^{2}}$ (ami azt jelenti, hogy $\frac{du}{dx}(x) = -2xe^{-x^{2}}$), akkor belátható, hogy

$$
e^{-1} - 1 = \int_{e^{-0}}^{e^{-1}} 1 \; dy = -2\int_0^{1} ye^{-y^2}\;dy,
$$

és ebből átrendezéssel

$$
\int_0^{1} ye^{-y^2}\; dy = \frac{1-e^{-1}}{2}.
$$

## Megjegyzés az előjelkonvencióról

Az éles szemű olvasók valami furcsát vehetnek észre a fenti számításokban. Mégpedig az, hogy az ehhez hasonló számítások

$$
\int_{e^{-0}}^{e^{-1}} 1 \; dy = e^{-1} -1 < 0,
$$

negatív számot adnak eredményül. Területekre gondolva különösnek tűnhet a negatív érték, ezért érdemes tisztázni, mi az elfogadott konvenció.

A matematikusok az előjeles terület fogalmát alkalmazzák. Ez két módon nyilvánul meg. Először: ha egy $f(x)$ függvény néha negatív értékeket vesz fel, a terület is negatív lesz. Például

$$
\int_0^{1} (-1)\;dx = -1.
$$

Hasonlóan, a jobbról balra haladó integrálok (azaz a fordított irányú integrálok) szintén negatív területnek számítanak:

$$
\int_0^{-1} 1\; dx = -1.
$$

A szokásos terület (balról jobbra, pozitív függvény esetén) mindig pozitív. Bármi, amit megfordítunk (például az $x$-tengelyre tükrözve negatív szám integrálját, vagy az $y$-tengelyre tükrözve fordított sorrendű integrált kapunk), negatív területet ad. Kétszeres tükrözés esetén a két negatív előjel kiejti egymást, és pozitív területet kapunk:

$$
\int_0^{-1} (-1)\;dx =  1.
$$

Ha ismerősnek tűnik ez a fejtegetés, az nem véletlen! A :numref:`sec_geometry-linear-algebraic-ops` szakaszban tárgyaltuk, hogy a determináns is pontosan ilyen módon fejezi ki az előjeles területet.

## Többszörös integrálok
Bizonyos esetekben magasabb dimenziókban kell dolgoznunk. Tegyük fel például, hogy adott egy kétváltozós $f(x, y)$ függvény, és szeretnénk meghatározni az $f$ alatti térfogatot, ahol $x$ az $[a, b]$ intervallumon, $y$ pedig a $[c, d]$ intervallumon fut.

```{.python .input}
#@tab mxnet
# Rács létrehozása és a függvény kiszámítása
x, y = np.meshgrid(np.linspace(-2, 2, 101), np.linspace(-2, 2, 101),
                   indexing='ij')
z = np.exp(- x**2 - y**2)

# A függvény ábrázolása
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy())
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

```{.python .input}
#@tab pytorch
# Rács létrehozása és a függvény kiszámítása
x, y = torch.meshgrid(torch.linspace(-2, 2, 101), torch.linspace(-2, 2, 101))
z = torch.exp(- x**2 - y**2)

# A függvény ábrázolása
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

```{.python .input}
#@tab tensorflow
# Rács létrehozása és a függvény kiszámítása
x, y = tf.meshgrid(tf.linspace(-2., 2., 101), tf.linspace(-2., 2., 101))
z = tf.exp(- x**2 - y**2)

# A függvény ábrázolása
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

Ezt az alábbiak szerint írjuk:

$$
\int_{[a, b]\times[c, d]} f(x, y)\;dx\;dy.
$$

Tegyük fel, hogy ki akarjuk számítani ezt az integrált. Állításom szerint ez elvégezhető úgy, hogy először $x$ szerint integrálunk, majd áttérünk $y$ szerinti integrálásra, azaz

$$
\int_{[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int_c^{d} \left(\int_a^{b} f(x, y) \;dx\right) \; dy.
$$

Nézzük meg, miért igaz ez!

Tekintsük a fenti ábrát, ahol a függvényt $\epsilon \times \epsilon$ méretű négyzetekre osztottuk, amelyeket $i, j$ egész koordinátákkal indexelünk. Ebben az esetben az integrálunk közelítőleg

$$
\sum_{i, j} \epsilon^{2} f(\epsilon i, \epsilon j).
$$

Miután diszkretizáltuk a feladatot, a négyzeteken lévő értékeket tetszőleges sorrendben adhatjuk össze anélkül, hogy az értékek változnának. Ezt szemlélteti :numref:`fig_sum-order`. Különösen, elmondhatjuk, hogy

$$
 \sum _ {j} \epsilon \left(\sum_{i} \epsilon f(\epsilon i, \epsilon j)\right).
$$

![Annak szemléltetése, hogy sok négyzet összegét hogyan bonthatjuk fel először az oszlopokon belüli összegre (1), majd az oszlopösszegek összegére (2).](../img/sum-order.svg)
:label:`fig_sum-order`

A belső összeg pontosan a következő integrál diszkretizálása:

$$
G(\epsilon j) = \int _a^{b} f(x, \epsilon j) \; dx.
$$

Végül, ha a két kifejezést összevonjuk:

$$
\sum _ {j} \epsilon G(\epsilon j) \approx \int _ {c}^{d} G(y) \; dy = \int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy.
$$

Így összességében

$$
\int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int _ c^{d} \left(\int _ a^{b} f(x, y) \;dx\right) \; dy.
$$

Vegyük észre, hogy a diszkretizáció után mindössze annyit tettünk, hogy megváltoztattuk a számok összeadásának sorrendjét. Bár ez jelentéktelennek tűnhet, ez az eredmény (amelyet *Fubini-tételnek* neveznek) nem mindig igaz! Az ebben a könyvben tárgyalt gépi tanulás matematikaigénye esetén (folytonos függvények) ez nem jelent gondot, azonban szerkeszthető olyan ellenpélda is, amelyre ez nem teljesül (például az $f(x, y) = xy(x^2-y^2)/(x^2+y^2)^3$ függvény a $[0,2]\times[0,1]$ téglalapon).

Megjegyezzük, hogy az a választás, hogy először $x$ szerint, majd $y$ szerint integráltunk, tetszőleges volt. Ugyanúgy elvégezhettük volna fordítva is:

$$
\int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int _ a^{b} \left(\int _ c^{d} f(x, y) \;dy\right) \; dx.
$$

Gyakran vektoros jelölésre tömörítjük, és azt mondjuk, hogy $U = [a, b]\times [c, d]$ esetén

$$
\int _ U f(\mathbf{x})\;d\mathbf{x}.
$$

## Változócsere többszörös integrálokban
Az :eqref:`eq_change_var` egyvariábilis esethez hasonlóan a magasabb dimenziójú integrálokban való változócsere is kulcsfontosságú eszköz. Foglaljuk össze az eredményt levezetés nélkül.

Szükségünk van egy olyan függvényre, amely átparametrizálja az integrálás tartományát. Legyen ez $\phi : \mathbb{R}^n \rightarrow \mathbb{R}^n$, azaz bármely olyan függvény, amely $n$ valós változót vesz fel, és $n$ valóst ad vissza. A kifejezések tisztán tartása érdekében feltesszük, hogy $\phi$ *injektív*, azaz soha nem hajtja össze önmagát ($\phi(\mathbf{x}) = \phi(\mathbf{y}) \implies \mathbf{x} = \mathbf{y}$).

Ebben az esetben elmondhatjuk, hogy

$$
\int _ {\phi(U)} f(\mathbf{x})\;d\mathbf{x} = \int _ {U} f(\phi(\mathbf{x})) \left|\det(D\phi(\mathbf{x}))\right|\;d\mathbf{x}.
$$

ahol $D\phi$ a $\phi$ *Jacobi-mátrixa*, amely $\boldsymbol{\phi} = (\phi_1(x_1, \ldots, x_n), \ldots, \phi_n(x_1, \ldots, x_n))$ parciális deriváltjainak mátrixa:

$$
D\boldsymbol{\phi} = \begin{bmatrix}
\frac{\partial \phi _ 1}{\partial x _ 1} & \cdots & \frac{\partial \phi _ 1}{\partial x _ n} \\
\vdots & \ddots & \vdots \\
\frac{\partial \phi _ n}{\partial x _ 1} & \cdots & \frac{\partial \phi _ n}{\partial x _ n}
\end{bmatrix}.
$$

Ha közelebbről megvizsgáljuk, látjuk, hogy ez hasonló az egyvariábilis láncszabályhoz :eqref:`eq_change_var`, azzal a különbséggel, hogy a $\frac{du}{dx}(x)$ tagot $\left|\det(D\phi(\mathbf{x}))\right|$-lel helyettesítettük. Nézzük meg, hogyan értelmezhetjük ezt a tagot! A $\frac{du}{dx}(x)$ tag azt fejezte ki, hogy az $u$ alkalmazásával mennyire nyújtjuk meg az $x$-tengelyt. Ugyanez a folyamat magasabb dimenzióban meghatározza, hogy egy kis négyzet (vagy kis *hiperkocka*) területét (vagy térfogatát, illetve hipertérfogatát) mennyire nyújtja meg $\boldsymbol{\phi}$ alkalmazása. Ha $\boldsymbol{\phi}$ egy mátrixszal való szorzás, a determináns már megadja a választ.

Némi munkával belátható, hogy a *Jacobi-mátrix* ugyanúgy adja a legjobb mátrixos közelítést egy többváltozós $\boldsymbol{\phi}$ függvényhez egy pontban, ahogy deriváltakkal és gradiensekkel közelíthetünk egyenessel vagy síkkal. Így a Jacobi-determináns pontosan azt a skálázási tényezőt tükrözi, amelyet az egydimenziós esetben azonosítottunk.

A részletek kitöltése némi munkát igényel, ezért ne aggódjunk, ha egyelőre nem teljesen érthetők. Nézzünk legalább egy példát, amellyel később találkozni fogunk. Tekintsük az alábbi integrált:

$$
\int _ {-\infty}^{\infty} \int _ {-\infty}^{\infty} e^{-x^{2}-y^{2}} \;dx\;dy.
$$

Ezzel az integrállal közvetlenül nem jutunk messzire, de változócserével jelentős haladást érhetünk el. Ha $\boldsymbol{\phi}(r, \theta) = (r \cos(\theta),  r\sin(\theta))$ (azaz $x = r \cos(\theta)$, $y = r \sin(\theta)$), akkor a változócsere-képlet alkalmazásával látjuk, hogy ez egyenértékű az alábbival:

$$
\int _ 0^\infty \int_0 ^ {2\pi} e^{-r^{2}} \left|\det(D\mathbf{\phi}(\mathbf{x}))\right|\;d\theta\;dr,
$$

ahol

$$
\left|\det(D\mathbf{\phi}(\mathbf{x}))\right| = \left|\det\begin{bmatrix}
\cos(\theta) & -r\sin(\theta) \\
\sin(\theta) & r\cos(\theta)
\end{bmatrix}\right| = r(\cos^{2}(\theta) + \sin^{2}(\theta)) = r.
$$

Így az integrál

$$
\int _ 0^\infty \int _ 0 ^ {2\pi} re^{-r^{2}} \;d\theta\;dr = 2\pi\int _ 0^\infty re^{-r^{2}} \;dr = \pi,
$$

ahol az utolsó egyenlőség ugyanolyan számítással következik, mint amelyet a :numref:`subsec_integral_example` szakaszban használtunk.

Ezzel az integrállal ismét találkozunk, amikor a :numref:`sec_random_variables` szakaszban a folytonos valószínűségi változókat tárgyaljuk.

## Összefoglalás

* Az integráció elmélete lehetővé teszi területekre vagy térfogatokra vonatkozó kérdések megválaszolását.
* A differenciálszámítás alaptétele lehetővé teszi, hogy a deriváltakra vonatkozó ismereteinket felhasználva területeket számítsunk ki, azon megfigyelés alapján, hogy egy adott pontig húzódó terület deriváltja az integrálandó függvény értéke.
* Magasabb dimenziójú integrálok egyvariábilis integrálok iterálásával számíthatók ki.

## Feladatok
1. Mi $\int_1^2 \frac{1}{x} \;dx$ értéke?
2. A változócsere-képlet segítségével integráljuk $\int_0^{\sqrt{\pi}}x\sin(x^2)\;dx$-et!
3. Mi $\int_{[0,1]^2} xy \;dx\;dy$ értéke?
4. A változócsere-képlet segítségével számítsuk ki $\int_0^2\int_0^1xy(x^2-y^2)/(x^2+y^2)^3\;dy\;dx$ és $\int_0^1\int_0^2f(x, y) = xy(x^2-y^2)/(x^2+y^2)^3\;dx\;dy$ értékét, és lássuk be, hogy különböznek!

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/414)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1092)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1093)
:end_tab:
