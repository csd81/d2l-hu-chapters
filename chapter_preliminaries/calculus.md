```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Analízis
:label:`sec_calculus`

Sokáig rejtély maradt, hogyan lehet kiszámítani egy kör területét.
Majd az ókori Görögországban Arkhimédész matematikus
azzal az ötlettel állt elő,
hogy egyre több csúcsú sokszögeket illeszt
a kör belsejébe
(:numref:`fig_circle_area`).
Egy $n$ csúcsú sokszögnél $n$ háromszöget kapunk.
Minden egyes háromszög magassága az $r$ sugárhoz közelít,
ahogy egyre finomabban osztjuk fel a kört.
Ugyanakkor az alapja $2 \pi r/n$-hez közelít,
mivel az ív és a szelő aránya nagy csúcsszám esetén 1-hez tart.
Így a sokszög területe
$n \cdot r \cdot \frac{1}{2} (2 \pi r/n) = \pi r^2$-hez közelít.

![Egy kör területének meghatározása határértékes eljárással.](../img/polygon-circle.svg)
:label:`fig_circle_area`

Ez a határértékes eljárás mind a *differenciálszámítás*,
mind az *integrálszámítás* alapját képezi.
Az előbbi megmutathatja, hogyan növelhetjük
vagy csökkenthetjük egy függvény értékét
az argumentumainak manipulálásával.
Ez hasznos az *optimalizálási problémáknál*,
amelyekkel a mélytanulásban találkozunk,
ahol a paramétereket ismételten frissítjük
a veszteségfüggvény csökkentése érdekében.
Az optimalizálás azt vizsgálja, hogyan illesztjük modelleinket a tanítási adatokhoz,
és a számítás ennek kulcsfontosságú előfeltétele.
Ne feledjük azonban, hogy végső célunk
a *korábban nem látott* adatokon való jó teljesítmény.
Ezt a problémát *általánosításnak* nevezik,
és a többi fejezet egyik fő témája lesz.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from matplotlib_inline import backend_inline
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

## Deriváltak és differenciálás

Egyszerűen fogalmazva, a *derivált* egy függvény változási sebessége
az argumentumainak változásához képest.
A deriváltak megmondják, milyen gyorsan növekedne vagy csökkenne a veszteségfüggvény,
ha minden paramétert egy infinitezimálisan kis mennyiséggel
*növelnénk* vagy *csökkentenénk*.
Formálisan, az $f: \mathbb{R} \rightarrow \mathbb{R}$ függvényekre,
amelyek skalárisokból skalárisokba képeznek,
[**az $f$ függvény $x$ pontbeli *deriváltját* a következőképpen definiáljuk:**]

(**$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}.$$**)
:eqlabel:`eq_derivative`

A jobb oldali kifejezést *határértéknek* nevezzük,
és megmutatja, mi történik egy kifejezés értékével,
ahogy egy adott változó egy bizonyos értékhez közelít.
Ez a határérték megmutatja, mihez konvergál
a $h$ perturbáció és az $f(x + h) - f(x)$ függvényérték-változás aránya,
ahogy a méretét nullához közelítjük.

Ha $f'(x)$ létezik, akkor $f$-et *differenciálhatónak* nevezzük az $x$ pontban;
és ha $f'(x)$ létezik minden $x$-re egy halmazon,
pl. az $[a,b]$ intervallumon,
azt mondjuk, hogy $f$ differenciálható ezen a halmazon.
Nem minden függvény differenciálható,
köztük sok olyan, amelyet optimalizálni szeretnénk,
mint például a pontosság és a vevőműködési
karakterisztika görbe alatti terület (AUC).
Mivel azonban a veszteség deriváltjának kiszámítása
szinte minden mély neurális hálózat tanítási algoritmusának kulcslépése,
gyakran egy differenciálható *helyettesítőt* optimalizálunk ehelyett.


Az $f'(x)$ deriváltat az $f(x)$ *pillanatnyi* változási sebességeként
értelmezhetjük $x$ vonatkozásában.
Fejlesszünk intuíciót egy példán keresztül.
(**Legyen $u = f(x) = 3x^2-4x$.**)

```{.python .input}
%%tab mxnet
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab pytorch
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab tensorflow
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab jax
def f(x):
    return 3 * x ** 2 - 4 * x
```

[**Ha $x=1$-et helyettesítünk, látjuk, hogy $\frac{f(x+h) - f(x)}{h}$**] (**$2$-höz közelít,
ahogy $h$ nullához tart.**)
Bár ez a kísérlet nélkülözi a matematikai bizonyítás szigorát,
gyorsan meggyőződhetünk arról, hogy valóban $f'(1) = 2$.

```{.python .input}
%%tab all
for h in 10.0**np.arange(-1, -6, -1):
    print(f'h={h:.5f}, numerical limit={(f(1+h)-f(1))/h:.5f}')
```

A deriváltak jelölésére több egyenértékű konvenció létezik.
Az $y = f(x)$ esetén a következő kifejezések egyenértékűek:

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

ahol a $\frac{d}{dx}$ és $D$ szimbólumok *differenciálási operátorok*.
Az alábbiakban bemutatjuk néhány általános függvény deriváltját:

$$\begin{aligned} \frac{d}{dx} C & = 0 && \textrm{for any constant $C$} \\ \frac{d}{dx} x^n & = n x^{n-1} && \textrm{for } n \neq 0 \\ \frac{d}{dx} e^x & = e^x \\ \frac{d}{dx} \ln x & = x^{-1}. \end{aligned}$$

A differenciálható függvényekből összetett függvények
maguk is gyakran differenciálhatók.
A következő szabályok hasznosak bármely differenciálható
$f$ és $g$ függvény, valamint $C$ konstans
kompozícióival való munkához.

$$\begin{aligned} \frac{d}{dx} [C f(x)] & = C \frac{d}{dx} f(x) && \textrm{Constant multiple rule} \\ \frac{d}{dx} [f(x) + g(x)] & = \frac{d}{dx} f(x) + \frac{d}{dx} g(x) && \textrm{Sum rule} \\ \frac{d}{dx} [f(x) g(x)] & = f(x) \frac{d}{dx} g(x) + g(x) \frac{d}{dx} f(x) && \textrm{Product rule} \\ \frac{d}{dx} \frac{f(x)}{g(x)} & = \frac{g(x) \frac{d}{dx} f(x) - f(x) \frac{d}{dx} g(x)}{g^2(x)} && \textrm{Quotient rule} \end{aligned}$$

Ezek segítségével alkalmazhatjuk a szabályokat
a $3 x^2 - 4x$ deriváltjának megtalálásához:

$$\frac{d}{dx} [3 x^2 - 4x] = 3 \frac{d}{dx} x^2 - 4 \frac{d}{dx} x = 6x - 4.$$

$x = 1$ behelyettesítése megmutatja, hogy a derivált valóban $2$ ennél a pontnál.
Megjegyezzük, hogy a deriváltak a függvény *meredekségét* mondják meg
egy adott pontban.

## Vizualizációs segédeszközök

[**A függvények meredekségét a `matplotlib` könyvtár segítségével vizualizálhatjuk**].
Néhány függvényt kell definiálnunk.
Ahogy a neve is mutatja, a `use_svg_display`
utasítja a `matplotlib`-et, hogy SVG formátumban adja ki a grafikákat
az élesebb képek érdekében.
A `#@save` megjegyzés egy speciális módosító,
amely lehetővé teszi, hogy bármely függvényt,
osztályt vagy más kódblokkot a `d2l` csomagba mentsük,
így később meg tudjuk hívni a kód megismétlése nélkül,
pl. a `d2l.use_svg_display()` segítségével.

```{.python .input}
%%tab all
def use_svg_display():  #@save
    """SVG formátumban jeleníti meg az ábrát Jupyterben."""
    backend_inline.set_matplotlib_formats('svg')
```

Kényelmesen beállíthatjuk az ábra méretét a `set_figsize` segítségével.
Mivel a `from matplotlib import pyplot as plt` importálási utasítás
`#@save` jelöléssel lett ellátva a `d2l` csomagban, meghívhatjuk a `d2l.plt`-t.

```{.python .input}
%%tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """Beállítja az ábra méretét a matplotlib számára."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

A `set_axes` függvény tulajdonságokat rendelhet a tengelyekhez,
beleértve a feliratokat, tartományokat és skálákat.

```{.python .input}
%%tab all
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Beállítja a tengelyeket a matplotlib számára."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```

Ezzel a három függvénnyel definiálhatunk egy `plot` függvényt
több görbe egymásra rajzolásához.
A kód nagy része csupán gondoskodik arról,
hogy a bemenetek méretei és alakjai megegyezzenek.

```{.python .input}
%%tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Adatpontokat ábrázol."""

    def has_one_axis(X):  # Igaz, ha X-nek (tenzor vagy lista) 1 tengelye van
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))
    
    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
        
    set_figsize(figsize)
    if axes is None:
        axes = d2l.plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

Most már [**ábrázolhatjuk az $u = f(x)$ függvényt és érintőjét $y = 2x - 3$ az $x=1$ pontban**],
ahol a $2$ együttható az érintő meredeksége.

```{.python .input}
%%tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

## Parciális deriváltak és gradiensek
:label:`subsec_calculus-grad`

Eddig csupán egyváltozós függvényeket differenciáltunk.
A mélytanulásban *sok* változós függvényekkel is kell dolgoznunk.
Röviden bemutatjuk a derivált fogalmát,
amely ilyen *többváltozós* függvényekre is alkalmazható.


Legyen $y = f(x_1, x_2, \ldots, x_n)$ egy $n$ változós függvény.
Az $y$ *parciális deriváltja*
az $i^\textrm{th}$ paramétere $x_i$ szerint:

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$


A $\frac{\partial y}{\partial x_i}$ kiszámításához
az $x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$ értékeket konstansként kezelhetjük,
és kiszámíthatjuk az $y$ deriváltját $x_i$ szerint.
A parciális deriváltak következő jelölési konvenciói
mind általánosak és mind ugyanazt jelentik:

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = \partial_{x_i} f = \partial_i f = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

Egy többváltozós függvény összes változója szerinti parciális deriváltjait
összefűzhetjük, hogy egy vektort kapjunk,
amelyet a függvény *gradiensének* nevezünk.
Tegyük fel, hogy az
$f: \mathbb{R}^n \rightarrow \mathbb{R}$
függvény bemenete egy $n$-dimenziós vektor
$\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$,
és a kimenet egy skaláris.
Az $f$ függvény $\mathbf{x}$ szerinti gradiense
$n$ parciális deriváltból álló vektor:

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \left[\partial_{x_1} f(\mathbf{x}), \partial_{x_2} f(\mathbf{x}), \ldots
\partial_{x_n} f(\mathbf{x})\right]^\top.$$ 

Ha nincs félreértés,
a $\nabla_{\mathbf{x}} f(\mathbf{x})$-t általában $\nabla f(\mathbf{x})$-szel helyettesítik.
A következő szabályok hasznosak a többváltozós függvények differenciálásánál:

* Minden $\mathbf{A} \in \mathbb{R}^{m \times n}$ esetén $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$ és $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$.
* Négyzetes $\mathbf{A} \in \mathbb{R}^{n \times n}$ mátrixokra $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ és különösen
$\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$.

Hasonlóan, bármely $\mathbf{X}$ mátrix esetén
$\nabla_{\mathbf{X}} \|\mathbf{X} \|_\textrm{F}^2 = 2\mathbf{X}$.



## Láncszabály

A mélytanulásban a fontos gradienseket gyakran nehéz kiszámítani,
mert mélyen egymásba ágyazott függvényekkel dolgozunk
(függvényeknek (függvényeknek...)).
Szerencsére a *lánc-szabály* gondoskodik erről.
Visszatérve az egyváltozós függvényekhez,
tegyük fel, hogy $y = f(g(x))$,
és az alapul szolgáló $y=f(u)$ és $u=g(x)$ függvények
mindkettő differenciálható.
A lánc-szabály szerint


$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$



Visszatérve a többváltozós függvényekhez,
tegyük fel, hogy $y = f(\mathbf{u})$ az $u_1, u_2, \ldots, u_m$ változókkal rendelkezik,
ahol minden $u_i = g_i(\mathbf{x})$
az $x_1, x_2, \ldots, x_n$ változókkal rendelkezik,
azaz $\mathbf{u} = g(\mathbf{x})$.
A lánc-szabály szerint ekkor

$$\frac{\partial y}{\partial x_{i}} = \frac{\partial y}{\partial u_{1}} \frac{\partial u_{1}}{\partial x_{i}} + \frac{\partial y}{\partial u_{2}} \frac{\partial u_{2}}{\partial x_{i}} + \ldots + \frac{\partial y}{\partial u_{m}} \frac{\partial u_{m}}{\partial x_{i}} \ \textrm{ and so } \ \nabla_{\mathbf{x}} y =  \mathbf{A} \nabla_{\mathbf{u}} y,$$

ahol $\mathbf{A} \in \mathbb{R}^{n \times m}$ egy *mátrix*,
amely tartalmazza az $\mathbf{u}$ vektor deriváltját
az $\mathbf{x}$ vektor szerint.
Így a gradiens kiértékeléséhez vektor--mátrix szorzatot kell kiszámítani.
Ez az egyik fő oka annak, hogy a lineáris algebra
olyan alapvető építőkő
a mélytanulás rendszerek felépítésében.



## Diszkusszió

Bár csupán a felszínét karcoltuk meg egy mély témának,
számos fogalom már élesedik:
először is, a differenciálás kompozíciós szabályai rutinszerűen alkalmazhatók,
lehetővé téve számunkra, hogy *automatikusan* számítsuk ki a gradienseket.
Ez a feladat nem igényel kreativitást,
ezért kognitív erőinket máshol összpontosíthatjuk.
Másodszor, vektoros értékű függvények deriváltjainak kiszámítása
mátrixok szorzatát igényli,
miközben nyomon követjük a változók függőségi gráfját
a kimenettől a bemenetig.
Különösen, ezt a gráfot *előre* irányban járjuk be
egy függvény kiértékelésekor,
és *hátrafelé* irányban, amikor gradienseket számítunk.
A későbbi fejezetek formálisan bemutatják a visszaterjesztést,
a lánc-szabály alkalmazásának számítási eljárását.

Az optimalizálás szempontjából a gradiensek lehetővé teszik számunkra,
hogy meghatározzuk, hogyan mozgassuk a modell paramétereit
a veszteség csökkentése érdekében,
és az ebben a könyvben használt optimalizálási algoritmusok minden lépése
megköveteli a gradiens kiszámítását.

## Feladatok

1. Eddig adottnak vettük a deriváltak szabályait.
   A definíció és a határértékek segítségével bizonyítsuk be a tulajdonságokat
   a következőkre: (i) $f(x) = c$, (ii) $f(x) = x^n$, (iii) $f(x) = e^x$ és (iv) $f(x) = \log x$.
1. Hasonlóképpen, bizonyítsuk be a szorzatszabályt, az összegszabályt
   és a hányadosszabályt az alapelvekből.
1. Bizonyítsuk be, hogy a konstans többszörösének szabálya
   a szorzatszabály speciális eseteként következik.
1. Számítsuk ki az $f(x) = x^x$ deriváltját.
1. Mit jelent, hogy $f'(x) = 0$ valamely $x$-re?
   Adjunk példát egy $f$ függvényre
   és egy $x$ helyre, amelyre ez teljesülhet.
1. Rajzoljuk fel az $y = f(x) = x^3 - \frac{1}{x}$ függvényt
   és érintőjét az $x = 1$ pontban.
1. Keressük meg az $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$ függvény gradiensét.
1. Mi az $f(\mathbf{x}) = \|\mathbf{x}\|_2$ függvény gradiense?
   Mi történik $\mathbf{x} = \mathbf{0}$ esetén?
1. Írjuk fel a lánc-szabályt arra az esetre,
   ahol $u = f(x, y, z)$ és $x = x(a, b)$, $y = y(a, b)$ és $z = z(a, b)$!
1. Adott egy invertálható $f(x)$ függvény;
   számítsuk ki az inverzének $f^{-1}(x)$ deriváltját.
   Tudjuk, hogy $f^{-1}(f(x)) = x$ és megfordítva $f(f^{-1}(y)) = y$.
   Tipp: használjuk ezeket a tulajdonságokat a levezetésben.

:begin_tab:`mxnet`
[Megbeszélések](https://discuss.d2l.ai/t/32)
:end_tab:

:begin_tab:`pytorch`
[Megbeszélések](https://discuss.d2l.ai/t/33)
:end_tab:

:begin_tab:`tensorflow`
[Megbeszélések](https://discuss.d2l.ai/t/197)
:end_tab:

:begin_tab:`jax`
[Megbeszélések](https://discuss.d2l.ai/t/17969)
:end_tab:
