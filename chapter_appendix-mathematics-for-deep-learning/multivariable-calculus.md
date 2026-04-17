# Többváltozós differenciálszámítás
:label:`sec_multivariable_calculus`

Most, hogy elég alapos ismereteink vannak az egyváltozós függvények deriváltjairól, térjünk vissza az eredeti kérdésünkhöz, ahol egy potenciálisan milliárd súlyt tartalmazó veszteségfüggvényt vizsgálunk.

## Magasabb dimenziós differenciálszámítás
A :numref:`sec_single_variable_calculus` azt mondja el nekünk, hogy ha e milliárd súly közül egyet megváltoztatunk, miközben az összes többit rögzítve tartjuk, pontosan tudjuk, mi fog történni! Ez nem más, mint egy egyváltozós függvény, így felírhatjuk:

$$L(w_1+\epsilon_1, w_2, \ldots, w_N) \approx L(w_1, w_2, \ldots, w_N) + \epsilon_1 \frac{d}{dw_1} L(w_1, w_2, \ldots, w_N).$$
:eqlabel:`eq_part_der`

Az egyetlen változó szerinti deriváltat, miközben a többi változót rögzítjük, *parciális deriváltnak* nevezzük, és az :eqref:`eq_part_der`-beli deriváltra a $\frac{\partial}{\partial w_1}$ jelölést használjuk.

Most változtassuk meg $w_2$-t is egy kicsit: legyen $w_2 + \epsilon_2$:

$$
\begin{aligned}
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N) & \approx L(w_1, w_2+\epsilon_2, \ldots, w_N) + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2+\epsilon_2, \ldots, w_N+\epsilon_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1\epsilon_2\frac{\partial}{\partial w_2}\frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N).
\end{aligned}
$$

Ismét felhasználtuk azt a tényt, hogy $\epsilon_1\epsilon_2$ magasabb rendű tag, amelyet elhagyhatunk – ugyanúgy, ahogyan az előző szakaszban elhagytuk az $\epsilon^{2}$ tagot –, valamint a :eqref:`eq_part_der` eredményét. Ezt az eljárást folytatva azt kapjuk, hogy

$$
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N+\epsilon_N) \approx L(w_1, w_2, \ldots, w_N) + \sum_i \epsilon_i \frac{\partial}{\partial w_i} L(w_1, w_2, \ldots, w_N).
$$

Ez első ránézésre bonyolultnak tűnhet, de ismerősebbé tehetjük, ha észrevesszük, hogy a jobb oldali összeg pontosan egy skaláris szorzat. Ha tehát bevezetjük a következő jelöléseket:

$$
\boldsymbol{\epsilon} = [\epsilon_1, \ldots, \epsilon_N]^\top \; \textrm{és} \;
\nabla_{\mathbf{w}} L = \left[\frac{\partial L}{\partial w_1}, \ldots, \frac{\partial L}{\partial w_N}\right]^\top,
$$

akkor

$$L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).$$
:eqlabel:`eq_nabla_use`

A $\nabla_{\mathbf{w}} L$ vektort $L$ *gradiensének* nevezzük.

Az :eqref:`eq_nabla_use` egyenlet megérdemli, hogy egy pillanatra megálljunk mellette. Pontosan ugyanolyan alakú, mint az egydimenziós eset, csak most mindent vektorokra és skaláris szorzatokra írtunk át. Segítségével megbecsülhetjük, hogy a $L$ függvény hogyan változik a bemenet tetszőleges kis perturbációjára. Amint a következő szakaszban látni fogjuk, ez fontos eszközül szolgál annak geometriai megértéséhez, hogyan tanulhatunk a gradiensben tárolt információ felhasználásával.

De előbb nézzük ezt a közelítést egy példán! Tegyük fel, hogy az alábbi függvénnyel dolgozunk:

$$
f(x, y) = \log(e^x + e^y) \textrm{ amelynek gradiense } \nabla f (x, y) = \left[\frac{e^x}{e^x+e^y}, \frac{e^y}{e^x+e^y}\right].
$$

Ha megnézzük a $(0, \log(2))$ pontot, azt kapjuk, hogy

$$
f(x, y) = \log(3) \textrm{ és a gradiens } \nabla f (x, y) = \left[\frac{1}{3}, \frac{2}{3}\right].
$$

Tehát ha $f$-et az $(\epsilon_1, \log(2) + \epsilon_2)$ pontban szeretnénk közelíteni, az :eqref:`eq_nabla_use` konkrét eseteként azt kapjuk, hogy

$$
f(\epsilon_1, \log(2) + \epsilon_2) \approx \log(3) + \frac{1}{3}\epsilon_1 + \frac{2}{3}\epsilon_2.
$$

Kóddal ellenőrizhetjük, milyen jó ez a közelítés.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mpl_toolkits import mplot3d
from mxnet import autograd, np, npx
npx.set_np()

def f(x, y):
    return np.log(np.exp(x) + np.exp(y))
def grad_f(x, y):
    return np.array([np.exp(x) / (np.exp(x) + np.exp(y)),
                     np.exp(y) / (np.exp(x) + np.exp(y))])

epsilon = np.array([0.01, -0.03])
grad_approx = f(0, np.log(2)) + epsilon.dot(grad_f(0, np.log(2)))
true_value = f(0 + epsilon[0], np.log(2) + epsilon[1])
f'közelítés: {grad_approx}, igaz érték: {true_value}'
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from mpl_toolkits import mplot3d
import torch
import numpy as np

def f(x, y):
    return torch.log(torch.exp(x) + torch.exp(y))
def grad_f(x, y):
    return torch.tensor([torch.exp(x) / (torch.exp(x) + torch.exp(y)),
                     torch.exp(y) / (torch.exp(x) + torch.exp(y))])

epsilon = torch.tensor([0.01, -0.03])
grad_approx = f(torch.tensor([0.]), torch.log(
    torch.tensor([2.]))) + epsilon.dot(
    grad_f(torch.tensor([0.]), torch.log(torch.tensor(2.))))
true_value = f(torch.tensor([0.]) + epsilon[0], torch.log(
    torch.tensor([2.])) + epsilon[1])
f'közelítés: {grad_approx}, igaz érték: {true_value}'
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from mpl_toolkits import mplot3d
import tensorflow as tf
import numpy as np

def f(x, y):
    return tf.math.log(tf.exp(x) + tf.exp(y))
def grad_f(x, y):
    return tf.constant([(tf.exp(x) / (tf.exp(x) + tf.exp(y))).numpy(),
                        (tf.exp(y) / (tf.exp(x) + tf.exp(y))).numpy()])

epsilon = tf.constant([0.01, -0.03])
grad_approx = f(tf.constant([0.]), tf.math.log(
    tf.constant([2.]))) + tf.tensordot(
    epsilon, grad_f(tf.constant([0.]), tf.math.log(tf.constant(2.))), axes=1)
true_value = f(tf.constant([0.]) + epsilon[0], tf.math.log(
    tf.constant([2.])) + epsilon[1])
f'közelítés: {grad_approx}, igaz érték: {true_value}'
```

## A gradiens és a gradienscsökkentés geometriája
Vegyük elő ismét az :eqref:`eq_nabla_use` kifejezést:

$$
L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).
$$

Tegyük fel, hogy ennek segítségével szeretnénk minimalizálni a $L$ veszteséget. Először értsük meg geometriailag a :numref:`sec_autograd`-ban elsőként leírt gradienscsökkentés algoritmust! Az eljárás a következő:

1. Véletlenszerűen választjuk meg a kezdeti $\mathbf{w}$ paramétereket.
2. Megkeressük azt a $\mathbf{v}$ irányt, amely mentén $L$ a leggyorsabban csökken $\mathbf{w}$-nél.
3. Kis lépést teszünk ebbe az irányba: $\mathbf{w} \rightarrow \mathbf{w} + \epsilon\mathbf{v}$.
4. Ismétlés.

Az egyetlen dolog, amelyet nem tudunk pontosan kiszámítani, a második lépésbeli $\mathbf{v}$ vektor. Ezt az irányt *legmeredekebb ereszkedés irányának* nevezzük. A :numref:`sec_geometry-linear-algebraic-ops`-ban megismert skaláris szorzat geometriai értelmezésének segítségével az :eqref:`eq_nabla_use` átírható:

$$
L(\mathbf{w} + \mathbf{v}) \approx L(\mathbf{w}) + \mathbf{v}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}) = L(\mathbf{w}) + \|\nabla_{\mathbf{w}} L(\mathbf{w})\|\cos(\theta).
$$

Vegyük észre, hogy az egyszerűség kedvéért az irányt egységhosszúnak választottuk, és $\theta$ jelöli $\mathbf{v}$ és $\nabla_{\mathbf{w}} L(\mathbf{w})$ közötti szöget. Ha azt szeretnénk, hogy $L$ a lehető leggyorsabban csökkenjen, ezt a kifejezést a lehető legnegatívabbá kell tennünk. A választott irány kizárólag a $\cos(\theta)$ tagon keresztül befolyásolja az egyenletet, tehát ezt a koszinuszt kell a lehető legnegatívabbá tennünk. A koszinusz alakjára emlékezve ez akkor valósul meg, ha $\cos(\theta) = -1$, vagyis ha a gradiens és a választott irány közötti szög $\pi$ radián, azaz $180$ fok. Ez egyetlen módon érhető el: $\mathbf{v}$-t pontosan a $\nabla_{\mathbf{w}} L(\mathbf{w})$-vel ellentétes irányba kell mutattatni!

Ez elvezet minket a gépi tanulás egyik legfontosabb matematikai fogalmához: a legmeredekebb ereszkedés iránya $-\nabla_{\mathbf{w}}L(\mathbf{w})$ irányába mutat. Az informális algoritmusunk tehát az alábbi alakot ölti:

1. Véletlenszerűen választjuk meg a kezdeti $\mathbf{w}$ paramétereket.
2. Kiszámítjuk $\nabla_{\mathbf{w}} L(\mathbf{w})$-t.
3. Kis lépést teszünk az ellentétes irányba: $\mathbf{w} \leftarrow \mathbf{w} - \epsilon\nabla_{\mathbf{w}} L(\mathbf{w})$.
4. Ismétlés.


Ezt az alap algoritmust sok kutató sokféleképpen módosította és alkalmazta, de az alapgondolat mindegyikükben ugyanaz marad: a gradiens segítségével megtaláljuk azt az irányt, amelyik a lehető leggyorsabban csökkenti a veszteséget, majd a paramétereket egy lépésnyit ebbe az irányba frissítjük.

## Megjegyzés a matematikai optimalizálásról
Ebben a könyvben végig numerikus optimalizálási módszerekre összpontosítunk, mivel minden mélytanulás-es függvényünk túl összetett ahhoz, hogy közvetlenül minimalizálható legyen.

Hasznos gyakorlat azonban megvizsgálni, mit árul el a fentebb szerzett geometriai megértés a függvények közvetlen optimalizálásáról.

Tegyük fel, hogy meg szeretnénk találni azt az $\mathbf{x}_0$ értéket, amely minimalizálja a $L(\mathbf{x})$ függvényt. Tegyük fel továbbá, hogy valaki megad nekünk egy értéket, és azt állítja, hogy ez minimalizálja $L$-t. Hogyan ellenőrizhetjük, hogy a válasz egyáltalán hihetőnek tűnik-e?

Vegyük elő ismét az :eqref:`eq_nabla_use` egyenletet:
$$
L(\mathbf{x}_0 + \boldsymbol{\epsilon}) \approx L(\mathbf{x}_0) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{x}} L(\mathbf{x}_0).
$$

Ha a gradiens nem nulla, tudunk lépni $-\epsilon \nabla_{\mathbf{x}} L(\mathbf{x}_0)$ irányba, és kisebb $L$-értéket kapni. Tehát ha valóban minimumban vagyunk, ez nem lehetséges! Arra juthatunk tehát, hogy ha $\mathbf{x}_0$ minimum, akkor $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$. Azokat a pontokat, amelyekre $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$, *kritikus pontoknak* nevezzük.

Ez azért hasznos, mert bizonyos ritka esetekben *kifejezetten megtalálhatjuk* az összes pontot, ahol a gradiens nulla, majd ezek közül kiválaszthatjuk a legkisebb értékűt.

Konkrét példaként tekintsük az alábbi függvényt:
$$
f(x) = 3x^4 - 4x^3 -12x^2.
$$

Ennek deriváltja:
$$
\frac{df}{dx} = 12x^3 - 12x^2 -24x = 12x(x-2)(x+1).
$$

A minimumok egyedüli lehetséges helyei $x = -1, 0, 2$, ahol a függvény értéke rendre $-5, 0, -32$, tehát a minimum $x = 2$-ben van. Ezt egy gyors ábrázolás is megerősíti.

```{.python .input}
#@tab mxnet
x = np.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

Ez rámutat egy fontos tényre, amelyet elméleti és numerikus munkánk során egyaránt érdemes szem előtt tartani: az egyetlen lehetséges pontok, ahol egy függvényt minimalizálni (vagy maximalizálni) lehet, azok, ahol a gradiens nulla; azonban nem minden nulla gradienssel rendelkező pont a valódi *globális* minimum (vagy maximum).

## Többváltozós láncszabály
Tegyük fel, hogy van egy négy változójú ($w, x, y$ és $z$) függvényünk, amelyet sok tag összetételéből kapunk:

$$\begin{aligned}f(u, v) & = (u+v)^{2} \\u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.\end{aligned}$$
:eqlabel:`eq_multi_func_def`

Az ilyen egyenletláncok gyakoriak a neurális hálózatokkal való munka során, ezért kulcsfontosságú megérteni, hogyan számítsuk ki az ilyen függvények gradienseit. Az összefüggések vizuális nyomait megtalálhatjuk :numref:`fig_chain-1`-ben, ha megnézzük, mely változók függnek közvetlenül egymástól.

![A fenti függvénykapcsolatok, ahol a csomópontok értékeket, az élek pedig funkcionális függéseket jelölnek.](../img/chain-net1.svg)
:label:`fig_chain-1`

Semmi sem akadályoz minket abban, hogy az :eqref:`eq_multi_func_def`-ből mindent összetegyünk, és felírjuk:

$$
f(w, x, y, z) = \left(\left((w+x+y+z)^2+(w+x-y-z)^2\right)^2+\left((w+x+y+z)^2-(w+x-y-z)^2\right)^2\right)^2.
$$

Ezután egyváltozós deriváltakkal elvégezhetnénk a deriválást, de hamarosan elárasztanának minket a tagok, amelyek közül sok ismétlődik! Valójában belátható, hogy például:

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = 2 \left(2 \left(2 (w + x + y + z) - 2 (w + x - y - z)\right) \left((w + x + y + z)^{2}- (w + x - y - z)^{2}\right) + \right.\\
& \left. \quad 2 \left(2 (w + x - y - z) + 2 (w + x + y + z)\right) \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)\right) \times \\
& \quad \left(\left((w + x + y + z)^{2}- (w + x - y - z)^2\right)^{2}+ \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)^{2}\right).
\end{aligned}
$$

Ha ezután $\frac{\partial f}{\partial x}$-et is ki akarnánk számítani, ismét hasonló egyenlethez jutnánk, sok ismétlődő taggal – és a két derivált között sok *közös* ismétlődő taggal. Ez hatalmas mennyiségű felesleges munkát jelent; ha így kellene deriválnunk, a mélytanulás forradalma még a kezdete előtt elakadt volna!


Bontsuk fel a problémát! Először próbáljuk megérteni, hogyan változik $f$, ha $a$-t változtatjuk, lényegében feltéve, hogy $w, x, y$ és $z$ nem léteznek. Ugyanúgy gondolkodjunk, ahogy a gradienssel való első találkozásunkkor tettük. Vegyük $a$-t, és adjunk hozzá egy kis $\epsilon$ értéket!

$$
\begin{aligned}
& f(u(a+\epsilon, b), v(a+\epsilon, b)) \\
\approx & f\left(u(a, b) + \epsilon\frac{\partial u}{\partial a}(a, b), v(a, b) + \epsilon\frac{\partial v}{\partial a}(a, b)\right) \\
\approx & f(u(a, b), v(a, b)) + \epsilon\left[\frac{\partial f}{\partial u}(u(a, b), v(a, b))\frac{\partial u}{\partial a}(a, b) + \frac{\partial f}{\partial v}(u(a, b), v(a, b))\frac{\partial v}{\partial a}(a, b)\right].
\end{aligned}
$$

Az első sor a parciális derivált definíciójából következik, a második a gradiens definíciójából. Jelölésileg nehézkes pontosan nyomon követni, hol értékeljük ki az összes deriváltat – mint a $\frac{\partial f}{\partial u}(u(a, b), v(a, b))$ kifejezésben –, ezért ezt sokkal könyebben megjegyezhető rövidítéssel szoktuk jelölni:

$$
\frac{\partial f}{\partial a} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}.
$$

Érdemes elgondolkodni a folyamat értelmén. Azt vizsgáljuk, hogyan változtatja meg az értékét a $f(u(a, b), v(a, b))$ alakú függvény $a$ megváltozásakor. Ennek két útja van: az $a \rightarrow u \rightarrow f$ és az $a \rightarrow v \rightarrow f$ ösvény. Mindkét hozzájárulást a láncszabállyal számíthatjuk ki: rendre $\frac{\partial w}{\partial u} \cdot \frac{\partial u}{\partial x}$ és $\frac{\partial w}{\partial v} \cdot \frac{\partial v}{\partial x}$, majd összeadjuk őket.

Képzeljük el, hogy van egy másféle függvényhálózatunk, amelyben a jobb oldali függvények a bal oldalról csatlakozó függvényektől függnek, ahogy :numref:`fig_chain-2` mutatja.

![A láncszabály egy másik, finomabb példája.](../img/chain-net2.svg)
:label:`fig_chain-2`

Az olyan mennyiség kiszámításához, mint $\frac{\partial f}{\partial y}$, össze kell adnunk az összes (ebben az esetben $3$) utat $y$-tól $f$-ig:

$$
\frac{\partial f}{\partial y} = \frac{\partial f}{\partial a} \frac{\partial a}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial b} \frac{\partial b}{\partial v} \frac{\partial v}{\partial y}.
$$

A láncszabály ilyen értelmezése nagy haszonnal jár, ha meg akarjuk érteni, hogyan áramlik a gradiens a hálózatokon keresztül, és miért segíthetnek bizonyos architekturális döntések – mint az LSTM-ekben (:numref:`sec_lstm`) vagy a reziduális rétegekben (:numref:`sec_resnet`) alkalmazottak – a tanítási folyamat irányításában a gradiens áramlásának szabályozásával.

## A visszaterjesztési algoritmus

Térjünk vissza az előző szakasz :eqref:`eq_multi_func_def` példájához, ahol

$$
\begin{aligned}
f(u, v) & = (u+v)^{2} \\
u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\
a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.
\end{aligned}
$$

Ha például $\frac{\partial f}{\partial w}$-t szeretnénk kiszámítani, a többváltozós láncszabályt alkalmazhatjuk:

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial w} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial w}, \\
\frac{\partial u}{\partial w} & = \frac{\partial u}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial u}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial v}{\partial w} & = \frac{\partial v}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial v}{\partial b}\frac{\partial b}{\partial w}.
\end{aligned}
$$

Próbáljuk meg ezzel a felbontással kiszámítani $\frac{\partial f}{\partial w}$-t! Vegyük észre, hogy mindössze az egylépéses parciális deriváltakra van szükségünk:

$$
\begin{aligned}
\frac{\partial f}{\partial u} = 2(u+v), & \quad\frac{\partial f}{\partial v} = 2(u+v), \\
\frac{\partial u}{\partial a} = 2(a+b), & \quad\frac{\partial u}{\partial b} = 2(a+b), \\
\frac{\partial v}{\partial a} = 2(a-b), & \quad\frac{\partial v}{\partial b} = -2(a-b), \\
\frac{\partial a}{\partial w} = 2(w+x+y+z), & \quad\frac{\partial b}{\partial w} = 2(w+x-y-z).
\end{aligned}
$$

Ha ezt kódba írjuk, elég kezelhető kifejezést kapunk.

```{.python .input}
#@tab all
# Számítsuk ki a függvény értékét a bemenetektől a kimenetig
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'    f at {w}, {x}, {y}, {z} is {f}')

# Számítsuk ki az egy lépéses parciális deriváltakat
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)

# Számítsuk ki a végső eredményt a bemenetektől a kimenetig
du_dw, dv_dw = du_da*da_dw + du_db*db_dw, dv_da*da_dw + dv_db*db_dw
df_dw = df_du*du_dw + df_dv*dv_dw
print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
```

Vegyük azonban észre, hogy ez még mindig nem teszi egyszerűvé például $\frac{\partial f}{\partial x}$ kiszámítását. Ennek oka az, *ahogyan* a láncszabályt alkalmaztuk. Ha megnézzük a fentieket, mindig $\partial w$ szerepel a nevező szerepében, amikor lehetett. Ezzel azt vizsgáltuk, hogyan változtat $w$ minden más változón. Ha ez lenne a célunk, ez helyes megközelítés lenne. Gondoljunk azonban vissza a mélytanulás motivációjára: azt szeretnénk látni, hogyan változtat minden egyes paraméter a *veszteségen*. Lényegében azt szeretnénk, ha a láncszabály alkalmazásakor $\partial f$ mindig a számlálóban szerepelne!

Pontosabban fogalmazva, jegyezzük meg, hogy felírhatjuk:

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial w} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial f}{\partial a} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}, \\
\frac{\partial f}{\partial b} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial b}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial b}.
\end{aligned}
$$

Vegyük észre, hogy a láncszabály ezen alkalmazásával kifejezetten kiszámítjuk $\frac{\partial f}{\partial u}, \frac{\partial f}{\partial v}, \frac{\partial f}{\partial a}, \frac{\partial f}{\partial b}, \; \textrm{és} \; \frac{\partial f}{\partial w}$ értékét. Semmi sem akadályoz abban, hogy belefoglaljuk az alábbi egyenleteket is:

$$
\begin{aligned}
\frac{\partial f}{\partial x} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial x} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial x}, \\
\frac{\partial f}{\partial y} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial y}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial y}, \\
\frac{\partial f}{\partial z} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial z}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial z}.
\end{aligned}
$$

és ezzel nyomon követni, hogyan változik $f$, ha a teljes hálózat *bármelyik* csomópontját módosítjuk. Valósítsuk meg!

```{.python .input}
#@tab all
# Számítsuk ki a függvény értékét a bemenetektől a kimenetig
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'f at {w}, {x}, {y}, {z} is {f}')

# A fenti felbontást használva számítsuk ki a deriváltat
# Először számítsuk ki az egy lépéses parciális deriváltakat
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)
da_dx, db_dx = 2*(w + x + y + z), 2*(w + x - y - z)
da_dy, db_dy = 2*(w + x + y + z), -2*(w + x - y - z)
da_dz, db_dz = 2*(w + x + y + z), -2*(w + x - y - z)

# Most számítsuk ki, hogyan változik f, ha bármely értéket a kimenettől a bemenet felé módosítunk
df_da, df_db = df_du*du_da + df_dv*dv_da, df_du*du_db + df_dv*dv_db
df_dw, df_dx = df_da*da_dw + df_db*db_dw, df_da*da_dx + df_db*db_dx
df_dy, df_dz = df_da*da_dy + df_db*db_dy, df_da*da_dz + df_db*db_dz

print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
print(f'df/dx at {w}, {x}, {y}, {z} is {df_dx}')
print(f'df/dy at {w}, {x}, {y}, {z} is {df_dy}')
print(f'df/dz at {w}, {x}, {y}, {z} is {df_dz}')
```

Az a tény, hogy a deriváltakat $f$-től visszafelé, a bemenetek felé számítjuk ki – nem pedig a bemenetektől előre a kimenet felé (ahogy az első kódrészletben tettük) –, adja az algoritmus nevét: *visszaterjesztés* (visszaterjesztés). Vegyük észre, hogy két lépésből áll:
1. Kiszámítjuk a függvény értékét és az egylépéses parciális deriváltakat előlről hátra. Bár fent nem így tettük, ez egyetlen *előre menetbe* kombinálható.
2. Kiszámítjuk $f$ gradiensét hátulról előre. Ezt *visszameneti lépésnek* nevezzük.

Ez pontosan az, amit minden mélytanulás algoritmus megvalósít, hogy egy menetben lehetővé tegye a veszteség gradiensének kiszámítását a hálózat minden egyes súlyára vonatkozóan. Megdöbbentő tény, hogy ilyen felbontás egyáltalán lehetséges.

Hogy lássuk, hogyan foglalható mindez egységbe, nézzük meg gyorsan ezt a példát!

```{.python .input}
#@tab mxnet
# Inicializáljuk ndarray-ként, majd csatoljuk a gradienseket
w, x, y, z = np.array(-1), np.array(0), np.array(-2), np.array(1)

w.attach_grad()
x.attach_grad()
y.attach_grad()
z.attach_grad()

# Számoljunk a megszokott módon, és kövessük a gradienseket
with autograd.record():
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# Hajtsuk végre a visszaterjesztést
f.backward()

print(f'df/dw at {w}, {x}, {y}, {z} is {w.grad}')
print(f'df/dx at {w}, {x}, {y}, {z} is {x.grad}')
print(f'df/dy at {w}, {x}, {y}, {z} is {y.grad}')
print(f'df/dz at {w}, {x}, {y}, {z} is {z.grad}')
```

```{.python .input}
#@tab pytorch
# Inicializáljuk ndarray-ként, majd csatoljuk a gradienseket
w = torch.tensor([-1.], requires_grad=True)
x = torch.tensor([0.], requires_grad=True)
y = torch.tensor([-2.], requires_grad=True)
z = torch.tensor([1.], requires_grad=True)
# Számoljunk a megszokott módon, és kövessük a gradienseket
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2

# Hajtsuk végre a visszaterjesztést
f.backward()

print(f'df/dw at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {w.grad.data.item()}')
print(f'df/dx at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {x.grad.data.item()}')
print(f'df/dy at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {y.grad.data.item()}')
print(f'df/dz at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {z.grad.data.item()}')
```

```{.python .input}
#@tab tensorflow
# Inicializáljuk ndarray-ként, majd csatoljuk a gradienseket
w = tf.Variable(tf.constant([-1.]))
x = tf.Variable(tf.constant([0.]))
y = tf.Variable(tf.constant([-2.]))
z = tf.Variable(tf.constant([1.]))
# Számoljunk a megszokott módon, és kövessük a gradienseket
with tf.GradientTape(persistent=True) as t:
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# Hajtsuk végre a visszaterjesztést
w_grad = t.gradient(f, w).numpy()
x_grad = t.gradient(f, x).numpy()
y_grad = t.gradient(f, y).numpy()
z_grad = t.gradient(f, z).numpy()

print(f'df/dw at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {w_grad}')
print(f'df/dx at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {x_grad}')
print(f'df/dy at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {y_grad}')
print(f'df/dz at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {z_grad}')
```

Mindazt, amit fent tettünk, a `f.backwards()` meghívásával automatikusan elvégezhetjük.


## Hessian-mátrix
Az egyváltozós számításhoz hasonlóan a magasabb rendű deriváltak vizsgálata itt is hasznos, hogy a gradiensen túl pontosabb közelítést kapjunk egy függvényről.

Több változós függvények magasabb rendű deriváltjaival való munka során azonnal felmerül egy probléma: nagyon sok ilyen derivált létezik. Ha van egy $f(x_1, \ldots, x_n)$ $n$ változós függvényünk, akkor $n^{2}$ darab másodfokú deriváltat vehetünk, mégpedig az $i$ és $j$ tetszőleges megválasztásával:

$$
\frac{d^2f}{dx_idx_j} = \frac{d}{dx_i}\left(\frac{d}{dx_j}f\right).
$$

Ezeket hagyományosan egy *Hessian-mátrixnak* nevezett mátrixba rendezzük:

$$\mathbf{H}_f = \begin{bmatrix} \frac{d^2f}{dx_1dx_1} & \cdots & \frac{d^2f}{dx_1dx_n} \\ \vdots & \ddots & \vdots \\ \frac{d^2f}{dx_ndx_1} & \cdots & \frac{d^2f}{dx_ndx_n} \\ \end{bmatrix}.$$
:eqlabel:`eq_hess_def`

A mátrix nem minden eleme független egymástól. Megmutatható, hogy ha mindkét *kevert parciális derivált* (egynél több változó szerinti parciális derivált) létezik és folytonos, akkor tetszőleges $i$ és $j$-re fennáll:

$$
\frac{d^2f}{dx_idx_j} = \frac{d^2f}{dx_jdx_i}.
$$

Ez belátható, ha először $x_i$ irányában, majd $x_j$ irányában perturbáljuk a függvényt, és összehasonlítjuk az eredményt azzal, amit akkor kapunk, ha fordítva, először $x_j$-t, majd $x_i$-t perturbáljuk – tudva, hogy mindkét sorrend ugyanolyan végső változást idéz elő $f$ kimenetén.

Az egyváltozós esethez hasonlóan ezeket a deriváltakat felhasználhatjuk arra, hogy sokkal jobb képet kapjunk a függvény viselkedéséről egy pont közelében. Különösen arra alkalmas, hogy megtaláljuk a legjobb közelítő másodfokú függvényt egy $\mathbf{x}_0$ pont körül, ahogy egyváltozós esetben is tettük.

Nézzünk egy példát! Legyen $f(x_1, x_2) = a + b_1x_1 + b_2x_2 + c_{11}x_1^{2} + c_{12}x_1x_2 + c_{22}x_2^{2}$. Ez a kétváltozós másodfokú függvény általános alakja. Ha megnézzük a függvény értékét, gradiensét és Hessian-mátrixát :eqref:`eq_hess_def` szerint a nullpontban:

$$
\begin{aligned}
f(0,0) & = a, \\
\nabla f (0,0) & = \begin{bmatrix}b_1 \\ b_2\end{bmatrix}, \\
\mathbf{H} f (0,0) & = \begin{bmatrix}2 c_{11} & c_{12} \\ c_{12} & 2c_{22}\end{bmatrix},
\end{aligned}
$$

visszakaphatjuk az eredeti polinomunkat:

$$
f(\mathbf{x}) = f(0) + \nabla f (0) \cdot \mathbf{x} + \frac{1}{2}\mathbf{x}^\top \mathbf{H} f (0) \mathbf{x}.
$$

Általánosan, ha ezt a kifejtést tetszőleges $\mathbf{x}_0$ pontban elvégezzük:

$$
f(\mathbf{x}) = f(\mathbf{x}_0) + \nabla f (\mathbf{x}_0) \cdot (\mathbf{x}-\mathbf{x}_0) + \frac{1}{2}(\mathbf{x}-\mathbf{x}_0)^\top \mathbf{H} f (\mathbf{x}_0) (\mathbf{x}-\mathbf{x}_0).
$$

Ez tetszőleges dimenziós bemenetre működik, és egy tetszőleges függvény legjobb közelítő másodfokú függvényét adja meg egy adott pont körül. Illusztrációképpen ábrázoljuk az alábbi függvényt:

$$
f(x, y) = xe^{-x^2-y^2}.
$$

Kiszámítható, hogy a gradiens és a Hessian:
$$
\nabla f(x, y) = e^{-x^2-y^2}\begin{pmatrix}1-2x^2 \\ -2xy\end{pmatrix} \; \textrm{és} \; \mathbf{H}f(x, y) = e^{-x^2-y^2}\begin{pmatrix} 4x^3 - 6x & 4x^2y - 2y \\ 4x^2y-2y &4xy^2-2x\end{pmatrix}.
$$

Így egy kis algebrával belátható, hogy a közelítő másodfokú függvény a $[-1,0]^\top$ pontban:

$$
f(x, y) \approx e^{-1}\left(-1 - (x+1) +(x+1)^2+y^2\right).
$$

```{.python .input}
#@tab mxnet
# Rácsot hozunk létre, és kiszámítjuk a függvényt
x, y = np.meshgrid(np.linspace(-2, 2, 101),
                   np.linspace(-2, 2, 101), indexing='ij')
z = x*np.exp(- x**2 - y**2)

# Számítsuk ki a közelítő másodfokú függvényt az (1, 0) pontban vett gradienssel és Hesse-mátrixszal
w = np.exp(-1)*(-1 - (x + 1) + (x + 1)**2 + y**2)

# Ábrázoljuk a függvényt
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), w.asnumpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab pytorch
# Rácsot hozunk létre, és kiszámítjuk a függvényt
x, y = torch.meshgrid(torch.linspace(-2, 2, 101),
                   torch.linspace(-2, 2, 101))

z = x*torch.exp(- x**2 - y**2)

# Számítsuk ki a közelítő másodfokú függvényt az (1, 0) pontban vett gradienssel és Hesse-mátrixszal
w = torch.exp(torch.tensor([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# Ábrázoljuk a függvényt
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab tensorflow
# Rácsot hozunk létre, és kiszámítjuk a függvényt
x, y = tf.meshgrid(tf.linspace(-2., 2., 101),
                   tf.linspace(-2., 2., 101))

z = x*tf.exp(- x**2 - y**2)

# Számítsuk ki a közelítő másodfokú függvényt az (1, 0) pontban vett gradienssel és Hesse-mátrixszal
w = tf.exp(tf.constant([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# Ábrázoljuk a függvényt
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

Ez képezi a :numref:`sec_gd`-ben tárgyalt Newton-algoritmus alapját, amelynek során iteratívan megkeressük a legjobb közelítő másodfokú függvényt, majd azt pontosan minimalizáljuk.

## Egy kis mátrixos számítás
Mátrixokat tartalmazó függvények deriváltjai különösen szép alakot öltenek. Ez a szakasz jelölésileg nehézkessé válhat, ezért az első olvasáskor akár ki is hagyható – ugyanakkor hasznos tudni, hogy a leggyakoribb mátrixos műveletek deriváltjai milyen sokszor sokkal letisztultabbak, mint várnánk, különösen, ha figyelembe vesszük, mennyire központi szerepet játszanak a mátrixos műveletek a mélytanulás alkalmazásokban.

Kezdjük egy példával! Tegyük fel, hogy adott egy rögzített $\boldsymbol{\beta}$ oszlopvektor, és vizsgálni szeretnénk az $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$ szorzatfüggvényt, és azt, hogyan változik a skaláris szorzat $\mathbf{x}$ megváltozásakor.

A mátrixos deriváltakkal való munkánál hasznos jelölést *nevező-elrendezésű mátrix-deriváltnak* hívják: a parciális deriváltakat abba az alakba rendezzük, amely megfelel a differenciálban szereplő vektor, mátrix vagy tenzor alakjának. Ebben az esetben így írjuk:

$$
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix},
$$

ahol igazodtunk az $\mathbf{x}$ oszlopvektor alakjához.

Ha az összetevőkre bontva írjuk fel a függvényt:

$$
f(\mathbf{x}) = \sum_{i = 1}^{n} \beta_ix_i = \beta_1x_1 + \cdots + \beta_nx_n.
$$

Ha most $\beta_1$ szerint vesszük a parciális deriváltat, minden tag nullává válik, kivéve az elsőt, amely $x_1$ szorozva $\beta_1$-gyel, így:

$$
\frac{df}{dx_1} = \beta_1,
$$

általánosan pedig:

$$
\frac{df}{dx_i} = \beta_i.
$$

Ezeket visszarendezve mátrixba:

$$
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix} = \begin{bmatrix}
\beta_1 \\
\vdots \\
\beta_n
\end{bmatrix} = \boldsymbol{\beta}.
$$

Ez rávilágít a mátrixos számítás néhány általunk ebben a szakaszban sokszor tapasztalandó sajátosságára:

* Először is: a számítások elég összetettekké válhatnak.
* Másodszor: a végeredmény sokkal letisztultabb a közbülső lépéseknél, és mindig hasonlít az egyváltozós esetre. Jelen esetben jegyezzük meg, hogy $\frac{d}{dx}(bx) = b$ és $\frac{d}{d\mathbf{x}} (\boldsymbol{\beta}^\top\mathbf{x}) = \boldsymbol{\beta}$ hasonló alakú.
* Harmadszor: transzponáltak látszólag véletlenszerűen bukkanhatnak fel. Ennek alapvető oka az a konvenció, hogy a nevező alakjához igazodunk; ezért mátrixok szorzásakor transzponálnunk kell, hogy visszakapjuk az eredeti tag alakját.

Az intuíció mélyítése érdekében próbáljunk meg egy kissé nehezebb számítást! Legyen adott egy $\mathbf{x}$ oszlopvektor és egy $A$ négyzetes mátrix, és számítsuk ki:

$$\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}).$$
:eqlabel:`eq_mat_goal_1`

A könnyebben kezelhető jelölés érdekében vizsgáljuk ezt a problémát Einstein-jelölés segítségével. Ebben az esetben a függvényt így írhatjuk:

$$
\mathbf{x}^\top A \mathbf{x} = x_ia_{ij}x_j.
$$

A derivált kiszámításához minden $k$-ra meg kell határoznunk:

$$
\frac{d}{dx_k}(\mathbf{x}^\top A \mathbf{x}) = \frac{d}{dx_k}x_ia_{ij}x_j.
$$

A szorzatszabály alapján ez:

$$
\frac{d}{dx_k}x_ia_{ij}x_j = \frac{dx_i}{dx_k}a_{ij}x_j + x_ia_{ij}\frac{dx_j}{dx_k}.
$$

Az olyan tagra, mint $\frac{dx_i}{dx_k}$, belátható, hogy értéke egy, ha $i=k$, egyébként nulla. Ez azt jelenti, hogy az összegből minden olyan tag kiesik, ahol $i$ és $k$ különbözik; az első összegben csak azok a tagok maradnak, ahol $i=k$. Ugyanez az érvelés érvényes a második tagra, ahol $j=k$ szükséges. Így:

$$
\frac{d}{dx_k}x_ia_{ij}x_j = a_{kj}x_j + x_ia_{ik}.
$$

Einstein-jelölésben az indexek nevei tetszőlegesek – az, hogy $i$ és $j$ különbözők, ezen a ponton irreleváns –, így átindexelhetünk, hogy mindkettő $i$ legyen:

$$
\frac{d}{dx_k}x_ia_{ij}x_j = a_{ki}x_i + x_ia_{ik} = (a_{ki} + a_{ik})x_i.
$$

Most már szükség van némi gyakorlatra a továbblépéshez. Próbáljuk mátrixos műveletek segítségével azonosítani ezt az eredményt! Az $a_{ki} + a_{ik}$ az $\mathbf{A} + \mathbf{A}^\top$ mátrix $k, i$-edik eleme. Így:

$$
\frac{d}{dx_k}x_ia_{ij}x_j = [\mathbf{A} + \mathbf{A}^\top]_{ki}x_i.
$$

Hasonlóképpen ez a tag az $\mathbf{A} + \mathbf{A}^\top$ mátrix és az $\mathbf{x}$ vektor szorzatának $k$-adik eleme:

$$
\left[\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x})\right]_k = \frac{d}{dx_k}x_ia_{ij}x_j = [(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}]_k.
$$

Ezzel beláttuk, hogy az :eqref:`eq_mat_goal_1`-beli kívánt derivált $k$-adik eleme megegyezik a jobb oldali vektor $k$-adik elemével, tehát a kettő egyenlő:

$$
\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}.
$$

Ez jóval több munkát igényelt, mint az előző számítás, de a végeredmény tömör. Ráadásul vegyük észre a következő hagyományos egyváltozós deriváltat:

$$
\frac{d}{dx}(xax) = \frac{dx}{dx}ax + xa\frac{dx}{dx} = (a+a)x.
$$

Egyenértékűen $\frac{d}{dx}(ax^2) = 2ax = (a+a)x$. Ismét olyan eredményt kaptunk, amely az egyváltozós esethez hasonlít, csupán egy transzponált jelenik meg benne.

Ezen a ponton a minta már gyanúsan felismerhető, érdemes megértenünk, miért. Ha ilyen mátrixos deriváltakat számítunk, először feltételezzük, hogy a kapott kifejezés szintén mátrixos kifejezés lesz: mátrixok szorzataiból és összegeiből, valamint azok transzponáltjaiból állítható össze. Ha ilyen kifejezés létezik, minden mátrixra teljesítenie kell. Így különösen az $1 \times 1$-es mátrixokra is, amelyekre a mátrixszorzás egyszerű számszorzás, a mátrixösszeadás számok összeadása, és a transzponálás mit sem változtat. Más szóval, bármely kifejezést kapjunk, annak *egyeznie kell* az egyváltozós kifejezéssel. Ez azt jelenti, hogy némi gyakorlattal sokszor megsejthetjük a mátrixos deriváltat puszta abból, hogyan kell kinéznie az egyváltozós megfelelőjének!

Próbáljuk ki! Legyen $\mathbf{X}$ egy $n \times m$-es mátrix, $\mathbf{U}$ egy $n \times r$-es és $\mathbf{V}$ egy $r \times m$-es mátrix. Számítsuk ki:

$$\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2} = \;?$$
:eqlabel:`eq_mat_goal_2`

Ez a számítás a mátrixfaktorizáció területén fontos. Számunkra azonban ez csupán egy kiszámítandó derivált. Képzeljük el, milyen lenne ez $1\times1$-es mátrixokra! Ekkor:

$$
\frac{d}{dv} (x-uv)^{2}= -2(x-uv)u,
$$

ahol a derivált meglehetősen szokványos. Ha ezt mátrixos kifejezéssé alakítjuk vissza:

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2(\mathbf{X} - \mathbf{U}\mathbf{V})\mathbf{U}.
$$

Ez azonban nem egészen helyes. Emlékezzünk, hogy $\mathbf{X}$ mérete $n \times m$, csakúgy, mint $\mathbf{U}\mathbf{V}$-é, tehát $2(\mathbf{X} - \mathbf{U}\mathbf{V})$ mérete $n \times m$. Másrészt $\mathbf{U}$ mérete $n \times r$, és egy $n \times m$-es mátrixot nem szorozhatunk egy $n \times r$-essel, mert a méretek nem egyeznek!

Azt szeretnénk, hogy $\frac{d}{d\mathbf{V}}$ ugyanolyan alakú legyen, mint $\mathbf{V}$, vagyis $r \times m$-es. Tehát valahogyan egy $n \times m$-es és egy $n \times r$-es mátrixból kell (esetleg transzponálással) $r \times m$-es mátrixot előállítani. Ezt úgy érhetjük el, ha $U^\top$-t szorozzuk $(\mathbf{X} - \mathbf{U}\mathbf{V})$-vel. Az :eqref:`eq_mat_goal_2` megoldásának tehát azt sejtjük, hogy:

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}).
$$

Hogy igazoljuk ezt, illendő részletes levezetést is adni. Ha már meggyőződtünk az ökölszabály helyességéről, nyugodtan ugorjuk át ezt a levezetést! Az alábbi mennyiség kiszámításához:

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^2,
$$

minden $a$-ra és $b$-re meg kell határoznunk:

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \frac{d}{dv_{ab}} \sum_{i, j}\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)^2.
$$

Mivel $\mathbf{X}$ és $\mathbf{U}$ összes eleme állandónak tekinthető $\frac{d}{dv_{ab}}$ szempontjából, a deriváltat bevihetjük az összeg alá, majd a négyzetgyök láncszabályát alkalmazva:

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \sum_{i, j}2\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)\left(-\sum_k u_{ik}\frac{dv_{kj}}{dv_{ab}} \right).
$$

Az előző levezetéshez hasonlóan megjegyezhetjük, hogy $\frac{dv_{kj}}{dv_{ab}}$ csak akkor nem nulla, ha $k=a$ és $j=b$. Ha e feltételek bármelyike nem teljesül, az adott tag nulla, és elhagyható. Így:

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}\left(x_{ib} - \sum_k u_{ik}v_{kb}\right)u_{ia}.
$$

Egy fontos finomság: a $k=a$ feltétel nem vonatkozik a belső összegre, mivel ott $k$ csupán egy összegzési dummy-változó. Jelölési szempontból tisztább példaként tekintsük, miért teljesül:

$$
\frac{d}{dx_1} \left(\sum_i x_i \right)^{2}= 2\left(\sum_i x_i \right).
$$

Ettől a ponttól kezdve az összeg összetevőit azonosíthatjuk. Először:

$$
\sum_k u_{ik}v_{kb} = [\mathbf{U}\mathbf{V}]_{ib}.
$$

Tehát az összeg belsejében szereplő teljes kifejezés:

$$
x_{ib} - \sum_k u_{ik}v_{kb} = [\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$

Így a derivált:

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}u_{ia}.
$$

Azt szeretnénk, ha ez egy mátrix $a, b$-edik elemeként lenne felírható – ahogy az előző példában is tettük –, ehhez az $u_{ia}$ indexeit kell felcserélnünk. Ha észrevesszük, hogy $u_{ia} = [\mathbf{U}^\top]_{ai}$, felírhatjuk:

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i} [\mathbf{U}^\top]_{ai}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$

Ez mátrixszorzat, tehát arra jutunk, hogy:

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2[\mathbf{U}^\top(\mathbf{X}-\mathbf{U}\mathbf{V})]_{ab}.
$$

Így az :eqref:`eq_mat_goal_2` megoldása:

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}).
$$

Ez egyezik a fentebb sejtett megoldással!

Joggal merülhet fel a kérdés: „Miért nem írom fel egyszerűen a mátrixos változatait az összes megtanult számítási szabálynak? Nyilvánvalóan mechanikusan is elvégezhető. Miért nem szabadítjuk meg magunkat ettől a munkától!" Valóban léteznek ilyen szabályok, és :cite:`Petersen.Pedersen.ea.2008` kiváló összefoglalót nyújt belőlük. Azonban az egyedi értékekhez képest a mátrixos műveletek kombinálásának sokkal több lehetséges módja van, ezért jóval több mátrixos derivált szabály létezik, mint egyváltozós megfelelőjük. Általában a legjobb megközelítés vagy az indexekkel való munka, vagy – ha lehetséges – az automatikus differenciálásra bízni a feladatot.

## Összefoglalás

* Magasabb dimenziókban definiálhatjuk a gradienseket, amelyek ugyanolyan szerepet töltenek be, mint az egydimenziós deriváltak. Segítségükkel meghatározható, hogyan változik egy többváltozós függvény a bemenetek tetszőleges kis módosítására.
* A visszaterjesztési algoritmus tekinthető a többváltozós láncszabály olyan szervezési módjaként, amely lehetővé teszi sok parciális derivált hatékony kiszámítását.
* A mátrixos számítás lehetővé teszi, hogy mátrixkifejezések deriváltjait tömör formában írjuk fel.

## Feladatok
1. Adott egy $\boldsymbol{\beta}$ oszlopvektor. Számítsuk ki mind az $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$, mind a $g(\mathbf{x}) = \mathbf{x}^\top\boldsymbol{\beta}$ deriváltját! Miért kapjuk ugyanazt az eredményt?
2. Legyen $\mathbf{v}$ egy $n$-dimenziós vektor! Mi $\frac{\partial}{\partial\mathbf{v}}\|\mathbf{v}\|_2$?
3. Legyen $L(x, y) = \log(e^x + e^y)$. Számítsuk ki a gradienst! Mennyi a gradiens összetevőinek összege?
4. Legyen $f(x, y) = x^2y + xy^2$. Mutassuk meg, hogy az egyetlen kritikus pont $(0,0)$! Az $f(x, x)$ vizsgálatával döntsük el, hogy $(0,0)$ maximum, minimum vagy egyik sem!
5. Tegyük fel, hogy egy $f(\mathbf{x}) = g(\mathbf{x}) + h(\mathbf{x})$ függvényt minimalizálunk. Hogyan értelmezhető geometriailag a $\nabla f = 0$ feltétel $g$ és $h$ segítségével?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/413)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1090)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1091)
:end_tab:
