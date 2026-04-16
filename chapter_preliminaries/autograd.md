```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Automatikus Differenciálás
:label:`sec_autograd`

Ahogyan a :numref:`sec_calculus` fejezetben láttuk,
a deriváltak kiszámítása döntő lépés
minden olyan optimalizálási algoritmusban,
amelyet mély hálózatok tanítására fogunk használni.
Bár maga a számítás elvben egyszerű,
kézzel elvégezni fáradságos és hibalehetőségekkel teli,
és ez a probléma csak nő,
ahogy modelljeink összetettebbé válnak.

Szerencsére minden modern deep learning keretrendszer
leveszi ezt a terhet a vállunkról
azáltal, hogy *automatikus differenciálást*
(gyakran *autograd*-nak rövidítve) kínál.
Miközben az adatokat átvezetjük az egymást követő függvényeken,
a keretrendszer felépít egy *számítási gráfot*,
amely nyomon követi, hogyan függ minden érték a többitől.
A deriváltak kiszámításához
az automatikus differenciálás
visszafelé halad ezen a gráfon,
alkalmazva a lánc-szabályt.
A lánc-szabálynak ezt a számítási algoritmusát
*backpropagation*-nak nevezzük.

Bár az autograd könyvtárak az elmúlt évtizedben
egyre nagyobb figyelmet kaptak,
hosszú történelemre tekintenek vissza.
A legkorábbi hivatkozások az autograd-ra
több mint fél évszázaddal ezelőttre nyúlnak vissza :cite:`Wengert.1964`.
A modern backpropagation mögötti alapötletek
egy 1980-as PhD-dolgozathoz köthetők :cite:`Speelpenning.1980`,
és a 1980-as évek végén fejlődtek tovább :cite:`Griewank.1989`.
Bár a backpropagation mára az alapértelmezett módszerré vált
a gradiensek kiszámítására, nem ez az egyetlen lehetőség.
Például a Julia programozási nyelv
előre terjedést (forward propagation) alkalmaz :cite:`Revels.Lubin.Papamarkou.2016`.
Mielőtt megvizsgálnánk a módszereket,
először sajátítsuk el az autograd csomagot.

```{.python .input}
%%tab mxnet
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from jax import numpy as jnp
```

## Egy Egyszerű Függvény

Tegyük fel, hogy szeretnénk
(**differenciálni a
$y = 2\mathbf{x}^{\top}\mathbf{x}$
függvényt a $\mathbf{x}$ oszlopvektorhoz képest.**)
Először adjunk `x`-nek egy kezdőértéket.

```{.python .input  n=1}
%%tab mxnet
x = np.arange(4.0)
x
```

```{.python .input  n=7}
%%tab pytorch
x = torch.arange(4.0)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(4, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(4.0)
x
```

:begin_tab:`mxnet, pytorch, tensorflow`
[**Mielőtt kiszámítanánk $y$ gradiensét
$\mathbf{x}$-hez képest,
szükségünk van egy helyre a tárolásához.**]
Általában kerüljük, hogy minden deriváltszámításnál
új memóriát foglaljunk le,
mivel a deep learning megköveteli,
hogy ugyanazon paraméterekhez képest
nagyszámú deriváltat számítsunk egymás után,
és fennáll a veszélye, hogy elfogy a memóriánk.
Megjegyezzük, hogy egy skaláris értékű függvény
gradiense egy $\mathbf{x}$ vektorhoz képest
vektor értékű, és ugyanolyan alakú, mint $\mathbf{x}$.
:end_tab:

```{.python .input  n=8}
%%tab mxnet
# Egy tenzor gradiensének memóriát az `attach_grad` meghívásával foglalunk le
x.attach_grad()
# Miután kiszámítottuk az `x`-hez képest vett gradienst, elérhetjük
# a `grad` attribútumon keresztül, amelynek értékei 0-val vannak inicializálva
x.grad
```

```{.python .input  n=9}
%%tab pytorch
# Alternatíva: x = torch.arange(4.0, requires_grad=True)
x.requires_grad_(True)
x.grad  # A gradiens alapértelmezés szerint None
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(x)
```

(**Most kiszámítjuk az `x`-re vonatkozó függvényünket, és az eredményt `y`-hoz rendeljük.**)

```{.python .input  n=10}
%%tab mxnet
# A kódunk egy `autograd.record` hatókörben van a számítási gráf felépítéséhez
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input  n=11}
%%tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
%%tab tensorflow
# Az összes számítást rögzítjük egy szalagra
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

```{.python .input}
%%tab jax
y = lambda x: 2 * jnp.dot(x, x)
y(x)
```

:begin_tab:`mxnet`
[**Most kiszámíthatjuk `y` gradiensét
`x`-hez képest**] a `backward` metódus meghívásával.
Ezután a gradienst a `x` `grad` attribútumán keresztül érhetjük el.
:end_tab:

:begin_tab:`pytorch`
[**Most kiszámíthatjuk `y` gradiensét
`x`-hez képest**] a `backward` metódus meghívásával.
Ezután a gradienst a `x` `grad` attribútumán keresztül érhetjük el.
:end_tab:

:begin_tab:`tensorflow`
[**Most kiszámíthatjuk `y` gradiensét
`x`-hez képest**] a `gradient` metódus meghívásával.
:end_tab:

:begin_tab:`jax`
[**Most kiszámíthatjuk `y` gradiensét
`x`-hez képest**] a `grad` transzformáción keresztül.
:end_tab:

```{.python .input}
%%tab mxnet
y.backward()
x.grad
```

```{.python .input  n=12}
%%tab pytorch
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

```{.python .input}
%%tab jax
from jax import grad
# A `grad` transzformáció egy Python függvényt ad vissza,
# amely az eredeti függvény gradiensét számítja ki
x_grad = grad(y)(x)
x_grad
```

(**Már tudjuk, hogy a $y = 2\mathbf{x}^{\top}\mathbf{x}$ függvény gradiense
$\mathbf{x}$-hez képest $4\mathbf{x}$ kell legyen.**)
Most ellenőrizhetjük, hogy az automatikus gradienszámítás
és a várt eredmény megegyezik-e.

```{.python .input  n=13}
%%tab mxnet
x.grad == 4 * x
```

```{.python .input  n=14}
%%tab pytorch
x.grad == 4 * x
```

```{.python .input}
%%tab tensorflow
x_grad == 4 * x
```

```{.python .input}
%%tab jax
x_grad == 4 * x
```

:begin_tab:`mxnet`
[**Most számítsunk ki `x`-re vonatkozó
egy másik függvényt,
és vegyük a gradiensét.**]
Megjegyezzük, hogy az MXNet alaphelyzetbe állítja a gradienspuffert,
valahányszor új gradienst rögzítünk.
:end_tab:

:begin_tab:`pytorch`
[**Most számítsunk ki `x`-re vonatkozó
egy másik függvényt,
és vegyük a gradiensét.**]
Megjegyezzük, hogy a PyTorch nem állítja vissza automatikusan
a gradienspuffert,
amikor új gradienst rögzítünk.
Ehelyett az új gradiens
hozzáadódik a már tárolt gradienshez.
Ez a viselkedés akkor hasznos,
ha több célfüggvény összegét szeretnénk optimalizálni.
A gradienspuffer visszaállításához
az `x.grad.zero_()` metódust hívhatjuk meg az alábbiak szerint:
:end_tab:

:begin_tab:`tensorflow`
[**Most számítsunk ki `x`-re vonatkozó
egy másik függvényt,
és vegyük a gradiensét.**]
Megjegyezzük, hogy a TensorFlow alaphelyzetbe állítja a gradienspuffert,
valahányszor új gradienst rögzítünk.
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Az újonnan kiszámított gradiens felülírja
```

```{.python .input  n=20}
%%tab pytorch
x.grad.zero_()  # A gradiens visszaállítása
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Az újonnan kiszámított gradiens felülírja
```

```{.python .input}
%%tab jax
y = lambda x: x.sum()
grad(y)(x)
```

## Visszaterjesztés Nem Skaláris Változókra

Ha `y` vektor,
akkor `y` deriváltjának legtermészetesebb ábrázolása
egy `x` vektorhoz képest
egy *Jacobi-mátrix* nevű mátrix,
amely tartalmazza `y` minden komponensének
`x` minden komponenséhez viszonyított parciális deriváltjait.
Hasonlóképpen, magasabb rendű `y` és `x` esetén
a differenciálás eredménye akár még magasabb rendű tenzor is lehet.

Bár a Jacobi-mátrixok megjelennek bizonyos
haladó gépi tanulási technikákban,
általában inkább össze akarjuk adni
`y` minden komponensének gradiensét
a teljes `x` vektorhoz képest,
és egy `x`-szel azonos alakú vektort kapni.
Például sokszor rendelkezünk egy vektorral,
amely a veszteségfüggvényünk értékét képviseli,
amelyet külön-külön számítunk ki
egy tanítópéldák *kötegének* (batch) minden egyes példájára.
Ebben az esetben egyszerűen (**össze akarjuk adni
az egyes példákra külön-külön kiszámított gradienseket**).

:begin_tab:`mxnet`
Az MXNet ezt a problémát úgy kezeli, hogy az összes tenzort
összegzéssel skalárissá redukálja a gradienszámítás előtt.
Más szóval a Jacobi $\partial_{\mathbf{x}} \mathbf{y}$ helyett
az összeg gradiensét adja vissza:
$\partial_{\mathbf{x}} \sum_i y_i$.
:end_tab:

:begin_tab:`pytorch`
Mivel a deep learning keretrendszerek eltérően értelmezik
a nem skaláris tenzorok gradienseit,
a PyTorch bizonyos lépéseket tesz a félreértések elkerülése érdekében.
A `backward` meghívása nem skaláris értéken hibát okoz,
hacsak nem mondjuk meg a PyTorch-nak, hogyan redukálja az objektumot skalárissá.
Pontosabban, meg kell adnunk egy $\mathbf{v}$ vektort,
hogy a `backward` a
$\mathbf{v}^\top \partial_{\mathbf{x}} \mathbf{y}$ értéket számítsa ki
a $\partial_{\mathbf{x}} \mathbf{y}$ helyett.
Ez a következő rész talán összezavaró lesz,
de a később nyilvánvalóvá váló okok miatt
ez az argumentum (amely $\mathbf{v}$-t képvisel) `gradient` nevet kapott.
Részletesebb leírásért lásd Yang Zhang
[Medium bejegyzését](https://zhang-yang.medium.com/the-gradient-argument-in-pytorchs-backward-function-explained-by-examples-68f266950c29).
:end_tab:

:begin_tab:`tensorflow`
Alapértelmezés szerint a TensorFlow az összeg gradiensét adja vissza.
Más szóval a Jacobi $\partial_{\mathbf{x}} \mathbf{y}$ helyett
az összeg gradiensét adja vissza:
$\partial_{\mathbf{x}} \sum_i y_i$.
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x  
y.backward()
x.grad  # Egyenlő az y = sum(x * x) gradiensével
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Gyorsabb: y.sum().backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Ugyanaz, mint y = tf.reduce_sum(x * x)
```

```{.python .input}
%%tab jax
y = lambda x: x * x
# a grad csak skaláris kimenetű függvényekre van értelmezve
grad(lambda x: y(x).sum())(x)
```

## A Számítás Leválasztása

Néha szeretnénk [**bizonyos számításokat
kiemelni a rögzített számítási gráfból.**]
Tegyük fel például, hogy a bemenetet felhasználjuk
néhány segéd közbenső kifejezés létrehozásához,
amelyekre nem kívánunk gradienst számítani.
Ebben az esetben *le kell választanunk*
a megfelelő számítási gráfot
a végeredménytől.
A következő egyszerű példa ezt teszi érthetőbbé:
tegyük fel, hogy `z = x * y` és `y = x * x`,
de csak az `x` *közvetlen* hatására vagyunk kíváncsiak `z`-re,
nem az `y`-on keresztül közvetített hatásra.
Ebben az esetben létrehozhatunk egy új `u` változót,
amelynek értéke megegyezik `y`-éval,
de amelynek *eredete* (hogyan jött létre)
törlésre kerül.
Így `u`-nak nincs őse a gráfban,
és a gradiensek nem folynak `u`-n keresztül `x`-hez.
Például a `z = x * u` gradiensének felvétele
`u` eredményt ad,
(nem `3 * x * x`-et, ahogyan vártuk volna,
hiszen `z = x * x * x`).

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input  n=21}
%%tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
%%tab tensorflow
# persistent=True beállítása megőrzi a számítási gráfot.
# Így a t.gradient-t egynél többször is futtathatjuk
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

```{.python .input}
%%tab jax
import jax

y = lambda x: x * x
# a jax.lax primitívek XLA műveletek köré épített Python burkolók
u = jax.lax.stop_gradient(y(x))
z = lambda x: u * x

grad(lambda x: z(x).sum())(x) == y(x)
```

Megjegyezzük, hogy bár ez az eljárás leválasztja
`y` őseit a `z`-hez vezető gráfból,
a `y`-hoz vezető számítási gráf megmarad,
így kiszámíthatjuk `y` gradiensét `x`-hez képest.

```{.python .input}
%%tab mxnet
y.backward()
x.grad == 2 * x
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
%%tab tensorflow
t.gradient(y, x) == 2 * x
```

```{.python .input}
%%tab jax
grad(lambda x: y(x).sum())(x) == 2 * x
```

## Gradiensek és a Python Vezérlési Folyam

Eddig olyan eseteket vizsgáltunk, ahol a bemenettől a kimenetig vezető út
jól meghatározott volt egy `z = x * x * x` jellegű függvényen keresztül.
A programozás sokkal nagyobb szabadságot ad abban, hogyan számítjuk ki az eredményeket.
Például függővé tehetjük azokat segédváltozóktól,
vagy feltételektől, amelyek közbenső eredményektől függenek.
Az automatikus differenciálás egyik előnye,
hogy [**még ha**] a számítási gráf felépítéséhez
(**egy Python vezérlési folyam labirintusán kellett is áthaladni**)
(pl. feltételek, ciklusok és tetszőleges függvényhívások),
(**az eredményváltozó gradiensét még mindig ki tudjuk számítani.**)
Ennek szemléltetéséhez tekintsük a következő kódrészletet, ahol
a `while` ciklus iterációinak száma
és az `if` utasítás kiértékelése
egyaránt az `a` bemeneti értékétől függ.

```{.python .input}
%%tab mxnet
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab jax
def f(a):
    b = a * 2
    while jnp.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

Az alábbiakban meghívjuk ezt a függvényt, egy véletlenszerű értéket adva be bemenetként.
Mivel a bemenet véletlenszerű változó,
nem tudjuk előre, milyen alakot ölt majd
a számítási gráf.
Mindazonáltal valahányszor egy adott bemeneten végrehajtjuk az `f(a)` hívást,
egy konkrét számítási gráfot realizálunk,
és ezt követően lefuttathatjuk a `backward` műveletet.

```{.python .input}
%%tab mxnet
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
%%tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
%%tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

```{.python .input}
%%tab jax
from jax import random
a = random.normal(random.PRNGKey(1), ())
d = f(a)
d_grad = grad(f)(a)
```

Bár az `f` függvényünk szemléltetési céllal egy kissé mesterséges,
a bemenettől való függése meglehetősen egyszerű:
az `a` *lineáris* függvénye
darabonként meghatározott skálával.
Ennélfogva az `f(a) / a` konstans bejegyzésekből álló vektor,
és ráadásul az `f(a) / a`-nak meg kell egyeznie
az `f(a)` gradiensével `a`-hoz képest.

```{.python .input}
%%tab mxnet
a.grad == d / a
```

```{.python .input}
%%tab pytorch
a.grad == d / a
```

```{.python .input}
%%tab tensorflow
d_grad == d / a
```

```{.python .input}
%%tab jax
d_grad == d / a
```

A dinamikus vezérlési folyam nagyon elterjedt a deep learning-ben.
Például szövegfeldolgozáskor a számítási gráf
a bemenet hosszától függ.
Ilyen esetekben az automatikus differenciálás
nélkülözhetetlenné válik a statisztikai modellezésben,
mivel a gradiens *a priori* kiszámítása lehetetlen.

## Megbeszélés

Most már kaptál egy ízlelítőt az automatikus differenciálás erejéből.
A deriváltak automatikus és hatékony kiszámítására szolgáló könyvtárak fejlesztése
óriási termelékenységnövekedést hozott
a deep learning szakemberei számára,
felszabadítva őket, hogy kevésbé unalmas feladatokra összpontosítsanak.
Ezenkívül az autograd lehetővé teszi, hogy hatalmas modelleket tervezzünk,
amelyeknél a kézzel történő gradienszámítás elfogadhatatlanul időigényes lenne.
Érdekesség, hogy bár az autograd-ot modellek *optimalizálására* használjuk
(statisztikai értelemben),
az autograd könyvtárak *optimalizálása* maguk
(számítástechnikai értelemben)
gazdag témakör,
amely létfontosságú a keretrendszer-tervezők számára.
Itt fordítóktól és gráfmanipulációtól vett eszközöket alkalmaznak,
hogy az eredményeket a leggyorsabb és legmemória-hatékonyabb módon számítsák ki.

Egyelőre próbáld megjegyezni ezeket az alapokat: (i) csatold a gradienseket azokhoz a változókhoz, amelyek szerint deriváltat szeretnél; (ii) rögzítsd a célérték számítását; (iii) hajtsd végre a backpropagation függvényt; és (iv) érd el az eredményt tartalmazó gradienst.


## Feladatok

1. Miért sokkal drágább a második derivált kiszámítása, mint az elsőé?
1. A backpropagation függvény lefuttatása után futtasd le azonnal újra, és figyeld meg, mi történik. Vizsgáld meg!
1. A vezérlési folyam példájában, ahol `d` deriváltját számítjuk `a`-hoz képest, mi történne, ha az `a` változót véletlenszerű vektorrá vagy mátrixszá változtatnád? Ekkor az `f(a)` számítás eredménye már nem skaláris. Mi történik az eredménnyel? Hogyan elemzed ezt?
1. Legyen $f(x) = \sin(x)$. Rajzold fel $f$ és $f'$ deriváltjának grafikonját. Ne használd ki, hogy $f'(x) = \cos(x)$, hanem automatikus differenciálással kapd meg az eredményt.
1. Legyen $f(x) = ((\log x^2) \cdot \sin x) + x^{-1}$. Írj fel egy függőségi gráfot, amely $x$-től $f(x)$-ig nyomkövet az eredményeken.
1. A lánc-szabály segítségével számítsd ki a fent említett függvény $\frac{df}{dx}$ deriváltját, minden tagot elhelyezve a korábban felépített függőségi gráfon.
1. A gráf és a közbenső derivált eredmények alapján számos lehetőséged van a gradiens kiszámítására. Értékeld az eredményt egyszer $x$-től $f$-ig haladva, majd egyszer $f$-től visszafelé $x$-ig. Az $x$-től $f$-ig vezető utat általában *előre irányú differenciálás*-nak, az $f$-től $x$-ig vezető utat pedig visszafelé irányú differenciálás-nak nevezzük.
1. Mikor érdemes előre irányú, és mikor visszafelé irányú differenciálást használni? Tipp: vedd figyelembe a szükséges közbenső adatok mennyiségét, a lépések párhuzamosíthatóságát, valamint az érintett mátrixok és vektorok méretét.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/200)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17970)
:end_tab: