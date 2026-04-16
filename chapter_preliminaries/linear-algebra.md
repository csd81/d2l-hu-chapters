```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Lineáris Algebra
:label:`sec_linear-algebra`

Eddig már tudunk adathalmazokat tenzorokba tölteni
és ezeket a tenzorokat
alapvető matematikai műveletekkel manipulálni.
A kifinomult modellek építéséhez
néhány lineáris algebrai eszközre is szükségünk lesz.
Ez a fejezet szelíd bevezetést nyújt
a leglényegesebb fogalmakba,
a skaláris aritmetikától kezdve
egészen a mátrixszorzásig.

```{.python .input}
%%tab mxnet
from mxnet import np, npx
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

## Skalárok


A mindennapos matematika nagy részét
egyenként végzett számmanipuláció alkotja.
Ezeket az értékeket formálisan *skalároknak* nevezzük.
Például Palo Alto hőmérséklete
kellemes $72$ Fahrenheit-fok.
Ha a hőmérsékletet Celsiusra szeretnénk átváltani,
kiértékeljük a $c = \frac{5}{9}(f - 32)$ kifejezést,
ahol $f$ értéke $72$.
Ebben az egyenletben az
$5$, $9$ és $32$ értékek állandó skalárok.
A $c$ és $f$ változók
általában ismeretlen skalárokat jelölnek.

A skalárokat
közönséges kisbetűkkel jelöljük
(pl. $x$, $y$ és $z$),
az összes (folytonos)
*valós értékű* skalárok terét pedig $\mathbb{R}$-rel.
A tömörség kedvéért kihagyjuk
a *terek* szigorú definícióját:
csupán jegyezzük meg, hogy az $x \in \mathbb{R}$ kifejezés
annak formális módja, hogy $x$ valós értékű skalár.
A $\in$ szimbólum (ejtsd: „eleme")
halmazhoz való tartozást jelöl.
Például $x, y \in \{0, 1\}$
azt jelzi, hogy $x$ és $y$ olyan változók,
amelyek csak $0$ vagy $1$ értéket vehetnek fel.

(**A skalárokat egyetlen elemet tartalmazó tenzorokként implementálják.**)
Az alábbiakban két skalárist rendelünk értékhez,
majd elvégezzük a megszokott összeadás, szorzás,
osztás és hatványozás műveleteket.

```{.python .input}
%%tab mxnet
x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab tensorflow
x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab jax
x = jnp.array(3.0)
y = jnp.array(2.0)

x + y, x * y, x / y, x**y
```

## Vektorok

Jelenlegi céljainkra [**egy vektort skalárok rögzített hosszúságú tömbjeként képzelhetünk el.**]
A kódbeli megfelelőjükhöz hasonlóan
a vektor *elemeit* nevezzük skalároknak
(szinonimák: *bejegyzések* és *komponensek*).
Amikor a vektorok valós adathalmazokból származó példákat képviselnek,
értékeiknek valóságos jelentésük van.
Például ha egy modellt tanítanánk arra, hogy megjósolja
egy hitel nemteljesítési kockázatát,
minden kérelmezőhöz hozzárendelhetnénk egy vektort,
amelynek komponensei olyan mennyiségeknek felelnek meg,
mint a jövedelmük, a foglalkoztatásuk hossza
vagy a korábbi nemteljesítések száma.
Ha szívroham kockázatát tanulmányoznánk,
minden vektor egy beteget képviselhetne,
komponensei pedig megfelelhetnek
a legutóbbi létfontosságú jeleknek, koleszterinszinteknek,
napi edzési perceknek stb.
A vektorokat félkövér kisbetűkkel jelöljük
(pl. $\mathbf{x}$, $\mathbf{y}$ és $\mathbf{z}$).

A vektorokat 1. rendű tenzorokként implementálják.
Általánosan, az ilyen tenzorok tetszőleges hosszúságúak lehetnek,
a memória korlátaitól függően. Figyelem: Pythonban, csakúgy mint a legtöbb programozási nyelvben, a vektorindexek $0$-tól kezdődnek, ezt *nulla alapú indexelésnek* is nevezzük, míg a lineáris algebrában az alindexek $1$-től kezdődnek (egy alapú indexelés).

```{.python .input}
%%tab mxnet
x = np.arange(3)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(3)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(3)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(3)
x
```

Egy vektor elemére alsó indexszel hivatkozhatunk.
Például $x_2$ a $\mathbf{x}$ második elemét jelöli.
Mivel $x_2$ skaláris, nem félkövérítjük.
Alapértelmezés szerint a vektorokat
elemeik függőleges felsorolásával szemléltetjük:

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\ \vdots  \\x_{n}\end{bmatrix}.$$
:eqlabel:`eq_vec_def`

Itt $x_1, \ldots, x_n$ a vektor elemei.
Később különbséget teszünk az ilyen *oszlopvektorok*
és a *sorvektorok* között, amelyek elemei vízszintesen vannak felsorolva.
Emlékezzünk: [**egy tenzor elemeit indexeléssel érjük el.**]

```{.python .input}
%%tab all
x[2]
```

Annak jelzésére, hogy egy vektor $n$ elemet tartalmaz,
a $\mathbf{x} \in \mathbb{R}^n$ jelölést használjuk.
Formálisan az $n$ értéket a vektor *dimenziószámának* nevezzük.
[**Kódban ez a tenzor hosszának felel meg**],
amely a Python beépített `len` függvényével érhető el.

```{.python .input}
%%tab all
len(x)
```

A hosszt a `shape` attribútumon keresztül is elérhetjük.
Az alak egy sor, amely megadja a tenzor hosszát minden tengely mentén.
(**Az egyetlen tengellyel rendelkező tenzorokban az alak egyetlen elemet tartalmaz.**)

```{.python .input}
%%tab all
x.shape
```

Sokszor a „dimenzió" szót túlterhelten használják,
hogy mind a tengelyek számát,
mind egy adott tengely mentén mért hosszt jelöljék.
E zavart elkerülendő,
a tengelyek számára a *rend* kifejezést használjuk,
a *dimenzionalitás* kizárólag a
komponensek számát jelenti.


## Mátrixok

Ahogy a skalárok 0. rendű tenzorok
és a vektorok 1. rendű tenzorok,
a mátrixok 2. rendű tenzorok.
A mátrixokat félkövér nagybetűkkel jelöljük
(pl. $\mathbf{X}$, $\mathbf{Y}$ és $\mathbf{Z}$),
és kódban két tengellyel rendelkező tenzorokként ábrázoljuk őket.
Az $\mathbf{A} \in \mathbb{R}^{m \times n}$ kifejezés
azt jelzi, hogy az $\mathbf{A}$ mátrix
$m \times n$ valós értékű skalárist tartalmaz,
amelyek $m$ sorba és $n$ oszlopba vannak rendezve.
Amikor $m = n$, azt mondjuk, hogy a mátrix *négyzetes*.
Vizuálisan bármely mátrixot táblázatként ábrázolhatunk.
Egy adott elemre való hivatkozáshoz
mind a sor-, mind az oszlopindexet alsó indexként írjuk, pl.
$a_{ij}$ az az érték, amely az $\mathbf{A}$
$i$. sorának és $j$. oszlopának metszéspontjában van:

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`


Kódban az $\mathbf{A} \in \mathbb{R}^{m \times n}$ mátrixot
egy ($m$, $n$) alakú 2. rendű tenzorral ábrázoljuk.
[**Bármely megfelelő méretű $m \times n$ tenzort
$m \times n$ mátrixszá alakíthatunk**]
a kívánt alak `reshape` függvénynek való átadásával:

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab pytorch
A = torch.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6), (3, 2))
A
```

```{.python .input}
%%tab jax
A = jnp.arange(6).reshape(3, 2)
A
```

Néha a tengelyeket fel akarjuk cserélni.
Amikor egy mátrix sorait és oszlopait felcseréljük,
az eredményt *transzponáltnak* nevezzük.
Formálisan az $\mathbf{A}$ mátrix transzponáltját
$\mathbf{A}^\top$-vel jelöljük, és ha $\mathbf{B} = \mathbf{A}^\top$,
akkor minden $i$-re és $j$-re $b_{ij} = a_{ji}$.
Így egy $m \times n$ mátrix transzponáltja
egy $n \times m$ mátrix:

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

Kódban bármely (**mátrix transzponáltját**) a következőképpen érhetjük el:

```{.python .input}
%%tab mxnet, pytorch, jax
A.T
```

```{.python .input}
%%tab tensorflow
tf.transpose(A)
```

[**A szimmetrikus mátrixok azoknak a négyzetes mátrixoknak a részhalmaza,
amelyek egyenlők a saját transzponáltjukkal:
$\mathbf{A} = \mathbf{A}^\top$.**]
A következő mátrix szimmetrikus:

```{.python .input}
%%tab mxnet
A = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab pytorch
A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab tensorflow
A = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == tf.transpose(A)
```

```{.python .input}
%%tab jax
A = jnp.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

A mátrixok hasznosak adathalmazok ábrázolásához.
Általában a sorok egyedi rekordoknak felelnek meg,
az oszlopok pedig különböző attribútumoknak.



## Tenzorok

Bár a gépi tanulás útján skalárokkal, vektorokkal és mátrixokkal
messze eljuthatunk,
végül szükségünk lehet magasabb rendű [**tenzorokkal**] dolgozni.
A tenzorok (**általános módot nyújtanak az $n$. rendű tömbök
kiterjesztéseinek leírásához.**)
A *tenzor osztály* szoftver objektumait „tenzoroknak" nevezzük,
pontosan azért, mert azok is tetszőleges számú tengellyel rendelkezhetnek.
Bár zavart okozhat a *tenzor* szó használata
mind a matematikai objektumra,
mind annak kódbeli megvalósítására,
a jelentés általában egyértelmű a szövegkörnyezetből.
Az általános tenzorokat nagybetűkkel jelöljük
speciális betűtípussal
(pl. $\mathsf{X}$, $\mathsf{Y}$ és $\mathsf{Z}$),
indexelési mechanizmusuk
(pl. $x_{ijk}$ és $[\mathsf{X}]_{1, 2i-1, 3}$)
természetes módon következik a mátrixokéból.

A tenzorok akkor válnak fontosabbá,
amikor képekkel kezdünk dolgozni.
Minden kép 3. rendű tenzorként érkezik,
amelynek tengelyei a magasságnak, szélességnek és a *csatornának* felelnek meg.
Minden térbeli helyen az egyes színek (piros, zöld és kék)
intenzitásai a csatorna mentén vannak egymásra halmozva.
Továbbá képek gyűjteményét
kódban 4. rendű tenzorral ábrázolják,
ahol az egyes képeket
az első tengely mentén indexelik.
A magasabb rendű tenzorokat, csakúgy mint a vektorokat és mátrixokat,
az alak komponenseinek növelésével állítják elő.

```{.python .input}
%%tab mxnet
np.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab pytorch
torch.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab tensorflow
tf.reshape(tf.range(24), (2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.arange(24).reshape(2, 3, 4)
```

## A tenzor aritmetika alapvető tulajdonságai

A skalárok, vektorok, mátrixok
és a magasabb rendű tenzorok
mind rendelkeznek néhány hasznos tulajdonsággal.
Például az elemenként végrehajtott műveletek
olyan kimeneteket adnak, amelyeknek alakja megegyezik
az operandusaik alakjával.

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(2, 3)
B = A.copy()  # A másolatát B-be tesszük új memória foglalásával
A, A + B
```

```{.python .input}
%%tab pytorch
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # A másolatát B-be tesszük új memória foglalásával
A, A + B
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3))
B = A  # A nem klónozódik B-be, nincs új memóriafoglalás
A, A + B
```

```{.python .input}
%%tab jax
A = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
B = A
A, A + B
```

A [**két mátrix elemenként vett szorzatát *Hadamard-szorzatnak* nevezzük**] (jelölése $\odot$).
Felírhatjuk a két
$\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$
mátrix Hadamard-szorzatának elemeit:



$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
%%tab all
A * B
```

[**Skaláris és tenzor összeadása vagy szorzása**] olyan eredményt ad,
amelynek alakja megegyezik az eredeti tenzorral.
Ilyenkor a tenzor minden egyes eleméhez hozzáadjuk (vagy megszorozzuk) a skalárist.

```{.python .input}
%%tab mxnet
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

```{.python .input}
%%tab jax
a = 2
X = jnp.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

## Redukció
:label:`subsec_lin-alg-reduction`

Sokszor ki akarjuk számítani [**egy tenzor elemeinek összegét.**]
Az $n$ hosszúságú $\mathbf{x}$ vektor elemeinek összegét
$\sum_{i=1}^n x_i$ alakban írjuk. Erre egyszerű függvény áll rendelkezésre:

```{.python .input}
%%tab mxnet
x = np.arange(3)
x, x.sum()
```

```{.python .input}
%%tab pytorch
x = torch.arange(3, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
%%tab tensorflow
x = tf.range(3, dtype=tf.float32)
x, tf.reduce_sum(x)
```

```{.python .input}
%%tab jax
x = jnp.arange(3, dtype=jnp.float32)
x, x.sum()
```

A [**tetszőleges alakú tenzorok elemeinek összeadásához**]
egyszerűen összeadjuk az összes tengelye mentén lévő elemeket.
Például egy $m \times n$ méretű $\mathbf{A}$ mátrix elemeinek összege
$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$ alakban írható.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum()
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A)
```

Alapértelmezés szerint az összeg függvény meghívása
*redukálja* a tenzort az összes tengelye mentén,
végül egy skalárist ad vissza.
A könyvtáraink lehetővé teszik számunkra, hogy [**megadjuk azokat a tengelyeket,
amelyek mentén a tenzort redukálni kell.**]
Az összes elem sormentén (0. tengely) való összegzéséhez
az `axis=0` paramétert adjuk meg a `sum` függvénynek.
Mivel a bemeneti mátrix a 0. tengely mentén redukálódik
a kimeneti vektor előállításához,
ez a tengely hiányzik a kimenet alakjából.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=0).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=0).shape
```

Az `axis=1` megadásával az oszlop dimenziót (1. tengely) redukáljuk az összes oszlop elemeinek összeadásával.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=1).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=1).shape
```

Egy mátrix sorok és oszlopok mentén való összeadással végzett redukciója
egyenértékű a mátrix összes elemének összegzésével.

```{.python .input}
%%tab mxnet, pytorch, jax
A.sum(axis=[0, 1]) == A.sum()  # Ugyanaz mint: A.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(A, axis=[0, 1]), tf.reduce_sum(A)  # Ugyanaz mint: tf.reduce_sum(A)
```

[**Egy kapcsolódó mennyiség az *átlag*, más néven *középérték*.**]
Az átlagot az összeg elosztásával számítjuk
az elemek teljes számával.
Mivel az átlag kiszámítása oly általános,
saját könyvtári függvényt kap,
amely a `sum` függvényhez hasonlóan működik.

```{.python .input}
%%tab mxnet, jax
A.mean(), A.sum() / A.size
```

```{.python .input}
%%tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

Hasonlóképpen az átlag kiszámítására szolgáló függvény
adott tengelyek mentén is redukálhat egy tenzort.

```{.python .input}
%%tab mxnet, pytorch, jax
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

## Nem-redukáló összeg
:label:`subsec_lin-alg-non-reduction`

Néha hasznos lehet [**megtartani a tengelyek számát változatlanul**]
az összeg vagy az átlag kiszámítására szolgáló függvény meghívásakor.
Ez akkor lényeges, amikor a broadcasting mechanizmust szeretnénk használni.

```{.python .input}
%%tab mxnet, pytorch, jax
sum_A = A.sum(axis=1, keepdims=True)
sum_A, sum_A.shape
```

```{.python .input}
%%tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A, sum_A.shape
```

Mivel például a `sum_A` megtartja két tengelyét minden sor összegzése után,
(**broadcasting segítségével eloszthatjuk `A`-t `sum_A`-val**)
egy olyan mátrix előállításához, ahol minden sor összege $1$.

```{.python .input}
%%tab all
A / sum_A
```

Ha ki akarjuk számítani [**az `A` elemeinek kumulatív összegét valamely tengely mentén**],
mondjuk `axis=0` (sorról sorra), meghívhatjuk a `cumsum` függvényt.
Ez a függvény tervezés szerint nem redukálja a bemeneti tenzort egyetlen tengely mentén sem.

```{.python .input}
%%tab mxnet, pytorch, jax
A.cumsum(axis=0)
```

```{.python .input}
%%tab tensorflow
tf.cumsum(A, axis=0)
```

## Skaláris szorzatok

Eddig csak elemenként végzett műveleteket, összegeket és átlagokat hajtottunk végre.
Ha ez minden, amit tehetnénk, a lineáris algebra
nem érdemelné meg a saját fejezetét.
Szerencsére itt válnak a dolgok érdekesebbé.
Az egyik legalapvetőbb művelet a skaláris szorzat.
Adott két $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ vektor esetén
*skaláris szorzatuk* $\mathbf{x}^\top \mathbf{y}$ (más néven *belső szorzat*, $\langle \mathbf{x}, \mathbf{y}  \rangle$)
az azonos pozíción lévő elemek szorzatainak összege:
$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$.

[~~A két vektor *skaláris szorzata* az azonos pozíción lévő elemek szorzatainak összege~~]

```{.python .input}
%%tab mxnet
y = np.ones(3)
x, y, np.dot(x, y)
```

```{.python .input}
%%tab pytorch
y = torch.ones(3, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
%%tab tensorflow
y = tf.ones(3, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

```{.python .input}
%%tab jax
y = jnp.ones(3, dtype = jnp.float32)
x, y, jnp.dot(x, y)
```

Egyenértékű módon (**két vektor skaláris szorzatát kiszámíthatjuk
elemenként vett szorzás, majd összegzés elvégzésével:**)

```{.python .input}
%%tab mxnet
np.sum(x * y)
```

```{.python .input}
%%tab pytorch
torch.sum(x * y)
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(x * y)
```

```{.python .input}
%%tab jax
jnp.sum(x * y)
```

A skaláris szorzatok számos összefüggésben hasznosak.
Például adott néhány érték,
amelyeket egy $\mathbf{x}  \in \mathbb{R}^n$ vektorral jelölünk,
és súlyok egy halmaza, amelyet $\mathbf{w} \in \mathbb{R}^n$-nel jelölünk,
az $\mathbf{x}$ értékeinek $\mathbf{w}$ súlyok szerinti súlyozott összege
kifejezhető a $\mathbf{x}^\top \mathbf{w}$ skaláris szorzattal.
Ha a súlyok nemnegatívak
és összegük $1$, azaz $\left(\sum_{i=1}^{n} {w_i} = 1\right)$,
a skaláris szorzat egy *súlyozott átlagot* fejez ki.
Miután két vektort egységnyi hosszúságúvá normalizálunk,
a skaláris szorzat a köztük lévő szög koszinuszát fejezi ki.
A szakasz egy későbbi pontján formálisan bevezetjük ezt a *hossz* fogalmat.


## Mátrix–vektor szorzás

Most, hogy tudjuk, hogyan számítsuk a skaláris szorzatokat,
megérthetjük egy $m \times n$ méretű $\mathbf{A}$ mátrix
és egy $n$-dimenziós $\mathbf{x}$ vektor közötti *szorzatot*.
Kezdetként a mátrixunkat
sorvektorai formájában szemléltetjük:

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

ahol minden $\mathbf{a}^\top_{i} \in \mathbb{R}^n$
egy sorvektor, amely az $\mathbf{A}$ mátrix $i$. sorát képviseli.

[**Az $\mathbf{A}\mathbf{x}$ mátrix–vektor szorzat
egyszerűen egy $m$ hosszúságú oszlopvektor,
amelynek $i$. eleme a
$\mathbf{a}^\top_i \mathbf{x}$ skaláris szorzat:**]

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

Az $\mathbf{A}\in \mathbb{R}^{m \times n}$ mátrixszal való szorzást
úgy is felfoghatjuk, mint egy transzformáció, amely vetíti a vektorokat
$\mathbb{R}^{n}$-ből $\mathbb{R}^{m}$-be.
Ezek a transzformációk rendkívül hasznosak.
Például az elforgatásokat
bizonyos négyzetes mátrixokkal való szorzásként ábrázolhatjuk.
A mátrix–vektor szorzatok a neurális hálózatok
egyes rétegeinek kimeneteit is leírják,
amelyeket az előző réteg kimenetéből számítunk.

:begin_tab:`mxnet`
A mátrix–vektor szorzat kódban való kifejezéséhez
ugyanazt a `dot` függvényt használjuk.
A műveletet az argumentumok típusa alapján következteti ki.
Vegyük figyelembe, hogy az `A` oszlopdimenziójának
(1. tengely menti hossza)
meg kell egyeznie az `x` dimenziójával (hosszával).
:end_tab:

:begin_tab:`pytorch`
A mátrix–vektor szorzat kódban való kifejezéséhez
az `mv` függvényt használjuk.
Vegyük figyelembe, hogy az `A` oszlopdimenziójának
(1. tengely menti hossza)
meg kell egyeznie az `x` dimenziójával (hosszával).
A Pythonban rendelkezésre áll a `@` kényelmi operátor,
amely mind a mátrix–vektor,
mind a mátrix–mátrix szorzatokat elvégezheti
(az argumentumaitól függően).
Így írhatjuk: `A@x`.
:end_tab:

:begin_tab:`tensorflow`
A mátrix–vektor szorzat kódban való kifejezéséhez
a `matvec` függvényt használjuk.
Vegyük figyelembe, hogy az `A` oszlopdimenziójának
(1. tengely menti hossza)
meg kell egyeznie az `x` dimenziójával (hosszával).
:end_tab:

```{.python .input}
%%tab mxnet
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
%%tab pytorch
A.shape, x.shape, torch.mv(A, x), A@x
```

```{.python .input}
%%tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

```{.python .input}
%%tab jax
A.shape, x.shape, jnp.matmul(A, x)
```

## Mátrix–mátrix szorzás

Ha már értjük a skaláris szorzatokat és a mátrix–vektor szorzatokat,
a *mátrix–mátrix szorzás* egyszerűen megérthető.

Tegyük fel, hogy adott két mátrix
$\mathbf{A} \in \mathbb{R}^{n \times k}$
és $\mathbf{B} \in \mathbb{R}^{k \times m}$:

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$


Legyen $\mathbf{a}^\top_{i} \in \mathbb{R}^k$ az a sorvektor, amely az
az $\mathbf{A}$ mátrix $i$. sorát képviseli,
és legyen $\mathbf{b}_{j} \in \mathbb{R}^k$ az a oszlopvektor, amely a
a $\mathbf{B}$ mátrix $j$. oszlopából származik:

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$


A $\mathbf{C} \in \mathbb{R}^{n \times m}$ mátrixszorzat előállításához
egyszerűen kiszámítjuk minden $c_{ij}$ elemet
az $\mathbf{A}$ $i$. sora és
a $\mathbf{B}$ $j$. oszlopa közötti skaláris szorzatként,
azaz $\mathbf{a}^\top_i \mathbf{b}_j$:

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

[**Az $\mathbf{AB}$ mátrix–mátrix szorzásra gondolhatunk úgy, mint
$m$ mátrix–vektor szorzás
vagy $m \times n$ skaláris szorzat elvégzésére,
majd az eredmények összeillesztésére
egy $n \times m$ mátrix előállításához.**]
A következő kódrészletben
mátrixszorzást hajtunk végre `A` és `B` esetén.
Az `A` egy két sorból és három oszlopból álló mátrix,
a `B` pedig egy három sorból és négy oszlopból álló mátrix.
A szorzás után két sorból és négy oszlopból álló mátrixot kapunk.

```{.python .input}
%%tab mxnet
B = np.ones(shape=(3, 4))
np.dot(A, B)
```

```{.python .input}
%%tab pytorch
B = torch.ones(3, 4)
torch.mm(A, B), A@B
```

```{.python .input}
%%tab tensorflow
B = tf.ones((3, 4), tf.float32)
tf.matmul(A, B)
```

```{.python .input}
%%tab jax
B = jnp.ones((3, 4))
jnp.matmul(A, B)
```

A *mátrix–mátrix szorzás* kifejezést
gyakran egyszerűen *mátrixszorzásra* rövidítik,
és nem szabad összekeverni a Hadamard-szorzattal.


## Normák
:label:`subsec_lin-algebra-norms`

A lineáris algebra leghasznosabb operátorai közé tartoznak a *normák*.
Informálisan egy vektor normája megmondja nekünk, milyen *nagy* az.
Például az $\ell_2$ norma méri
egy vektor (euklideszi) hosszát.
Itt a *méret* fogalmát alkalmazzuk, amely a vektor komponenseinek nagyságára vonatkozik
(nem a dimenzionalitására).

A norma egy $\| \cdot \|$ függvény, amely egy vektort
skalárisra képez le, és teljesíti az alábbi három tulajdonságot:

1. Bármely $\mathbf{x}$ vektor esetén, ha a vektort
   (összes elemét) egy $\alpha \in \mathbb{R}$ skalárissal skálázzuk, normája ennek megfelelően skálázódik:
   $$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|.$$
2. Bármely $\mathbf{x}$ és $\mathbf{y}$ vektorra:
   a normák teljesítik a háromszög-egyenlőtlenséget:
   $$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|.$$
3. Egy vektor normája nemnegatív, és csak akkor nulla, ha a vektor maga nulla:
   $$\|\mathbf{x}\| > 0 \textrm{ for all } \mathbf{x} \neq 0.$$

Számos függvény érvényes normaként, és különböző normák
különböző méretfogalmakat kódolnak.
Az euklideszi norma, amelyet mindannyian általános iskolai mértantanulmányaink során tanultunk
egy derékszögű háromszög átfogójának kiszámításakor,
egy vektor elemeinek négyzetösszegének négyzetgyöke.
Formálisan ezt [**az $\ell_2$ *normának* nevezzük**], és a következőképpen fejezzük ki:

(**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.$$**)

A `norm` metódus kiszámítja az $\ell_2$ normát.

```{.python .input}
%%tab mxnet
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
%%tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
%%tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

```{.python .input}
%%tab jax
u = jnp.array([3.0, -4.0])
jnp.linalg.norm(u)
```

[**Az $\ell_1$ norma**] szintén elterjedt,
a kapcsolódó mérték neve Manhattan-távolság.
Definíció szerint az $\ell_1$ norma összegzi
egy vektor elemeinek abszolút értékeit:

(**$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**)

Az $\ell_2$ normához képest kevésbé érzékeny a kiugró értékekre.
Az $\ell_1$ norma kiszámításához
az abszolút értéket az összeg művelettel kombináljuk.

```{.python .input}
%%tab mxnet
np.abs(u).sum()
```

```{.python .input}
%%tab pytorch
torch.abs(u).sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(tf.abs(u))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(u, ord=1) # ugyanaz mint: jnp.abs(u).sum()
```

Az $\ell_2$ és $\ell_1$ normák az általánosabb $\ell_p$ *normák* speciális esetei:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

Mátrixok esetén bonyolultabb a helyzet.
A mátrixok egyszerre tekinthetők egyedi bejegyzések gyűjteményeként
*és* vektorokon ható, azokat más vektorokká alakító objektumokként.
Például megkérdezhetjük, hogy mennyivel lehet hosszabb
a $\mathbf{X} \mathbf{v}$ mátrix–vektor szorzat $\mathbf{v}$-hez képest.
Ez a gondolat a *spektrális* norma fogalmához vezet.
Egyelőre bevezetjük [**a *Frobenius-normát*,
amelynek kiszámítása sokkal egyszerűbb**], és amelyet úgy definiálunk, mint
egy mátrix elemeinek négyzetösszegének négyzetgyökét:

[**$$\|\mathbf{X}\|_\textrm{F} = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**]

A Frobenius-norma úgy viselkedik, mintha az $\ell_2$ norma
egy mátrix alakú vektorra lenne alkalmazva.
A következő függvény hívása kiszámítja
egy mátrix Frobenius-normáját.

```{.python .input}
%%tab mxnet
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
%%tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
%%tab tensorflow
tf.norm(tf.ones((4, 9)))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(jnp.ones((4, 9)))
```

Bár nem akarunk túl messzire előreszaladni,
már most elültethetünk némi intuíciót arról, miért hasznosak ezek a fogalmak.
A mélytanulásban gyakran optimalizálási problémákat próbálunk megoldani:
*maximalizálni* a megfigyelt adatokhoz rendelt valószínűséget;
*maximalizálni* egy ajánlómodellhez kapcsolódó bevételt;
*minimalizálni* az előrejelzések és a valódi megfigyelések közötti távolságot;
*minimalizálni* ugyanazon személy fényképeinek ábrázolásai közötti távolságot,
miközben *maximalizáljuk* a különböző személyek fényképeinek
ábrázolásai közötti távolságot.
Ezek a távolságok, amelyek a mélytanulás algoritmusok
célfüggvényeit alkotják,
gyakran normák formájában fejeződnek ki.


## Összefoglalás

Ebben a fejezetben áttekintettük mindazt a lineáris algebrát,
amelyre szükség van a modern mélytanulás
egy jelentős részének megértéséhez.
Természetesen a lineáris algebrának sokkal több ágát lehetne tárgyalni,
amelyek nagy része a gépi tanulás számára is hasznos.
Például a mátrixok felbonthatók tényezőkre,
és ezek a felbontások alacsony dimenziójú struktúrákat tárhatnak fel
valós adathalmazokban.
A gépi tanulásnak vannak egész részterületei,
amelyek a mátrixfelbontásokra és
azok magas rendű tenzorokra való általánosításaira összpontosítanak
az adathalmazokban lévő struktúrák feltárásához
és előrejelzési problémák megoldásához.
Ám ez a könyv a mélytanulásra összpontosít.
Úgy véljük, hogy hajlamosabb leszel több matematikát tanulni,
ha már rátettük a kezünket és valós adathalmazokon alkalmaztuk a gépi tanulást.
Így bár fenntartjuk a jogot
a matematika bővebb bevezetésére a könyv további részeiben,
itt lezárjuk ezt a fejezetet.

Ha szívesen tanulnál több lineáris algebrát,
számos kiváló könyv és online forrás áll rendelkezésre.
Egy haladóbb összefoglalóért érdemes megnézni
:citet:`Strang.1993`, :citet:`Kolter.2008` és :citet:`Petersen.Pedersen.ea.2008` munkáit.

Összefoglalás:

* A skalárok, vektorok, mátrixok és tenzorok
  a lineáris algebrában használt alapvető matematikai objektumok,
  amelyek rendre nulla, egy, kettő, illetve tetszőleges számú tengellyel rendelkeznek.
* A tenzorok szeletelhetők vagy redukálhatók adott tengelyek mentén
  indexeléssel, illetve `sum` és `mean` típusú műveletekkel.
* Az elemenként vett szorzatokat Hadamard-szorzatnak nevezzük.
  Ezzel szemben a skaláris szorzatok, a mátrix–vektor szorzatok és a mátrix–mátrix szorzatok
  nem elemenkénti műveletek, és általában olyan objektumokat adnak vissza,
  amelyek alakja különbözik az operandusokétól.
* A Hadamard-szorzathoz képest a mátrix–mátrix szorzatok
  lényegesen több számítást igényelnek (köbös, nem négyzetes idő).
* A normák egy vektor (vagy mátrix) nagyságrendjének különböző fogalmait ragadják meg,
  és általában két vektor különbségére alkalmazzák őket a köztük lévő távolság mérésére.
* A leggyakoribb vektornormák az $\ell_1$ és $\ell_2$ normák,
  a leggyakoribb mátrixnormák pedig a *spektrális* és a *Frobenius*-norma.


## Feladatok

1. Bizonyítsuk be, hogy egy mátrix transzponáltjának transzponáltja maga a mátrix: $(\mathbf{A}^\top)^\top = \mathbf{A}$.
1. Adott két $\mathbf{A}$ és $\mathbf{B}$ mátrix esetén mutassuk meg, hogy az összeg és a transzponálás felcserélhető: $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$.
1. Bármely $\mathbf{A}$ négyzetes mátrix esetén az $\mathbf{A} + \mathbf{A}^\top$ mindig szimmetrikus-e? Bebizonyíthatjuk-e az eredményt kizárólag az előző két feladat eredményeinek felhasználásával?
1. Ebben a fejezetben definiáltuk a (2, 3, 4) alakú `X` tenzort. Mi a `len(X)` kimenete? Válaszoljunk kód implementálása nélkül, majd ellenőrizzük a választ kóddal.
1. Tetszőleges alakú `X` tenzor esetén a `len(X)` mindig az `X` valamely tengelyének hosszának felel-e meg? Melyik az a tengely?
1. Futtassuk az `A / A.sum(axis=1)` kifejezést, és vizsgáljuk meg, mi történik. El tudjuk-e elemezni az eredményeket?
1. Manhattan belvárosában két pont között utazva mekkora távolságot kell megtennünk a koordináták szerint, azaz az utak és sugárutak mentén? Lehet-e átlósan közlekedni?
1. Tekintsünk egy (2, 3, 4) alakú tenzort. Mik az összegzési kimenetek alakjai a 0., 1. és 2. tengely mentén?
1. Adjunk egy három vagy több tengellyel rendelkező tenzort a `linalg.norm` függvénynek, és figyeljük meg a kimenetét. Mit számít ez a függvény tetszőleges alakú tenzorokra?
1. Tekintsünk három nagy mátrixot: $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ és $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{14}}$, amelyeket Gauss-eloszlású véletlen változókkal inicializáltak. Ki akarjuk számítani az $\mathbf{A} \mathbf{B} \mathbf{C}$ szorzatot. Van-e különbség a memóriahasználatban és a sebességben attól függően, hogy $(\mathbf{A} \mathbf{B}) \mathbf{C}$ vagy $\mathbf{A} (\mathbf{B} \mathbf{C})$ sorrendben számítjuk? Miért?
1. Tekintsünk három nagy mátrixot: $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ és $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{16}}$. Van-e sebességbeli különbség attól függően, hogy $\mathbf{A} \mathbf{B}$-t vagy $\mathbf{A} \mathbf{C}^\top$-t számítjuk? Miért? Mi változik, ha memória klónozása nélkül inicializáljuk $\mathbf{C} = \mathbf{B}^\top$? Miért?
1. Tekintsünk három mátrixot: $\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{R}^{100 \times 200}$. Hozzunk létre egy három tengellyel rendelkező tenzort az $[\mathbf{A}, \mathbf{B}, \mathbf{C}]$ egymásra helyezésével. Mi a dimenzionalitás? A harmadik tengely második koordinátájának kiszeletelésével nyerjük vissza $\mathbf{B}$-t. Ellenőrizzük, hogy a válaszunk helyes.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/196)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17968)
:end_tab:
