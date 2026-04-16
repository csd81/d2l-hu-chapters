```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Adatmanipuláció
:label:`sec_ndarray`

Ahhoz, hogy bármit is elvégezhessünk,
szükségünk van valamilyen módra az adatok tárolására és manipulálására.
Általában két fontos dolgot
kell tennünk az adatokkal:
(i) megszereznünk őket;
és (ii) feldolgoznunk őket, miután a számítógépen vannak.
Nincs értelme adatokat megszerezni
anélkül, hogy lenne módunk tárolni őket,
ezért kezdjük azzal, hogy belemártjuk a kezünket
az $n$-dimenziós tömbökbe,
amelyeket *tenzoroknak* is nevezünk.
Ha már ismered a NumPy
tudományos számítási csomagot,
ez gyerekjáték lesz.
Minden modern deep learning keretrendszernél
a *tenzor osztály* (`ndarray` az MXNet-ben,
`Tensor` a PyTorch-ban és TensorFlow-ban)
hasonlít a NumPy `ndarray`-jére,
néhány remek funkcióval kiegészítve.
Először is, a tenzor osztály
támogatja az automatikus differenciálást.
Másodszor, GPU-kat használ
a numerikus számítások felgyorsítására,
míg a NumPy csak CPU-kon fut.
Ezek a tulajdonságok a neurális hálózatokat
egyszerűvé teszik kódolni és gyorssá futtatni.



## Első lépések

:begin_tab:`mxnet`
Kezdésként importáljuk az MXNet `np` (`numpy`) és
`npx` (`numpy_extension`) moduljait.
Az `np` modul a NumPy által támogatott
függvényeket tartalmazza,
míg az `npx` modul egy bővítménykészletet kínál,
amelyet a deep learning támogatására
fejlesztettek NumPy-szerű környezetben.
Tenzorok használatakor szinte mindig
meghívjuk a `set_np` függvényt:
ez az MXNet más összetevőivel való
tenzorfeldolgozási kompatibilitás miatt szükséges.
:end_tab:

:begin_tab:`pytorch`
(**Kezdésként importáljuk a PyTorch könyvtárat.
Megjegyezzük, hogy a csomag neve `torch`.**)
:end_tab:

:begin_tab:`tensorflow`
Kezdésként importáljuk a `tensorflow`-t.
A rövidség kedvéért a szakemberek
gyakran a `tf` aliast használják.
:end_tab:

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
import jax
from jax import numpy as jnp
```

[**A tenzor numerikus értékek (esetleg többdimenziós) tömbjét jelöli.**]
Az egydimenziós esetben, azaz ha az adatokhoz csak egy tengely szükséges,
a tenzort *vektornak* nevezzük.
Két tengellyel a tenzort *mátrixnak* hívjuk.
$k > 2$ tengellyel elhagyjuk a speciális elnevezéseket,
és az objektumot egyszerűen $k^\textrm{th}$-*rendű tenzornak* nevezzük.

:begin_tab:`mxnet`
Az MXNet számos függvényt biztosít
értékekkel előre feltöltött
új tenzorok létrehozásához.
Például az `arange(n)` meghívásával
egyenlően elosztott értékekből álló vektort hozhatunk létre,
amelynek kezdete 0 (beleértve)
és vége `n` (nem beleértve).
Alapértelmezés szerint az intervallum mérete $1$.
Hacsak másképp nincs megadva,
az új tenzorok a főmemóriában tárolódnak,
és CPU-alapú számításra vannak kijelölve.
:end_tab:

:begin_tab:`pytorch`
A PyTorch számos függvényt biztosít
értékekkel előre feltöltött
új tenzorok létrehozásához.
Például az `arange(n)` meghívásával
egyenlően elosztott értékekből álló vektort hozhatunk létre,
amelynek kezdete 0 (beleértve)
és vége `n` (nem beleértve).
Alapértelmezés szerint az intervallum mérete $1$.
Hacsak másképp nincs megadva,
az új tenzorok a főmemóriában tárolódnak,
és CPU-alapú számításra vannak kijelölve.
:end_tab:

:begin_tab:`tensorflow`
A TensorFlow számos függvényt biztosít
értékekkel előre feltöltött
új tenzorok létrehozásához.
Például a `range(n)` meghívásával
egyenlően elosztott értékekből álló vektort hozhatunk létre,
amelynek kezdete 0 (beleértve)
és vége `n` (nem beleértve).
Alapértelmezés szerint az intervallum mérete $1$.
Hacsak másképp nincs megadva,
az új tenzorok a főmemóriában tárolódnak,
és CPU-alapú számításra vannak kijelölve.
:end_tab:

```{.python .input}
%%tab mxnet
x = np.arange(12)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(12, dtype=torch.float32)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(12, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(12)
x
```

:begin_tab:`mxnet`
Ezeket az értékeket a tenzor *elemének* nevezzük.
Az `x` tenzor 12 elemet tartalmaz.
A tenzor elemeinek teljes számát
a `size` attribútumán keresztül vizsgálhatjuk meg.
:end_tab:

:begin_tab:`pytorch`
Ezeket az értékeket a tenzor *elemének* nevezzük.
Az `x` tenzor 12 elemet tartalmaz.
A tenzor elemeinek teljes számát
a `numel` metódusán keresztül vizsgálhatjuk meg.
:end_tab:

:begin_tab:`tensorflow`
Ezeket az értékeket a tenzor *elemének* nevezzük.
Az `x` tenzor 12 elemet tartalmaz.
A tenzor elemeinek teljes számát
a `size` függvényen keresztül vizsgálhatjuk meg.
:end_tab:

```{.python .input}
%%tab mxnet, jax
x.size
```

```{.python .input}
%%tab pytorch
x.numel()
```

```{.python .input}
%%tab tensorflow
tf.size(x)
```

Egy tenzor *alakját* elérhetjük
az egyes tengelyek mentén mért hosszát
a `shape` attribútum vizsgálatával.
Mivel itt vektorral van dolgunk,
az `shape` egyetlen elemet tartalmaz,
és megegyezik a mérettel.

```{.python .input}
%%tab all
x.shape
```

Egy tenzor alakját [**megváltoztathatjuk
anélkül, hogy méretét vagy értékeit módosítanánk**],
a `reshape` meghívásával.
Például az (12,) alakú `x` vektort
átalakíthatjuk (3, 4) alakú `X` mátrixszá.
Ez az új tenzor megőrzi az összes elemet,
de mátrixba rendezi őket.
Figyeljük meg, hogy a vektor elemei
soronként vannak elrendezve, ezért
`x[3] == X[0, 3]`.

```{.python .input}
%%tab mxnet, pytorch, jax
X = x.reshape(3, 4)
X
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

Megjegyezzük, hogy az összes alak-összetevő megadása
a `reshape`-nek redundáns.
Mivel már ismerjük a tenzorunk méretét,
a többi alapján ki tudjuk számítani az egyik összetevőt.
Például egy $n$ méretű tenzor esetén
és ($h$, $w$) célalak esetén
tudjuk, hogy $w = n/h$.
Ahhoz, hogy az alaknak egy összetevőjét automatikusan következtessük ki,
a `-1`-et helyezhetjük arra az alak-összetevőre,
amelyet automatikusan kell kikövetkeztetni.
Esetünkben az `x.reshape(3, 4)` hívás helyett
egyenértékűen hívhattuk volna az `x.reshape(-1, 4)` vagy `x.reshape(3, -1)` alakot.

A szakembereknek gyakran kell dolgozniuk
csupa 0-ból vagy 1-ből álló tenzorokkal.
[**Létrehozhatunk egy tenzort, amelynek minden eleme 0**] (~~vagy egy~~)
és alakja (2, 3, 4) a `zeros` függvény segítségével.

```{.python .input}
%%tab mxnet
np.zeros((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.zeros((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.zeros((2, 3, 4))
```

Hasonlóképpen, csupa 1-es tenzort hozhatunk létre
az `ones` meghívásával.

```{.python .input}
%%tab mxnet
np.ones((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.ones((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.ones((2, 3, 4))
```

Gyakran szeretnénk
[**minden elemet véletlenszerűen (és egymástól függetlenül) mintavételezni**]
egy adott valószínűségi eloszlásból.
Például a neurális hálózatok paraméterei
általában véletlenszerűen inicializálódnak.
A következő kódrészlet olyan tenzort hoz létre,
amelynek elemei egy standard Gauss- (normál-) eloszlásból
vannak kivéve, 0 várható értékkel és 1 szórással.

```{.python .input}
%%tab mxnet
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
%%tab pytorch
torch.randn(3, 4)
```

```{.python .input}
%%tab tensorflow
tf.random.normal(shape=[3, 4])
```

```{.python .input}
%%tab jax
# JAX-ban minden véletlenszám-függvény hívásakor meg kell adni egy kulcsot;
# ha ugyanazt a kulcsot adjuk meg, a függvény mindig ugyanazt a mintát
# állítja elő
jax.random.normal(jax.random.PRNGKey(0), (3, 4))
```

Végül tenzorokat úgy is létrehozhatunk, hogy
[**minden elemhez pontos értékeket adunk meg**]
numerikus literálokat tartalmazó
(esetleg egymásba ágyazott) Python lista(ák) megadásával.
Itt listák listájával hozunk létre mátrixot,
ahol a legkülső lista a 0-s tengelynek felel meg,
a belső lista pedig az 1-es tengelynek.

```{.python .input}
%%tab mxnet
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab jax
jnp.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## Indexelés és szeletelés

A Python listákhoz hasonlóan
a tenzor elemeihez
indexeléssel (0-tól kezdve) is hozzáférhetünk.
Egy elem eléréséhez a lista végéhez viszonyított
pozíciója alapján
negatív indexelést is használhatunk.
Végül egész indextartományokhoz is hozzáférhetünk
szeletelés útján (pl. `X[start:stop]`),
ahol a visszaadott érték tartalmazza
az első indexet (`start`), *de az utolsót nem* (`stop`).
Továbbá, ha egy $k^\textrm{th}$-rendű tenzorhoz
csak egy index (vagy szelet) van megadva,
azt a 0-s tengely mentén alkalmazzuk.
Így a következő kódban
[**`[-1]` az utolsó sort, `[1:3]`
a második és harmadik sort választja ki**].

```{.python .input}
%%tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
Az olvasáson túl (**a mátrix elemeit indexek megadásával *írhatjuk* is.**)
:end_tab:

:begin_tab:`tensorflow`
A TensorFlow `Tensor`-ai nem módosíthatók, és nem rendelhetők hozzájuk értékek.
A TensorFlow `Variable`-jei az állapot módosítható tárolói, amelyek támogatják
az értékadást. Ne feledjük, hogy a TensorFlow-ban a gradiensek nem áramlanak vissza
a `Variable` értékadásokon keresztül.

Az egész `Variable`-nek értéket adásán túl a `Variable` elemeit
indexek megadásával is írhatjuk.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X[1, 2] = 17
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

```{.python .input}
%%tab jax
# A JAX tömbök nem módosíthatók. A jax.numpy.ndarray.at index-
# frissítő operátorok új tömböt hoznak létre a megfelelő
# módosításokkal
X_new_1 = X.at[1, 2].set(17)
X_new_1
```

Ha [**több elemhez ugyanazt az értéket akarjuk rendelni,
az indexelést az értékadási művelet bal oldalán
alkalmazzuk.**]
Például `[:2, :]` az első és második sorhoz fér hozzá,
ahol `:` az összes elemet veszi az 1-es tengely (oszlop) mentén.
Bár az indexelést mátrixokra tárgyaltuk,
ez vektorokra és kettőnél több dimenziójú
tenzorokra is működik.

```{.python .input}
%%tab mxnet, pytorch
X[:2, :] = 12
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[:2, :].assign(tf.ones(X_var[:2,:].shape, dtype=tf.float32) * 12)
X_var
```

```{.python .input}
%%tab jax
X_new_2 = X_new_1.at[:2, :].set(12)
X_new_2
```

## Műveletek

Most, hogy tudjuk, hogyan kell tenzorokat létrehozni,
és hogyan lehet olvasni és írni az elemeiket,
megkezdhetjük a különféle matematikai
műveletekkel való manipulálásukat.
Ezek közül a leghasznosabbak
az *elemenként* elvégzett műveletek.
Ezek egy standard skaláris műveletet alkalmaznak
a tenzor minden elemére.
Két tenzort bemenetként fogadó függvények esetén
az elemenként elvégzett műveletek valamilyen standard bináris operátort alkalmaznak
a megfelelő elempárokra.
Bármely skalárisból skalárisba képező függvényből
létrehozhatunk elemenként elvégzett függvényt.

Matematikai jelölésben az ilyen
*unáris* skaláris operátorokat (egy bemenetet fogadva)
az alábbi szignaturával jelöljük:
$f: \mathbb{R} \rightarrow \mathbb{R}$.
Ez csupán azt jelenti, hogy a függvény
bármely valós számból egy másik valós számba képez.
A legtöbb standard operátor, köztük az unárisak, mint például az $e^x$, alkalmazható elemenként.

```{.python .input}
%%tab mxnet
np.exp(x)
```

```{.python .input}
%%tab pytorch
torch.exp(x)
```

```{.python .input}
%%tab tensorflow
tf.exp(x)
```

```{.python .input}
%%tab jax
jnp.exp(x)
```

Hasonlóképpen jelöljük a *bináris* skaláris operátorokat,
amelyek valós számok párjait
egyetlen valós számba képezik,
az alábbi szignaturával:
$f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$.
Adott bármely két *azonos alakú* $\mathbf{u}$
és $\mathbf{v}$ vektor,
valamint egy $f$ bináris operátor esetén előállíthatjuk a
$\mathbf{c} = F(\mathbf{u},\mathbf{v})$ vektort
úgy, hogy minden $i$-re $c_i \gets f(u_i, v_i)$,
ahol $c_i, u_i$, és $v_i$ a $\mathbf{c}, \mathbf{u}$, és $\mathbf{v}$
vektorok $i^\textrm{th}$ elemei.
Itt vektorértékű
$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$
függvényt állítottunk elő a skaláris függvény
elemenként vektoros műveletté *emelésével*.
Az összeadás (`+`), kivonás (`-`),
szorzás (`*`), osztás (`/`)
és hatványozás (`**`)
szokásos aritmetikai operátorai
mind *emelve* lettek elemenként elvégzett műveletekké
azonos alakú, tetszőleges méretű tenzorokhoz.

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab jax
x = jnp.array([1.0, 2, 4, 8])
y = jnp.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

Az elemenként elvégzett számítások mellett
lineáris algebrai műveleteket is végezhetünk,
például skaláris szorzatot és mátrixszorzást.
Ezeket részletesen tárgyaljuk
a :numref:`sec_linear-algebra` részben.

[***Össze is fűzhetünk* több tenzort,**]
egymás végére rakva őket egy nagyobb tenzor alkotásához.
Csak egy tenzorokból álló listát kell megadnunk,
és meg kell mondanunk a rendszernek, melyik tengely mentén fűzze össze.
Az alábbi példa megmutatja, mi történik, ha két mátrixot
sorok mentén (0-s tengely)
fűzünk össze, nem oszlopok mentén (1-es tengely).
Láthatjuk, hogy az első kimenet 0-s tengelye ($6$)
a két bemeneti tenzor 0-s tengelyeinek összege ($3 + 3$);
míg a második kimenet 1-es tengelye ($8$)
a két bemeneti tenzor 1-es tengelyeinek összege ($4 + 4$).

```{.python .input}
%%tab mxnet
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
%%tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

```{.python .input}
%%tab jax
X = jnp.arange(12, dtype=jnp.float32).reshape((3, 4))
Y = jnp.array([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
jnp.concatenate((X, Y), axis=0), jnp.concatenate((X, Y), axis=1)
```

Néha szeretnénk
[***logikai állítások* segítségével bináris tenzort létrehozni.**]
Vegyük például az `X == Y` kifejezést.
Minden `i, j` pozícióra, ha `X[i, j]` és `Y[i, j]` egyenlő,
akkor az eredmény megfelelő bejegyzése `1` értéket kap,
egyébként `0` értéket.

```{.python .input}
%%tab all
X == Y
```

[**A tenzor összes elemének összegzése**] egy egyetlen elemet tartalmazó tenzort eredményez.

```{.python .input}
%%tab mxnet, pytorch, jax
X.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(X)
```

## Sugárzás (Broadcasting)
:label:`subsec_broadcasting`

Mostanra már tudod, hogyan kell
elemenként elvégzett bináris műveleteket végezni
azonos alakú két tenzorral.
Bizonyos feltételek mellett,
még akkor is, ha az alakok eltérnek,
[**elvégezhetünk elemenként elvégzett bináris műveleteket
a *sugárzási mechanizmus* meghívásával.**]
A sugárzás az alábbi kétlépéses eljárás szerint működik:
(i) bővítsük ki az egyik vagy mindkét tömböt
az elemek másolásával az 1-es hosszú tengelyek mentén,
hogy az átalakítás után
a két tenzornak azonos alakja legyen;
(ii) végezzünk elemenként elvégzett műveletet
a kapott tömbökön.

```{.python .input}
%%tab mxnet
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
%%tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
%%tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

```{.python .input}
%%tab jax
a = jnp.arange(3).reshape((3, 1))
b = jnp.arange(2).reshape((1, 2))
a, b
```

Mivel `a` és `b` rendre $3\times1$ és $1\times2$ méretű mátrixok,
alakjuk nem egyezik meg.
A sugárzás egy nagyobb $3\times2$ mátrixot állít elő
azáltal, hogy az `a` mátrixot az oszlopok mentén,
a `b` mátrixot a sorok mentén sokszorozza meg,
majd elemenként összeadja őket.

```{.python .input}
%%tab all
a + b
```

## Memória megtakarítása

[**Műveletek végrehajtása miatt az eredmények tárolásához
új memória kerülhet lefoglalásra.**]
Például, ha `Y = X + Y`-t írunk,
megszüntetjük a hivatkozást arra a tenzorra, amelyre `Y` mutatott,
és ehelyett `Y`-t az újonnan lefoglalt memóriára irányítjuk.
Ezt a problémát a Python `id()` függvényével szemléltethetjük,
amely a hivatkozott objektum pontos memóriacímét adja meg.
Figyeljük meg, hogy miután lefuttatjuk az `Y = Y + X` utasítást,
az `id(Y)` más helyre mutat.
Ennek oka, hogy Python először kiértékeli az `Y + X`-et,
új memóriát foglal az eredmény számára,
majd az `Y`-t erre az új memóriaterületre irányítja.

```{.python .input}
%%tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

Ez két okból is nemkívánatos lehet.
Először is, nem akarjuk folyamatosan
szükségtelenül lefoglalni a memóriát.
A gépi tanulásban gyakran
több száz megabájtnyi paraméterünk van,
amelyeket másodpercenként többször frissítünk.
Ahol lehetséges, ezeket a frissítéseket *helyben* szeretnénk elvégezni.
Másodszor, több változóból is mutathatunk
ugyanazokra a paraméterekre.
Ha nem frissítünk helyben,
gondosan kell frissítenünk az összes hivatkozást,
különben memóriaszivárgást okozunk,
vagy véletlenül elavult paraméterekre hivatkozunk.

:begin_tab:`mxnet, pytorch`
Szerencsére (**a helyben végzett műveletek**) végrehajtása egyszerű.
Egy művelet eredményét hozzárendelhetjük
egy korábban lefoglalt `Y` tömbhöz
a szeletes jelölés használatával: `Y[:] = <kifejezés>`.
Ennek szemléltetéséhez felülírjuk a `Z` tenzor értékeit,
amelyet a `zeros_like` segítségével inicializálunk,
hogy azonos alakja legyen, mint az `Y`-nak.
:end_tab:

:begin_tab:`tensorflow`
A `Variable`-ök TensorFlow-ban az állapot módosítható tárolói. Lehetővé teszik
a modell paramétereinek tárolását.
Egy művelet eredményét az `assign` segítségével rendelhetjük
egy `Variable`-höz.
Ennek szemléltetéséhez felülírjuk a `Z` `Variable` értékeit,
amelyet a `zeros_like` segítségével inicializálunk,
hogy azonos alakja legyen, mint az `Y`-nak.
:end_tab:

```{.python .input}
%%tab mxnet
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

```{.python .input}
%%tab jax
# A JAX tömbök nem támogatják a helyben végzett műveleteket
```

:begin_tab:`mxnet, pytorch`
[**Ha az `X` értékét nem használjuk fel a következő számításokban,
az `X[:] = X + Y` vagy `X += Y` kifejezéssel is csökkenthetjük
a művelet memóriaigényét.**]
:end_tab:

:begin_tab:`tensorflow`
Még ha az állapotot tartósan tároltuk is egy `Variable`-ben,
érdemes lehet tovább csökkenteni a memóriahasználatot azzal, hogy elkerüljük
a modell paramétereitől eltérő tenzorokhoz szükséges felesleges lefoglalásokat.
Mivel a TensorFlow `Tensor`-ai nem módosíthatók,
és a gradiensek nem áramlanak át a `Variable` értékadásokon,
a TensorFlow nem biztosít explicit módot arra,
hogy egy egyes műveletet helyben futtassunk.

Ugyanakkor a TensorFlow biztosítja a `tf.function` dekorátort,
amely a számítást egy TensorFlow-gráfba csomagolja,
ami futtatás előtt lefordításra és optimalizálásra kerül.
Ez lehetővé teszi a TensorFlow számára a nem használt értékek eltávolítását,
és a már nem szükséges korábbi lefoglalások újrafelhasználását.
Ez minimalizálja a TensorFlow számítások memóriaigényét.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
%%tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # Ez a nem használt érték ki lesz pruningolva
    A = X + Y  # A lefoglalások újrafelhasználódnak, ha már nem kellenek
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## Konverzió más Python objektumokká

:begin_tab:`mxnet, tensorflow`
[**NumPy tenzorrá (`ndarray`) való konvertálás**], vagy fordítva, egyszerű.
A konvertált eredmény nem osztja meg a memóriát.
Ez az apró kellemetlenség valójában meglehetősen fontos:
ha CPU-n vagy GPU-kon végzel műveleteket,
nem akarod leállítani a számítást, várva, hogy
a Python NumPy csomagja
esetleg mást akarna csinálni
ugyanazon a memóriaterületen.
:end_tab:

:begin_tab:`pytorch`
[**NumPy tenzorrá (`ndarray`) való konvertálás**], vagy fordítva, egyszerű.
A torch tenzor és a NumPy tömb
megosztja az alatta lévő memóriát,
és az egyiken helyben végzett művelettel végrehajtott módosítás
a másikat is megváltoztatja.
:end_tab:

```{.python .input}
%%tab mxnet
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
%%tab pytorch
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)
```

```{.python .input}
%%tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

```{.python .input}
%%tab jax
A = jax.device_get(X)
B = jax.device_put(A)
type(A), type(B)
```

Egy 1-es méretű tenzor (**Python skalárrá való konvertálásához**)
meghívhatjuk az `item` függvényt vagy a Python beépített függvényeit.

```{.python .input}
%%tab mxnet
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab jax
a = jnp.array([3.5])
a, a.item(), float(a), int(a)
```

## Összefoglalás

A tenzor osztály a fő felület az adatok tárolásához és kezeléséhez a deep learning könyvtárakban.
A tenzorok számos funkciót biztosítanak, többek között: létrehozási eljárásokat; indexelést és szeletelést; alapvető matematikai műveleteket; sugárzást; memóriahatékony értékadást; és konverziót más Python objektumokra, illetve azokból.


## Feladatok

1. Futtasd le a kódot ebben a szakaszban. Változtasd meg az `X == Y` feltételes kifejezést `X < Y`-ra vagy `X > Y`-ra, majd nézd meg, milyen tenzort kapsz.
1. Cseréld le a sugárzási mechanizmusban elemenként működő két tenzort más alakúakra, például 3-dimenziós tenzorokra. Az eredmény megfelel a várakozásoknak?

:begin_tab:`mxnet`
[Megbeszélések](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[Megbeszélések](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[Megbeszélések](https://discuss.d2l.ai/t/187)
:end_tab:

:begin_tab:`jax`
[Megbeszélések](https://discuss.d2l.ai/t/17966)
:end_tab:
