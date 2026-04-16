# Sajátérték-felbontás
:label:`sec_eigendecompositions`

A sajátértékek a lineáris algebra tanulmányozása során az egyik leghasznosabb fogalom,
amellyel találkozunk, azonban kezdőként könnyű alábecsülni a jelentőségüket.
Az alábbiakban bevezetjük a sajátérték-felbontást,
és megpróbáljuk érzékeltetni, miért is olyan fontos.

Tegyük fel, hogy adott egy $A$ mátrix a következő elemekkel:

$$
\mathbf{A} = \begin{bmatrix}
2 & 0 \\
0 & -1
\end{bmatrix}.
$$

Ha $A$-t alkalmazzuk egy tetszőleges $\mathbf{v} = [x, y]^\top$ vektorra,
a $\mathbf{A}\mathbf{v} = [2x, -y]^\top$ vektort kapjuk.
Ennek szemléletes értelmezése van:
a vektort az $x$-irányban kétszeresére nyújtjuk,
majd az $y$-irányban megfordítjuk.

Vannak azonban *bizonyos* vektorok, amelyek esetén valami változatlan marad.
Nevezetesen a $[1, 0]^\top$ vektor a $[2, 0]^\top$ vektorra képeződik,
a $[0, 1]^\top$ pedig a $[0, -1]^\top$ vektorra.
Ezek a vektorok ugyanazon az egyenesen maradnak,
az egyetlen változás, hogy a mátrix rendre $2$-szeres, illetve $-1$-szeres
faktorral nyújtja őket.
Az ilyen vektorokat *sajátvektoroknak* nevezzük,
a nyújtás mértékét pedig *sajátértéknek*.

Általánosan fogalmazva: ha találunk egy $\lambda$ számot
és egy $\mathbf{v}$ vektort olyanokat, hogy

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}.
$$

Azt mondjuk, hogy $\mathbf{v}$ az $A$ mátrix sajátvektora, és $\lambda$ a hozzá tartozó sajátérték.

## Sajátértékek meghatározása
Nézzük meg, hogyan lehet ezeket megtalálni. Ha mindkét oldalból kivonjuk a $\lambda \mathbf{v}$ tagot,
majd kiemeljük a vektort,
láthatjuk, hogy a fenti feltétel ekvivalens a következővel:

$$(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = 0.$$
:eqlabel:`eq_eigvalue_der`

Ahhoz, hogy :eqref:`eq_eigvalue_der` teljesüljön, szükséges, hogy $(\mathbf{A} - \lambda \mathbf{I})$
valamely irányt nullává zsugorítson,
tehát nem invertálható, és így determinánsa nulla.
Ezért a *sajátértékeket* úgy találjuk meg,
hogy megkeressük, melyek azok a $\lambda$ értékek, amelyekre $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$.
Ha megvan a sajátérték, megoldhatjuk az
$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$
egyenletet a hozzá tartozó *sajátvektor(ok)* meghatározásához.

### Egy példa
Lássuk ezt egy kissé bonyolultabb mátrixon:

$$
\mathbf{A} = \begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}.
$$

Ha megvizsgáljuk a $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$ feltételt,
látjuk, hogy ez ekvivalens a
$0 = (2-\lambda)(3-\lambda)-2 = (4-\lambda)(1-\lambda)$
polinomiális egyenlettel.
Tehát a két sajátérték $4$ és $1$.
A hozzájuk tartozó vektorok megtalálásához meg kell oldanunk a következőt:

$$
\begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix}x \\ y\end{bmatrix}  \; \textrm{and} \;
\begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix}  = \begin{bmatrix}4x \\ 4y\end{bmatrix} .
$$

A megoldások rendre a $[1, -1]^\top$ és a $[1, 2]^\top$ vektorok.

Ezt kódban ellenőrizhetjük a beépített `numpy.linalg.eig` rutinnal.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
import numpy as np

np.linalg.eig(np.array([[2, 1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch

torch.linalg.eig(torch.tensor([[2, 1], [2, 3]], dtype=torch.float64))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

tf.linalg.eig(tf.constant([[2, 1], [2, 3]], dtype=tf.float64))
```

Megjegyzendő, hogy a `numpy` a sajátvektorokat egységnyi hosszúságúra normálja,
míg mi tetszőleges hosszúságú vektorokat választottunk.
Emellett az előjel megválasztása is önkényes.
Mindazonáltal a számítással kapott vektorok párhuzamosak
a kézzel meghatározott vektorokkal, és ugyanazok a sajátértékek tartoznak hozzájuk.

## Mátrixok felbontása
Folytassuk az előző példát egy lépéssel tovább. Legyen

$$
\mathbf{W} = \begin{bmatrix}
1 & 1 \\
-1 & 2
\end{bmatrix},
$$

az a mátrix, amelynek oszlopai az $\mathbf{A}$ mátrix sajátvektorai. Legyen

$$
\boldsymbol{\Sigma} = \begin{bmatrix}
1 & 0 \\
0 & 4
\end{bmatrix},
$$

az a mátrix, amelynek főátlójában a megfelelő sajátértékek állnak.
Ekkor a sajátértékek és sajátvektorok definíciójából következik, hogy

$$
\mathbf{A}\mathbf{W} =\mathbf{W} \boldsymbol{\Sigma} .
$$

A $W$ mátrix invertálható, ezért mindkét oldalt jobbról megszorozhatjuk $W^{-1}$-gyel,
és a következőt írhatjuk:

$$\mathbf{A} = \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^{-1}.$$
:eqlabel:`eq_eig_decomp`

A következő szakaszban látni fogjuk ennek néhány szép következményét,
de egyelőre elég tudni, hogy ilyen felbontás mindig létezik,
ha található elegendő, lineárisan független sajátvektor (úgy hogy $W$ invertálható legyen).

## Műveletek sajátérték-felbontásokkal
A sajátérték-felbontás :eqref:`eq_eig_decomp` egyik szép tulajdonsága,
hogy számos szokásos műveletet tömören felírhatunk
a sajátérték-felbontás segítségével. Első példaként tekintsük:

$$
\mathbf{A}^n = \overbrace{\mathbf{A}\cdots \mathbf{A}}^{\textrm{$n$ times}} = \overbrace{(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})\cdots(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})}^{\textrm{$n$ times}} =  \mathbf{W}\overbrace{\boldsymbol{\Sigma}\cdots\boldsymbol{\Sigma}}^{\textrm{$n$ times}}\mathbf{W}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^n \mathbf{W}^{-1}.
$$

Ez azt mondja ki, hogy egy mátrix bármely pozitív hatványának sajátérték-felbontásához
elég a sajátértékeket ugyanakkora hatványra emelni.
Ugyanez negatív hatványokra is igazolható,
tehát ha egy mátrixot invertálni akarunk, elég megvizsgálni a következőt:

$$
\mathbf{A}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^{-1} \mathbf{W}^{-1},
$$

azaz egyszerűen minden sajátértéket invertálunk.
Ez akkor működik, ha minden sajátérték nemnulla,
tehát egy mátrix pontosan akkor invertálható, ha nincs nulla sajátértéke.

Tovább vizsgálva: ha $\lambda_1, \ldots, \lambda_n$
a mátrix sajátértékei, akkor a mátrix determinánsa

$$
\det(\mathbf{A}) = \lambda_1 \cdots \lambda_n,
$$

vagyis az összes sajátérték szorzata.
Ez szemléletesen is érthető: bármit is végez $\mathbf{W}$,
azt $W^{-1}$ visszacsinái, így végül csak a $\boldsymbol{\Sigma}$ diagonális mátrix
okoz nyújtást, mégpedig a térfogatot az átlós elemek szorzatával szorozza.

Végül emlékezzünk, hogy a rang a mátrix lineárisan független oszlopainak maximális száma.
A sajátérték-felbontást közelebbről megvizsgálva láthatjuk,
hogy a rang egyenlő az $\mathbf{A}$ nemnulla sajátértékeinek számával.

A példák sora folytatható, de remélhetőleg a lényeg már világos:
a sajátérték-felbontás sok lineáris algebrai számítást egyszerűsíthet,
és számos numerikus algoritmus, valamint a lineáris algebrai elemzések
alapvető művelete.

## Szimmetrikus mátrixok sajátérték-felbontása
Nem mindig lehetséges elegendő, lineárisan független sajátvektort találni
ahhoz, hogy a fenti eljárás működjön. Például a

$$
\mathbf{A} = \begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix},
$$

mátrixnak csak egyetlen sajátvektora van, nevezetesen $(1, 0)^\top$.
Az ilyen mátrixok kezeléséhez haladottabb technikák szükségesek
(például a Jordan-féle normálalak vagy a szingulárisérték-felbontás),
amelyek meghaladják e könyv kereteit.
Ezért gyakran azon mátrixokra korlátozzuk figyelmünket,
amelyek esetén garantálható a teljes sajátvektor-készlet létezése.

A legtöbbször előforduló ilyen mátrixcsalád a *szimmetrikus mátrixoké*,
amelyekre $\mathbf{A} = \mathbf{A}^\top$ teljesül.
Ebben az esetben $W$ választható *ortogonális mátrixnak* — olyan mátrixnak, amelynek oszlopai egységnyi hosszúságú, egymásra merőleges vektorok, és amelyre
$\mathbf{W}^\top = \mathbf{W}^{-1}$ — és az összes sajátérték valós.
Így ebben a speciális esetben :eqref:`eq_eig_decomp` a következőképpen írható:

$$
\mathbf{A} = \mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top .
$$

## Gersgorin-körök tétele
A sajátértékekkel nehéz intuitív módon boldogulni.
Ha egy tetszőleges mátrixot kapunk, a sajátértékekre kiszámítás nélkül
alig mondható valami.
Létezik azonban egy tétel, amely megkönnyíti a közelítést,
ha a legnagyobb értékek az átlón helyezkednek el.

Legyen $\mathbf{A} = (a_{ij})$ egy tetszőleges négyzetes mátrix ($n\times n$).
Definiáljuk a $r_i = \sum_{j \neq i} |a_{ij}|$ mennyiséget.
Jelölje $\mathcal{D}_i$ a komplex síkon azt a körlemezt,
amelynek középpontja $a_{ii}$ és sugara $r_i$.
Ekkor $\mathbf{A}$ minden sajátértéke valamelyik $\mathcal{D}_i$ körlemezben található.

Ez egy kicsit elvont, ezért nézzünk egy példát.
Tekintsük a következő mátrixot:

$$
\mathbf{A} = \begin{bmatrix}
1.0 & 0.1 & 0.1 & 0.1 \\
0.1 & 3.0 & 0.2 & 0.3 \\
0.1 & 0.2 & 5.0 & 0.5 \\
0.1 & 0.3 & 0.5 & 9.0
\end{bmatrix}.
$$

Ekkor $r_1 = 0.3$, $r_2 = 0.6$, $r_3 = 0.8$ és $r_4 = 0.9$.
A mátrix szimmetrikus, tehát minden sajátértéke valós.
Ez azt jelenti, hogy minden sajátértékünk az alábbi intervallumok egyikébe esik:

$$[a_{11}-r_1, a_{11}+r_1] = [0.7, 1.3], $$

$$[a_{22}-r_2, a_{22}+r_2] = [2.4, 3.6], $$

$$[a_{33}-r_3, a_{33}+r_3] = [4.2, 5.8], $$

$$[a_{44}-r_4, a_{44}+r_4] = [8.1, 9.9]. $$


A numerikus számítás elvégzésével a sajátértékek közelítőleg $0.99$, $2.97$, $4.95$, $9.08$,
amelyek mind jól belülre esnek a megadott intervallumokon.

```{.python .input}
#@tab mxnet
A = np.array([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = np.linalg.eig(A)
v
```

```{.python .input}
#@tab pytorch
A = torch.tensor([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = torch.linalg.eig(A)
v
```

```{.python .input}
#@tab tensorflow
A = tf.constant([[1.0, 0.1, 0.1, 0.1],
                [0.1, 3.0, 0.2, 0.3],
                [0.1, 0.2, 5.0, 0.5],
                [0.1, 0.3, 0.5, 9.0]])

v, _ = tf.linalg.eigh(A)
v
```

Ily módon a sajátértékek közelíthetők,
és a közelítések meglehetősen pontosak lesznek abban az esetben,
ha az átló elemei lényegesen nagyobbak az összes többi elemnél.

Ez apróságnak tűnhet, de egy olyan összetett és finom témánál, mint a sajátérték-felbontás,
minden szemléletes megragadhatóság értékes.

## Egy hasznos alkalmazás: ismételt leképezések növekedése

Most, hogy alapvetően értjük, mik a sajátvektorok,
nézzük meg, hogyan használhatók arra, hogy mélyebb megértést nyerjünk
egy neurális hálózatok viselkedésének szempontjából kulcsfontosságú problémáról:
a megfelelő súlyinicializálásról.

### Sajátvektorok mint hosszú távú viselkedés

A mélyneurális hálózatok inicializálásának teljes matematikai vizsgálata
meghaladja e könyv kereteit,
de egy játékszerű változaton keresztül megérthetjük,
hogyan segíthetnek a sajátértékek e modellek működésének megértésében.
Mint tudjuk, a neurális hálózatok lineáris transzformációs rétegek
és nemlineáris műveletek váltakozásával működnek.
Az egyszerűség kedvéért tegyük fel, hogy nincs nemlinearitás,
és a transzformáció egyetlen, ismételten alkalmazott $A$ mátrix,
így a modell kimenete

$$
\mathbf{v}_{out} = \mathbf{A}\cdot \mathbf{A}\cdots \mathbf{A} \mathbf{v}_{in} = \mathbf{A}^N \mathbf{v}_{in}.
$$

E modellek inicializálásakor $A$-t véletlenszerű, Gauss-eloszlású elemekkel rendelkező mátrixnak vesszük, tehát hozzunk létre egyet.
Konkrétan: egy nulla várható értékű, egységnyi varianciájú Gauss-eloszlású $5 \times 5$-ös mátrixszal indítunk.

```{.python .input}
#@tab mxnet
np.random.seed(8675309)

k = 5
A = np.random.randn(k, k)
A
```

```{.python .input}
#@tab pytorch
torch.manual_seed(42)

k = 5
A = torch.randn(k, k, dtype=torch.float64)
A
```

```{.python .input}
#@tab tensorflow
k = 5
A = tf.random.normal((k, k), dtype=tf.float64)
A
```

### Viselkedés véletlen adatokon
A játékmodell egyszerűsítése érdekében tegyük fel,
hogy a bemenő $\mathbf{v}_{in}$ adatvektor
egy véletlenszerű, ötdimenziós Gauss-vektor.
Gondoljuk át, milyen viselkedést szeretnénk látni.
Kontextusként képzeljük el egy általános gépi tanulási feladatot,
ahol bemeneti adatból — például egy képből — szeretnénk előrejelzést készíteni,
mondjuk annak valószínűségét, hogy a kép egy macskáról szól.
Ha $\mathbf{A}$ ismételt alkalmazása egy véletlenszerű vektort nagyon hosszúvá nyújt,
akkor a bemenet kis változásai a kimenetben nagy változásokat okoznak —
a bemeneti kép apró módosítása teljesen eltérő előrejelzésekhez vezet.
Ez nem tűnik helyesnek!

Ha viszont $\mathbf{A}$ a véletlenszerű vektorokat rövidebbre zsugorítja,
akkor sok rétegen átfutva a vektor lényegében nullává zsugorodik,
és a kimenet nem fog függeni a bemenettől. Ez nyilvánvalóan szintén nem helyes!

A növekedés és a csökkenés között szűk ösvényen kell haladnunk,
hogy kimenetünk a bemenettől függjön, de ne túlságosan!

Nézzük meg, mi történik, ha $\mathbf{A}$ mátrixunkat ismételten alkalmazzuk
egy véletlenszerű bemeneti vektorra, és nyomon követjük a normát.

```{.python .input}
#@tab mxnet
# Az egymás után alkalmazott `A` esetén kapott normák sorozatának kiszámítása
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Az egymás után alkalmazott `A` esetén kapott normák sorozatának kiszámítása
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Az egymás után alkalmazott `A` esetén kapott normák sorozatának kiszámítása
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

A norma ellenőrizhetetlenül növekszik!
Valóban, ha megvizsgáljuk a hányadosok listáját, egy mintázatot fedezünk fel.

```{.python .input}
#@tab mxnet
# A normák skálázási tényezőjének kiszámítása
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# A normák skálázási tényezőjének kiszámítása
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# A normák skálázási tényezőjének kiszámítása
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

Ha megnézzük a fenti számítás utolsó részét,
látjuk, hogy a véletlenszerű vektort `1.974459321485[...]`-szoros faktorral nyújtja,
ahol a szám vége kissé változik,
de a nyújtási faktor stabil.

### Visszakapcsolás a sajátvektorokhoz

Láttuk, hogy a sajátvektorok és sajátértékek
a nyújtás mértékének felelnek meg,
de ez csak adott vektorokra és adott nyújtásokra volt igaz.
Vizsgáljuk meg, mik ezek $\mathbf{A}$ esetén.
Egy kis figyelmeztetés: kiderül, hogy az összes megtekintéséhez
komplex számokra van szükségünk.
Ezeket nyújtásként és forgatásként is felfoghatjuk.
A komplex szám normájának vételével
(a valós és képzetes részek négyzetösszegének négyzetgyöke)
mérhetjük a nyújtás mértékét. Rendezzük is sorba őket.

```{.python .input}
#@tab mxnet
# A sajátértékek kiszámítása
eigs = np.linalg.eigvals(A).tolist()
norm_eigs = [np.absolute(x) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab pytorch
# A sajátértékek kiszámítása
eigs = torch.linalg.eig(A).eigenvalues.tolist()
norm_eigs = [torch.abs(torch.tensor(x)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab tensorflow
# A sajátértékek kiszámítása
eigs = tf.linalg.eigh(A)[0].numpy().tolist()
norm_eigs = [tf.abs(tf.constant(x, dtype=tf.float64)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

### Egy megfigyelés

Valami kissé váratlan dologra figyelhetünk fel:
az a szám, amelyet korábban az $\mathbf{A}$ mátrix
egy véletlenszerű vektorra alkalmazott hosszú távú nyújtásaként azonosítottunk,
*pontosan* (tizenhárom tizedesjegy pontossággal!)
$\mathbf{A}$ legnagyobb sajátértékével egyezik meg.
Ez nyilvánvalóan nem véletlen!

Ha azonban most geometriailag gondolkodunk arról, ami történik,
ez kezd érthetővé válni. Tekintsünk egy véletlenszerű vektort.
Ez a véletlenszerű vektor minden irányban mutat valamennyire,
tehát különösen legalább egy kicsit abba az irányba is mutat,
amelybe $\mathbf{A}$ legnagyobb sajátértékéhez tartozó sajátvektora mutat.
Ez olyannyira fontos, hogy *domináns sajátértéknek* és *domináns sajátvektornak* nevezzük.
$\mathbf{A}$ alkalmazása után a véletlenszerű vektorunkat
minden lehetséges irányban megnyújtja,
ahogyan az minden lehetséges sajátvektorhoz tartozik,
de leginkább a domináns sajátvektorhoz tartozó irányban nyújtja.
Ez azt jelenti, hogy $A$ alkalmazása után
a véletlenszerű vektorunk hosszabb lesz, és olyan irányba mutat,
amely közelebb van a domináns sajátvektorhoz való igazodáshoz.
A mátrix sokszori alkalmazása után
az igazodás a domináns sajátvektorhoz egyre szorosabb lesz, olyannyira, hogy
minden gyakorlati szempontból a véletlenszerű vektorunk
a domináns sajátvektorrá alakult át!
Ez az algoritmus valójában az alapja a *hatványiterációnak*,
amellyel egy mátrix legnagyobb sajátértékét és sajátvektorát lehet meghatározni. Részletekért lásd például :cite:`Golub.Van-Loan.1996`.

### A normálás javítása

A fenti megbeszélés alapján arra jutottunk,
hogy nem szeretnénk, ha egy véletlenszerű vektort megnyújtanának vagy összezsugorítanának;
azt szeretnénk, hogy a véletlenszerű vektorok az egész folyamat során nagyjából azonos méretűek maradjanak.
Ehhez most átskálázzuk a mátrixunkat ezzel a domináns sajátértékkel,
úgy hogy a legnagyobb sajátérték ezentúl éppen egy legyen.
Nézzük meg, mi történik ebben az esetben.

```{.python .input}
#@tab mxnet
# Az `A` mátrix újraskálázása
A /= norm_eigs[-1]

# Ugyanaz a kísérlet ismét
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Az `A` mátrix újraskálázása
A /= norm_eigs[-1]

# Ugyanaz a kísérlet ismét
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Az `A` mátrix újraskálázása
A /= norm_eigs[-1]

# Ugyanaz a kísérlet ismét
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

Az egymást követő normák hányadosát is ábrázolhatjuk, ahogyan korábban is, és láthatjuk, hogy ez valóban stabilizálódik.

```{.python .input}
#@tab mxnet
# Az arány kirajzolása is
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Az arány kirajzolása is
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Az arány kirajzolása is
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

## Megbeszélés

Most pontosan azt látjuk, amire számítottunk!
Miután a mátrixokat a domináns sajátértékkel normáltuk,
a véletlenszerű adat már nem robban fel, mint korábban,
hanem végül egy adott értékre egyensúlyozódik be.
Szép lenne, ha ezeket első elvekből is levezethetnénk,
és valóban: ha mélyebben belenézünk a matematikájába,
láthatjuk, hogy egy nagy, független, nulla várható értékű, egységnyi varianciájú
Gauss-elemekből álló véletlenszerű mátrix legnagyobb sajátértéke átlagosan $\sqrt{n}$ körül van,
esetünkben $\sqrt{5} \approx 2.2$,
egy lenyűgöző tény következtében, amelyet *cirkuláris törvénynek* neveznek :cite:`Ginibre.1965`.
A véletlenszerű mátrixok sajátértékei (és egy kapcsolódó mennyiség, a szinguláris értékek) és a neurális hálózatok megfelelő inicializálása közötti összefüggést :citet:`Pennington.Schoenholz.Ganguli.2017` és az azt követő munkák tárgyalják.

## Összefoglalás
* A sajátvektorok olyan vektorok, amelyeket a mátrix az irányuk megváltoztatása nélkül nyújt.
* A sajátértékek azt fejezik ki, hogy a mátrix alkalmazása mekkora mértékben nyújtja a sajátvektorokat.
* Egy mátrix sajátérték-felbontása lehetővé teszi, hogy sok műveletet a sajátértékekre vonatkozó műveletre vezessük vissza.
* A Gersgorin-körök tétele közelítő értékeket ad egy mátrix sajátértékeire.
* Az ismételten alkalmazott mátrixhatványok viselkedése elsősorban a legnagyobb sajátérték nagyságától függ. Ez a felismerés számos alkalmazást talál a neurális hálózatok inicializálásának elméletében.

## Feladatok
1. Mik az alábbi mátrix sajátértékei és sajátvektorai?
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}?
$$
1. Mik az alábbi mátrix sajátértékei és sajátvektorai, és mi a különös ebben a példában az előzőhöz képest?
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
0 & 2
\end{bmatrix}.
$$
1. A sajátértékek kiszámítása nélkül lehetséges-e, hogy az alábbi mátrix legkisebb sajátértéke kisebb $0.5$-nél? *Megjegyzés*: ez a feladat fejben is megoldható.
$$
\mathbf{A} = \begin{bmatrix}
3.0 & 0.1 & 0.3 & 1.0 \\
0.1 & 1.0 & 0.1 & 0.2 \\
0.3 & 0.1 & 5.0 & 0.0 \\
1.0 & 0.2 & 0.0 & 1.8
\end{bmatrix}.
$$

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/411)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1086)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1087)
:end_tab:
