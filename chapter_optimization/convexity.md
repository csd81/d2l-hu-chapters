# Konvexitás
:label:`sec_convexity`

A konvexitás kulcsszerepet játszik az optimalizálási algoritmusok tervezésében. 
Ez nagyrészt annak köszönhető, hogy az ilyen kontextusban sokkal könnyebb az algoritmusokat elemezni és tesztelni. 
Más szóval, ha az algoritmus még konvex esetben is rosszul teljesít, általában nem számíthatunk jobb eredményekre más esetekben sem. 
Ráadásul, bár a mélytanulásban az optimalizálási problémák általában nemkonvexek, a lokális minimumok közelében sokszor konvex tulajdonságokat mutatnak. Ez izgalmas új optimalizálási változatokhoz vezethet, például :cite:`Izmailov.Podoprikhin.Garipov.ea.2018`.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

## Definíciók

A konvex analízis előtt meg kell határoznunk a *konvex halmazok* és a *konvex függvények* fogalmát.
Ezek olyan matematikai eszközökhöz vezetnek, amelyeket széles körben alkalmaznak a gépi tanulásban.


### Konvex halmazok

A halmazok a konvexitás alapjai. Egyszerűen fogalmazva: egy vektortérben lévő $\mathcal{X}$ halmaz *konvex*, ha bármely $a, b \in \mathcal{X}$ esetén az $a$-t és $b$-t összekötő szakasz szintén $\mathcal{X}$-ben van. Matematikai értelemben ez azt jelenti, hogy minden $\lambda \in [0, 1]$ esetén teljesül:

$$\lambda  a + (1-\lambda)  b \in \mathcal{X} \textrm{ whenever } a, b \in \mathcal{X}.$$

Ez kissé elvont. Tekintsük a :numref:`fig_pacman` ábrát. Az első halmaz nem konvex, mivel léteznek olyan szakaszok, amelyek nem találhatók benne.
A másik két halmaz nem szenved ilyen problémától.

![Az első halmaz nemkonvex, a másik kettő konvex.](../img/pacman.svg)
:label:`fig_pacman`

A definíciók önmagukban nem különösebben hasznosak, hacsak nem tudunk valamit kezdeni velük.
Ebben az esetben megvizsgálhatjuk a metszeteket, ahogy a :numref:`fig_convex_intersect` ábra mutatja.
Tegyük fel, hogy $\mathcal{X}$ és $\mathcal{Y}$ konvex halmazok. Ekkor $\mathcal{X} \cap \mathcal{Y}$ is konvex. Ennek belátásához tekintsünk bármely $a, b \in \mathcal{X} \cap \mathcal{Y}$ elemet. Mivel $\mathcal{X}$ és $\mathcal{Y}$ konvexek, az $a$-t és $b$-t összekötő szakasz mindkét halmazban benne van. Ebből következően benne kell lennie $\mathcal{X} \cap \mathcal{Y}$-ban is, ezzel bizonyítva a tételt.

![Két konvex halmaz metszete konvex.](../img/convex-intersect.svg)
:label:`fig_convex_intersect`

Ezt az eredményt kis erőfeszítéssel megerősíthetjük: adott $\mathcal{X}_i$ konvex halmazok esetén metszetük $\cap_{i} \mathcal{X}_i$ konvex.
Annak belátásához, hogy az ellenkezője nem igaz, tekintsünk két diszjunkt halmazt: $\mathcal{X} \cap \mathcal{Y} = \emptyset$. Vegyük fel $a \in \mathcal{X}$ és $b \in \mathcal{Y}$ elemeket. A :numref:`fig_nonconvex` ábrán látható, $a$-t és $b$-t összekötő szakasznak tartalmaznia kell egy olyan részt, amely sem $\mathcal{X}$-ben, sem $\mathcal{Y}$-ban nincs, mivel feltételeztük, hogy $\mathcal{X} \cap \mathcal{Y} = \emptyset$. Ezért a szakasz $\mathcal{X} \cup \mathcal{Y}$-ban sincs benne, ezzel bizonyítva, hogy a konvex halmazok uniója általában nem konvex.

![Két konvex halmaz uniója nem feltétlenül konvex.](../img/nonconvex.svg)
:label:`fig_nonconvex`

A mélytanulásban a problémák általában konvex halmazokon vannak értelmezve. Például $\mathbb{R}^d$, a $d$-dimenziós valós vektorok halmaza, konvex halmaz (elvégre $\mathbb{R}^d$ bármely két pontja közötti szakasz $\mathbb{R}^d$-ben marad). Egyes esetekben korlátos hosszúságú változókkal dolgozunk, például $r$ sugarú gömbökkel, amelyeket a következőképpen definiálunk: $\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \textrm{ and } \|\mathbf{x}\| \leq r\}$.

### Konvex függvények

Most, hogy rendelkezünk konvex halmazokkal, bevezethetjük a *konvex függvényeket* $f$.
Adott egy $\mathcal{X}$ konvex halmaz; az $f: \mathcal{X} \to \mathbb{R}$ függvény *konvex*, ha minden $x, x' \in \mathcal{X}$ és minden $\lambda \in [0, 1]$ esetén teljesül:

$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$

Ennek szemléltetéséhez rajzoljunk fel néhány függvényt, és ellenőrizzük, melyek teljesítik a feltételt.
Az alábbiakban néhány függvényt definiálunk, köztük konvex és nemkonvex változatokat.

```{.python .input}
#@tab all
f = lambda x: 0.5 * x**2  # Konvex
g = lambda x: d2l.cos(np.pi * x)  # Nemkonvex
h = lambda x: d2l.exp(0.5 * x)  # Konvex

x, segment = d2l.arange(-2, 2, 0.01), d2l.tensor([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
```

Ahogy várható, a koszinuszfüggvény *nemkonvex*, míg a parabola és az exponenciális függvény konvex. Fontos megjegyezni, hogy az $\mathcal{X}$ konvex halmazra vonatkozó feltétel szükséges ahhoz, hogy a feltétel értelmes legyen. Ellenkező esetben az $f(\lambda x + (1-\lambda) x')$ eredménye nem lenne jól definiált.


### Jensen-egyenlőtlenség

Adott egy $f$ konvex függvény, az egyik leghasznosabb matematikai eszköz a *Jensen-egyenlőtlenség*.
Ez lényegében a konvexitás definíciójának általánosítása:

$$\sum_i \alpha_i f(x_i)  \geq f\left(\sum_i \alpha_i x_i\right)    \textrm{ and }    E_X[f(X)]  \geq f\left(E_X[X]\right),$$
:eqlabel:`eq_jensens-inequality`

ahol $\alpha_i$ nemnegatív valós számok, amelyekre $\sum_i \alpha_i = 1$, és $X$ egy véletlen változó.
Más szóval: egy konvex függvény várható értéke nem kisebb, mint a várható érték konvex függvénye – ahol az utóbbi általában egyszerűbb kifejezés. 
Az első egyenlőtlenség bizonyításához a konvexitás definícióját ismételten alkalmazzuk az összeg egy-egy tagjára.


A Jensen-egyenlőtlenség egyik leggyakoribb alkalmazása egy bonyolultabb kifejezés egyszerűbb kifejezéssel való felső becslése.
Például alkalmazható részben megfigyelt véletlen változók log-valószínűségéhez. Vagyis:

$$E_{Y \sim P(Y)}[-\log P(X \mid Y)] \geq -\log P(X),$$

mivel $\int P(Y) P(X \mid Y) dY = P(X)$.
Ez variációs módszerekben is alkalmazható. A $Y$ általában a meg nem figyelt véletlen változó, $P(Y)$ az eloszlásának legjobb becslése, és $P(X)$ az $Y$-t kiintegrált eloszlás. Például klaszterezésnél $Y$ lehet a klasztercímke, és $P(X \mid Y)$ a generatív modell a klasztercímkék alkalmazásakor.



## Tulajdonságok

A konvex függvényeknek számos hasznos tulajdonsága van. Az alábbiakban néhány általánosan használtat ismertetünk.


### A lokális minimumok egyben globális minimumok

Mindenekelőtt a konvex függvények lokális minimumai egyben globális minimumok is. 
Ezt ellentmondással bizonyíthatjuk az alábbiak szerint.

Tekintsünk egy $\mathcal{X}$ konvex halmazon értelmezett $f$ konvex függvényt.
Tegyük fel, hogy $x^{\ast} \in \mathcal{X}$ lokális minimum:
létezik egy kis pozitív $p$ érték, amelyre minden $x \in \mathcal{X}$ esetén, ahol $0 < |x - x^{\ast}| \leq p$, teljesül, hogy $f(x^{\ast}) < f(x)$.

Tegyük fel, hogy a $x^{\ast}$ lokális minimum nem globális minimuma $f$-nek:
létezik $x' \in \mathcal{X}$, amelyre $f(x') < f(x^{\ast})$. 
Létezik olyan 
$\lambda \in [0, 1)$, például $\lambda = 1 - \frac{p}{|x^{\ast} - x'|}$,
amelyre
$0 < |\lambda x^{\ast} + (1-\lambda) x' - x^{\ast}| \leq p$. 

Azonban a konvex függvények definíciója szerint:

$$\begin{aligned}
    f(\lambda x^{\ast} + (1-\lambda) x') &\leq \lambda f(x^{\ast}) + (1-\lambda) f(x') \\
    &< \lambda f(x^{\ast}) + (1-\lambda) f(x^{\ast}) \\
    &= f(x^{\ast}),
\end{aligned}$$

ami ellentmond annak, hogy $x^{\ast}$ lokális minimum.
Tehát nincs olyan $x' \in \mathcal{X}$, amelyre $f(x') < f(x^{\ast})$. A $x^{\ast}$ lokális minimum egyben globális minimum is.

Például az $f(x) = (x-1)^2$ konvex függvénynek $x=1$-ben van lokális minimuma, amely egyben globális minimum is.

```{.python .input}
#@tab all
f = lambda x: (x - 1) ** 2
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
```

Az a tény, hogy a konvex függvények lokális minimumai egyben globális minimumok is, rendkívül kényelmes. 
Ez azt jelenti, hogy a függvények minimalizálásának során nem „ragadhatunk be". 
Fontos azonban megjegyezni, hogy ez nem zárja ki, hogy egynél több globális minimum létezzen, vagy hogy egyáltalán létezzen globális minimum. Például az $f(x) = \mathrm{max}(|x|-1, 0)$ függvény a $[-1, 1]$ intervallumon veszi fel minimumát. Ezzel szemben az $f(x) = \exp(x)$ függvénynek nincs minimuma $\mathbb{R}$-en: $x \to -\infty$ esetén $0$-hoz tart aszimptotikusan, de nincs olyan $x$, amelyre $f(x) = 0$.

### A konvex függvények alatti halmazok konvexek

Konvex halmazokat kényelmesen definiálhatunk konvex függvények *alatti halmazai* segítségével.
Konkrétan: adott egy $\mathcal{X}$ konvex halmazon értelmezett $f$ konvex függvény, bármely alatti halmaz

$$\mathcal{S}_b \stackrel{\textrm{def}}{=} \{x | x \in \mathcal{X} \textrm{ and } f(x) \leq b\}$$

konvex. 

Ennek gyors bizonyítása: bármely $x, x' \in \mathcal{S}_b$ esetén be kell mutatni, hogy $\lambda x + (1-\lambda) x' \in \mathcal{S}_b$, ha $\lambda \in [0, 1]$. 
Mivel $f(x) \leq b$ és $f(x') \leq b$,
a konvexitás definíciója alapján: 

$$f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b.$$


### Konvexitás és második deriváltak

Ha az $f: \mathbb{R}^n \rightarrow \mathbb{R}$ függvény második deriváltja létezik, akkor könnyen ellenőrizhetjük, hogy $f$ konvex-e. 
Mindössze azt kell ellenőrizni, hogy $f$ Hesse-mátrixa pozitív szemidefinit-e: $\nabla^2f \succeq 0$, vagyis a $\nabla^2f$ Hesse-mátrixot $\mathbf{H}$-val jelölve:
$\mathbf{x}^\top \mathbf{H} \mathbf{x} \geq 0$
minden $\mathbf{x} \in \mathbb{R}^n$ esetén.
Például az $f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2$ függvény konvex, mivel $\nabla^2 f = \mathbf{1}$, vagyis Hesse-mátrixa az egységmátrix.


Formálisan: egy kétszer differenciálható egydimenziós $f: \mathbb{R} \rightarrow \mathbb{R}$ függvény akkor és csak akkor konvex, ha második deriváltja $f'' \geq 0$. Bármely kétszer differenciálható többdimenziós $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$ függvény akkor és csak akkor konvex, ha Hesse-mátrixa $\nabla^2f \succeq 0$.

Először az egydimenziós esetet kell bebizonyítani.
Annak belátásához, hogy $f$ konvexitása implikálja $f'' \geq 0$, felhasználjuk a következőt:

$$\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left(\frac{x + \epsilon}{2} + \frac{x - \epsilon}{2}\right) = f(x).$$

Mivel a második derivált véges differenciák határértékeként adott, következik:

$$f''(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0.$$

Annak belátásához, hogy $f'' \geq 0$ implikálja $f$ konvexitását, felhasználjuk, hogy $f'' \geq 0$ azt jelenti, hogy $f'$ monoton nemcsökkenő függvény. Legyenek $a < x < b$ három pont $\mathbb{R}$-ben,
ahol $x = (1-\lambda)a + \lambda b$ és $\lambda \in (0, 1)$.
A középértéktétel szerint
léteznek $\alpha \in [a, x]$ és $\beta \in [x, b]$,
amelyekre:

$$f'(\alpha) = \frac{f(x) - f(a)}{x-a} \textrm{ and } f'(\beta) = \frac{f(b) - f(x)}{b-x}.$$


Monotonitás alapján $f'(\beta) \geq f'(\alpha)$, ebből következik:

$$\frac{x-a}{b-a}f(b) + \frac{b-x}{b-a}f(a) \geq f(x).$$

Mivel $x = (1-\lambda)a + \lambda b$,
fennáll:

$$\lambda f(b) + (1-\lambda)f(a) \geq f((1-\lambda)a + \lambda b),$$

ezzel bizonyítva a konvexitást.

Másodsorban szükségünk van egy lemmára a többdimenziós eset bizonyítása előtt:
$f: \mathbb{R}^n \rightarrow \mathbb{R}$
akkor és csak akkor konvex, ha minden $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ esetén

$$g(z) \stackrel{\textrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y}) \textrm{ where } z \in [0,1]$$ 

konvex.

Annak bizonyítására, hogy $f$ konvexitása implikálja $g$ konvexitását,
megmutatjuk, hogy minden $a, b, \lambda \in [0, 1]$ esetén (tehát
$0 \leq \lambda a + (1-\lambda) b \leq 1$)

$$\begin{aligned} &g(\lambda a + (1-\lambda) b)\\
=&f\left(\left(\lambda a + (1-\lambda) b\right)\mathbf{x} + \left(1-\lambda a - (1-\lambda) b\right)\mathbf{y} \right)\\
=&f\left(\lambda \left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) \left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \right)\\
\leq& \lambda f\left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) f\left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \\
=& \lambda g(a) + (1-\lambda) g(b).
\end{aligned}$$

A megfordítás bizonyításához megmutatjuk, hogy minden $\lambda \in [0, 1]$ esetén

$$\begin{aligned} &f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y})\\
=&g(\lambda \cdot 1 + (1-\lambda) \cdot 0)\\
\leq& \lambda g(1)  + (1-\lambda) g(0) \\
=& \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y}).
\end{aligned}$$


Végül a fenti lemma és az egydimenziós eset eredményét felhasználva a többdimenziós eset a következőképpen bizonyítható.
Egy $f: \mathbb{R}^n \rightarrow \mathbb{R}$ többdimenziós függvény akkor és csak akkor konvex, ha minden $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ esetén $g(z) \stackrel{\textrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y})$, ahol $z \in [0,1]$,
konvex.
Az egydimenziós eset szerint
ez akkor és csak akkor teljesül, ha
$g'' = (\mathbf{x} - \mathbf{y})^\top \mathbf{H}(\mathbf{x} - \mathbf{y}) \geq 0$ ($\mathbf{H} \stackrel{\textrm{def}}{=} \nabla^2f$)
minden $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ esetén,
ami egyenértékű azzal, hogy $\mathbf{H} \succeq 0$
a pozitív szemidefinit mátrixok definíciója szerint.


## Korlátok

A konvex optimalizálás egyik szép tulajdonsága, hogy lehetővé teszi a korlátok hatékony kezelését. Vagyis lehetővé teszi a *korlátozott optimalizálási* problémák megoldását az alábbi formában:

$$\begin{aligned} \mathop{\textrm{minimize~}}_{\mathbf{x}} & f(\mathbf{x}) \\
    \textrm{ subject to } & c_i(\mathbf{x}) \leq 0 \textrm{ for all } i \in \{1, \ldots, n\},
\end{aligned}$$

ahol $f$ a célfüggvény, a $c_i$ függvények pedig a kényszerfüggvények. Ennek szemléltetéséhez tekintsük azt az esetet, ahol $c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$. Ekkor a $\mathbf{x}$ paraméterek az egységgömbre korlátozódnak. Ha a második kényszer $c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$, ez minden $\mathbf{x}$-et egy féltérre korlátoz. Mindkét kényszer egyidejű teljesítése egy gömbszelet kiválasztásának felel meg.

### Lagrange-féle függvény

Általánosságban egy korlátozott optimalizálási probléma megoldása nehéz. Az egyik megközelítés a fizikából ered, meglehetősen egyszerű intuícióval. Képzeljünk el egy labdát egy dobozban. A labda oda gördül, ahol a legmélyebb pont van, és a gravitáció erőit kiegyensúlyozza a doboz oldalainak a labdára ható ereje. Röviden: a célfüggvény gradiensét (vagyis a gravitációt) kiegyensúlyozza a kényszerfüggvény gradiense (a labdának a falak „visszalökése" miatt a dobozban kell maradnia). 
Fontos megjegyezni, hogy egyes korlátok nem feltétlenül aktívak:
azok a falak, amelyeket a labda nem érint,
nem tudnak erőt kifejteni rá.


A *Lagrange-féle* $L$ levezetését mellőzve a fenti gondolatmenet a következő nyeregpontos optimalizálási problémaként fejezhető ki:

$$L(\mathbf{x}, \alpha_1, \ldots, \alpha_n) = f(\mathbf{x}) + \sum_{i=1}^n \alpha_i c_i(\mathbf{x}) \textrm{ where } \alpha_i \geq 0.$$

Az $\alpha_i$ ($i=1,\ldots,n$) változók az ún. *Lagrange-szorzók*, amelyek biztosítják a kényszerfeltételek megfelelő érvényesítését. Éppen olyan nagyok, hogy minden $i$-re teljesüljön $c_i(\mathbf{x}) \leq 0$. Például ha valamely $\mathbf{x}$-re $c_i(\mathbf{x}) < 0$ természetesen teljesül, $\alpha_i = 0$ lesz. Ráadásul ez egy nyeregpontos optimalizálási probléma, amelyben $L$-t *maximalizálni* szeretnénk az összes $\alpha_i$ tekintetében, és egyidejűleg *minimalizálni* $\mathbf{x}$ tekintetében. Gazdag irodalom tárgyalja, hogyan jutunk el az $L(\mathbf{x}, \alpha_1, \ldots, \alpha_n)$ függvényhez. Céljainkhoz elegendő tudni, hogy $L$ nyeregpontján az eredeti korlátozott optimalizálási probléma optimálisan megoldott.

### Büntetőtagok

A korlátozott optimalizálási problémák legalább *közelítőleges* kielégítésének egyik módja a Lagrange-féle $L$ adaptálása. 
A $c_i(\mathbf{x}) \leq 0$ teljesítése helyett egyszerűen hozzáadjuk $\alpha_i c_i(\mathbf{x})$-t az $f(x)$ célfüggvényhez. Ez biztosítja, hogy a korlátok ne sérüljenek túlságosan.

Valójában ezt a trükköt már korábban is alkalmaztuk. Gondoljunk a súlycsökkentésre a :numref:`sec_weight_decay` szakaszban. Ott $\frac{\lambda}{2} \|\mathbf{w}\|^2$-et adtunk a célfüggvényhez, hogy $\mathbf{w}$ ne nőjön túl nagyra. A korlátozott optimalizálás szempontjából ez biztosítja, hogy $\|\mathbf{w}\|^2 - r^2 \leq 0$ teljesüljön valamely $r$ sugárra. A $\lambda$ értékének változtatásával módosíthatjuk $\mathbf{w}$ méretét.

Általánosságban a büntetőtagok hozzáadása jó módszer a közelítő kényszerkielégítés biztosítására. A gyakorlatban ez sokkal robusztusabbnak bizonyul a pontos kielégítésnél. Ráadásul nemkonvex problémáknál a pontos megközelítés számos, konvex esetben vonzó tulajdonsága (pl. optimalitás) elvész.

### Vetítések

A korlátok kielégítésének alternatív stratégiája a vetítés. Ezzel korábban is találkoztunk, például a gradiens vágásnál a :numref:`sec_rnn-scratch` szakaszban. Ott biztosítottuk, hogy a gradiens hossza $\theta$ alatt maradjon:

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \mathrm{min}(1, \theta/\|\mathbf{g}\|).$$

Ez $\mathbf{g}$-nek a $\theta$ sugarú gömbre való *vetítése*. Általánosabban: egy $\mathcal{X}$ konvex halmazra való vetítés definíciója:

$$\textrm{Proj}_\mathcal{X}(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in \mathcal{X}} \|\mathbf{x} - \mathbf{x}'\|,$$

vagyis $\mathcal{X}$-nek $\mathbf{x}$-hez legközelebbi pontja. 

![Konvex vetítések.](../img/projections.svg)
:label:`fig_projections`

A vetítések matematikai definíciója kissé elvontnak tűnhet. A :numref:`fig_projections` ábra érthetőbbé teszi. Az ábrán két konvex halmaz látható: egy kör és egy rombusz. 
A mindkét halmazba eső pontok (sárga) nem változnak a vetítés során. 
A mindkét halmazon kívüli pontok (fekete) a halmazok belsejébe (piros) vetítődnek, az eredeti pontokhoz (fekete) legközelebb eső pontokra.
Míg $\ell_2$-es gömbök esetén az irány változatlan marad, ez általánosan nem szükségszerű, ahogy a rombusz eseténél is látható.


A konvex vetítések egyik alkalmazási területe a ritka súlyvektorok kiszámítása. Ebben az esetben a súlyvektorokat egy $\ell_1$-es gömbre vetítjük,
ami a :numref:`fig_projections` ábrán látható rombusz eset általánosítása.


## Összefoglalás

A mélytanulás kontextusában a konvex függvények fő célja az optimalizálási algoritmusok motiválása és részletes megértésük elősegítése. A következőkben látni fogjuk, hogyan vezethető le ennek megfelelően a gradient descent és a sztochasztikus gradient descent.


* Konvex halmazok metszetei konvexek. Unióik nem feltétlenül azok.
* Konvex függvény várható értéke nem kisebb, mint a várható érték konvex függvénye (Jensen-egyenlőtlenség).
* Egy kétszer differenciálható függvény akkor és csak akkor konvex, ha Hesse-mátrixa (a második deriváltak mátrixa) pozitív szemidefinit.
* Konvex korlátok hozzáadhatók a Lagrange-függvény segítségével. A gyakorlatban egyszerűen büntetőtagként adhatjuk hozzá őket a célfüggvényhez.
* A vetítések az eredeti pontokhoz legközelebbi konvex halmaz-beli pontokra képeznek le.

## Gyakorló feladatok

1. Tegyük fel, hogy egy halmaz konvexitását szeretnénk ellenőrizni azáltal, hogy az összes belső pont közötti egyenest megrajzoljuk és ellenőrizzük, hogy az egyenesek benne vannak-e a halmazban.
    1. Bizonyítsuk be, hogy elegendő csak a határon lévő pontokat ellenőrizni.
    1. Bizonyítsuk be, hogy elegendő csak a halmaz csúcsait ellenőrizni.
1. Jelöljük $\mathcal{B}_p[r] \stackrel{\textrm{def}}{=} \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \textrm{ and } \|\mathbf{x}\|_p \leq r\}$-rel az $r$ sugarú gömböt a $p$-normában. Bizonyítsuk be, hogy $\mathcal{B}_p[r]$ konvex minden $p \geq 1$ esetén.
1. Adott $f$ és $g$ konvex függvények esetén mutassuk meg, hogy $\mathrm{max}(f, g)$ szintén konvex. Bizonyítsuk be, hogy $\mathrm{min}(f, g)$ nem konvex.
1. Bizonyítsuk be, hogy a softmax függvény normalizációja konvex. Konkrétabban bizonyítsuk be az
    $f(x) = \log \sum_i \exp(x_i)$ konvexitását.
1. Bizonyítsuk be, hogy a lineáris alterterek, vagyis $\mathcal{X} = \{\mathbf{x} | \mathbf{W} \mathbf{x} = \mathbf{b}\}$, konvex halmazok.
1. Bizonyítsuk be, hogy $\mathbf{b} = \mathbf{0}$-ra vonatkozó lineáris alterterek esetén a $\textrm{Proj}_\mathcal{X}$ vetítés felírható $\mathbf{M} \mathbf{x}$ alakban valamely $\mathbf{M}$ mátrixra.
1. Mutassuk meg, hogy kétszer differenciálható konvex $f$ függvényekre felírható: $f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''(x + \xi)$ valamely $\xi \in [0, \epsilon]$ esetén.
1. Adott egy $\mathcal{X}$ konvex halmaz és két $\mathbf{x}$ és $\mathbf{y}$ vektor esetén bizonyítsuk be, hogy a vetítések soha nem növelik a távolságot, vagyis $\|\mathbf{x} - \mathbf{y}\| \geq \|\textrm{Proj}_\mathcal{X}(\mathbf{x}) - \textrm{Proj}_\mathcal{X}(\mathbf{y})\|$.


[Discussions](https://discuss.d2l.ai/t/350)
