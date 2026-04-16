# Maximális valószínűség
:label:`sec_maximum_likelihood`

A gépi tanulásban az egyik legelterjedtebb gondolkodásmód a maximum likelihood elve. Ez azt jelenti, hogy ha ismeretlen paraméterekkel rendelkező valószínűségi modellel dolgozunk, akkor azok a paraméterek a legvalószínűbbek, amelyek mellett az adatok a legnagyobb valószínűséggel fordulnak elő.

## A maximális valószínűség elve

Ennek van egy Bayes-i értelmezése is, amely hasznos lehet. Tegyük fel, hogy van egy $\boldsymbol{\theta}$ paraméterekkel rendelkező modellünk és egy $X$ adatgyűjteményünk. Konkrétan elképzelhetjük, hogy $\boldsymbol{\theta}$ egyetlen érték, amely annak valószínűségét jelöli, hogy egy pénzfeldobásnál fej jön ki, $X$ pedig egymástól független pénzfeldobások sorozata. Ezt a példát részletesen megvizsgáljuk.

Ha meg akarjuk találni a modell paramétereinek legvalószínűbb értékét, akkor a következőt keressük:

$$\mathop{\mathrm{argmax}} P(\boldsymbol{\theta}\mid X).$$
:eqlabel:`eq_max_like`

A Bayes-tétel alapján ez ugyanaz, mint

$$
\mathop{\mathrm{argmax}} \frac{P(X \mid \boldsymbol{\theta})P(\boldsymbol{\theta})}{P(X)}.
$$

A $P(X)$ kifejezés — az adatok paraméterektől független generálási valószínűsége — egyáltalán nem függ $\boldsymbol{\theta}$-tól, így kihagyható anélkül, hogy $\boldsymbol{\theta}$ legjobb megválasztása megváltozna. Hasonlóan, most azt is feltételezhetjük, hogy nincsen előzetes feltevésünk arról, hogy melyik paraméteregyüttes jobb bármely másiknál, ezért kijelenthetjük, hogy $P(\boldsymbol{\theta})$ sem függ théta-tól! Ez például a pénzfeldobásos példánkban is logikus: a fej valószínűsége bármely $[0,1]$ értéket felvehet, anélkül hogy előzetesen azt hinnénk, hogy az érme szabályos-e vagy sem (ezt szokás *nem informatív prior*nak nevezni). Így a Bayes-tétel alkalmazása megmutatja, hogy $\boldsymbol{\theta}$ legjobb megválasztása a $\boldsymbol{\theta}$ maximum likelihood becslése:

$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}} _ {\boldsymbol{\theta}} P(X \mid \boldsymbol{\theta}).
$$

Az általánosan elterjedt szóhasználatban az adatok valószínűségét az adott paraméterek mellett ($P(X \mid \boldsymbol{\theta})$) *likelihood*-nak nevezzük.

### Egy konkrét példa

Nézzük meg, hogyan működik ez egy konkrét példán. Tegyük fel, hogy egyetlen $\theta$ paraméterünk van, amely annak valószínűségét jelöli, hogy pénzfeldobásnál fej jön ki. Ekkor az írás valószínűsége $1-\theta$, és ha a megfigyelt $X$ adataink egy sorozat $n_H$ fejjel és $n_T$ írással, a független valószínűségek szorzatának tulajdonsága alapján:

$$
P(X \mid \theta) = \theta^{n_H}(1-\theta)^{n_T}.
$$

Ha $13$-szor dobunk pénzt és a "HHHTHTTHHHHHT" sorozatot kapjuk, ahol $n_H = 9$ és $n_T = 4$, ez a következőt adja:

$$
P(X \mid \theta) = \theta^9(1-\theta)^4.
$$

Ez a példa azért jó, mert a választ előre tudjuk. Ha valaki azt mondaná: „13-szor dobtam fel egy pénzérmét, és 9-szer jött fej — mi a legjobb becslésünk arra, hogy mekkora a fej valószínűsége?" — mindenki helyesen mondaná, hogy $9/13$. A maximum likelihood módszer ezt az értéket első elvekből vezeti le, és általánosítható sokkal bonyolultabb helyzetekre is.

A $P(X \mid \theta)$ függvény grafikonja a következő:

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

theta = np.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

theta = torch.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

theta = tf.range(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

A maximuma közel van a várt $9/13 \approx 0{,}7\ldots$ értékhez. Hogy pontosan ott van-e, a differenciálszámítás segítségével ellenőrizhetjük. A maximumban a függvény gradiense lapos. Ezért a maximum likelihood becslést :eqref:`eq_max_like` úgy találhatjuk meg, hogy megkeressük azokat a $\theta$ értékeket, ahol a derivált nulla, majd kiválasztjuk azt, amelyik a legnagyobb valószínűséget adja. Számítsuk ki:

$$
\begin{aligned}
0 & = \frac{d}{d\theta} P(X \mid \theta) \\
& = \frac{d}{d\theta} \theta^9(1-\theta)^4 \\
& = 9\theta^8(1-\theta)^4 - 4\theta^9(1-\theta)^3 \\
& = \theta^8(1-\theta)^3(9-13\theta).
\end{aligned}
$$

Ennek három megoldása van: $0$, $1$ és $9/13$. Az első kettő egyértelműen minimum, nem maximum, hiszen $0$ valószínűséget rendelnek a sorozathoz. Az utolsó érték *nem* rendel nulla valószínűséget a sorozathoz, tehát ez a maximum likelihood becslés: $\hat \theta = 9/13$.

## Numerikus optimalizálás és a negatív log-likelihood

Az előző példa szemléletes, de mi van akkor, ha milliárdnyi paraméterünk és adatpéldánk van?

Először is vegyük észre, hogy ha feltételezzük az összes adatpélda függetlenségét, a likelihood-et már nem lehet közvetlenül kiszámítani, mivel az sok valószínűség szorzata. Minden valószínűség $[0,1]$-ben van, jellemzően kb. $1/2$ értékkel, és $(1/2)^{1000000000}$ szorzata jóval a gépi pontosság alatt van — nem dolgozhatunk ezzel közvetlenül.

Viszont az emlékezetünkbe idézve, hogy a logaritmus szorzatokat összegekre cserél:

$$
\log((1/2)^{1000000000}) = 1000000000\cdot\log(1/2) \approx -301029995{,}6\ldots
$$

Ez az érték még egyetlen $32$-bites egyszeres pontosságú lebegőpontos számban is elfér. Ezért a *log-likelihood*-del kell dolgoznunk:

$$
\log(P(X \mid \boldsymbol{\theta})).
$$

Mivel az $x \mapsto \log(x)$ függvény monoton növekvő, a likelihood maximalizálása ugyanaz, mint a log-likelihood maximalizálása. Ezt a gondolatmenetet a :numref:`sec_naive_bayes` részben a naiv Bayes-osztályozó kapcsán láthatjuk alkalmazni.

Veszteségfüggvényekkel dolgozva általában minimalizálni szeretnénk a veszteséget. A maximum likelihood minimalizálási feladattá alakítható a $-\log(P(X \mid \boldsymbol{\theta}))$ negatív log-likelihood segítségével.

Ennek szemléltetéséhez vegyük újra a pénzfeldobásos példát, és tegyük fel, hogy nem ismerjük a zárt formájú megoldást. Kiszámíthatjuk:

$$
-\log(P(X \mid \boldsymbol{\theta})) = -\log(\theta^{n_H}(1-\theta)^{n_T}) = -(n_H\log(\theta) + n_T\log(1-\theta)).
$$

Ez kódba írható és szabadon optimalizálható akár milliárdnyi pénzfeldobásra is.

```{.python .input}
#@tab mxnet
# Adataink előkészítése
n_H = 8675309
n_T = 256245

# Paraméterünk inicializálása
theta = np.array(0.5)
theta.attach_grad()

# Gradienscsökkentés végrehajtása
lr = 1e-9
for iter in range(100):
    with autograd.record():
        loss = -(n_H * np.log(theta) + n_T * np.log(1 - theta))
    loss.backward()
    theta -= lr * theta.grad

# Eredmény ellenőrzése
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab pytorch
# Adataink előkészítése
n_H = 8675309
n_T = 256245

# Paraméterünk inicializálása
theta = torch.tensor(0.5, requires_grad=True)

# Gradienscsökkentés végrehajtása
lr = 1e-9
for iter in range(100):
    loss = -(n_H * torch.log(theta) + n_T * torch.log(1 - theta))
    loss.backward()
    with torch.no_grad():
        theta -= lr * theta.grad
    theta.grad.zero_()

# Eredmény ellenőrzése
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab tensorflow
# Adataink előkészítése
n_H = 8675309
n_T = 256245

# Paraméterünk inicializálása
theta = tf.Variable(tf.constant(0.5))

# Gradienscsökkentés végrehajtása
lr = 1e-9
for iter in range(100):
    with tf.GradientTape() as t:
        loss = -(n_H * tf.math.log(theta) + n_T * tf.math.log(1 - theta))
    theta.assign_sub(lr * t.gradient(loss, theta))

# Eredmény ellenőrzése
theta, n_H / (n_H + n_T)
```

A negatív log-likelihood használatának azonban nem csupán numerikus kényelmi okai vannak — több más indok is szól mellette.

A log-likelihood alkalmazásának második oka a differenciálszámítási szabályok egyszerűsödése. Ahogy fentebb tárgyaltuk, a függetlenségi feltételezések miatt a gépi tanulásban előforduló legtöbb valószínűség egyedi valószínűségek szorzata:

$$
P(X\mid\boldsymbol{\theta}) = p(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta})\cdots p(x_n\mid\boldsymbol{\theta}).
$$

Ha közvetlenül a szorzatszabályt alkalmazzuk a derivált kiszámításához:

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{\theta}} P(X\mid\boldsymbol{\theta}) & = \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right)\cdot P(x_2\mid\boldsymbol{\theta})\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_2\mid\boldsymbol{\theta})\right)\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot P(x_2\mid\boldsymbol{\theta}) \cdots \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
\end{aligned}
$$

Ehhez $n(n-1)$ szorzás és $(n-1)$ összeadás szükséges, vagyis a bemenetek méretéhez képest négyzetes időigényű! Az összetagok ügyes csoportosításával ez lineárisra csökkenthető, de ez némi leleményességet igényel. A negatív log-likelihood esetében viszont:

$$
-\log\left(P(X\mid\boldsymbol{\theta})\right) = -\log(P(x_1\mid\boldsymbol{\theta})) - \log(P(x_2\mid\boldsymbol{\theta})) \cdots - \log(P(x_n\mid\boldsymbol{\theta})),
$$

amelyből:

$$
- \frac{\partial}{\partial \boldsymbol{\theta}} \log\left(P(X\mid\boldsymbol{\theta})\right) = \frac{1}{P(x_1\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right) + \cdots + \frac{1}{P(x_n\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
$$

Ez csupán $n$ osztást és $n-1$ összeadást igényel, tehát a bemenetekben lineáris időigényű.

A negatív log-likelihood vizsgálatának harmadik és egyben utolsó oka az információelmélethez fűződő kapcsolat, amelyet részletesen a :numref:`sec_information_theory` részben tárgyalunk. Ez egy szigorú matematikai elmélet, amely módszert ad egy véletlen változó információtartalmának vagy véletlenszerűségének mérésére. Az elmélet központi fogalma az entrópia:

$$
H(p) = -\sum_{i} p_i \log_2(p_i),
$$

amely egy forrás véletlenszerűségét méri. Ez nem más, mint az átlagos $-\log$ valószínűség, ezért ha a negatív log-likelihood-et elosztjuk az adatpéldák számával, az entrópiával rokon, keresztentrópiának nevezett mennyiséget kapjuk. Ez az elméleti összefüggés önmagában is elégséges indok arra, hogy a modell teljesítményének mérésére az adathalmaz feletti átlagos negatív log-likelihood-et használjuk.

## Maximális valószínűség folytonos változók esetén

Mindeddig feltételeztük, hogy diszkrét véletlen változókkal dolgozunk — de mi van, ha folytonos változókkal szeretnénk dolgozni?

Röviden: semmi sem változik, csupán a valószínűségeket mindenhol valószínűségsűrűséggel helyettesítjük. Mivel a sűrűséget kisbetűs $p$-vel jelöljük, ez azt jelenti, hogy például most azt írjuk:

$$
-\log\left(p(X\mid\boldsymbol{\theta})\right) = -\log(p(x_1\mid\boldsymbol{\theta})) - \log(p(x_2\mid\boldsymbol{\theta})) \cdots - \log(p(x_n\mid\boldsymbol{\theta})) = -\sum_i \log(p(x_i \mid \theta)).
$$

Felmerülhet a kérdés: „Miért helyes ez?" Elvégre a sűrűségek bevezetésének éppen az volt az oka, hogy egyes kimenetelekre vonatkozó valószínűségek nullák lesznek — és nem nulla-e ekkor az adataink generálásának valószínűsége bármely paraméterhalmaz esetén?

Valóban, ez így van, és az, hogy miért térhetünk át sűrűségekre, egy olyan gondolatmenet eredménye, amely nyomon követi, mi történik az epszilonokkal.

Fogalmazzuk meg újra a célunkat. Tegyük fel, hogy folytonos véletlen változók esetén nem a pontos értékek eltalálásának valószínűségét akarjuk maximalizálni, hanem azt, hogy valamely $\epsilon$ sugarú intervallumon belülre essen. Az egyszerűség kedvéért feltesszük, hogy adataink azonos eloszlású $X_1, \ldots, X_N$ véletlen változók $x_1, \ldots, x_N$ ismételt megfigyelései. Ahogy korábban láttuk, ez felírható mint:

$$
\begin{aligned}
&P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta}) \\
\approx &\epsilon^Np(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta}) \cdots p(x_n\mid\boldsymbol{\theta}).
\end{aligned}
$$

Ha ennek negatív logaritmusát vesszük:

$$
\begin{aligned}
&-\log(P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta})) \\
\approx & -N\log(\epsilon) - \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
\end{aligned}
$$

Ha megvizsgáljuk ezt a kifejezést, az $\epsilon$ kizárólag a $-N\log(\epsilon)$ additív konstansban szerepel. Ez egyáltalán nem függ $\boldsymbol{\theta}$ paraméterektől, így $\boldsymbol{\theta}$ optimális megválasztása nem függ $\epsilon$ megválasztásától! Akár négy, akár négyszáz tizedesjegy pontosságot követelünk, $\boldsymbol{\theta}$ legjobb megválasztása ugyanaz marad — ezért az epszilont szabadon elhagyhatjuk, és azt optimalizáljuk, ami megmarad:

$$
- \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
$$

Így látható, hogy a maximum likelihood szemlélete folytonos véletlen változókra éppoly könnyedén alkalmazható, mint diszkrétekre, pusztán a valószínűségeket valószínűségsűrűségekkel helyettesítve.

## Összefoglalás
* A maximum likelihood elve azt mondja ki, hogy egy adott adathalmazhoz legjobban illeszkedő modell az, amelyik a legnagyobb valószínűséggel generálja az adatokat.
* Sok esetben a negatív log-likelihood-del dolgoznak számos okból kifolyólag: numerikus stabilitás, szorzatok összegekké alakítása (ami egyszerűsíti a gradiens számítást), valamint az információelmélethez fűződő elméleti kapcsolat.
* Bár a diszkrét esetben a legkönnyebb motiválni, a folytonos esetre is szabadon általánosítható az adatpontokhoz rendelt valószínűségsűrűség maximalizálásával.

## Feladatok
1. Tegyük fel, hogy tudjuk, egy nemnegatív véletlen változó sűrűsége $\alpha e^{-\alpha x}$ valamely $\alpha>0$ esetén. Egyetlen $3$ értékű megfigyelést kapunk a véletlen változóból. Mi az $\alpha$ maximum likelihood becslése?
2. Tegyük fel, hogy van egy $\{x_i\}_{i=1}^N$ mintaadathalmazunk, amelyet ismeretlen várható értékű, de $1$ varianciájú normális eloszlásból húztunk. Mi a várható érték maximum likelihood becslése?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/416)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1096)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1097)
:end_tab:
