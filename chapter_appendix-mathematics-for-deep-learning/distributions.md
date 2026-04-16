# Eloszlások
:label:`sec_distributions`

Most, hogy megtanultuk, hogyan kell valószínűséggel dolgozni mind diszkrét, mind folytonos esetben, ismerkedjünk meg néhány gyakran előforduló eloszlással. A gépi tanulás különböző területeitől függően szükségünk lehet jóval több ilyen eloszlás ismeretére, a mélytanulás egyes területein pedig esetleg egyikre sem. Ez azonban egy hasznos alaplista, amellyel érdemes megismerkedni. Először importáljuk a szükséges könyvtárakat.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from math import erf, factorial
import numpy as np
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from math import erf, factorial
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  # A pi érték meghatározása PyTorch-ban
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from math import erf, factorial
import tensorflow as tf
import tensorflow_probability as tfp

tf.pi = tf.acos(tf.zeros(1)) * 2  # A pi érték meghatározása TensorFlow-ban
```

## Bernoulli

Ez a legegyszerűbb általánosan előforduló valószínűségi változó. Ez a valószínűségi változó egy érmefejdobást kódol: $1$ értéket vesz fel $p$ valószínűséggel, és $0$ értéket $1-p$ valószínűséggel. Ha $X$ egy ilyen eloszlású valószínűségi változó, azt így jelöljük:

$$
X \sim \textrm{Bernoulli}(p).
$$

A kumulatív eloszlásfüggvény:

$$F(x) = \begin{cases} 0 & x < 0, \\ 1-p & 0 \le x < 1, \\ 1 & x >= 1 . \end{cases}$$
:eqlabel:`eq_bernoulli_cdf`

Az alábbiakban látható a valószínűségi tömegfüggvény.

```{.python .input}
#@tab all
p = 0.3

d2l.set_figsize()
d2l.plt.stem([0, 1], [1 - p, p], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Most ábrázoljuk a kumulatív eloszlásfüggvényt :eqref:`eq_bernoulli_cdf`.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, tf.constant([F(y) for y in x]), 'x', 'c.d.f.')
```

Ha $X \sim \textrm{Bernoulli}(p)$, akkor:

* $\mu_X = p$,
* $\sigma_X^2 = p(1-p)$.

Tetszőleges alakú tömböt mintavételezhetünk egy Bernoulli-féle valószínűségi változóból az alábbi módon.

```{.python .input}
#@tab mxnet
1*(np.random.rand(10, 10) < p)
```

```{.python .input}
#@tab pytorch
1*(torch.rand(10, 10) < p)
```

```{.python .input}
#@tab tensorflow
tf.cast(tf.random.uniform((10, 10)) < p, dtype=tf.float32)
```

## Diszkrét egyenletes eloszlás

A következő gyakran előforduló valószínűségi változó a diszkrét egyenletes eloszlás. Az itt következő tárgyalásban feltesszük, hogy az $\{1, 2, \ldots, n\}$ egész számokon értelmezett, bár bármely más értékkészlet szabadon megválasztható. Az *egyenletes* szó ebben az összefüggésben azt jelenti, hogy minden lehetséges érték egyforma valószínűséggel fordul elő. Az egyes $i \in \{1, 2, 3, \ldots, n\}$ értékek valószínűsége $p_i = \frac{1}{n}$. Az ilyen eloszlású $X$ valószínűségi változót így jelöljük:

$$
X \sim U(n).
$$

A kumulatív eloszlásfüggvény:

$$F(x) = \begin{cases} 0 & x < 1, \\ \frac{k}{n} & k \le x < k+1 \textrm{ with } 1 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_discrete_uniform_cdf`

Először ábrázoljuk a valószínűségi tömegfüggvényt.

```{.python .input}
#@tab all
n = 5

d2l.plt.stem([i+1 for i in range(n)], n*[1 / n], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Most ábrázoljuk a kumulatív eloszlásfüggvényt :eqref:`eq_discrete_uniform_cdf`.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else np.floor(x) / n

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else torch.floor(x) / n

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else tf.floor(x) / n

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

Ha $X \sim U(n)$, akkor:

* $\mu_X = \frac{1+n}{2}$,
* $\sigma_X^2 = \frac{n^2-1}{12}$.

Tetszőleges alakú tömböt mintavételezhetünk egy diszkrét egyenletes valószínűségi változóból az alábbi módon.

```{.python .input}
#@tab mxnet
np.random.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.uniform((10, 10), 1, n, dtype=tf.int32)
```

## Folytonos egyenletes eloszlás

Következőként tárgyaljuk a folytonos egyenletes eloszlást. Ennek a valószínűségi változónak az alapötlete az, hogy ha a diszkrét egyenletes eloszlásban szereplő $n$-t növeljük, majd az $[a, b]$ intervallumra skálázzuk, egy olyan folytonos valószínűségi változóhoz jutunk, amely egyforma valószínűséggel vesz fel bármilyen értéket az $[a, b]$ intervallumon. Ezt az eloszlást így jelöljük:

$$
X \sim U(a, b).
$$

A valószínűségi sűrűségfüggvény:

$$p(x) = \begin{cases} \frac{1}{b-a} & x \in [a, b], \\ 0 & x \not\in [a, b].\end{cases}$$
:eqlabel:`eq_cont_uniform_pdf`

A kumulatív eloszlásfüggvény:

$$F(x) = \begin{cases} 0 & x < a, \\ \frac{x-a}{b-a} & x \in [a, b], \\ 1 & x >= b . \end{cases}$$
:eqlabel:`eq_cont_uniform_cdf`

Először ábrázoljuk a valószínűségi sűrűségfüggvényt :eqref:`eq_cont_uniform_pdf`.

```{.python .input}
#@tab mxnet
a, b = 1, 3

x = np.arange(0, 4, 0.01)
p = (x > a)*(x < b)/(b - a)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
a, b = 1, 3

x = torch.arange(0, 4, 0.01)
p = (x > a).type(torch.float32)*(x < b).type(torch.float32)/(b-a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
a, b = 1, 3

x = tf.range(0, 4, 0.01)
p = tf.cast(x > a, tf.float32) * tf.cast(x < b, tf.float32) / (b - a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

Most ábrázoljuk a kumulatív eloszlásfüggvényt :eqref:`eq_cont_uniform_cdf`.

```{.python .input}
#@tab mxnet
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

Ha $X \sim U(a, b)$, akkor:

* $\mu_X = \frac{a+b}{2}$,
* $\sigma_X^2 = \frac{(b-a)^2}{12}$.

Tetszőleges alakú tömböt mintavételezhetünk egy egyenletes valószínűségi változóból az alábbi módon. Megjegyezzük, hogy alapértelmezés szerint $U(0,1)$-ből vesz mintát, ezért ha más tartomány kell, skálázni kell.

```{.python .input}
#@tab mxnet
(b - a) * np.random.rand(10, 10) + a
```

```{.python .input}
#@tab pytorch
(b - a) * torch.rand(10, 10) + a
```

```{.python .input}
#@tab tensorflow
(b - a) * tf.random.uniform((10, 10)) + a
```

## Binomiális eloszlás

Tegyük bonyolultabbá a dolgot, és vizsgáljuk meg a *binomiális* valószínűségi változót. Ez a valószínűségi változó abból ered, hogy $n$ egymástól független kísérletet végzünk, amelyek mindegyike $p$ valószínűséggel sikeres, és azt kérdezzük, hány sikert várhatunk.

Fogalmazzuk meg ezt matematikailag. Minden kísérlet egy-egy független valószínűségi változó $X_i$, ahol $1$-et kódolunk a sikernek és $0$-t a kudarcnak. Mivel minden egyes kísérlet egy független érmefejdobás, amely $p$ valószínűséggel sikeres, azt mondhatjuk, hogy $X_i \sim \textrm{Bernoulli}(p)$. A binomiális valószínűségi változó ekkor:

$$
X = \sum_{i=1}^n X_i.
$$

Ebben az esetben azt írjuk:

$$
X \sim \textrm{Binomial}(n, p).
$$

A kumulatív eloszlásfüggvény levezetéséhez észre kell vennünk, hogy pontosan $k$ siker $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ különböző módon következhet be, és minden egyes eset $p^k(1-p)^{n-k}$ valószínűséggel fordul elő. A kumulatív eloszlásfüggvény tehát:

$$F(x) = \begin{cases} 0 & x < 0, \\ \sum_{m \le k} \binom{n}{m} p^m(1-p)^{n-m}  & k \le x < k+1 \textrm{ with } 0 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_binomial_cdf`

Először ábrázoljuk a valószínűségi tömegfüggvényt.

```{.python .input}
#@tab mxnet
n, p = 10, 0.2

# Binomiális együttható kiszámítása
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = np.array([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
n, p = 10, 0.2

# Binomiális együttható kiszámítása
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = d2l.tensor([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
n, p = 10, 0.2

# Binomiális együttható kiszámítása
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = tf.constant([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Most ábrázoljuk a kumulatív eloszlásfüggvényt :eqref:`eq_binomial_cdf`.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 11, 0.01)
cmf = np.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 11, 0.01)
cmf = torch.cumsum(pmf, dim=0)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 11, 0.01)
cmf = tf.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

Ha $X \sim \textrm{Binomial}(n, p)$, akkor:

* $\mu_X = np$,
* $\sigma_X^2 = np(1-p)$.

Ez a várható érték $n$ Bernoulli-változó összegére vonatkozó linearitásából, valamint abból következik, hogy független valószínűségi változók összegének varianciája egyenlő a varianciák összegével. A mintavételezés a következőképpen végezhető.

```{.python .input}
#@tab mxnet
np.random.binomial(n, p, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.binomial.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

## Poisson-eloszlás
Végezzünk most egy gondolatkísérletet. Egy buszmegállóban állunk, és tudni szeretnénk, hány busz érkezik a következő percben. Kezdjük azzal, hogy megvizsgáljuk az $X^{(1)} \sim \textrm{Bernoulli}(p)$ változót, amely egyszerűen annak valószínűsége, hogy az egyperces ablakban megérkezik egy busz. Városi központoktól távolabb eső buszmegállókban ez meglehetősen jó közelítés lehet: ott valószínűleg egypercenténként legfeljebb egy busz érkezik.

Ha azonban forgalmas területen vagyunk, lehetséges, sőt valószínű, hogy két busz is megérkezik. Ezt úgy modellezhetjük, hogy a változónkat két részre bontjuk: az első 30 másodpercre és a második 30 másodpercre. Ekkor felírhatjuk:

$$
X^{(2)} \sim X^{(2)}_1 + X^{(2)}_2,
$$

ahol $X^{(2)}$ a teljes összeg, és $X^{(2)}_i \sim \textrm{Bernoulli}(p/2)$. A teljes eloszlás ekkor $X^{(2)} \sim \textrm{Binomial}(2, p/2)$.

Miért álljunk meg itt? Folytassuk az egész percet $n$ részre osztva. A fenti érveléssel analóg módon:

$$X^{(n)} \sim \textrm{Binomial}(n, p/n).$$
:eqlabel:`eq_eq_poisson_approx`

Vizsgáljuk meg ezeket a valószínűségi változókat. Az előző szakasz alapján tudjuk, hogy :eqref:`eq_eq_poisson_approx` várható értéke $\mu_{X^{(n)}} = n(p/n) = p$, varianciája pedig $\sigma_{X^{(n)}}^2 = n(p/n)(1-(p/n)) = p(1-p/n)$. Ha $n \rightarrow \infty$, ezek az értékek $\mu_{X^{(\infty)}} = p$ és $\sigma_{X^{(\infty)}}^2 = p$ határértékekhez tartanak. Ez arra utal, hogy *létezhet* valamilyen valószínűségi változó, amelyet ebben a végtelen felosztási határesetben definiálhatunk.

Ez nem meglepő, hiszen a valós világban egyszerűen megszámlálhatjuk a buszok érkezési számát; mégis megnyugtató látni, hogy matematikai modellünk jól definiált. Ez az érvelés formalizálható a *ritka események törvényeként*.

Az érvelést gondosan végigkövetve a következő modellhez jutunk. Azt mondjuk, hogy $X \sim \textrm{Poisson}(\lambda)$, ha egy olyan valószínűségi változóról van szó, amely a $\{0,1,2, \ldots\}$ értékeket veszi fel a következő valószínűséggel:

$$p_k = \frac{\lambda^ke^{-\lambda}}{k!}.$$
:eqlabel:`eq_poisson_mass`

A $\lambda > 0$ értéket *rátának* (vagy *alakparaméternek*) nevezzük, és az egy időegység alatt várható átlagos eseményszámot jelöli.

A valószínűségi tömegfüggvény összegzésével megkapjuk a kumulatív eloszlásfüggvényt.

$$F(x) = \begin{cases} 0 & x < 0, \\ e^{-\lambda}\sum_{m = 0}^k \frac{\lambda^m}{m!} & k \le x < k+1 \textrm{ with } 0 \le k. \end{cases}$$
:eqlabel:`eq_poisson_cdf`

Először ábrázoljuk a valószínűségi tömegfüggvényt :eqref:`eq_poisson_mass`.

```{.python .input}
#@tab mxnet
lam = 5.0

xs = [i for i in range(20)]
pmf = np.array([np.exp(-lam) * lam**k / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
lam = 5.0

xs = [i for i in range(20)]
pmf = torch.tensor([torch.exp(torch.tensor(-lam)) * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
lam = 5.0

xs = [i for i in range(20)]
pmf = tf.constant([tf.exp(tf.constant(-lam)).numpy() * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Most ábrázoljuk a kumulatív eloszlásfüggvényt :eqref:`eq_poisson_cdf`.

```{.python .input}
#@tab mxnet
x = np.arange(-1, 21, 0.01)
cmf = np.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 21, 0.01)
cmf = torch.cumsum(pmf, dim=0)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 21, 0.01)
cmf = tf.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

Amint fentebb láttuk, a várható érték és a variancia különösen tömören írható. Ha $X \sim \textrm{Poisson}(\lambda)$, akkor:

* $\mu_X = \lambda$,
* $\sigma_X^2 = \lambda$.

A mintavételezés a következőképpen végezhető.

```{.python .input}
#@tab mxnet
np.random.poisson(lam, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.poisson.Poisson(lam)
m.sample((10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Poisson(lam)
m.sample((10, 10))
```

## Gauss-eloszlás
Próbáljunk ki egy másik, de hasonló kísérletet. Tegyük fel, hogy ismét $n$ egymástól független $\textrm{Bernoulli}(p)$ mérést végzünk: $X_i$. Ezek összegének eloszlása $X^{(n)} \sim \textrm{Binomial}(n, p)$. Most ne a Poisson-esethez hasonlóan a határátmenetet vizsgáljuk $n \rightarrow \infty$-re és $p \rightarrow 0$-ra, hanem rögzítsük $p$-t, majd küldjük $n \rightarrow \infty$-be. Ekkor $\mu_{X^{(n)}} = np \rightarrow \infty$ és $\sigma_{X^{(n)}}^2 = np(1-p) \rightarrow \infty$, tehát nincs okunk azt gondolni, hogy ez a határátmenet jól definiált lesz.

Mégsem veszett el minden remény! Tegyük a várható értéket és a varianciát jól viselkedővé a következő definícióval:

$$
Y^{(n)} = \frac{X^{(n)} - \mu_{X^{(n)}}}{\sigma_{X^{(n)}}}.
$$

Belátható, hogy ennek várható értéke nulla, varianciája egy, így valószínűsíthető, hogy valamely határeloszláshoz tart. Ha ábrázoljuk ezeket az eloszlásokat, még inkább meggyőződhettünk erről.

```{.python .input}
#@tab mxnet
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = np.array([p**i * (1-p)**(n-i) * binom(n, i) for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/np.sqrt(n*p*(1 - p)) for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = torch.tensor([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/torch.sqrt(torch.tensor(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = tf.constant([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/tf.sqrt(tf.constant(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

Érdemes megjegyezni: a Poisson-esethez képest most a szórással osztunk, ami azt jelenti, hogy a lehetséges kimeneteleket egyre kisebb és kisebb tartományba szorítjuk össze. Ez arra utal, hogy a határeloszlás már nem diszkrét, hanem folytonos lesz.

Annak levezetése, hogy mi is következik be, meghaladja e dokumentum kereteit, azonban a *centrális határeloszlás-tétel* kimondja, hogy $n \rightarrow \infty$ esetén ez a Gauss-eloszláshoz (más nevén normális eloszláshoz) tart. Pontosabban, bármely $a, b$ esetén:

$$
\lim_{n \rightarrow \infty} P(Y^{(n)} \in [a, b]) = P(\mathcal{N}(0,1) \in [a, b]),
$$

ahol azt mondjuk, hogy egy valószínűségi változó normális eloszlású $\mu$ várható értékkel és $\sigma^2$ varianciával – jelölése $X \sim \mathcal{N}(\mu, \sigma^2)$ –, ha $X$ sűrűségfüggvénye:

$$p_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}.$$
:eqlabel:`eq_gaussian_pdf`

Először ábrázoljuk a valószínűségi sűrűségfüggvényt :eqref:`eq_gaussian_pdf`.

```{.python .input}
#@tab mxnet
mu, sigma = 0, 1

x = np.arange(-3, 3, 0.01)
p = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
mu, sigma = 0, 1

x = torch.arange(-3, 3, 0.01)
p = 1 / torch.sqrt(2 * torch.pi * sigma**2) * torch.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
mu, sigma = 0, 1

x = tf.range(-3, 3, 0.01)
p = 1 / tf.sqrt(2 * tf.pi * sigma**2) * tf.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

Most ábrázoljuk a kumulatív eloszlásfüggvényt. Ez a függelék kereteit meghaladja, de a Gauss-féle kumulatív eloszlásfüggvénynek nincs zárt alakú képlete elemi függvények segítségével. Az `erf` függvényt fogjuk használni, amellyel ez az integrál numerikusan kiszámítható.

```{.python .input}
#@tab mxnet
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * np.sqrt(2)))) / 2.0

d2l.plot(x, np.array([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * torch.sqrt(d2l.tensor(2.))))) / 2.0

d2l.plot(x, torch.tensor([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * tf.sqrt(tf.constant(2.))))) / 2.0

d2l.plot(x, [phi(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

Az éles szemű olvasók felismerhetnek néhány tagot. Valóban, ezzel az integrállal :numref:`sec_integral_calculus`-ban találkoztunk. Pontosan erre a számításra van szükségünk annak igazolásához, hogy ez a $p_X(x)$ összfelülete egy, tehát érvényes sűrűségfüggvény.

Az érmefejdobással való munka rövidítette a számításokat, de semmi alapvető nem kötötte hozzá a választást. Valójában, ha bármely egymástól független, azonos eloszlású valószínűségi változók $X_i$ gyűjteményét vesszük, és képezzük az összeget:

$$
X^{(N)} = \sum_{i=1}^N X_i.
$$

Ekkor

$$
\frac{X^{(N)} - \mu_{X^{(N)}}}{\sigma_{X^{(N)}}}
$$

közelítőleg Gauss-eloszlású lesz. Ehhez néhány további feltétel szükséges, leggyakrabban az $E[X^4] < \infty$ teljesülése, de az elv egyértelmű.

A centrális határeloszlás-tétel magyarázza, miért alapvető a Gauss-eloszlás a valószínűségszámításban, a statisztikában és a gépi tanulásban. Valahányszor kimondhatjuk, hogy valami, amit mértünk, sok kis független hozzájárulás összege, feltételezhetjük, hogy a mért mennyiség közel Gauss-eloszlású lesz.

A Gauss-eloszlásnak számos lenyűgöző tulajdonsága van; ezek közül még egyet szeretnénk itt megemlíteni. A Gauss-eloszlás az úgynevezett *maximális entrópiájú eloszlás*. Az entrópiát mélyebben :numref:`sec_information_theory`-ban tárgyaljuk, de most elegendő annyit tudni, hogy a véletlenszerűség mértéke. Szigorú matematikai értelemben a Gauss-eloszlásra úgy tekinthetünk, mint a rögzített várható értékű és varianciájú valószínűségi változók *legvéletlenszerűbb* választására. Tehát ha tudjuk, hogy valószínűségi változónknak van valamilyen várható értéke és varianciája, a Gauss-eloszlás bizonyos értelemben a legkonzervatívabb eloszlásválasztás, amelyet tehetünk.

A szakasz lezárásaként jegyezzük meg, hogy ha $X \sim \mathcal{N}(\mu, \sigma^2)$, akkor:

* $\mu_X = \mu$,
* $\sigma_X^2 = \sigma^2$.

A Gauss-eloszlásból (vagy standard normális eloszlásból) az alábbiak szerint vehetünk mintát.

```{.python .input}
#@tab mxnet
np.random.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.normal((10, 10), mu, sigma)
```

## Exponenciális eloszláscsalád
:label:`subsec_exponential_family`

A fent felsorolt összes eloszlásnak van egy közös tulajdonsága: mindegyik az úgynevezett *exponenciális eloszláscsaládhoz* tartozik. Az exponenciális eloszláscsalád olyan eloszlások halmaza, amelyek sűrűségfüggvénye a következő alakban írható:

$$p(\mathbf{x} \mid \boldsymbol{\eta}) = h(\mathbf{x}) \cdot \exp \left( \boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) - A(\boldsymbol{\eta}) \right)$$
:eqlabel:`eq_exp_pdf`

Mivel ez a definíció némileg finomságokat rejt, vizsgáljuk meg alaposabban.

Először, $h(\mathbf{x})$ az úgynevezett *alámérték* vagy *alapmérték*. Ez tekinthető az eredeti mértékválasztásnak, amelyet az exponenciális súlyozással módosítunk.

Másodszor, adott a $\boldsymbol{\eta} = (\eta_1, \eta_2, ..., \eta_l) \in \mathbb{R}^l$ vektor, amelyet *természetes paramétereknek* vagy *kanonikus paramétereknek* nevezünk. Ezek határozzák meg, hogyan módosul az alapmérték. A természetes paraméterek az új mértékbe úgy kerülnek be, hogy skaláris szorzatot veszünk e paraméterek és az $\mathbf{x}= (x_1, x_2, ..., x_n) \in \mathbb{R}^n$ valamilyen $T(\cdot)$ függvénye között, majd ezt hatványozzuk. A $T(\mathbf{x})= (T_1(\mathbf{x}), T_2(\mathbf{x}), ..., T_l(\mathbf{x}))$ vektort $\boldsymbol{\eta}$ *elégséges statisztikájának* nevezzük. Ezt a nevet azért viseli, mert a $T(\mathbf{x})$ által hordozott információ elegendő a valószínűségi sűrűség kiszámításához, és az $\mathbf{x}$ mintából semmilyen más információra nincs szükség.

Harmadszor, adott $A(\boldsymbol{\eta})$, amelyet *kumuláns-függvénynek* nevezünk, és azt biztosítja, hogy a fenti :eqref:`eq_exp_pdf` eloszlás egy-re integrálódjon, vagyis:

$$A(\boldsymbol{\eta})  = \log \left[\int h(\mathbf{x}) \cdot \exp
\left(\boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) \right) d\mathbf{x} \right].$$

Szemléltetésül vizsgáljuk meg a Gauss-eloszlást. Feltéve, hogy $\mathbf{x}$ egyváltozós, sűrűségfüggvénye:

$$
\begin{aligned}
p(x \mid \mu, \sigma) &= \frac{1}{\sqrt{2 \pi \sigma^2}} \cdot \exp 
\left\{ \frac{-(x-\mu)^2}{2 \sigma^2} \right\} \\
&= \frac{1}{\sqrt{2 \pi}} \cdot \exp \left\{ \frac{\mu}{\sigma^2}x
-\frac{1}{2 \sigma^2} x^2 - \left( \frac{1}{2 \sigma^2} \mu^2
+\log(\sigma) \right) \right\}.
\end{aligned}
$$

Ez megfelel az exponenciális eloszláscsalád definíciójának, ahol:

* *alapmérték*: $h(x) = \frac{1}{\sqrt{2 \pi}}$,
* *természetes paraméterek*: $\boldsymbol{\eta} = \begin{bmatrix} \eta_1 \\ \eta_2
\end{bmatrix} = \begin{bmatrix} \frac{\mu}{\sigma^2} \\
\frac{1}{2 \sigma^2} \end{bmatrix}$,
* *elégséges statisztika*: $T(x) = \begin{bmatrix}x\\-x^2\end{bmatrix}$, és
* *kumuláns-függvény*: $A({\boldsymbol\eta}) = \frac{1}{2 \sigma^2} \mu^2 + \log(\sigma)
= \frac{\eta_1^2}{4 \eta_2} - \frac{1}{2}\log(2 \eta_2)$.

Érdemes megjegyezni, hogy a fenti tagok konkrét megválasztása némiképp önkényes. A lényeges tulajdonság az, hogy az eloszlás kifejezhető ebben az alakban, nem maga a konkrét forma.

Amint arra :numref:`subsec_softmax_and_derivatives`-ban utalunk, egy széles körben alkalmazott technika az, hogy feltesszük: a végső kimenet $\mathbf{y}$ exponenciális eloszláscsaládba tartozó eloszlást követ. Az exponenciális eloszláscsalád a gépi tanulásban rendkívül elterjedt és hatékony eloszláscsalád.


## Összefoglalás
* A Bernoulli-féle valószínűségi változók igen/nem kimenetelű események modellezésére alkalmasak.
* A diszkrét egyenletes eloszlások véges lehetséges értékkészletből való egyenletes választást modelleznek.
* A folytonos egyenletes eloszlások egy intervallumból való választást modelleznek.
* A binomiális eloszlások Bernoulli-féle valószínűségi változók sorozatát modellezik, és a sikerek számát számolják meg.
* A Poisson-féle valószínűségi változók ritka események bekövetkezését modellezik.
* A Gauss-féle valószínűségi változók nagyszámú független valószínűségi változó összegének eredményét modellezik.
* A fent felsorolt összes eloszlás az exponenciális eloszláscsaládhoz tartozik.

## Feladatok

1. Mi a szórása annak a valószínűségi változónak, amely két független binomiális valószínűségi változó, $X, Y \sim \textrm{Binomial}(16, 1/2)$ különbségeként áll elő: $X-Y$?
2. Ha adott egy $X \sim \textrm{Poisson}(\lambda)$ Poisson-féle valószínűségi változó, és megvizsgáljuk $(X - \lambda)/\sqrt{\lambda}$ viselkedését $\lambda \rightarrow \infty$ esetén, megmutatható, hogy ez közelítőleg Gauss-eloszláshoz tart. Miért van ez így?
3. Mi az valószínűségi tömegfüggvénye két, $n$ elemű diszkrét egyenletes eloszlású valószínűségi változó összegének?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/417)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1098)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1099)
:end_tab:
