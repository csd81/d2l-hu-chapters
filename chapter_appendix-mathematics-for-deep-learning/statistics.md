# Statisztika
:label:`sec_statistics`

Kétségtelen, hogy a mélytanulás legjobb gyakorlóivá váláshoz elengedhetetlen a csúcstechnológiás, nagy pontosságú modellek tanításának képessége. Ugyanakkor gyakran nem egyértelmű, mikor számít egy javulás valóban szignifikánsnak, és mikor csupán a tanítási folyamat véletlen ingadozásának eredménye. Ahhoz, hogy a becsült értékek bizonytalanságáról érdemben tudjunk beszélni, meg kell ismernünk néhány statisztikai alapfogalmat.

A *statisztika* legkorábbi előzményei Al-Kindi arab tudóshoz nyúlnak vissza a $9.$ században, aki részletes leírást adott arról, hogyan lehet statisztikát és frekvenciaelemzést alkalmazni titkosított üzenetek megfejtéséhez. Nyolcszáz évvel később a modern statisztika Németországban jelent meg az 1700-as években, amikor a kutatók a demográfiai és gazdasági adatok gyűjtésére és elemzésére összpontosítottak. Ma a statisztika az a tudományterület, amely az adatok gyűjtésével, feldolgozásával, elemzésével, értelmezésével és vizualizációjával foglalkozik. Sőt, a statisztika alapelméletét széles körben alkalmazzák az akadémiai, ipari és kormányzati kutatásokban.

Pontosabban fogalmazva, a statisztika *leíró statisztikára* és *statisztikai következtetésre* bontható. Az előbbi a megfigyelt adatok gyűjteményének – az úgynevezett *mintának* – összegzésére és szemléltetésére összpontosít. A mintát egy *populációból* veszik, amely kísérletünk érdeklődési körébe tartozó összes hasonló egyedet, elemet vagy eseményt jelöli. A leíró statisztikával ellentétben a *statisztikai következtetés* az adott *mintákból* levezeti a populáció jellemzőit, azon feltételezés alapján, hogy a mintaeloszlás bizonyos mértékig képes leírni a populációeloszlást.

Felmerülhet a kérdés: „Mi az alapvető különbség a gépi tanulás és a statisztika között?" Alapvetően a statisztika a következtetési problémákra összpontosít. Ilyen jellegű problémák közé tartozik a változók közötti kapcsolat modellezése – például az ok-okozati következtetés –, valamint a modellparaméterek statisztikai szignifikanciájának tesztelése, mint például az A/B-tesztelés. Ezzel szemben a gépi tanulás a pontos előrejelzések elvégzésére helyezi a hangsúlyt, anélkül hogy minden egyes paraméter funkcióját explicit módon programoznánk vagy megértenénk.

Ebben a fejezetben a statisztikai következtetés három típusát mutatjuk be: a becslők kiértékelését és összehasonlítását, a hipotézisvizsgálatot, valamint a konfidenciaintervallumok meghatározását. Ezek a módszerek segítenek egy adott populáció jellemzőinek, vagyis az igazi $\theta$ paraméternek a következtetéses meghatározásában. Az egyszerűség kedvéért feltételezzük, hogy egy adott populáció igazi paramétere, $\theta$, skaláris érték. A $\theta$ vektorra vagy tenzorra való kiterjesztés egyértelmű, ezért azt a tárgyalásban elhagyjuk.



## Becslők kiértékelése és összehasonlítása

A statisztikában a *becslő* az adott minták egy olyan függvénye, amelyet az igazi $\theta$ paraméter becslésére használnak. A $\theta$ becslését az $\{x_1, x_2, \ldots, x_n\}$ minták megfigyelése után $\hat{\theta}_n = \hat{f}(x_1, \ldots, x_n)$ formában jelöljük.

Már láttunk egyszerű becslőpéldákat a :numref:`sec_maximum_likelihood` fejezetben. Ha egy Bernoulli-féle valószínűségi változóból több mintánk van, akkor az annak valószínűségére vonatkozó maximum-likelihood-becslés, hogy a valószínűségi változó értéke egy, úgy kapható meg, hogy megszámoljuk a megfigyelt egyek számát, és elosztjuk a minták összesített számával. Hasonlóan, egy feladat arra kért, hogy mutassuk meg: egy Gauss-eloszlás várható értékének maximum-likelihood-becslése adott számú minta esetén az összes minta átlagaként adódik. Ezek a becslők szinte soha nem adják meg a paraméter tényleges értékét, de ideális esetben, nagy mintaszámnál a becslés közel lesz hozzá.

Példaként az alábbiakban bemutatjuk egy nulla várható értékű és egységnyi varianciájú Gauss-féle valószínűségi változó valódi sűrűségfüggvényét, valamint az abból vett minták gyűjteményét. Az $y$ koordinátát úgy szerkesztettük meg, hogy minden pont látható legyen, és az eredeti sűrűségfüggvénnyel való kapcsolat egyértelműbb legyen.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()

# Mintapontok és y-koordináta létrehozása
epsilon = 0.1
random.seed(8675309)
xs = np.random.normal(loc=0, scale=1, size=(300,))

ys = [np.sum(np.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))
             / np.sqrt(2*np.pi*epsilon**2)) / len(xs) for i in range(len(xs))]

# Az igazi sűrűség kiszámítása
xd = np.arange(np.min(xs), np.max(xs), 0.01)
yd = np.exp(-xd**2/2) / np.sqrt(2 * np.pi)

# Az eredmények ábrázolása
d2l.plot(xd, yd, 'x', 'sűrűség')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=np.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'mintaátlag: {float(np.mean(xs)):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  # Definiáljuk a pí-t a torch-ban

# Mintapontok és y-koordináta létrehozása
epsilon = 0.1
torch.manual_seed(8675309)
xs = torch.randn(size=(300,))

ys = torch.tensor(
    [torch.sum(torch.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))\
               / torch.sqrt(2*torch.pi*epsilon**2)) / len(xs)\
     for i in range(len(xs))])

# Az igazi sűrűség kiszámítása
xd = torch.arange(torch.min(xs), torch.max(xs), 0.01)
yd = torch.exp(-xd**2/2) / torch.sqrt(2 * torch.pi)

# Az eredmények ábrázolása
d2l.plot(xd, yd, 'x', 'sűrűség')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=torch.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'mintaátlag: {float(torch.mean(xs).item()):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

tf.pi = tf.acos(tf.zeros(1)) * 2  # Definiáljuk a pí-t a TensorFlow-ban

# Mintapontok és y-koordináta létrehozása
epsilon = 0.1
xs = tf.random.normal((300,))

ys = tf.constant(
    [(tf.reduce_sum(tf.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2)) \
               / tf.sqrt(2*tf.pi*epsilon**2)) / tf.cast(
        tf.size(xs), dtype=tf.float32)).numpy() \
     for i in range(tf.size(xs))])

# Az igazi sűrűség kiszámítása
xd = tf.range(tf.reduce_min(xs), tf.reduce_max(xs), 0.01)
yd = tf.exp(-xd**2/2) / tf.sqrt(2 * tf.pi)

# Az eredmények ábrázolása
d2l.plot(xd, yd, 'x', 'sűrűség')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=tf.reduce_mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'mintaátlag: {float(tf.reduce_mean(xs).numpy()):.2f}')
d2l.plt.show()
```

Egy $\hat{\theta}_n$ paraméternek számos módon lehet becslőjét kiszámítani. Ebben a fejezetben három általánosan használt módszert mutatunk be a becslők kiértékelésére és összehasonlítására: a négyzetesközép-hibát, a szórást és a statisztikai torzítást.

### Négyzetesközép-hiba

A becslők kiértékelésére talán a legegyszerűbb mérőszám a *négyzetesközép-hiba (MSE)* (más néven $l_2$-veszteség), amely a következőképpen definiálható:

$$\textrm{MSE} (\hat{\theta}_n, \theta) = E[(\hat{\theta}_n - \theta)^2].$$
:eqlabel:`eq_mse_est`

Ez lehetővé teszi, hogy számszerűsítsük az igazi értéktől való átlagos négyzetes eltérést. Az MSE mindig nemnegatív. Ha elolvasta a :numref:`sec_linear_regression` fejezetet, felismeri, hogy ez a legelterjedtebb regressziós veszteségfüggvény. Becslő kiértékelésének mérőszámaként: minél közelebb van az értéke nullához, annál közelebb van a becslő az igazi $\theta$ paraméterhez.


### Statisztikai torzítás

Az MSE természetes mérőszámot biztosít, de könnyen elképzelhetjük, hogy több különböző jelenség is okozhatja nagyságának növekedését. A két alapvetően fontos jelenség: az adathalmazban lévő véletlenszerűségből eredő ingadozás a becslőben, illetve a becslési eljárásból eredő szisztematikus hiba a becslőben.

Először mérjük meg a szisztematikus hibát. Egy $\hat{\theta}_n$ becslő esetén a *statisztikai torzítás* matematikai definíciója a következő:

$$\textrm{bias}(\hat{\theta}_n) = E(\hat{\theta}_n - \theta) = E(\hat{\theta}_n) - \theta.$$
:eqlabel:`eq_bias`

Megjegyezzük, hogy ha $\textrm{bias}(\hat{\theta}_n) = 0$, akkor a $\hat{\theta}_n$ becslő várható értéke egyenlő a paraméter igazi értékével. Ebben az esetben azt mondjuk, hogy $\hat{\theta}_n$ torzítatlan becslő. Általában a torzítatlan becslő jobb, mint a torzított, mivel várható értéke megegyezik az igazi paraméterrel.

Érdemes azonban tudatában lenni annak, hogy a torzított becslőket a gyakorlatban is gyakran alkalmazzák. Vannak esetek, amikor torzítatlan becslők nem léteznek további feltételezések nélkül, vagy kiszámításuk nem megvalósítható. Ez jelentős hiányosságnak tűnhet egy becslőnél, azonban a gyakorlatban előforduló becslők többsége legalább aszimptotikusan torzítatlan abban az értelemben, hogy a torzítás nullához tart, ahogy a rendelkezésre álló minták száma végtelenhez tart: $\lim_{n \rightarrow \infty} \textrm{bias}(\hat{\theta}_n) = 0$.


### Variancia és szórás

Másodszor mérjük meg a becslő véletlenszerűségét. Ahogy a :numref:`sec_random_variables` fejezetből felidézhetjük, a *szórás* (más néven *standard hiba*) a variancia négyzetgyökeként van definiálva. Egy becslő ingadozásának mértékét az adott becslő szórásával vagy varianciájával mérhetjük.

$$\sigma_{\hat{\theta}_n} = \sqrt{\textrm{Var} (\hat{\theta}_n )} = \sqrt{E[(\hat{\theta}_n - E(\hat{\theta}_n))^2]}.$$
:eqlabel:`eq_var_est`

Fontos összevetni a :eqref:`eq_var_est` és a :eqref:`eq_mse_est` összefüggéseket. Ebben az egyenletben nem az igazi populációs $\theta$ értékhez hasonlítunk, hanem $E(\hat{\theta}_n)$-hez, azaz a várható mintaátlaghoz. Így nem azt mérjük, mennyivel tér el a becslő általában az igazi értéktől, hanem magának a becslőnek az ingadozását vizsgáljuk.


### A torzítás–variancia kompromisszum

Intuitívan egyértelmű, hogy ez a két fő összetevő hozzájárul a négyzetesközép-hibához. Ami némiképp meglepő, az az, hogy megmutatható: ez valójában a négyzetesközép-hiba *felbontása* e két összetevőre, plusz egy harmadikra. Vagyis a négyzetesközép-hiba felírható a torzítás négyzetének, a varianciának és a visszafordíthatatlan hibának összegeként.

$$
\begin{aligned}
\textrm{MSE} (\hat{\theta}_n, \theta) &= E[(\hat{\theta}_n - \theta)^2] \\
 &= E[(\hat{\theta}_n)^2] + E[\theta^2] - 2E[\hat{\theta}_n\theta] \\
 &= \textrm{Var} [\hat{\theta}_n] + E[\hat{\theta}_n]^2 + \textrm{Var} [\theta] + E[\theta]^2 - 2E[\hat{\theta}_n]E[\theta] \\
 &= (E[\hat{\theta}_n] - E[\theta])^2 + \textrm{Var} [\hat{\theta}_n] + \textrm{Var} [\theta] \\
 &= (E[\hat{\theta}_n - \theta])^2 + \textrm{Var} [\hat{\theta}_n] + \textrm{Var} [\theta] \\
 &= (\textrm{bias} [\hat{\theta}_n])^2 + \textrm{Var} (\hat{\theta}_n) + \textrm{Var} [\theta].\\
\end{aligned}
$$

A fenti képletre *torzítás–variancia kompromisszumként* hivatkozunk. A négyzetesközép-hiba három hibaforrásra bontható: a nagy torzításból eredő hibára, a nagy varianciából eredő hibára és a visszafordíthatatlan hibára. A torzítási hiba jellemzően egyszerű modellekben (például lineáris regressziós modellekben) fordul elő, amelyek nem képesek kinyerni a jellemzők és a kimenetek közötti magas dimenziójú összefüggéseket. Ha egy modellt nagy torzítási hiba jellemez, azt gyakran *alultanításnak* vagy a *rugalmasság* hiányának nevezzük, ahogy azt a (:numref:`sec_generalization_basics`) fejezetben bemutattuk. A nagy variancia általában egy túlságosan összetett modellből ered, amely túltanulja a tanítási adatokat. Ennek következtében egy *túltanult* modell érzékeny az adatok kis ingadozásaira. Ha egy modell nagy varianciával rendelkezik, azt gyakran *túltanulásnak* és az *általánosítóképesség* hiányának nevezzük, ahogy azt a (:numref:`sec_generalization_basics`) fejezetben bemutattuk. A visszafordíthatatlan hiba maga a $\theta$-ban lévő zajból ered.


### Becslők kiértékelése kóddal

Mivel egy becslő szórása egyszerűen egy `a` tenzor esetén az `a.std()` hívásával valósítható meg, ezt kihagyjuk, de implementáljuk a statisztikai torzítást és a négyzetesközép-hibát.

```{.python .input}
#@tab mxnet
# Statisztikai torzítás
def stat_bias(true_theta, est_theta):
    return(np.mean(est_theta) - true_theta)

# Négyzetesközép-hiba
def mse(data, true_theta):
    return(np.mean(np.square(data - true_theta)))
```

```{.python .input}
#@tab pytorch
# Statisztikai torzítás
def stat_bias(true_theta, est_theta):
    return(torch.mean(est_theta) - true_theta)

# Négyzetesközép-hiba
def mse(data, true_theta):
    return(torch.mean(torch.square(data - true_theta)))
```

```{.python .input}
#@tab tensorflow
# Statisztikai torzítás
def stat_bias(true_theta, est_theta):
    return(tf.reduce_mean(est_theta) - true_theta)

# Négyzetesközép-hiba
def mse(data, true_theta):
    return(tf.reduce_mean(tf.square(data - true_theta)))
```

A torzítás–variancia kompromisszum egyenletének szemléltetéséhez szimuláljunk egy $\mathcal{N}(\theta, \sigma^2)$ normáleloszlást $10\,000$ mintával. Itt $\theta = 1$-et és $\sigma = 4$-et használunk. Mivel a becslő az adott minták függvénye, itt a minták átlagát alkalmazzuk becslőként a $\mathcal{N}(\theta, \sigma^2)$ normáleloszlásban lévő igazi $\theta$ értékére.

```{.python .input}
#@tab mxnet
theta_true = 1
sigma = 4
sample_len = 10000
samples = np.random.normal(theta_true, sigma, sample_len)
theta_est = np.mean(samples)
theta_est
```

```{.python .input}
#@tab pytorch
theta_true = 1
sigma = 4
sample_len = 10000
samples = torch.normal(theta_true, sigma, size=(sample_len, 1))
theta_est = torch.mean(samples)
theta_est
```

```{.python .input}
#@tab tensorflow
theta_true = 1
sigma = 4
sample_len = 10000
samples = tf.random.normal((sample_len, 1), theta_true, sigma)
theta_est = tf.reduce_mean(samples)
theta_est
```

Ellenőrizzük az átváltási egyenletet a becslőnk négyzetes torzítása és varianciája összegének kiszámításával. Először számítsuk ki a becslőnk MSE-értékét.

```{.python .input}
#@tab all
mse(samples, theta_true)
```

Ezt követően az alábbiakban kiszámítjuk $\textrm{Var} (\hat{\theta}_n) + [\textrm{bias} (\hat{\theta}_n)]^2$ értékét. Ahogy látható, a két érték numerikus pontossággal egyezik.

```{.python .input}
#@tab mxnet
bias = stat_bias(theta_true, theta_est)
np.square(samples.std()) + np.square(bias)
```

```{.python .input}
#@tab pytorch
bias = stat_bias(theta_true, theta_est)
torch.square(samples.std(unbiased=False)) + torch.square(bias)
```

```{.python .input}
#@tab tensorflow
bias = stat_bias(theta_true, theta_est)
tf.square(tf.math.reduce_std(samples)) + tf.square(bias)
```

## Hipotézisvizsgálat

A statisztikai következtetésben leggyakrabban előforduló téma a hipotézisvizsgálat. Bár a hipotézisvizsgálatot a $20^{th}$. század elején terjesztették el szélesebb körben, első alkalmazása John Arbuthnot-ig vezethető vissza az 1700-as évekbe. John 80 éven át követte nyomon a londoni születési anyakönyvi adatokat, és arra a következtetésre jutott, hogy minden évben több fiú születik, mint lány. Ezt követően a modern szignifikanciavizsgálat Karl Pearson szellemi örökségéből táplálkozik, aki megalkotta a $p$-értéket és a Pearson-féle khi-négyzet-próbát; William Gosset-től, aki a Student-féle t-eloszlás atyja; valamint Ronald Fisher-től, aki bevezette a nullhipotézis és a szignifikanciavizsgálat fogalmát.

A *hipotézisvizsgálat* egy olyan módszer, amellyel egy populációra vonatkozó alapértelmezett állítással szemben egyes bizonyítékokat értékelünk ki. Az alapértelmezett állítást *nullhipotézisnek* ($H_0$) nevezzük, amelyet a megfigyelt adatok alapján igyekszünk elvetni. A $H_0$-t a statisztikai szignifikanciavizsgálat kiindulópontjaként alkalmazzuk. Az *alternatív hipotézis* ($H_A$ vagy $H_1$) a nullhipotézissel ellentétes állítás. A nullhipotézist általában kijelentő formában fogalmazzák meg, amely a változók közötti összefüggést rögzíti. Lehetőleg a lehető legpontosabban tükrözze a feltételezéseket, és statisztikai elmélet segítségével ellenőrizhető legyen.

Képzeljük el, hogy vegyész vagyunk. Több ezer laborban töltött óra után kifejlesztünk egy új gyógyszert, amely drámaian javítja a matematikai megértési képességet. Hogy bemutassuk csodálatos hatását, tesztelni kell. Természetszerűleg szükségünk lesz néhány önkéntesre, akik beveszik a gyógyszert, és megvizsgáljuk, hogy valóban segít-e nekik jobban megtanulni a matematikát. Hogyan kezdjünk hozzá?

Először gondosan véletlenszerűen ki kell választanunk két önkéntescsoportot, amelyek között nincs különbség valamilyen mérőszámmal mért matematikai megértési képességük tekintetében. A két csoportot általában kísérleti csoportnak és kontrollcsoportnak nevezik. A *kísérleti csoport* (vagy *kezelési csoport*) azon személyek csoportja, akik megkapják a gyógyszert, míg a *kontrollcsoport* azon felhasználók csoportját jelenti, akiket referenciacsoportként tartunk fenn – vagyis azonos környezeti feltételek között vannak, kivéve a gyógyszer szedését. Így az összes változó hatása minimalizálható, kivéve a kezelésben lévő független változó hatását.

Másodszor, a gyógyszer szedésének egy bizonyos időszaka után mindkét csoport matematikai megértési szintjét ugyanazokkal a mérőeszközökkel kell mérni – például az önkéntesek egy új matematikai képlet megtanulása után ugyanolyan teszteket végezzenek el. Ezután összegyűjthetjük teljesítményüket és összehasonlíthatjuk az eredményeket. Ebben az esetben nullhipotézisünk az lesz, hogy nincs különbség a két csoport között, az alternatív hipotézis pedig az, hogy van.

Ez még mindig nem teljesen formális. Számos részletet kell alaposan átgondolni. Például: mi az a megfelelő mérőszám a matematikai megértési képességük teszteléséhez? Hány önkéntes kell a teszthez ahhoz, hogy biztonsággal állíthassuk a gyógyszer hatékonyságát? Mennyi ideig kell futtatni a tesztet? Hogyan döntjük el, hogy van-e különbség a két csoport között? Csupán az átlagos teljesítménnyel foglalkozunk, vagy a pontszámok szóródásának mértékével is? És így tovább.

A hipotézisvizsgálat így keretet biztosít a kísérleti tervezéshez és a megfigyelt eredményekkel kapcsolatos bizonyosságra vonatkozó érveléshez. Ha most megmutatjuk, hogy a nullhipotézis nagyon valószínűtlen, magabiztosan elvethetjük.

Az összes, a hipotézisvizsgálattal kapcsolatos tudnivaló összefoglalásához be kell vezetnünk néhány további fogalmat, és formálissá kell tennünk a fentiekben ismertetett koncepciók némelyikét.


### Statisztikai szignifikancia

A *statisztikai szignifikancia* annak valószínűségét méri, hogy a $H_0$ nullhipotézist tévesen vetjük el, vagyis akkor is elvetjük, amikor nem kellene elvetni, azaz

$$ \textrm{statisztikai szignifikancia }= 1 - \alpha = 1 - P(\textrm{elveti } H_0 \mid H_0 \textrm{ igaz} ).$$

Ezt *I. típusú hibának* vagy *téves pozitívnak* is nevezik. Az $\alpha$-t *szignifikanciaszintnek* hívják, és általánosan használt értéke $5\%$, vagyis $1-\alpha = 95\%$. A szignifikanciaszint értelmezhető azon kockázat szintjeként, amelyet hajlandók vagyunk vállalni, amikor egy igaz nullhipotézist elvetünk.

A :numref:`fig_statistical_significance` ábra egy kétmintás hipotézisvizsgálat keretében megmutatja a megfigyelt értékeket és egy adott normáleloszlás valószínűségeit. Ha egy megfigyelési adatpont a $95\%$-os küszöbértéken kívül esik, az a nullhipotézis feltételezése mellett nagyon valószínűtlen megfigyelés. Ezért valami gond lehet a nullhipotézissel, és el fogjuk vetni.

![Statisztikai szignifikancia.](../img/statistical-significance.svg)
:label:`fig_statistical_significance`


### Statisztikai erő

A *statisztikai erő* (más néven *érzékenység*) annak valószínűségét méri, hogy a $H_0$ nullhipotézist elvetjük, amikor valóban el kell vetni, azaz

$$ \textrm{statisztikai erő }= 1 - \beta = 1 - P(\textrm{ nem veti el } H_0  \mid H_0 \textrm{ hamis} ).$$

Emlékeztetőül: az *I. típusú hiba* az a hiba, amely akkor keletkezik, ha a nullhipotézist akkor vetjük el, amikor az igaz, míg a *II. típusú hiba* akkor keletkezik, ha nem vetjük el a nullhipotézist, amikor az hamis. A II. típusú hibát általában $\beta$-val jelöljük, és ennek megfelelően a statisztikai erő $1-\beta$.

Intuitívan a statisztikai erő úgy értelmezhető, mint annak valószínűsége, hogy tesztünk egy adott minimális nagyságú tényleges eltérést kívánt statisztikai szignifikanciaszinten képes kimutatni. A $80\%$ egy általánosan használt statisztikai erőküszöb. Minél nagyobb a statisztikai erő, annál valószínűbb, hogy valós különbségeket mutatunk ki.

A statisztikai erő egyik legelterjedtebb felhasználási területe a szükséges mintaszám meghatározása. Annak valószínűsége, hogy a nullhipotézist elvetjük, amikor az hamis, függ attól, mennyire hamis (ezt *hatásnagyságnak* nevezzük), illetve a rendelkezésre álló minták számától. Ahogy várható, kis hatásnagyságok esetén nagyon nagy mintaszámra van szükség, hogy az eltérés nagy valószínűséggel kimutatható legyen. Bár e rövid függelék keretein belül nem részletezzük a levezetést, példaként: ha egy nulla várható értékű, egységnyi varianciájú Gauss-eloszlásból vett mintára vonatkozó nullhipotézist akarunk elvetni, és úgy gondoljuk, hogy a mintánk várható értéke valójában közel van egyhez, ezt elfogadható hibaráta mellett mindössze $8$ elemű mintával is megtehetjük. Ha azonban a mintapopuláció igazi várható értéke közel van $0{,}01$-hez, akkor az eltérés kimutatásához közel $80\,000$ elemű mintára lenne szükségünk.

A statisztikai erőt egy vízszűrőhöz hasonlíthatjuk. Ebben az analógiában a nagy erejű hipotézisvizsgálat olyan, mint egy kiváló minőségű vízszűrő rendszer, amely a lehető legnagyobb mértékben csökkenti a vízben lévő káros anyagokat. Ezzel szemben a kisebb eltérés olyan, mint egy gyengébb minőségű vízszűrő, amelyen az aránylag kis részecskék könnyen átcsúszhatnak a réseken. Hasonlóan, ha a statisztikai erő nem elég nagy, a teszt esetleg nem képes kimutatni a kisebb eltéréseket.


### Tesztstatisztika

A *tesztstatisztika* $T(x)$ egy skalár, amely a mintaadatok valamely jellemzőjét foglalja össze. Az ilyen statisztika definiálásának célja, hogy lehetővé tegye különböző eloszlások megkülönböztetését és a hipotézisvizsgálat elvégzését. Ha visszagondolunk a vegyész példájára: ha azt szeretnénk kimutatni, hogy az egyik populáció jobban teljesít, mint a másik, ésszerű lehet az átlagot tesztstatisztikaként alkalmazni. A tesztstatisztika különböző megválasztásai drasztikusan eltérő statisztikai erejű vizsgálatokhoz vezethetnek.

Gyakran $T(X)$ (a tesztstatisztika eloszlása a nullhipotézis feltételezése mellett) legalább közelítőleg egy ismert valószínűségeloszlást – például normáleloszlást – követ a nullhipotézis feltételezése esetén. Ha ezt az eloszlást explicit módon le tudjuk vezetni, majd mérjük a tesztstatisztikát az adathalmazunkon, biztonsággal elvethetjük a nullhipotézist, ha a statisztikánk messze esik a várható tartományon kívülre. Ennek számszerűsítése vezet el a $p$-érték fogalmához.


### $p$-érték

A $p$-érték (vagy *valószínűségi érték*) annak valószínűsége, hogy $T(X)$ legalább olyan szélsőséges, mint a megfigyelt $T(x)$ tesztstatisztika, feltéve, hogy a nullhipotézis *igaz*, azaz

$$ p\textrm{-érték} = P_{H_0}(T(X) \geq T(x)).$$

Ha a $p$-érték kisebb vagy egyenlő egy előre meghatározott és rögzített $\alpha$ statisztikai szignifikanciaszintnél, elvethetjük a nullhipotézist. Ellenkező esetben arra a következtetésre jutunk, hogy nincs elegendő bizonyítékunk a nullhipotézis elvetéséhez. Egy adott populációeloszlás esetén az *elvetési tartomány* az összes olyan pontot tartalmazó intervallum, amelynek $p$-értéke kisebb az $\alpha$ statisztikai szignifikanciaszintnél.


### Egyoldali és kétoldali vizsgálat

Általában kétféle szignifikanciavizsgálat létezik: az egyoldali és a kétoldali vizsgálat. Az *egyoldali vizsgálat* (más néven *egyoldalas teszt*) akkor alkalmazható, ha a nullhipotézis és az alternatív hipotézis csak egy irányba mutat. Például a nullhipotézis kimondhatja, hogy az igazi $\theta$ paraméter kisebb vagy egyenlő egy $c$ értéknél. Az alternatív hipotézis az lenne, hogy $\theta$ nagyobb $c$-nél. Vagyis az elvetési tartomány a mintaeloszlásnak csupán az egyik oldalán található. Az egyoldali vizsgálattal ellentétben a *kétoldali vizsgálat* (más néven *kétoldalas teszt*) akkor alkalmazható, ha az elvetési tartomány a mintaeloszlás mindkét oldalán megtalálható. Erre példa lehet egy olyan nullhipotézis, amely kimondja, hogy az igazi $\theta$ paraméter egyenlő egy $c$ értékkel. Az alternatív hipotézis ekkor az lenne, hogy $\theta$ nem egyenlő $c$-vel.


### A hipotézisvizsgálat általános lépései

A fenti fogalmak megismerése után nézzük végig a hipotézisvizsgálat általános lépéseit.

1. Fogalmazzuk meg a kérdést, és állapítsunk meg egy $H_0$ nullhipotézist.
2. Állítsuk be az $\alpha$ statisztikai szignifikanciaszintet és a statisztikai erőt ($1 - \beta$).
3. Gyűjtsünk mintákat kísérleteken keresztül. A szükséges minták száma a statisztikai erőtől és a várt hatásnagyságtól függ.
4. Számítsuk ki a tesztstatisztikát és a $p$-értéket.
5. A $p$-érték és az $\alpha$ statisztikai szignifikanciaszint alapján hozzuk meg a döntést, hogy megtartjuk vagy elvetjük a nullhipotézist.

A hipotézisvizsgálat elvégzéséhez először definiálunk egy nullhipotézist és meghatározzuk a vállalni kívánt kockázat szintjét. Ezután kiszámítjuk a minta tesztstatisztikáját, a tesztstatisztika szélsőséges értékét a nullhipotézis elleni bizonyítékként kezelve. Ha a tesztstatisztika az elvetési tartományba esik, elvethetjük a nullhipotézist az alternatív hipotézis javára.

A hipotézisvizsgálat számos területen alkalmazható, például klinikai vizsgálatokban és A/B-tesztelésben.


## Konfidenciaintervallumok szerkesztése

Egy $\theta$ paraméter értékének becslésekor a $\hat \theta$-hoz hasonló pontbecslők korlátozott hasznosságúak, mivel nem hordoznak semmilyen bizonytalansági információt. Sokkal előnyösebb lenne, ha olyan intervallumot tudnánk előállítani, amely nagy valószínűséggel tartalmazza az igazi $\theta$ paramétert. Ha egy évszázaddal ezelőtt érdeklődött volna ilyen ötletek iránt, izgatottan olvasta volna Jerzy Neyman :cite:`Neyman.1937` „Vázlat a statisztikai becslés elméletéről a valószínűség klasszikus elmélete alapján" című munkáját, amely 1937-ben bevezette a konfidenciaintervallum fogalmát.

Ahhoz, hogy hasznos legyen, a konfidenciaintervallumnak adott bizonyossági szint mellett a lehető legkisebb méretűnek kell lennie. Nézzük meg, hogyan vezethető le.


### Definíció

Matematikailag az igazi $\theta$ paraméterre vonatkozó *konfidenciaintervallum* egy $C_n$ intervallum, amelyet a mintaadatokból számítunk ki, és amely teljesíti:

$$P_{\theta} (C_n \ni \theta) \geq 1 - \alpha, \forall \theta.$$
:eqlabel:`eq_confidence`

Itt $\alpha \in (0, 1)$, és $1 - \alpha$ neve az intervallum *konfidenciaszintje* vagy *lefedési valószínűsége*. Ez ugyanaz az $\alpha$, mint a fentebb tárgyalt szignifikanciaszint.

Megjegyezzük, hogy a :eqref:`eq_confidence` a $C_n$ változóra vonatkozik, nem az rögzített $\theta$-ra. Ennek hangsúlyozása érdekében $P_{\theta} (C_n \ni \theta)$ helyett $P_{\theta} (\theta \in C_n)$ jelölést alkalmazzuk.

### Értelmezés

Kísértő egy $95\%$-os konfidenciaintervallumot úgy értelmezni, mint egy olyan intervallumot, amelyen belül $95\%$ valószínűséggel található az igazi paraméter – ez azonban sajnos nem helyes. Az igazi paraméter rögzített, és az intervallum az, amely véletlenszerű. Ezért egy jobb értelmezés az lenne, hogy ha ezt az eljárást alkalmazva nagyszámú konfidenciaintervallumot generálunk, a generált intervallumok $95\%$-a tartalmazná az igazi paramétert.

Ez akadékoskodónak tűnhet, de valós következményekkel járhat az eredmények értelmezésére nézve. Különösen: kielégíthetjük a :eqref:`eq_confidence` feltételt úgy, hogy olyan intervallumokat szerkesztünk, amelyekről *szinte biztosak* vagyunk, hogy nem tartalmazzák az igazi értéket – feltéve, hogy ezt elég ritkán tesszük. Zárjuk ezt a fejezetet három csábítóan hangzó, de téves állítással. Ezeknek a pontoknak részletes tárgyalása megtalálható :citet:`Morey.Hoekstra.Rouder.ea.2016` munkájában.

* **1. tévhit.** Szűk konfidenciaintervallumok azt jelentik, hogy pontosan meg tudjuk becsülni a paramétert.
* **2. tévhit.** A konfidenciaintervallumban lévő értékek nagyobb valószínűséggel az igazi értékek, mint az intervallumon kívüliek.
* **3. tévhit.** Annak valószínűsége, hogy egy adott megfigyelt $95\%$-os konfidenciaintervallum tartalmazza az igazi értéket, $95\%$.

Elmondható, hogy a konfidenciaintervallumok finom fogalmak. Ha azonban az értelmezés egyértelmű marad, hatékony eszközök lehetnek.

### Gauss-példa

Tárgyaljuk a leghíresebb példát: egy ismeretlen várható értékű és varianciájú Gauss-eloszlás várható értékének konfidenciaintervallumát. Tegyük fel, hogy $n$ mintát gyűjtünk $\{x_i\}_{i=1}^n$ formában a $\mathcal{N}(\mu, \sigma^2)$ Gauss-eloszlásból. A várható értékre és a varianciára vonatkozó becslőket a következőképpen számíthatjuk:

$$\hat\mu_n = \frac{1}{n}\sum_{i=1}^n x_i \;\textrm{és}\; \hat\sigma^2_n = \frac{1}{n-1}\sum_{i=1}^n (x_i - \hat\mu)^2.$$

Ha most tekintjük a valószínűségi változót

$$
T = \frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}},
$$

egy jól ismert eloszlást – az *$n-1$ szabadsági fokú Student-féle t-eloszlást* – követő valószínűségi változót kapunk.

Ezt az eloszlást alaposan tanulmányozták, és például ismeretes, hogy $n\rightarrow \infty$ esetén közelítőleg standard Gauss-eloszlást követ. Így a Gauss kumulatív eloszlásfüggvényének tábláiból megállapítható, hogy $T$ értéke legalább $95\%$-os valószínűséggel a $[-1{,}96; 1{,}96]$ intervallumba esik. Véges $n$ esetén az intervallumot valamivel nagyobbra kell venni, de ezek jól ismertek és előre ki vannak számítva táblázatokban.

Így arra a következtetésre juthatunk, hogy nagy $n$ esetén

$$
P\left(\frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}} \in [-1.96, 1.96]\right) \ge 0.95.
$$

Ezt átrendezve – mindkét oldalt megszorozzuk $\hat\sigma_n/\sqrt{n}$-nel, majd hozzáadjuk $\hat\mu_n$-t – a következőt kapjuk:

$$
P\left(\mu \in \left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right]\right) \ge 0.95.
$$

Tehát megtaláltuk a $95\%$-os konfidenciaintervallumot:
$$\left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right].$$
:eqlabel:`eq_gauss_confidence`

Elmondható, hogy a :eqref:`eq_gauss_confidence` az egyik legtöbbet használt képlet a statisztikában. Zárjuk a statisztikáról szóló tárgyalásunkat ennek implementálásával. Az egyszerűség kedvéért feltételezzük, hogy aszimptotikus tartományban vagyunk. Kis $N$ értékek esetén a `t_star` helyes értékét programozottan vagy t-táblázatból kell meghatározni.

```{.python .input}
#@tab mxnet
# Minták száma
N = 1000

# Mintahalmaz
samples = np.random.normal(loc=0, scale=1, size=(N,))

# Keressük ki a Student-féle t-eloszlás kumulatív eloszlásfüggvényét.
t_star = 1.96

# Intervallum felépítése
mu_hat = np.mean(samples)
sigma_hat = samples.std(ddof=1)
(mu_hat - t_star*sigma_hat/np.sqrt(N), mu_hat + t_star*sigma_hat/np.sqrt(N))
```

```{.python .input}
#@tab pytorch
# A PyTorch alapértelmezetten Bessel-korrekciót használ, ami azt jelenti, hogy
# ddof=1-et alkalmaz a NumPy alapértelmezett ddof=0 értéke helyett. Az
# unbiased=False használatával a ddof=0 viselkedést tudjuk utánozni.

# Minták száma
N = 1000

# Mintahalmaz
samples = torch.normal(0, 1, size=(N,))

# Keressük ki a Student-féle t-eloszlás kumulatív eloszlásfüggvényét.
t_star = 1.96

# Intervallum felépítése
mu_hat = torch.mean(samples)
sigma_hat = samples.std(unbiased=True)
(mu_hat - t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)),\
 mu_hat + t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)))
```

```{.python .input}
#@tab tensorflow
# Minták száma
N = 1000

# Mintahalmaz
samples = tf.random.normal((N,), 0, 1)

# Keressük ki a Student-féle t-eloszlás kumulatív eloszlásfüggvényét.
t_star = 1.96

# Intervallum felépítése
mu_hat = tf.reduce_mean(samples)
sigma_hat = tf.math.reduce_std(samples)
(mu_hat - t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)), \
 mu_hat + t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)))
```

## Összefoglalás

* A statisztika a következtetési problémákra összpontosít, míg a mélytanulás a pontos előrejelzések elvégzésére helyezi a hangsúlyt, explicit programozás és megértés nélkül.
* Három általánosan használt statisztikai következtetési módszer létezik: a becslők kiértékelése és összehasonlítása, a hipotézisvizsgálat elvégzése és a konfidenciaintervallumok szerkesztése.
* A becslők értékelésére használt három legfontosabb mérőszám a statisztikai torzítás, a szórás és a négyzetesközép-hiba.
* A konfidenciaintervallum egy igazi populációs paraméter becsült tartománya, amelyet a minták alapján szerkesztünk meg.
* A hipotézisvizsgálat egy olyan módszer, amellyel egy populációra vonatkozó alapértelmezett állítással szemben egyes bizonyítékokat értékelünk ki.


## Feladatok

1. Legyen $X_1, X_2, \ldots, X_n \overset{\textrm{iid}}{\sim} \textrm{Unif}(0, \theta)$, ahol az „iid" rövidítés a *független és azonos eloszlású* (independent and identically distributed) kifejezésre utal. Tekintsük $\theta$ következő becslőit:
$$\hat{\theta} = \max \{X_1, X_2, \ldots, X_n \};$$
$$\tilde{\theta} = 2 \bar{X_n} = \frac{2}{n} \sum_{i=1}^n X_i.$$
    * Keressük meg $\hat{\theta}$ statisztikai torzítását, szórását és négyzetesközép-hibáját.
    * Keressük meg $\tilde{\theta}$ statisztikai torzítását, szórását és négyzetesközép-hibáját.
    * Melyik becslő a jobb?
1. A bevezetőben szereplő vegyész példájában le tudja-e vezetni a kétoldali hipotézisvizsgálat elvégzéséhez szükséges 5 lépést? Adott az $\alpha = 0{,}05$ statisztikai szignifikanciaszint és az $1 - \beta = 0{,}8$ statisztikai erő.
1. Futtassa a konfidenciaintervallum kódját $N=2$ és $\alpha = 0{,}5$ értékekkel $100$ independently generált adathalmazon, és ábrázolja a kapott intervallumokat (ebben az esetben `t_star = 1.0`). Néhány nagyon rövid intervallumot fog látni, amelyek messze esnek az igazi $0$ várható értéktől. Ez ellentmond-e a konfidenciaintervallum értelmezésének? Kényelmes-e rövid intervallumokat használni a nagy pontosságú becslések jelzésére?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/419)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1102)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1103)
:end_tab:
