# Valószínűségi változók
:label:`sec_random_variables`

A :numref:`sec_prob` részben megismertük a diszkrét valószínűségi változókkal való munka alapjait, ahol a valószínűségi változók véges értékkészletből vagy az egész számok halmazából veszik fel értékeiket. Ebben a szakaszban a *folytonos valószínűségi változók* elméletét dolgozzuk ki, amelyek tetszőleges valós értéket felvehetnek.

## Folytonos valószínűségi változók

A folytonos valószínűségi változók lényegesen összetettebb témakört alkotnak, mint a diszkrét társaik. Jó analógia, hogy az elmélet nehézségi szintjének ugrása hasonló ahhoz, mint számsorok összeadásától a függvények integrálásáig vezető lépés. Ezért szükséges egy kis időt szánnunk az elmélet kiépítésére.

### A diszkréttől a folytonosig

Hogy megértsük a folytonos valószínűségi változókkal való munka során felmerülő technikai kihívásokat, végezzünk el egy gondolatkísérletet. Tegyük fel, hogy nyilat dobunk egy céltáblára, és tudni szeretnénk annak valószínűségét, hogy pontosan $2 \textrm{cm}$-re találja el a tábla középpontját.

Kezdetben egyetlen tizedes jegynyi pontossággal mérjük az eredményt, azaz $0 \textrm{cm}$, $1 \textrm{cm}$, $2 \textrm{cm}$ stb. rekeszekkel dolgozunk. Tegyük fel, hogy $100$ nyilat dobunk a céltáblára, és ha $20$ a $2\textrm{cm}$-es rekeszbe esik, akkor azt a következtetést vonjuk le, hogy a nyilak $20\%$-a $2 \textrm{cm}$-re találja a tábla középpontját.

Ám ha jobban megnézzük, ez nem egyezik meg az eredeti kérdésünkkel! Pontos egyenlőséget akartunk, míg ezek a rekeszek mindent tartalmaznak, ami $1.5\textrm{cm}$ és $2.5\textrm{cm}$ közé esik.

Nem tántorítva magunkat, folytatjuk tovább. Még pontosabban mérünk: $1.9\textrm{cm}$, $2.0\textrm{cm}$, $2.1\textrm{cm}$, és most azt látjuk, hogy talán $3$ a $100$ nyílból a $2.0\textrm{cm}$-es rekeszbe ér. Így az valószínűséget $3\%$-ra becsüljük.

Ez azonban nem oldja meg a problémát! Csupán egy tizedes jeggyel mélyebbre toltuk a kérdést. Vonjuk el egy kicsit. Képzeljük el, hogy ismerjük annak valószínűségét, hogy az első $k$ jegy megegyezik a $2.00000\ldots$ értékkel, és tudni szeretnénk az első $k+1$ jegyre vonatkozó egyezés valószínűségét. Elég ésszerű feltételezni, hogy a ${k+1}^{\textrm{th}}$ jegy lényegében véletlenszerű választás a $\{0, 1, 2, \ldots, 9\}$ halmazból. Legalábbis nem tudunk elképzelni olyan fizikailag értelmes folyamatot, amely arra kényszerítené a középponttól mért mikrométerszámot, hogy inkább $7$-re, mint $3$-ra végződjön.

Mindez azt jelenti, hogy minden további pontossági jegy lényegében $10$-szeres faktorral csökkenti az egyezés valószínűségét. Másképpen fogalmazva, azt várjuk, hogy

$$
P(\textrm{a távolság}\; 2.00\ldots, \;\textrm{pontosan}\; k \;\textrm{jegyre} ) \approx p\cdot10^{-k}.
$$

A $p$ érték lényegében azt kódolja, ami az első néhány jeggyel történik, a $10^{-k}$ pedig a többit kezeli.

Vegyük észre, hogy ha $k=4$ tizedes jegyre ismerjük a pozíciót, akkor tudjuk, hogy az érték a $[1.99995,2.00005]$ intervallumba esik, amelynek hossza $2.00005-1.99995 = 10^{-4}$. Ha tehát ezt az intervallumhosszt $\epsilon$-nak jelöljük, akkor azt mondhatjuk:

$$
P(\textrm{a távolság egy}\; \epsilon\textrm{-méretű intervallumba esik}\; 2 \;\textrm{körül} ) \approx \epsilon \cdot p.
$$

Tegyük meg ezt az utolsó lépést is. Mindvégig a $2$ pontot vizsgáltuk, de sohasem gondolkoztunk más pontokról. Alapvetően semmi különbség nincs, de a $p$ értéke valószínűleg eltér. Legalábbis azt remélnénk, hogy a nyíl dobója nagyobb valószínűséggel találja el a középponthoz közeli pontokat, mint például $2\textrm{cm}$ helyett $20\textrm{cm}$-t. Tehát a $p$ érték nem rögzített, hanem az $x$ ponttól függ. Ez azt mondja nekünk:

$$P(\textrm{a távolság egy}\; \epsilon \textrm{-méretű intervallumba esik}\; x \;\textrm{körül} ) \approx \epsilon \cdot p(x).$$
:eqlabel:`eq_pdf_deriv`

Valóban, a :eqref:`eq_pdf_deriv` egyenlet pontosan a *valószínűségi sűrűségfüggvényt* definiálja. Ez egy $p(x)$ függvény, amely kódolja az egyik pont közelébe való találás relatív valószínűségét egy másik ponthoz képest. Vizualizáljuk, hogyan nézhet ki egy ilyen függvény.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# Néhány valószínűségi változó sűrűségfüggvényének ábrázolása
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2)/np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2)/np.sqrt(2 * np.pi)

d2l.plot(x, p, 'x', 'Sűrűség')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # A pi definiálása a Torchban

# Néhány valószínűségi változó sűrűségfüggvényének ábrázolása
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi)) + \
    0.8*torch.exp(-(x + 1)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi))

d2l.plot(x, p, 'x', 'Sűrűség')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)).numpy() * 2  # A pi definiálása a TensorFlow-ban

# Néhány valószínűségi változó sűrűségfüggvényének ábrázolása
x = tf.range(-5, 5, 0.01)
p = 0.2*tf.exp(-(x - 3)**2 / 2)/tf.sqrt(2 * tf.constant(tf.pi)) + \
    0.8*tf.exp(-(x + 1)**2 / 2)/tf.sqrt(2 * tf.constant(tf.pi))

d2l.plot(x, p, 'x', 'Sűrűség')
```

Ahol a függvényérték nagy, ott nagyobb valószínűséggel találjuk meg a véletlen értéket. Az alacsony részek azok a területek, ahol a véletlen értéket ritkán találjuk.

### Valószínűségi sűrűségfüggvények

Most vizsgáljuk meg ezt mélyebben. Már láttuk intuitívan, mi a valószínűségi sűrűségfüggvény egy $X$ valószínűségi változóra: a sűrűségfüggvény egy $p(x)$ függvény, amelyre

$$P(X \; \textrm{egy}\; \epsilon \textrm{-méretű intervallumba esik}\; x \;\textrm{körül} ) \approx \epsilon \cdot p(x).$$
:eqlabel:`eq_pdf_def`

De mit jelent ez $p(x)$ tulajdonságaira nézve?

Először is, a valószínűségek sohasem negatívak, ezért elvárjuk, hogy $p(x) \ge 0$ is teljesüljön.

Másodszor, képzeljük el, hogy a $\mathbb{R}$ számegyenest végtelen sok, $\epsilon$ szélességű szeletre osztjuk fel, például $(\epsilon\cdot i, \epsilon \cdot (i+1)]$ alakú szeletekre. Mindegyikre tudjuk :eqref:`eq_pdf_def` alapján, hogy a valószínűség közelítőleg

$$
P(X \; \textrm{egy}\; \epsilon\textrm{-méretű intervallumba esik}\; x \;\textrm{körül} ) \approx \epsilon \cdot p(\epsilon \cdot i),
$$

tehát összegezve az összeset:

$$
P(X\in\mathbb{R}) \approx \sum_i \epsilon \cdot p(\epsilon\cdot i).
$$

Ez pontosan a :numref:`sec_integral_calculus` részben tárgyalt integrálközelítés, így azt mondhatjuk:

$$
P(X\in\mathbb{R}) = \int_{-\infty}^{\infty} p(x) \; dx.
$$

Tudjuk, hogy $P(X\in\mathbb{R}) = 1$, hiszen a valószínűségi változónak *valamely* számot fel kell vennie, ezért minden sűrűségfüggvényre teljesül:

$$
\int_{-\infty}^{\infty} p(x) \; dx = 1.
$$

Mélyebbre ásva azt is beláthatjuk, hogy tetszőleges $a$ és $b$ esetén:

$$
P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx.
$$

Ezt kódban is közelíthetjük ugyanolyan diszkrét közelítéssel, mint korábban. Így a kék területre eső valószínűséget közelíthetjük.

```{.python .input}
#@tab mxnet
# A valószínűség közelítése numerikus integrálással
epsilon = 0.01
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2) / np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2) / np.sqrt(2 * np.pi)

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'közelítő valószínűség: {np.sum(epsilon*p[300:800])}'
```

```{.python .input}
#@tab pytorch
# A valószínűség közelítése numerikus integrálással
epsilon = 0.01
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi)) +\
    0.8*torch.exp(-(x + 1)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi))

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'közelítő valószínűség: {torch.sum(epsilon*p[300:800])}'
```

```{.python .input}
#@tab tensorflow
# A valószínűség közelítése numerikus integrálással
epsilon = 0.01
x = tf.range(-5, 5, 0.01)
p = 0.2*tf.exp(-(x - 3)**2 / 2) / tf.sqrt(2 * tf.constant(tf.pi)) +\
    0.8*tf.exp(-(x + 1)**2 / 2) / tf.sqrt(2 * tf.constant(tf.pi))

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.numpy().tolist()[300:800], p.numpy().tolist()[300:800])
d2l.plt.show()

f'közelítő valószínűség: {tf.reduce_sum(epsilon*p[300:800])}'
```

Kiderül, hogy ez a két tulajdonság pontosan leírja a lehetséges valószínűségi sűrűségfüggvények (*p.d.f.*-ek) terét. Ezek nemnegatív $p(x) \ge 0$ függvények, amelyekre

$$\int_{-\infty}^{\infty} p(x) \; dx = 1.$$
:eqlabel:`eq_pdf_int_one`

Ezt a függvényt integrálással értelmezzük: a valószínűségi változónk valamely intervallumba esésének valószínűségét így kapjuk:

$$P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx.$$
:eqlabel:`eq_pdf_int_int`

A :numref:`sec_distributions` részben számos elterjedt eloszlást fogunk látni, de most folytassuk az absztrakt tárgyalást.

### Kumulatív eloszlásfüggvények

Az előző részben megismertük a p.d.f. fogalmát. A gyakorlatban ez egy elterjedt módszer a folytonos valószínűségi változók tárgyalására, de van egy lényeges csapdája: a p.d.f. értékei önmagukban nem valószínűségek, hanem egy olyan függvény, amelyet integrálnunk kell a valószínűségek előállításához. Nincs semmi baj azzal, ha a sűrűség nagyobb $10$-nél, feltéve, hogy nem marad $10$-nél nagyobb egy $1/10$ hosszúságnál rövidebb intervallumon. Ez ellenintuitív lehet, ezért sokan inkább a *kumulatív eloszlásfüggvénnyel*, azaz a c.d.f.-fel gondolkodnak, amely *maga is* valószínűség.

Különösen, a :eqref:`eq_pdf_int_int` felhasználásával, az $X$ valószínűségi változó $p(x)$ sűrűségfüggvényű c.d.f.-jét így definiáljuk:

$$
F(x) = \int _ {-\infty}^{x} p(x) \; dx = P(X \le x).
$$

Figyeljük meg néhány tulajdonságát.

* $F(x) \rightarrow 0$, ha $x\rightarrow -\infty$.
* $F(x) \rightarrow 1$, ha $x\rightarrow \infty$.
* $F(x)$ nem csökkenő ($y > x \implies F(y) \ge F(x)$).
* $F(x)$ folytonos (nincs ugrása), ha $X$ folytonos valószínűségi változó.

A negyedik ponthoz megjegyezzük, hogy ez nem teljesülne, ha $X$ diszkrét lenne, például ha $0$ és $1$ értékeket egyenlő $1/2$ valószínűséggel vesz fel. Ebben az esetben

$$
F(x) = \begin{cases}
0 & x < 0, \\
\frac{1}{2} & x < 1, \\
1 & x \ge 1.
\end{cases}
$$

Ebből a példából kitűnik a c.d.f. egyik előnye: ugyanabban a keretrendszerben tudunk kezelni folytonos és diszkrét valószínűségi változókat, sőt ezek keverékét is (például: dobjunk fel egy érmét; ha fej, adjuk vissza egy dobókocka dobásának eredményét; ha írás, a céltáblától mért távolságot).

### Várható értékek

Tegyük fel, hogy egy $X$ valószínűségi változóval dolgozunk. Az eloszlás önmagában nehezen értelmezhető. Sokszor hasznos tömören összefoglalni a valószínűségi változó viselkedését. Az ilyen összefoglalásra szolgáló számokat *összefoglaló statisztikáknak* nevezzük. A leggyakrabban előforduló ezek közül a *várható érték*, a *szórásnégyzet* és a *szórás*.

A *várható érték* a valószínűségi változó átlagos értékét kódolja. Ha $X$ diszkrét valószínűségi változó, amely $x_i$ értékeket vesz fel $p_i$ valószínűséggel, akkor a várható érték a súlyozott átlag: az értékek összege, szorozva azzal a valószínűséggel, amellyel a változó felveszi az adott értéket:

$$\mu_X = E[X] = \sum_i x_i p_i.$$
:eqlabel:`eq_exp_def`

A várható értéket (óvatossággal) úgy értelmezhetjük, hogy megmutatja, hol tartózkodik jellemzően a valószínűségi változó.

Mint minimális példát, amelyet az egész szakaszon végigkövetünk, legyen $X$ az a valószínűségi változó, amely $a-2$ értéket vesz fel $p$ valószínűséggel, $a+2$ értéket $p$ valószínűséggel és $a$ értéket $1-2p$ valószínűséggel. A :eqref:`eq_exp_def` segítségével kiszámítható, hogy $a$ és $p$ bármely lehetséges választásánál a várható érték:

$$
\mu_X = E[X] = \sum_i x_i p_i = (a-2)p + a(1-2p) + (a+2)p = a.
$$

Láthatjuk tehát, hogy a várható érték $a$. Ez megfelel az intuíciónak, mivel $a$ az a hely, amely köré a valószínűségi változónkat centráltuk.

Mivel hasznosak, foglaljuk össze néhány tulajdonságukat.

* Bármely $X$ valószínűségi változóra és $a$, $b$ számokra: $\mu_{aX+b} = a\mu_X + b$.
* Ha $X$ és $Y$ két valószínűségi változó, akkor $\mu_{X+Y} = \mu_X+\mu_Y$.

A várható értékek hasznosak a valószínűségi változó átlagos viselkedésének megértéséhez, de a várható érték önmagában nem elegendő a teljes intuitív megértéshez. Értékesítésenként $\$10 \pm \$1$ nyereség alapvetően más, mint $\$10 \pm \$15$, annak ellenére, hogy az átlagos értékük azonos. Az utóbbinak sokkal nagyobb az ingadozása, és így lényegesen nagyobb kockázatot jelent. Ezért a valószínűségi változó viselkedésének megértéséhez legalább még egy mértékre van szükségünk: valamilyen mértékre arról, hogy mennyire ingadozik a valószínűségi változó.

### Szórásnégyzetek

Ez elvezet bennünket a valószínűségi változó *szórásnégyzetének* fogalmához. Ez egy kvantitatív mérőszáma annak, mennyire tér el a valószínűségi változó a várható értékétől. Tekintsük az $X - \mu_X$ kifejezést. Ez a valószínűségi változó eltérése a várható értéktől. Ez az érték pozitív vagy negatív is lehet, ezért valamit tennünk kell, hogy pozitívvá tegyük, és mérjük a szórásnégyzet nagyságát.

Ésszerű próbálkozás a $\left|X-\mu_X\right|$ vizsgálata, ami valóban egy hasznos mennyiséget, a *közepes abszolút eltérést* adja. A matematika és a statisztika más területeivel való összefüggések miatt azonban az emberek általában más megoldást alkalmaznak.

Konkrétan a $(X-\mu_X)^2$ kifejezést vizsgálják. Ha ennek jellemző méretét a várható érték segítségével nézzük, megkapjuk a szórásnégyzetet:

$$\sigma_X^2 = \textrm{Var}(X) = E\left[(X-\mu_X)^2\right] = E[X^2] - \mu_X^2.$$
:eqlabel:`eq_var_def`

A :eqref:`eq_var_def` utolsó egyenlősége a középső definíció kifejtéséből és a várható érték tulajdonságainak alkalmazásából következik.

Nézzük meg a példánkat, ahol $X$ az a valószínűségi változó, amely $a-2$ értéket vesz fel $p$ valószínűséggel, $a+2$ értéket $p$ valószínűséggel és $a$ értéket $1-2p$ valószínűséggel. Ebben az esetben $\mu_X = a$, tehát csak $E\left[X^2\right]$-t kell kiszámolnunk. Ez könnyen elvégezhető:

$$
E\left[X^2\right] = (a-2)^2p + a^2(1-2p) + (a+2)^2p = a^2 + 8p.
$$

Így :eqref:`eq_var_def` alapján a szórásnégyzet:

$$
\sigma_X^2 = \textrm{Var}(X) = E[X^2] - \mu_X^2 = a^2 + 8p - a^2 = 8p.
$$

Ez az eredmény ismét értelmes. A $p$ maximális értéke $1/2$, amely megfelel annak, ha $a-2$ vagy $a+2$ között pénzfeldobással döntünk. Hogy a szórásnégyzet ekkor $4$, megfelel annak, hogy mind $a-2$, mind $a+2$ pontosan $2$ egységnyire van a várható értéktől, és $2^2 = 4$. A másik végponton, ha $p=0$, a valószínűségi változó mindig $0$ értéket vesz fel, tehát egyáltalán nincs szórásnégyzete.

A szórásnégyzet néhány tulajdonságát az alábbiakban soroljuk fel:

* Bármely $X$ valószínűségi változóra $\textrm{Var}(X) \ge 0$, ahol $\textrm{Var}(X) = 0$ akkor és csak akkor, ha $X$ konstans.
* Bármely $X$ valószínűségi változóra és $a$, $b$ számokra: $\textrm{Var}(aX+b) = a^2\textrm{Var}(X)$.
* Ha $X$ és $Y$ két *független* valószínűségi változó, akkor $\textrm{Var}(X+Y) = \textrm{Var}(X) + \textrm{Var}(Y)$.

Az értékek értelmezésekor kis akadályba ütközhetünk. Különösen, próbáljuk meg elképzelni, mi történik, ha végigkövetjük a mértékegységeket a számítás során. Tegyük fel, hogy egy weblapon lévő termék csillaggal értékelt minősítésével dolgozunk. Ekkor $a$, $a-2$ és $a+2$ mind csillag mértékegységben értendő. Hasonlóképpen a $\mu_X$ várható érték is csillagban mérhető (lévén súlyozott átlag). Ám ha eljutunk a szórásnégyzethez, azonnal gonddal szembesülünk: a $(X-\mu_X)^2$ kifejezés *négyzetcsillag* mértékegységű. Ez azt jelenti, hogy a szórásnégyzet önmagában nem hasonlítható össze az eredeti mérésekkel. Az értelmezhetőség érdekében vissza kell térnünk az eredeti mértékegységhez.

### Szórások

Ez az összefoglaló statisztika mindig levezethető a szórásnégyzetből, ha gyököt vonunk! Így a *szórást* a következőképpen definiáljuk:

$$
\sigma_X = \sqrt{\textrm{Var}(X)}.
$$

A mi példánkban ez azt jelenti, hogy a szórás $\sigma_X = 2\sqrt{2p}$. Ha az értékelési példánkban csillag mértékegységgel dolgozunk, akkor $\sigma_X$ ismét csillagban mérhető.

A szórásnégyzetre vonatkozó tulajdonságok a szórásra is átfogalmazhatók.

* Bármely $X$ valószínűségi változóra $\sigma_{X} \ge 0$.
* Bármely $X$ valószínűségi változóra és $a$, $b$ számokra: $\sigma_{aX+b} = |a|\sigma_{X}$.
* Ha $X$ és $Y$ két *független* valószínűségi változó, akkor $\sigma_{X+Y} = \sqrt{\sigma_{X}^2 + \sigma_{Y}^2}$.

Természetes kérdés felmerülhet: „Ha a szórás az eredeti valószínűségi változó mértékegységében van, akkor valami olyat fejez-e ki, amit lerajzolhatunk az adott valószínűségi változóra?" A válasz határozottan igen! Valóban, ahogy a várható érték megmondja a valószínűségi változó jellemző helyzetét, a szórás megadja az ingadozás jellemző tartományát. Ezt formálisan a Csebisev-egyenlőtlenséggel tehetjük rigorózussá:

$$P\left(X \not\in [\mu_X - \alpha\sigma_X, \mu_X + \alpha\sigma_X]\right) \le \frac{1}{\alpha^2}.$$
:eqlabel:`eq_chebyshev`

Más szóval, $\alpha=10$ esetén bármely valószínűségi változó mintáinak $99\%$-a a várható értéktől $10$ szóráson belül esik. Ez azonnali értelmezést ad az összefoglaló statisztikáinknak.

Hogy lássuk, ez az állítás mennyire árnyalt, nézzük meg ismét a futó példánkat, ahol $X$ az a valószínűségi változó, amely $a-2$ értéket $p$ valószínűséggel, $a+2$ értéket $p$ valószínűséggel és $a$ értéket $1-2p$ valószínűséggel vesz fel. Láttuk, hogy a várható érték $a$ és a szórás $2\sqrt{2p}$. Ez azt jelenti, hogy ha a Csebisev-egyenlőtlenséget :eqref:`eq_chebyshev` $\alpha = 2$-vel alkalmazzuk:

$$
P\left(X \not\in [a - 4\sqrt{2p}, a + 4\sqrt{2p}]\right) \le \frac{1}{4}.
$$

Ez azt jelenti, hogy az esetek $75\%$-ában ez a valószínűségi változó ebbe az intervallumba esik, $p$ bármely értékére. Figyeljük meg, hogy ha $p \rightarrow 0$, ez az intervallum konvergál az $a$ ponthoz. De tudjuk, hogy a valószínűségi változónk csak az $a-2, a$ és $a+2$ értékeket veheti fel, tehát végül biztosak lehetünk abban, hogy $a-2$ és $a+2$ kiesnek az intervallumból! A kérdés az, hogy melyik $p$-nél történik ez. Tehát meg kell oldani: milyen $p$-re teljesül $a+4\sqrt{2p} = a+2$, amelynek megoldása $p=1/8$. Ez pontosan az első olyan $p$, amelynél lehetséges az, hogy nem több mint $1/4$ minta esik az intervallumon kívülre ($1/8$ a bal oldalon és $1/8$ a jobb oldalon).

Vizualizáljuk ezt. Megmutatjuk a három érték valószínűségét három függőleges oszloppal, amelyek magassága arányos a valószínűséggel. Az intervallum egy vízszintes vonalként jelenik meg a közepén. Az első ábra azt mutatja, mi történik $p > 1/8$ esetén, ahol az intervallum biztonságosan tartalmaz minden pontot.

```{.python .input}
#@tab mxnet
# Segédfüggvény a fenti ábrák rajzolásához
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * np.sqrt(2 * p),
                   a + 4 * np.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * np.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * np.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Intervallum ábrázolása, amikor p > 1/8
plot_chebyshev(0.0, 0.2)
```

```{.python .input}
#@tab pytorch
# Segédfüggvény a fenti ábrák rajzolásához
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * torch.sqrt(2 * p),
                   a + 4 * torch.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * torch.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * torch.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Intervallum ábrázolása, amikor p > 1/8
plot_chebyshev(0.0, torch.tensor(0.2))
```

```{.python .input}
#@tab tensorflow
# Segédfüggvény a fenti ábrák rajzolásához
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * tf.sqrt(2 * p),
                   a + 4 * tf.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * tf.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * tf.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Intervallum ábrázolása, amikor p > 1/8
plot_chebyshev(0.0, tf.constant(0.2))
```

A második ábra azt mutatja, hogy $p = 1/8$ esetén az intervallum pontosan érinti a két pontot. Ez azt jelzi, hogy az egyenlőtlenség *éles*, hiszen nem lehet kisebb intervallumot venni úgy, hogy az egyenlőtlenség még teljesüljön.

```{.python .input}
#@tab mxnet
# Intervallum ábrázolása, amikor p = 1/8
plot_chebyshev(0.0, 0.125)
```

```{.python .input}
#@tab pytorch
# Intervallum ábrázolása, amikor p = 1/8
plot_chebyshev(0.0, torch.tensor(0.125))
```

```{.python .input}
#@tab tensorflow
# Intervallum ábrázolása, amikor p = 1/8
plot_chebyshev(0.0, tf.constant(0.125))
```

A harmadik ábra azt mutatja, hogy $p < 1/8$ esetén az intervallum csak a középső pontot tartalmazza. Ez nem mond ellent az egyenlőtlenségnek, mivel csupán azt kellett biztosítani, hogy a valószínűség legfeljebb $1/4$-e essen az intervallumon kívülre; ezért ha $p < 1/8$, az $a-2$ és $a+2$ pontok kihagyhatók.

```{.python .input}
#@tab mxnet
# Intervallum ábrázolása, amikor p < 1/8
plot_chebyshev(0.0, 0.05)
```

```{.python .input}
#@tab pytorch
# Intervallum ábrázolása, amikor p < 1/8
plot_chebyshev(0.0, torch.tensor(0.05))
```

```{.python .input}
#@tab tensorflow
# Intervallum ábrázolása, amikor p < 1/8
plot_chebyshev(0.0, tf.constant(0.05))
```

### Várható értékek és szórásnégyzetek folytonos esetben

Mindez eddig diszkrét valószínűségi változókra vonatkozott, de a folytonos eset hasonló. Az intuíció megértéséhez képzeljük el, hogy a valós számegyenest $\epsilon$ hosszú intervallumokra osztjuk fel: $(\epsilon i, \epsilon (i+1)]$. Ezzel a folytonos valószínűségi változónk diszkréttá vált, és a :eqref:`eq_exp_def` felhasználásával írhatjuk, hogy

$$
\begin{aligned}
\mu_X & \approx \sum_{i} (\epsilon i)P(X \in (\epsilon i, \epsilon (i+1)]) \\
& \approx \sum_{i} (\epsilon i)p_X(\epsilon i)\epsilon, \\
\end{aligned}
$$

ahol $p_X$ az $X$ sűrűségfüggvénye. Ez közelíti az $xp_X(x)$ függvény integráljait, ezért azt kapjuk:

$$
\mu_X = \int_{-\infty}^\infty xp_X(x) \; dx.
$$

Hasonlóképpen, a :eqref:`eq_var_def` felhasználásával a szórásnégyzet:

$$
\sigma^2_X = E[X^2] - \mu_X^2 = \int_{-\infty}^\infty x^2p_X(x) \; dx - \left(\int_{-\infty}^\infty xp_X(x) \; dx\right)^2.
$$

A várható értékre, szórásnégyzetre és szórásra vonatkozóan minden korábban elmondott ebben az esetben is érvényes. Például, ha a sűrűségfüggvény

$$
p(x) = \begin{cases}
1 & x \in [0,1], \\
0 & \textrm{otherwise}.
\end{cases}
$$

kiszámíthatjuk:

$$
\mu_X = \int_{-\infty}^\infty xp(x) \; dx = \int_0^1 x \; dx = \frac{1}{2}.
$$

és

$$
\sigma_X^2 = \int_{-\infty}^\infty x^2p(x) \; dx - \left(\frac{1}{2}\right)^2 = \frac{1}{3} - \frac{1}{4} = \frac{1}{12}.
$$

Figyelmeztetésképpen vizsgáljunk meg egy másik példát, a *Cauchy-eloszlást*. Ennek p.d.f.-je:

$$
p(x) = \frac{1}{1+x^2}.
$$

```{.python .input}
#@tab mxnet
# A Cauchy-eloszlás p.d.f.-jének ábrázolása
x = np.arange(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
# A Cauchy-eloszlás p.d.f.-jének ábrázolása
x = torch.arange(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
# A Cauchy-eloszlás p.d.f.-jének ábrázolása
x = tf.range(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

Ez a függvény ártatlannak tűnik, és egy integrálok táblázatában ellenőrizve valóban látható, hogy alatta a terület egységnyi, tehát egy folytonos valószínűségi változót definiál.

Hogy lássuk, mi a baj, próbáljuk meg kiszámítani a szórásnégyzetet. Ehhez a :eqref:`eq_var_def` alapján a következő integrált kellene kiszámítanunk:

$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx.
$$

A belső függvény így néz ki:

```{.python .input}
#@tab mxnet
# A szórásnégyzet kiszámításához szükséges integrandus ábrázolása
x = np.arange(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

```{.python .input}
#@tab pytorch
# A szórásnégyzet kiszámításához szükséges integrandus ábrázolása
x = torch.arange(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

```{.python .input}
#@tab tensorflow
# A szórásnégyzet kiszámításához szükséges integrandus ábrázolása
x = tf.range(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

Ennek a függvénynek nyilván végtelen a területe alatta, hiszen lényegében az egyes konstans kis mélyedéssel közel a nullához, és valóban belátható, hogy

$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx = \infty.
$$

Ez azt jelenti, hogy nincs jól meghatározott véges szórásnégyzete.

Mélyebbre ásva azonban még aggasztóbb eredményre jutunk. Próbáljuk meg kiszámítani a várható értéket a :eqref:`eq_exp_def` alapján. A változócsere-formulával:

$$
\mu_X = \int_{-\infty}^{\infty} \frac{x}{1+x^2} \; dx = \frac{1}{2}\int_1^\infty \frac{1}{u} \; du.
$$

A belső integrál a logaritmus definíciója, tehát ez lényegében $\log(\infty) = \infty$, vagyis nincs jól meghatározott átlagérték sem!

A gépi tanulás kutatói úgy definiálják a modelleiket, hogy a legtöbb esetben nem kell ilyen problémákkal szembesülniük, és az esetek túlnyomó többségében jól meghatározott várható értékű és szórásnégyzetű valószínűségi változókkal dolgoznak. Ugyanakkor időnként a *nehéz farkú* valószínűségi változók (azok, amelyeknél a nagy értékek valószínűsége elég nagy ahhoz, hogy például a várható érték vagy a szórásnégyzet meghatározatlan legyen) hasznosak fizikai rendszerek modellezésénél, ezért érdemes tudni róluk.

### Együttes sűrűségfüggvények

A fentiek mind egyetlen valós értékű valószínűségi változóra vonatkoztak. De mi a helyzet, ha két vagy több, esetleg erősen korrelált valószínűségi változóval dolgozunk? Ez a helyzet a gépi tanulásban általános: gondoljunk az $R_{i, j}$ valószínűségi változókra, amelyek egy kép $(i, j)$ koordinátájú pixelének pirosértékét kódolják, vagy a $P_t$ valószínűségi változóra, amelyet a $t$ időpontbeli részvényár ad. A szomszédos pixelek általában hasonló színűek, a közeli időpontok általában hasonló árakat mutatnak. Nem kezelhetjük őket külön valószínűségi változóként, ha sikeres modellt szeretnénk alkotni (a :numref:`sec_naive_bayes` részben látni fogunk egy modellt, amely egy ilyen feltételezés miatt gyengén teljesít). Szükségünk van a matematikai eszköztárra, amellyel kezelhetjük ezeket a korrelált folytonos valószínűségi változókat.

Szerencsére a :numref:`sec_integral_calculus` többváltozós integráljainak segítségével kifejleszthetjük ezt az eszköztárt. Tegyük fel, hogy az egyszerűség kedvéért két korrelált valószínűségi változóval, $X$-szel és $Y$-nal dolgozunk. Ekkor, hasonlóan az egységváltozós esethez, feltehetjük a kérdést:

$$
P(X \;\textrm{egy}\; \epsilon \textrm{-méretű intervallumba esik}\; x \;\textrm{körül és}\; Y \;\textrm{egy}\; \epsilon \textrm{-méretű intervallumba esik}\; y \;\textrm{körül} ).
$$

Az egységváltozós esethez hasonló érveléssel ez közelítőleg

$$
P(X \;\textrm{egy}\; \epsilon \textrm{-méretű intervallumba esik}\; x \;\textrm{körül és}\; Y \;\textrm{egy}\; \epsilon \textrm{-méretű intervallumba esik}\; y \;\textrm{körül} ) \approx \epsilon^{2}p(x, y),
$$

valamely $p(x, y)$ függvényre. Ezt $X$ és $Y$ *együttes sűrűségének* nevezzük. Az egységváltozós esethez hasonló tulajdonságok érvényesek erre is:

* $p(x, y) \ge 0$;
* $\int _ {\mathbb{R}^2} p(x, y) \;dx \;dy = 1$;
* $P((X, Y) \in \mathcal{D}) = \int _ {\mathcal{D}} p(x, y) \;dx \;dy$.

Ily módon kezelhetünk több, esetleg korrelált valószínűségi változót. Ha kettőnél több valószínűségi változóval kívánunk dolgozni, a többváltozós sűrűséget tetszőleges számú koordinátára kiterjeszthetjük: $p(\mathbf{x}) = p(x_1, \ldots, x_n)$. A nemnegatívitás és az egységnyi összterület tulajdonságai megmaradnak.

### Marginális eloszlások
Több változóval való munka során sokszor el akarjuk hanyagolni a kapcsolatokat, és csak azt kérdezzük: „Hogyan oszlik el ez az egyetlen változó?" Az ilyen eloszlást *marginális eloszlásnak* nevezzük.

Konkrét formában: legyen adott két $X, Y$ valószínűségi változó $p _ {X, Y}(x, y)$ együttes sűrűséggel. Az index azt jelzi, mely valószínűségi változókhoz tartozik a sűrűség. A marginális eloszlás meghatározásának feladata ebből a függvényből kinyerni a $p _ X(x)$ függvényt.

A legtöbb esetben érdemes visszatérni az intuitív képhez. Idézzük fel, hogy a sűrűség az a $p _ X$ függvény, amelyre

$$
P(X \in [x, x+\epsilon]) \approx \epsilon \cdot p _ X(x).
$$

$Y$-ról nincs szó, de ha csak $p _{X, Y}$ adott, akkor $Y$-t valahogy be kell vonnunk. Először megfigyelhetjük, hogy ez ugyanaz, mint

$$
P(X \in [x, x+\epsilon] \textrm{, és } Y \in \mathbb{R}) \approx \epsilon \cdot p _ X(x).
$$

A sűrűségfüggvényünk közvetlenül nem ad felvilágosítást erről, ezért $y$-ban is kis intervallumokra osztjuk fel, így írhatjuk:

$$
\begin{aligned}
\epsilon \cdot p _ X(x) & \approx \sum _ {i} P(X \in [x, x+\epsilon] \textrm{, és } Y \in [\epsilon \cdot i, \epsilon \cdot (i+1)]) \\
& \approx \sum _ {i} \epsilon^{2} p _ {X, Y}(x, \epsilon\cdot i).
\end{aligned}
$$

![Ha a valószínűségek tömbjeinek oszlopai mentén összegezünk, megkapjuk az $\mathit{x}$-tengelyen ábrázolt valószínűségi változó marginális eloszlását.](../img/marginal.svg)
:label:`fig_marginal`

Ez azt mondja, hogy az egymást követő négyzeteken a sűrűség értékeit kell összeadni egy egyenes mentén, ahogy az :numref:`fig_marginal` ábrán látható. Valóban, miután mindkét oldalról ejtünk egy epsilon faktort, és felismerjük, hogy a jobb oldali összeg $y$ szerinti integrál, azt kapjuk:

$$
\begin{aligned}
 p _ X(x) &  \approx \sum _ {i} \epsilon p _ {X, Y}(x, \epsilon\cdot i) \\
 & \approx \int_{-\infty}^\infty p_{X, Y}(x, y) \; dy.
\end{aligned}
$$

Tehát:

$$
p _ X(x) = \int_{-\infty}^\infty p_{X, Y}(x, y) \; dy.
$$

Ez azt mondja, hogy a marginális eloszlás előállításához a nem kívánt változók szerint integrálunk. Ezt a folyamatot a nem kívánt változók *kintegrálásának* vagy *marginalizálásának* is nevezik.

### Kovarianciák

Több valószínűségi változóval való munka során van egy további hasznos összefoglaló statisztika: a *kovariancia*. Ez azt méri, hogy két valószínűségi változó mennyire ingadozik együtt.

Tegyük fel, hogy van két $X$ és $Y$ valószínűségi változónk; először legyenek diszkrétek, amelyek $(x_i, y_j)$ értékpárokat vesznek fel $p_{ij}$ valószínűséggel. Ekkor a kovariancia:

$$\sigma_{XY} = \textrm{Cov}(X, Y) = \sum_{i, j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij}. = E[XY] - E[X]E[Y].$$
:eqlabel:`eq_cov_def`

Az intuíció megértéséhez tekintsük a következő valószínűségi változópárt. Tegyük fel, hogy $X$ az $1$ és $3$ értékeket veszi fel, $Y$ pedig a $-1$ és $3$ értékeket. Legyenek a valószínűségek a következők:

$$
\begin{aligned}
P(X = 1 \; \textrm{és} \; Y = -1) & = \frac{p}{2}, \\
P(X = 1 \; \textrm{és} \; Y = 3) & = \frac{1-p}{2}, \\
P(X = 3 \; \textrm{és} \; Y = -1) & = \frac{1-p}{2}, \\
P(X = 3 \; \textrm{és} \; Y = 3) & = \frac{p}{2},
\end{aligned}
$$

ahol $p$ egy $[0,1]$ értékű paraméter, amelyet mi választhatunk. Vegyük észre, hogy ha $p=1$, akkor mindkét változó egyszerre veszi fel minimális vagy maximális értékét; ha $p=0$, akkor garantáltan felcserélt értékeket vesznek fel (az egyik nagy, amikor a másik kicsi, és fordítva). Ha $p=1/2$, akkor a négy lehetőség egyforma valószínűségű, és a kettő között nincs kapcsolat. Számítsuk ki a kovarianciát! Először: $\mu_X = 2$ és $\mu_Y = 1$, így :eqref:`eq_cov_def` alapján:

$$
\begin{aligned}
\textrm{Cov}(X, Y) & = \sum_{i, j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij} \\
& = (1-2)(-1-1)\frac{p}{2} + (1-2)(3-1)\frac{1-p}{2} + (3-2)(-1-1)\frac{1-p}{2} + (3-2)(3-1)\frac{p}{2} \\
& = 4p-2.
\end{aligned}
$$

Ha $p=1$ (mindkét változó egyszerre maximálisan pozitív vagy negatív), a kovariancia $2$. Ha $p=0$ (felcserélt eset), a kovariancia $-2$. Végül ha $p=1/2$ (a változók nem kapcsolódnak egymáshoz), a kovariancia $0$. Tehát a kovariancia azt méri, hogyan függenek össze a valószínűségi változók.

Fontos megjegyezni, hogy a kovariancia csak lineáris kapcsolatokat mér. Összetettebb összefüggések, mint például az $X = Y^2$, ahol $Y$ egyenlő valószínűséggel veszi fel a $\{-2, -1, 0, 1, 2\}$ értékeket, elveszhetnek. Valóban, gyors számítással látható, hogy ezeknek a valószínűségi változóknak zéró a kovarianciájuk, jóllehet az egyik a másik determinisztikus függvénye.

Folytonos valószínűségi változóknál hasonló a helyzet. Ezen a ponton már elég jártasak vagyunk a diszkrét és folytonos eset közötti átmenetben, ezért a :eqref:`eq_cov_def` folytonos megfelelőjét levezetés nélkül közöljük:

$$
\sigma_{XY} = \int_{\mathbb{R}^2} (x-\mu_X)(y-\mu_Y)p(x, y) \;dx \;dy.
$$

A szemléltetés érdekében nézzük meg a különféle kovarianciaértékű valószínűségi változók gyűjteményét.

```{.python .input}
#@tab mxnet
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = np.random.normal(0, 1, 500)
    Y = covs[i]*X + np.random.normal(0, 1, (500))

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.asnumpy(), Y.asnumpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = torch.randn(500)
    Y = covs[i]*X + torch.randn(500)

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = tf.random.normal((500, ))
    Y = covs[i]*X + tf.random.normal((500, ))

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

A kovariancia néhány tulajdonsága:

* Bármely $X$ valószínűségi változóra $\textrm{Cov}(X, X) = \textrm{Var}(X)$.
* Bármely $X, Y$ valószínűségi változóra és $a$, $b$ számokra: $\textrm{Cov}(aX+b, Y) = \textrm{Cov}(X, aY+b) = a\textrm{Cov}(X, Y)$.
* Ha $X$ és $Y$ függetlenek, akkor $\textrm{Cov}(X, Y) = 0$.

Ezen kívül a kovarianciával kibővíthetjük a korábban megismert összefüggést. Ha $X$ és $Y$ két független valószínűségi változó, akkor

$$
\textrm{Var}(X+Y) = \textrm{Var}(X) + \textrm{Var}(Y).
$$

A kovariancia ismeretével ezt az összefüggést általánosíthatjuk. Valóban, némi algebrával belátható, hogy általánosan:

$$
\textrm{Var}(X+Y) = \textrm{Var}(X) + \textrm{Var}(Y) + 2\textrm{Cov}(X, Y).
$$

Ez lehetővé teszi a szórásnégyzet összeadási szabályának általánosítását korrelált valószínűségi változókra.

### Korreláció

Ahogy a várható értékek és szórásnégyzetek esetén is tettük, most vizsgáljuk meg a mértékegységeket. Ha $X$ egy mértékegységben (pl. hüvelykben), $Y$ pedig egy másikban (pl. dollárban) mérhető, akkor a kovariancia a két mértékegység szorzatában, $\textrm{hüvelyk} \times \textrm{dollár}$ értékben fejezhető ki. Ezek a mértékegységek nehezen értelmezhetők. Sokszor mértékegység nélküli kapcsolati mértékre van szükségünk. Általában nem az egzakt kvantitatív korrelációra vagyunk kíváncsiak, hanem arra, hogy a korreláció ugyanabba az irányba mutat-e, és milyen erős a kapcsolat.

Hogy megértsük, mi az ésszerű, végezzünk el egy gondolatkísérletet. Tegyük fel, hogy a hüvelykben és dollárban mért valószínűségi változókat hüvelykre és centre váltjuk. Ekkor $Y$ valószínűségi változót $100$-szorosára növeljük. A definícióból következik, hogy $\textrm{Cov}(X, Y)$ is $100$-szorosára nő. Tehát egy mértékegység-változás a kovariancia $100$-szoros változásával jár. A mértékegységtől független korrelációs mértékhez tehát el kell osztani valamivel, ami szintén $100$-szorosára változik. Kézenfekvő jelölt a szórás! Valóban, ha a *korrelációs együtthatót* így definiáljuk:

$$\rho(X, Y) = \frac{\textrm{Cov}(X, Y)}{\sigma_{X}\sigma_{Y}},$$
:eqlabel:`eq_cor_def`

látjuk, hogy ez mértékegység nélküli. Egy kis matematikával belátható, hogy ez a szám $-1$ és $1$ közé esik, ahol $1$ a maximálisan pozitív korreláció, $-1$ pedig a maximálisan negatív korreláció.

Visszatérve a korábbi diszkrét példánkhoz, $\sigma_X = 1$ és $\sigma_Y = 2$, tehát :eqref:`eq_cor_def` alapján a két valószínűségi változó korrelációs együtthatója:

$$
\rho(X, Y) = \frac{4p-2}{1\cdot 2} = 2p-1.
$$

Ez $-1$ és $1$ közé esik, ahol $1$ a legszorosabb pozitív, $-1$ a legszorosabb negatív korreláció – az elvárt viselkedéssel.

Másik példaként tekintsük $X$-et tetszőleges valószínűségi változónak, $Y=aX+b$-t pedig $X$ tetszőleges lineáris determinisztikus függvényének. Ekkor kiszámítható, hogy

$$\sigma_{Y} = \sigma_{aX+b} = |a|\sigma_{X},$$

$$\textrm{Cov}(X, Y) = \textrm{Cov}(X, aX+b) = a\textrm{Cov}(X, X) = a\textrm{Var}(X),$$

és így :eqref:`eq_cor_def` alapján:

$$
\rho(X, Y) = \frac{a\textrm{Var}(X)}{|a|\sigma_{X}^2} = \frac{a}{|a|} = \textrm{sign}(a).
$$

Tehát a korreláció $+1$ minden $a > 0$ esetén, és $-1$ minden $a < 0$ esetén, ami azt szemlélteti, hogy a korreláció a két valószínűségi változó kapcsolatának mértékét és irányát méri, nem az ingadozás léptékét.

Rajzoljuk fel ismét a különféle korrelációjú valószínűségi változók gyűjteményét.

```{.python .input}
#@tab mxnet
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = np.random.normal(0, 1, 500)
    Y = cors[i] * X + np.sqrt(1 - cors[i]**2) * np.random.normal(0, 1, 500)

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.asnumpy(), Y.asnumpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = torch.randn(500)
    Y = cors[i] * X + torch.sqrt(torch.tensor(1) -
                                 cors[i]**2) * torch.randn(500)

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = tf.random.normal((500, ))
    Y = cors[i] * X + tf.sqrt(tf.constant(1.) -
                                 cors[i]**2) * tf.random.normal((500, ))

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

A korreláció néhány tulajdonsága:

* Bármely $X$ valószínűségi változóra $\rho(X, X) = 1$.
* Bármely $X, Y$ valószínűségi változóra és $a$, $b$ számokra: $\rho(aX+b, Y) = \rho(X, aY+b) = \rho(X, Y)$.
* Ha $X$ és $Y$ függetlenek és nem nulla szórásnégyzetűek, akkor $\rho(X, Y) = 0$.

Végül megjegyezzük, hogy ezek a képletek ismerősnek tűnhetnek. Valóban, ha mindent kifejtünk azzal a feltételezéssel, hogy $\mu_X = \mu_Y = 0$, azt kapjuk:

$$
\rho(X, Y) = \frac{\sum_{i, j} x_iy_ip_{ij}}{\sqrt{\sum_{i, j}x_i^2 p_{ij}}\sqrt{\sum_{i, j}y_j^2 p_{ij}}}.
$$

Ez egy tagok szorzatainak összege osztva tagok négyzetösszegének négyzetgyökeivel. Ez pontosan a $\mathbf{v}, \mathbf{w}$ vektorok szögének koszinuszának képlete, ahol a különböző koordinátákat $p_{ij}$-vel súlyozzuk:

$$
\cos(\theta) = \frac{\mathbf{v}\cdot \mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|} = \frac{\sum_{i} v_iw_i}{\sqrt{\sum_{i}v_i^2}\sqrt{\sum_{i}w_i^2}}.
$$

Valóban, ha a normákat szórásokhoz, a korrelációkat szögek koszinuszaihoz kötjük, a geometriai intuíciók nagy része alkalmazható a valószínűségi változókra való gondolkodásban.

## Összefoglalás
* A folytonos valószínűségi változók olyan valószínűségi változók, amelyek értékek folytonos halmazát vehetik fel. Technikai nehézségeik miatt bonyolultabban kezelhetők, mint a diszkrét valószínűségi változók.
* A valószínűségi sűrűségfüggvény lehetővé teszi a folytonos valószínűségi változókkal való munkát: egy olyan függvényt ad, amelynek egy intervallumon vett területe megadja a mintapontnak abba az intervallumba esésének valószínűségét.
* A kumulatív eloszlásfüggvény annak valószínűsége, hogy a valószínűségi változó egy adott küszöbértéknél kisebb értéket vesz fel. Hasznos alternatív nézőpontot kínál, amely egységes keretbe foglalja a diszkrét és folytonos változókat.
* A várható érték a valószínűségi változó átlagos értéke.
* A szórásnégyzet a valószínűségi változó és várható értéke különbségének várható négyzetértéke.
* A szórás a szórásnégyzet négyzetgyöke. Felfogható a valószínűségi változó által felvehető értékek tartományának mérőszámaként.
* A Csebisev-egyenlőtlenség ezt az intuíciót rigorózussá teszi, explicit intervallumot adva, amelybe a valószínűségi változó az esetek nagy részében esik.
* Az együttes sűrűségek lehetővé teszik a korrelált valószínűségi változókkal való munkát. A nem kívánt valószínűségi változók szerinti integrálással marginalizálhatjuk az együttes sűrűségeket, és megkaphatjuk a kívánt valószínűségi változó eloszlását.
* A kovariancia és a korrelációs együttható módszert ad két korrelált valószínűségi változó lineáris kapcsolatának mérésére.

## Feladatok
1. Tegyük fel, hogy van egy valószínűségi változónk, amelynek sűrűségfüggvénye $p(x) = \frac{1}{x^2}$, ha $x \ge 1$, egyébként $p(x) = 0$. Mi $P(X > 2)$?
2. A Laplace-eloszlás egy olyan valószínűségi változó, amelynek sűrűségfüggvénye $p(x) = \frac{1}{2}e^{-|x|}$. Mi ennek a függvénynek a várható értéke és szórása? Segítség: $\int_0^\infty xe^{-x} \; dx = 1$ és $\int_0^\infty x^2e^{-x} \; dx = 2$.
3. Az utcán megállítasz valakit, aki azt mondja: „Van egy valószínűségi változóm, amelynek várható értéke $1$, szórása $2$, és megfigyeltem, hogy a minták $25\%$-a $9$-nél nagyobb értéket vett fel." Elhiszed? Miért vagy miért nem?
4. Tegyük fel, hogy van két $X, Y$ valószínűségi változód, amelyek együttes sűrűségfüggvénye $p_{XY}(x, y) = 4xy$, ha $x, y \in [0,1]$, egyébként $p_{XY}(x, y) = 0$. Mi $X$ és $Y$ kovarianciája?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/415)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1094)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1095)
:end_tab:
