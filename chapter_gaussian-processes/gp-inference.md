```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("gpytorch")
```

# Gauss-folyamat inferencia

Ebben a szakaszban megmutatjuk, hogyan kell poszterior inferenciát elvégezni és előrejelzéseket készíteni az előző szakaszban bevezetett GP priorok segítségével. Regresszióval kezdünk, ahol az inferenciát _zárt formában_ végezhetjük el. Ez egy „GP-k dióhéjban" szakasz, amellyel gyorsan elkezdhetjük a Gauss-folyamatok gyakorlati alkalmazását. Alap operációkat kódolunk nulláról, majd bevezetjük a [GPyTorch](https://gpytorch.ai/) könyvtárat, amely sokkal kényelmesebbé teszi a legkorszerűbb Gauss-folyamatokkal való munkát és a mély neurális hálózatokkal való integrációt. Ezeket a fejlettebb témákat mélyebben a következő szakaszban tárgyaljuk. Abban a szakaszban olyan eseteket is megvizsgálunk, ahol közelítő inferencia szükséges — osztályozás, pontfolyamatok, vagy bármely nem Gauss-féle valószínűség esetén.

## Poszterior inferencia regresszióhoz

Egy _megfigyelési_ modell az általunk megtanulni kívánt függvényt, $f(x)$-et, a megfigyeléseinkkel $y(x)$-szel kapcsolja össze, mindkettőt valamely $x$ bemenet indexeli. Az osztályozásban $x$ lehet egy kép pixeleinek halmaza, $y$ pedig a hozzá tartozó osztálycímke. A regresszióban $y$ általában egy folytonos kimenetet képvisel, például a felszíni hőmérsékletet, a tengerszintet, a $CO_2$-koncentrációt stb.

A regresszióban gyakran feltételezzük, hogy a kimenetek egy látens zajmentes $f(x)$ függvényből és i.i.d. Gauss-zajból $\epsilon(x)$ adódnak:

$$y(x) = f(x) + \epsilon(x),$$
:eqlabel:`eq_gp-regression`

ahol $\epsilon(x) \sim \mathcal{N}(0,\sigma^2)$. Legyen $\mathbf{y} = y(X) = (y(x_1),\dots,y(x_n))^{\top}$ a tanítási megfigyeléseink vektora, és $\textbf{f} = (f(x_1),\dots,f(x_n))^{\top}$ a látens zajmentes függvényértékek vektora, a tanítási bemeneteknél $X = {x_1, \dots, x_n}$ lekérdezve.

Feltesszük, hogy $f(x) \sim \mathcal{GP}(m,k)$, ami azt jelenti, hogy bármely függvényértékek gyűjteménye $\textbf{f}$ együttes multivariát Gauss-eloszlással rendelkezik, $\mu_i = m(x_i)$ átlagvektorral és $K_{ij} = k(x_i,x_j)$ kovariancia-mátrixszal. Az RBF kernel $k(x_i,x_j) = a^2 \exp\left(-\frac{1}{2\ell^2}||x_i-x_j||^2\right)$ standard kovariancia-függvény választás lenne. A jelölés egyszerűsítése érdekében feltételezzük, hogy az átlagfüggvény $m(x)=0$; a levezetések könnyen általánosíthatók később.

Tegyük fel, hogy egy bemeneti halmaz $X_* = x_{*1},x_{*2},\dots,x_{*m}$ pontjainál szeretnénk előrejelzést készíteni. Ekkor meg szeretnénk találni $p(\mathbf{f}_* | \mathbf{y}, X)$-et. A regresszióban ezt az eloszlást kényelmesen megtalálhatjuk Gauss-azonosságok segítségével, miután megtaláltuk $\mathbf{f}_* = f(X_*)$ és $\mathbf{y}$ együttes eloszlását.

Ha a :eqref:`eq_gp-regression` egyenletet a tanítási bemeneteknél $X$-nél értékeljük ki, akkor $\mathbf{y} = \mathbf{f} + \mathbf{\epsilon}$. A Gauss-folyamat definíciója szerint (lásd az előző szakaszt) $\mathbf{f} \sim \mathcal{N}(0,K(X,X))$, ahol $K(X,X)$ egy $n \times n$ mátrix, amelyet a kovariancia-függvényünk (más néven _kernel_) összes lehetséges $x_i, x_j \in X$ bemeneti párra való kiértékelésével kapunk. $\mathbf{\epsilon}$ egyszerűen egy vektor, amely $\mathcal{N}(0,\sigma^2)$-ből vett i.i.d. mintákból áll, és ezért eloszlása $\mathcal{N}(0,\sigma^2I)$. $\mathbf{y}$ tehát két független multivariát Gauss-változó összege, és eloszlása $\mathcal{N}(0, K(X,X) + \sigma^2I)$. Megmutatható továbbá, hogy $\textrm{cov}(\mathbf{f}_*, \mathbf{y}) = \textrm{cov}(\mathbf{y},\mathbf{f}_*)^{\top} = K(X_*,X)$, ahol $K(X_*,X)$ egy $m \times n$ mátrix, amelyet a kernel kiértékelésével kapunk az összes teszt- és tanítási bemeneti párra.

$$
\begin{bmatrix}
\mathbf{y} \\
\mathbf{f}_*
\end{bmatrix}
\sim
\mathcal{N}\left(0, 
\mathbf{A} = \begin{bmatrix}
K(X,X)+\sigma^2I & K(X,X_*) \\
K(X_*,X) & K(X_*,X_*)
\end{bmatrix}
\right)
$$

Ezután standard Gauss-azonosságokkal megtalálhatjuk a feltételes eloszlást az együttes eloszlásból (lásd például Bishop, 2. fejezet):
$\mathbf{f}_* | \mathbf{y}, X, X_* \sim \mathcal{N}(m_*,S_*)$, ahol $m_* = K(X_*,X)[K(X,X)+\sigma^2I]^{-1}\textbf{y}$, és $S = K(X_*,X_*) - K(X_*,X)[K(X,X)+\sigma^2I]^{-1}K(X,X_*)$.

Általában nem szükséges a teljes prediktív kovariancia-mátrixot $S$-t alkalmazni, ehelyett $S$ diagonálisát használjuk az egyes előrejelzések bizonytalanságához. Ezért az előrejelzési eloszlást általában egy teszt ponthoz $x_*$ írjuk fel, nem tesztpontok gyűjteményéhez.

A kernel mátrixnak $\theta$ paraméterei vannak, amelyeket szintén meg szeretnénk becsülni, például az RBF kernel fenti $a$ amplitúdóját és $\ell$ hosszskáláját. Erre a célra a _marginális valószínűséget_ $p(\textbf{y} | \theta, X)$ használjuk, amelyet már levezettük a marginális eloszlások meghatározásakor, amikor megkerestük $\textbf{y},\textbf{f}_*$ együttes eloszlását. Amint látni fogjuk, a marginális valószínűség modellilleszkedési és modelkomplexitási tagokra bomlik, és automatikusan kódolja az Occam borotvájának elvét a hiperparaméterek tanulásához. A teljes tárgyaláshoz lásd MacKay 28. fejezete :cite:`mackay2003information`, és Rasmussen és Williams 5. fejezete :cite:`rasmussen2006gaussian`.

```{.python .input}
from d2l import torch as d2l
import numpy as np
from scipy.spatial import distance_matrix
from scipy import optimize
import matplotlib.pyplot as plt
import math
import torch
import gpytorch
import os

d2l.set_figsize()
```

## Egyenletek az előrejelzésekhez és a kernel hiperparaméterek tanulásához GP regresszióban

Az alábbiak a GP regresszióban a hiperparaméterek tanulásához és előrejelzések készítéséhez szükséges egyenletek. Feltételezzük az $X = \{x_1,\dots,x_n\}$ bemenetek által indexelt $\textbf{y}$ regressziós célvektort, és $x_*$ tesztbemenethez szeretnénk előrejelzést készíteni. I.i.d. additív, nulla átlagú Gauss-zajt feltételezünk $\sigma^2$ varianciával. A látens zajmentes függvényhez GP priort $f(x) \sim \mathcal{GP}(m,k)$ alkalmazunk $m$ átlagfüggvénnyel és $k$ kernelfüggvénnyel. A kernelnek $\theta$ paraméterei vannak, amelyeket meg szeretnénk tanulni. Például RBF kernel esetén $k(x_i,x_j) = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right)$, a $\theta = \{a^2, \ell^2\}$ értékeket akarjuk megtanulni. Legyen $K(X,X)$ egy $n \times n$ mátrix, amely az összes $n$ tanítási bemeneti pár kernelértékéből áll. Legyen $K(x_*,X)$ egy $1 \times n$ vektor $k(x_*, x_i)$ kiértékelésével, $i=1,\dots,n$. Legyen $\mu$ egy átlagvektor, amelyet az $m(x)$ átlagfüggvény kiértékelésével kapunk minden egyes $x$ tanítási pontban.

A Gauss-folyamatokkal való munkában általában kétlépéses eljárást követünk:
1. A kernel hiperparamétereit $\hat{\theta}$ a marginális valószínűség maximalizálásával tanítjuk meg ezekre a hiperparaméterekre vonatkozóan.
2. A prediktív átlagot pontbecslőként, a prediktív szórás 2-szeresét pedig 95\%-os megbízhatósági tartomány kialakítására használjuk, kondicionálva az ezeken a tanult $\hat{\theta}$ hiperparamétereken.

A log marginális valószínűség egyszerűen egy log Gauss-sűrűség, amelynek alakja:
$$\log p(\textbf{y} | \theta, X) = -\frac{1}{2}\textbf{y}^{\top}[K_{\theta}(X,X) + \sigma^2I]^{-1}\textbf{y} - \frac{1}{2}\log|K_{\theta}(X,X)| + c$$

A prediktív eloszlás alakja:
$$p(y_* | x_*, \textbf{y}, \theta) = \mathcal{N}(a_*,v_*)$$
$$a_* = k_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}(\textbf{y}-\mu) + \mu$$
$$v_* = k_{\theta}(x_*,x_*) - K_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}k_{\theta}(X,x_*)$$

## A tanuláshoz és előrejelzésekhez szükséges egyenletek értelmezése

A Gauss-folyamatok prediktív eloszlásaival kapcsolatban néhány kulcsfontosságú megjegyzést kell tenni:

* A modelosztály rugalmassága ellenére a GP regresszió esetén lehetséges _egzakt_ Bayesi inferenciát végezni _zárt formában_. A kernel hiperparaméterek tanulásán kívül nincs _tanítás_. Pontosan felírhatjuk, hogy milyen egyenleteket szeretnénk alkalmazni az előrejelzések elkészítéséhez. A Gauss-folyamatok ebből a szempontból viszonylag kivételesek, ami nagyban hozzájárult kényelmes, sokoldalú és tartósan népszerű voltukhoz. 

* A prediktív átlag $a_*$ a tanítási célok $\textbf{y}$ lineáris kombinációja, a kernel $k_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}$ súlyozásával. Amint látni fogjuk, a kernel (és hiperparaméterei) ezért kulcsszerepet játszik a modell általánosítási tulajdonságaiban.

* A prediktív átlag explicit módon függ a $\textbf{y}$ célértékektől, de a prediktív variancia nem. A prediktív bizonytalanság ehelyett növekszik, ahogy a $x_*$ tesztbemenet eltávolodik az $X$ célhelyzetektől, ahogy azt a kernelfüggvény szabályozza. A bizonytalanság azonban implicit módon függ a $\textbf{y}$ célok értékeitől a $\theta$ kernel hiperparamétereken keresztül, amelyeket az adatokból tanulunk meg.

* A marginális valószínűség modellilleszkedési és modelkomplexitási (log determináns) tagokra bomlik. A marginális valószínűség hajlamos olyan hiperparamétereket kiválasztani, amelyek az adatokkal még konzisztens legegyszerűbb illeszkedést biztosítják. 

* A fő számítási szűk keresztmetszetek a lineáris rendszer megoldásából és egy log determináns kiszámításából adódnak egy $n \times n$ szimmetrikus pozitív definit $K(X,X)$ mátrixon $n$ tanítási ponthoz. Naivan ezek a műveletek mindegyike $\mathcal{O}(n^3)$ számítást igényel, valamint $\mathcal{O}(n^2)$ tárhelyet a kernel (kovariancia) mátrix minden bejegyzéséhez, amelyet általában Cholesky-felbontással kezdünk. Történelmileg ezek a szűk keresztmetszetek körülbelül 10 000-nél kevesebb tanítási ponttal rendelkező feladatokra korlátozták a GP-ket, és „lassú” hírnevet adtak a GP-knek, ami már közel egy évtizede pontatlan. A fejlettebb témákban tárgyaljuk, hogyan lehet a GP-ket milliós pontszámú feladatokra skálázni.
* A kernelfüggvények népszerű választásainál $K(X,X)$ sokszor közel szinguláris, ami numerikus problémákat okozhat Cholesky-felbontásoknál vagy más lineáris rendszerek megoldására tervezett műveleteknél. Szerencsére a regresszióban általában $K_{\theta}(X,X)+\sigma^2I$-vel dolgozunk, úgyhogy a zajvariancia $\sigma^2$ hozzáadódik $K(X,X)$ diagonálisához, jelentősen javítva a kondicionálást. Ha a zajvariancia kicsi, vagy zajmentes regressziót végzünk, bevett gyakorlat egy kis mennyiségű "jitter" hozzáadása a diagonálishoz, körülbelül $10^{-6}$ nagyságrendben, a kondicionálás javítása érdekében.

## Nulláról elkészített munkált példa

Hozzunk létre regressziós adatokat, majd illesszünk rájuk egy GP-t, minden lépést nulláról implementálva.
Adatokat fogunk mintavételezni az
$$y(x) = \sin(x) + \frac{1}{2}\sin(4x) + \epsilon,$$ egyenletből, ahol $\epsilon \sim \mathcal{N}(0,\sigma^2)$. A megtalálni kívánt zajmentes függvény $f(x) = \sin(x) + \frac{1}{2}\sin(4x)$. Kezdetben $\sigma = 0.25$ zajszórással dolgozunk.

```{.python .input}
def data_maker1(x, sig):
    return np.sin(x) + 0.5 * np.sin(4 * x) + np.random.randn(x.shape[0]) * sig

sig = 0.25
train_x, test_x = np.linspace(0, 5, 50), np.linspace(0, 5, 500)
train_y, test_y = data_maker1(train_x, sig=sig), data_maker1(test_x, sig=0.)

d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y)
d2l.plt.xlabel("x", fontsize=20)
d2l.plt.ylabel("Observations y", fontsize=20)
d2l.plt.show()
```

Itt a zajos megfigyelések körökként láthatók, a megtalálni kívánt zajmentes függvény pedig kékkel jelölve.

Most adjunk meg egy GP priort a látens zajmentes függvényre, $f(x)\sim \mathcal{GP}(m,k)$. Nulla átlagfüggvényt $m(x) = 0$ és RBF kovariancia-függvényt (kernelt) alkalmazunk:
$$k(x_i,x_j) = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right).$$

```{.python .input}
mean = np.zeros(test_x.shape[0])
cov = d2l.rbfkernel(test_x, test_x, ls=0.2)
```

0.2-es hosszskálával indítunk. Mielőtt az adatokhoz illesztenénk, fontos megvizsgálni, hogy ésszerű priort adtunk-e meg. Vizualizáljunk néhány mintafüggvényt erről a priorról, valamint a 95%-os hihető tartományt (azt hisszük, 95% valószínűséggel a valódi függvény ezen a területen belül van).

```{.python .input}
prior_samples = np.random.multivariate_normal(mean=mean, cov=cov, size=5)
d2l.plt.plot(test_x, prior_samples.T, color='black', alpha=0.5)
d2l.plt.plot(test_x, mean, linewidth=2.)
d2l.plt.fill_between(test_x, mean - 2 * np.diag(cov), mean + 2 * np.diag(cov), 
                 alpha=0.25)
d2l.plt.show()
```

Ésszerűnek tűnnek ezek a minták? A függvények magas szintű tulajdonságai összhangban vannak a modellezni kívánt adatok típusával?

Most alakítsuk ki a poszterior prediktív eloszlás átlagát és varianciáját egy tetszőleges $x_*$ tesztpontban.

$$
\bar{f}_{*} = K(x, x_*)^T (K(x, x) + \sigma^2 I)^{-1}y
$$

$$
V(f_{*}) = K(x_*, x_*) - K(x, x_*)^T (K(x, x) + \sigma^2 I)^{-1}K(x, x_*)
$$

Mielőtt előrejelzéseket készítünk, meg kell tanulnunk a kernel hiperparamétereit $\theta$ és a zajvarianciát $\sigma^2$. Inicializáljuk a hosszskálát 0.75-re, mivel a priorfüggvényeink túl gyorsan változónak tűntek az illeszteni kívánt adatokhoz képest. Zajszórásnak $\sigma = 0.75$-öt tippelünk.

Ezeknek a paramétereknek a megtanulásához a marginális valószínűséget maximalizáljuk a paraméterekre nézve.

$$
\log p(y | X) = \log \int p(y | f, X)p(f | X)df
$$
$$
\log p(y | X) = -\frac{1}{2}y^T(K(x, x) + \sigma^2 I)^{-1}y - \frac{1}{2}\log |K(x, x) + \sigma^2 I| - \frac{n}{2}\log 2\pi
$$


Talán a priorfüggvényeink túl gyorsan változtak. Próbáljunk 0.4-es hosszskálát. Zajszórásnak 0.75-öt tippelünk. Ezek csupán hiperparaméter-inicializálások — ezeket a paramétereket a marginális valószínűségből tanuljuk meg.

```{.python .input}
ell_est = 0.4
post_sig_est = 0.5

def neg_MLL(pars):
    K = d2l.rbfkernel(train_x, train_x, ls=pars[0])
    kernel_term = -0.5 * train_y @ \
        np.linalg.inv(K + pars[1] ** 2 * np.eye(train_x.shape[0])) @ train_y
    logdet = -0.5 * np.log(np.linalg.det(K + pars[1] ** 2 * \
                                         np.eye(train_x.shape[0])))
    const = -train_x.shape[0] / 2. * np.log(2 * np.pi)
    
    return -(kernel_term + logdet + const)


learned_hypers = optimize.minimize(neg_MLL, x0=np.array([ell_est,post_sig_est]), 
                                   bounds=((0.01, 10.), (0.01, 10.)))
ell = learned_hypers.x[0]
post_sig_est = learned_hypers.x[1]
```

Ebben az esetben 0.299-es hosszskálát és 0.24-es zajszórást tanulunk. Fontos megjegyezni, hogy a tanult zaj rendkívül közel van a valódi zajhoz, ami azt jelzi, hogy a GP-nk nagyon jól illeszkedik ehhez a feladathoz.

Általánosságban alapvető fontosságú gondosan megválasztani a kernelt és inicializálni a hiperparamétereket. Bár a marginális valószínűség optimalizálása viszonylag robusztus az inicializáláshoz képest, nem mentes a rossz inicializálásoktól. Próbáld meg futtatni a fenti szkriptet különböző inicializálásokkal, és nézd meg, milyen eredményeket kapsz.

Most készítsünk előrejelzéseket ezekkel a tanult hiperparaméterekkel.

```{.python .input}
K_x_xstar = d2l.rbfkernel(train_x, test_x, ls=ell)
K_x_x = d2l.rbfkernel(train_x, train_x, ls=ell)
K_xstar_xstar = d2l.rbfkernel(test_x, test_x, ls=ell)

post_mean = K_x_xstar.T @ np.linalg.inv((K_x_x + \
                post_sig_est ** 2 * np.eye(train_x.shape[0]))) @ train_y
post_cov = K_xstar_xstar - K_x_xstar.T @ np.linalg.inv((K_x_x + \
                post_sig_est ** 2 * np.eye(train_x.shape[0]))) @ K_x_xstar

lw_bd = post_mean - 2 * np.sqrt(np.diag(post_cov))
up_bd = post_mean + 2 * np.sqrt(np.diag(post_cov))

d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y, linewidth=2.)
d2l.plt.plot(test_x, post_mean, linewidth=2.)
d2l.plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
d2l.plt.legend(['Observed Data', 'True Function', 'Predictive Mean', '95% Set on True Func'])
d2l.plt.show()
```

A narancssárga poszterior átlag szinte tökéletesen illeszkedik a valódi zajmentes függvényhez! Fontos megjegyezni, hogy az általunk megjelenített 95%-os hihető tartomány a látens _zajmentes_ (valódi) függvényre vonatkozik, nem az adatpontokra. Látjuk, hogy ez a hihető tartomány teljes egészében tartalmazza a valódi függvényt, és nem tűnik se túl szélesnek, se túl szűknek. Nem várjuk és nem is kívánjuk, hogy az adatpontokat is tartalmazza. Ha a megfigyelésekre vonatkozó hihető tartományt szeretnénk, a következőt kell kiszámítani:

```{.python .input}
lw_bd_observed = post_mean - 2 * np.sqrt(np.diag(post_cov) + post_sig_est ** 2)
up_bd_observed = post_mean + 2 * np.sqrt(np.diag(post_cov) + post_sig_est ** 2)
```

A bizonytalanságnak két forrása van: az _episztemikus_ bizonytalanság, amely a _csökkenthető_ bizonytalanságot képviseli, és az _aleatórikus_, vagyis _nem csökkenthető_ bizonytalanság. Az _episztemikus_ bizonytalanság itt a zajmentes függvény valódi értékeivel kapcsolatos bizonytalanságot jelenti. Ennek a bizonytalanságnak növekednie kell, ahogy az adatpontoktól távolodunk, mivel az adatoktól messze nagyobb változatossága van az adatainkkal összeegyeztethető függvényértékeknek. Ahogy egyre több adatot megfigyelünk, a valódi függvényről alkotott meggyőződésünk egyre magabiztosabbá válik, és az episztemikus bizonytalanság eltűnik. Az _aleatórikus_ bizonytalanság ebben az esetben a megfigyelési zaj, mivel az adatokat ezzel a zajjal kapjuk, és ez nem csökkenthető.

Az adatokban lévő _episztemikus_ bizonytalanságot a látens zajmentes függvény varianciája, np.diag(post\_cov), ragadja meg. Az _aleatórikus_ bizonytalanságot a zajvariancia post_sig_est**2 fejezi ki.

Sajnos az emberek gyakran hanyagul bánnak a bizonytalanság megjelenítésével: számos cikk teljesen definiálatlan hibasávokat mutat, nem derül ki egyértelműen, hogy episztemikus vagy aleatórikus bizonytalanságot vizualizálnak-e (vagy mindkettőt), és összekeverik a zajvarianciát a zajszórással, a szórást a standard hibával, a konfidenciaintervallumokat a hihető tartományokkal és így tovább. Ha nem pontosítjuk, mit jelképez a bizonytalanság, az lényegében értelmetlen.

Annak érdekében, hogy pontosan figyeljük, mit képvisel a bizonytalanságunk, alapvetően fontos megjegyezni, hogy a zajmentes függvény varianciabecsléséből _kétszeres_ _négyzetgyököt_ veszünk. Mivel a prediktív eloszlásunk Gauss-eloszlás, ez a mennyiség lehetővé teszi számunkra egy 95%-os hihető tartomány kialakítását, amely a valódi függvényt 95% valószínűséggel tartalmazó intervallumba vetett meggyőződésünket fejezi ki. A zaj _varianciája_ teljesen más skálán mozog, és sokkal nehezebben értelmezhető.

Végül nézzük meg a 20 poszterior mintát. Ezek a minták megmutatják, milyen típusú függvényeket gondolunk, hogy a posteriori illeszkedhetnek az adatainkhoz.

```{.python .input}
post_samples = np.random.multivariate_normal(post_mean, post_cov, size=20)
d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y, linewidth=2.)
d2l.plt.plot(test_x, post_mean, linewidth=2.)
d2l.plt.plot(test_x, post_samples.T, color='gray', alpha=0.25)
d2l.plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
plt.legend(['Observed Data', 'True Function', 'Predictive Mean', 'Posterior Samples'])
d2l.plt.show()
```

Alapvető regressziós alkalmazásokban a poszterior prediktív átlagot és szórást leggyakrabban pontbecslőként, illetve a bizonytalanság mérőszámaként alkalmazzuk. Fejlettebb alkalmazásokban, például Monte Carlo akvizíciós függvényeket alkalmazó Bayes-optimalizálásban vagy modelalapú megerősítéses tanuláshoz használt Gauss-folyamatokban, gyakran szükséges poszterior mintákat venni. Ugyanakkor még ha az alapvető alkalmazásokban nem is feltétlenül szükségesek, ezek a minták több intuíciót adnak az adatokra kapott illeszkedésről, és vizualizációkban is hasznosak lehetnek.

## Az élet megkönnyítése GPyTorch segítségével

Amint láttuk, az alapvető Gauss-folyamat regresszió nulláról való implementálása valójában egészen egyszerű. Azonban amint különféle kernelválasztásokat szeretnénk vizsgálni, közelítő inferenciát alkalmazni (ami már az osztályozásnál is szükséges), GP-ket neurális hálózatokkal kombinálni, vagy akár kb. 10 000-nél több pontból álló adathalmazt kezelni, a nulláról való implementálás kezelhetetlenné és körülményessé válik. A skálázható GP-inferencia leghatékonyabb módszereinek némelyike, mint például az SKI (más néven KISS-GP), akár több száz sornyi fejlett numerikus lineáris algebrai rutinokat igényelhet.

Ilyen esetekben a _GPyTorch_ könyvtár sokat könnyít a dolgainkon. A GPyTorch-ot a Gauss-folyamatok numerikájáról és a fejlett módszerekről szóló jövőbeli notebookban bővebben tárgyaljuk. A GPyTorch könyvtár [számos példát](https://github.com/cornellius-gp/gpytorch/tree/master/examples) tartalmaz. A csomag megismeréséhez végigmegyünk az [egyszerű regressziós példán](https://github.com/cornellius-gp/gpytorch/blob/master/examples/01_Exact_GPs/Simple_GP_Regression.ipynb), bemutatva, hogyan alkalmazható a fenti eredményeink reprodukálására GPyTorch segítségével. Ez sok kódnak tűnhet az alapvető regresszió reprodukálásához, és bizonyos értelemben az is. De az alábbiakból csupán néhány sor módosításával azonnal különféle kerneleket, skálázható inferencia-technikákat és közelítő inferenciát alkalmazhatunk, ahelyett hogy potenciálisan több ezer sor új kódot kellene írnunk.

```{.python .input}
# Először alakítsuk az adatainkat tenzorokká a PyTorch-hoz
train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y)
test_y = torch.tensor(test_y)

# Egzakt GP inferenciát alkalmazunk nulla átlaggal és RBF kernellel
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
```

Ez a kódblokk az adatokat a GPyTorch számára megfelelő formátumba helyezi, és megadja, hogy egzakt inferenciát alkalmazunk, valamint a használni kívánt átlagfüggvényt (nulla) és kernelfüggvényt (RBF). Bármely más kernelt nagyon könnyen alkalmazhatunk, például a gpytorch.kernels.matern_kernel() vagy gpytorch.kernels.spectral_mixture_kernel() hívásával. Eddig csak az egzakt inferenciát tárgyaltuk, ahol közelítések nélkül lehetséges prediktív eloszlást levezetni. Gauss-folyamatok esetén egzakt inferenciát csak akkor végezhetünk, ha Gauss-valószínűséggel rendelkezünk; pontosabban akkor, ha feltételezzük, hogy megfigyeléseink egy Gauss-folyamattal reprezentált zajmentes függvényből, valamint Gauss-zajból keletkeznek. A jövőbeli notebookban más eseteket is megvizsgálunk, például az osztályozást, ahol ezek a feltételezések nem teljesíthetők.

```{.python .input}
# Gauss-valószínűség inicializálása
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)
training_iter = 50
# Optimális modell-hiperparaméterek megkeresése
model.train()
likelihood.train()
# Adam optimalizáló használata, tartalmazza a GaussianLikelihood paramétereit
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  
# A veszteségfüggvény a negatív log GP marginális valószínűség
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
```

Itt explicit módon megadjuk a használni kívánt valószínűséget (Gauss), a kernel hiperparaméterek tanításához alkalmazott célfüggvényt (marginális valószínűség), valamint a célfüggvény optimalizálásához választott eljárást (jelen esetben Adam). Megjegyezzük, hogy bár Adam-ot használunk, amely „sztochasztikus" optimalizáló, itt ez teljes-batch Adam. Mivel a marginális valószínűség nem faktorizálható az adatpéldányok felett, nem alkalmazhatunk adatok „mini-batch"-ei felett működő optimalizálót garantált konvergenciával. Más optimalizálók, például az L-BFGS, szintén támogatottak a GPyTorch-ban. A standard mélytanulással ellentétben a marginális valószínűség jó optimalizálása szorosan összefügg a jó általánosítással, ami gyakran a hatékony optimalizálók, például az L-BFGS felé terel bennünket, feltéve, hogy nem túlzottan drágák.

```{.python .input}
for i in range(training_iter):
    # Gradiensek nullázása az előző iterációból
    optimizer.zero_grad()
    # Modell kimenetének kiszámítása
    output = model(train_x)
    # Veszteség kiszámítása és gradiensek visszaterjesztése
    loss = -mll(output, train_y)
    loss.backward()
    if i % 10 == 0:
        print(f'Iter {i+1:d}/{training_iter:d} - Loss: {loss.item():.3f} '
              f'squared lengthscale: '
              f'{model.covar_module.base_kernel.lengthscale.item():.3f} '
              f'noise variance: {model.likelihood.noise.item():.3f}')
    optimizer.step()
```

Itt ténylegesen futtatjuk az optimalizálási eljárást, minden 10. iterációban kiírva a veszteség értékét.

```{.python .input}
# Kiértékelési (prediktív poszterior) módba váltás
test_x = torch.tensor(test_x)
model.eval()
likelihood.eval()
observed_pred = likelihood(model(test_x)) 
```

A fenti kódblokk lehetővé teszi, hogy előrejelzéseket készítsünk a tesztbemeneteinken.

```{.python .input}
with torch.no_grad():
    # Grafikon inicializálása
    f, ax = d2l.plt.subplots(1, 1, figsize=(4, 3))
    # A 95\%-os hihető tartomány felső és alsó határainak meghatározása
    # (itt a megfigyelési térben)
    lower, upper = observed_pred.confidence_region()
    ax.scatter(train_x.numpy(), train_y.numpy())
    ax.plot(test_x.numpy(), test_y.numpy(), linewidth=2.)
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), linewidth=2.)
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.25)
    ax.set_ylim([-1.5, 1.5])
    ax.legend(['True Function', 'Predictive Mean', 'Observed Data',
               '95% Credible Set'])
```

Végül ábrázoljuk az illeszkedést.

Az illeszkedések szinte azonosak. Néhány megjegyzés: a GPyTorch _négyzetes_ hosszskálákkal és megfigyelési zajjal dolgozik. Például a nulláról írt kódban tanult zajszórásunk körülbelül 0.283. A GPyTorch által talált zajvariancia $0.81 \approx 0.283^2$. A GPyTorch grafikonon a hihető tartományt a _megfigyelési térben_ mutatjuk, nem a látens függvénytérben, annak szemléltetésére, hogy valóban lefedik a megfigyelt adatpontokat.

## Összefoglalás

A Gauss-folyamat priort az adatokkal kombinálva poszteriort alkothatunk, amelyet előrejelzések készítésére használunk. Marginális valószínűséget is alkothatunk, amely hasznos a kernel hiperparaméterek automatikus tanulásához; ezek a hiperparaméterek olyan tulajdonságokat szabályoznak, mint a Gauss-folyamat változásának mértéke. A poszterior kialakításának és a kernel hiperparaméterek tanításának regresszióhoz szükséges mechanizmusa egyszerű, mindössze néhány tucat sor kódot igényel. Ez a notebook jó referencia minden olvasónak, aki gyorsan szeretne „belevágni" a Gauss-folyamatok alkalmazásába. Bemutattuk a GPyTorch könyvtárat is. Bár az alapvető regresszióhoz szükséges GPyTorch kód viszonylag hosszú, triviálisan módosítható más kernelfüggvényekre, vagy a jövőbeli notebookban tárgyalandó fejlettebb funkciókra, mint a skálázható inferencia vagy nem Gauss-valószínűségek osztályozáshoz.


## Feladatok

1. Hangsúlyoztuk a kernel hiperparaméterek _tanulásának_ fontosságát, valamint a hiperparaméterek és kernelek hatását a Gauss-folyamatok általánosítási tulajdonságaira. Próbáld meg kihagyni a hiperparaméterek tanulásának lépését, és ehelyett találgass különféle hosszskálákat és zajvarianciákat, majd ellenőrizd, milyen hatással vannak az előrejelzésekre. Mi történik nagy hosszskála esetén? Kis hosszskálánál? Nagy zajvarianciánál? Kis zajvarianciánál?
2. Azt mondtuk, hogy a marginális valószínűség nem konvex célfüggvény, de olyan hiperparaméterek, mint a hosszskála és a zajvariancia, megbízhatóan becsülhetők GP regresszióban. Ez általánosan igaz — sőt, a marginális valószínűség _sokkal_ jobb a hosszskála hiperparaméterek tanulásában, mint a térbeli statisztika hagyományos megközelítései, amelyek empirikus autokorrelációs függvények („kovariogramok") illesztését tartalmazzák. Vitathatóan a gépi tanulás legnagyobb hozzájárulása a Gauss-folyamat-kutatáshoz, legalábbis a skálázható inferenciáról szóló közelmúltbeli munkák előtt, a marginális valószínűség bevezetése volt a hiperparaméter-tanuláshoz.

*Azonban* még ezeknek a paramétereknek a különböző párosítása is értelmezhetően különböző plauzibilis magyarázatokat nyújt sok adathalmazra, ami lokális optimumokhoz vezet a célfüggvényünkben. Ha nagy hosszskálát alkalmazunk, feltételezzük, hogy a valódi mögöttes függvény lassan változik. Ha a megfigyelt adatok _valóban_ jelentősen változnak, akkor a nagy hosszskála csak nagy zajvarianciával együtt plauzibilis. Ha kis hosszskálát alkalmazunk, ezzel szemben az illeszkedésünk nagyon érzékeny lesz az adatok változásaira, kevés teret hagyva a variáció zajjal való magyarázatára (aleatórikus bizonytalanság).

Próbáld megtalálni ezeket a lokális optimumokat: inicializálj nagyon nagy hosszskálával és nagy zajjal, majd kis hosszskálával és kis zajjal. Különböző megoldásokhoz konvergálsz?

3. Azt mondtuk, hogy a Bayes-módszerek alapvető előnye az _episztemikus_ bizonytalanság természetes megjelenítése. A fenti példában az episztemikus bizonytalanság hatásait nem láthatjuk teljesen. Próbálj inkább `test_x = np.linspace(0, 10, 1000)` értékkel előrejelzést készíteni. Mi történik a 95%-os hihető tartománnyal, ahogy az előrejelzések az adatokon túlra mozdulnak? Lefedi a valódi függvényt azon az intervallumon? Mi történik, ha abban a régióban csak aleatórikus bizonytalanságot vizualizálsz?

4. Próbáld meg futtatni a fenti példát, de ezúttal 10 000, 20 000 és 40 000 tanítóponttal, majd mérd meg a futási időket. Hogyan skálázódik a tanítási idő? Hogyan skálázódnak a futási idők a tesztpontok számával? Különbözik ez a prediktív átlag és a prediktív variancia esetén? Válaszold meg ezt a kérdést mind elméleti úton, kiszámítva a tanítási és tesztelési időkomplexitásokat, mind a fenti kód különböző pontszámmal való futtatásával.

5. Próbáld meg a GPyTorch példát különböző kovariancia-függvényekkel futtatni, például a Matérn-kernellel. Hogyan változnak az eredmények? Mi a helyzet a GPyTorch könyvtárban megtalálható spektrális keverékkernellel? Vannak olyanok, amelyekkel könnyebb a marginális valószínűséget tanítani? Vannak olyanok, amelyek értékesebbek a hosszú távú előrejelzéseknél a rövidtávúakhoz képest?

6. A GPyTorch példánkban a prediktív eloszlást a megfigyelési zajjal együtt ábrázoltuk, míg a „nulláról" készített példánkban csak az episztemikus bizonytalanságot vettük figyelembe. Végezd el újra a GPyTorch példát, de ezúttal csak az episztemikus bizonytalanságot ábrázolva, és hasonlítsd össze a nulláról készített eredményekkel. Ugyanolyannak tűnnek most a prediktív eloszlások? (Tűnniük kellene.)

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12117)
:end_tab:
