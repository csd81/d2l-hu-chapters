```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['pytorch'])
```

# Gauss-folyamat priorok

A Gauss-folyamatok (GP-k) megértése fontos a modellkonstrukcióról és az általánosításról való gondolkodáshoz, valamint a korszerű teljesítmény eléréséhez számos alkalmazásban, beleértve az aktív tanulást és a deep learningben a hiperparaméter-hangolást. A Gauss-folyamatok mindenütt jelen vannak, és érdekünk megérteni, mik azok és hogyan alkalmazhatók.

Ebben a szakaszban bevezetjük a Gauss-folyamat _priorokat_ a függvények felett. A következő notebookban megmutatjuk, hogyan használhatók ezek a priorok _poszterior inferenciára_ és előrejelzések készítésére. A következő szakasz „GP-k dióhéjban" megtekinthető, amely gyorsan megadja, ami szükséges a Gauss-folyamatok gyakorlatban való alkalmazásához.

```{.python .input}
from d2l import torch as d2l
import numpy as np
from scipy.spatial import distance_matrix

d2l.set_figsize()
```

## Definíció

A Gauss-folyamat _véletlenszerű változók gyűjteményeként_ van definiálva, _amelyeknek bármely véges száma együttes Gauss-eloszlással rendelkezik_. Ha az $f(x)$ függvény Gauss-folyamat, $m(x)$ _átlagfüggvénnyel_ és $k(x,x')$ _kovariancia-függvénnyel_ vagy _kernellel_, $f(x) \sim \mathcal{GP}(m, k)$, akkor bármely, bármely bemeneti pontoknál $x$ (időpontok, térbeli helyek, képpontok stb.) lekérdezett függvényértékek gyűjteménye $\mu$ átlagvektorral és $K$ kovariancia-mátrixszal rendelkező együttes multivariát Gauss-eloszlással rendelkezik: $f(x_1),\dots,f(x_n) \sim \mathcal{N}(\mu, K)$, ahol $\mu_i = E[f(x_i)] = m(x_i)$ és $K_{ij} = \textrm{Cov}(f(x_i),f(x_j)) = k(x_i,x_j)$.

Ez a definíció elvontnak és elérhetetlennek tűnhet, de a Gauss-folyamatok valójában nagyon egyszerű objektumok. Bármely

$$f(x) = w^{\top} \phi(x) = \langle w, \phi(x) \rangle,$$ :eqlabel:`eq_gp-function`

alakú függvény, ahol $w$ Gauss (normál) eloszlásból vett, és $\phi$ bázisfüggvények bármely vektora, például $\phi(x) = (1, x, x^2, ..., x^d)^{\top}$, Gauss-folyamat. Ráadásul bármely Gauss-folyamat $f(x)$ felírható a :eqref:`eq_gp-function` egyenlet alakjában. Nézzünk néhány konkrét példát, hogy megismerkedjünk a Gauss-folyamatokkal, amelyek után értékelhetjük, milyen egyszerűek és hasznosak valójában.

## Egy egyszerű Gauss-folyamat

Tegyük fel, hogy $f(x) = w_0 + w_1 x$, és $w_0, w_1 \sim \mathcal{N}(0,1)$, ahol $w_0, w_1, x$ mind egydimenziós. Ezt a függvényt egyenértékűen felírhatjuk belső szorzatként: $f(x) = (w_0, w_1)(1, x)^{\top}$. A fenti :eqref:`eq_gp-function` egyenletben $w = (w_0, w_1)^{\top}$ és $\phi(x) = (1,x)^{\top}$.

Bármely $x$-re $f(x)$ két Gauss-változó összege. Mivel a Gauss-eloszlások zártak az összeadásra, $f(x)$ is Gauss-változó bármely $x$-re. Valójában kiszámíthatjuk, hogy bármely $x$-re $f(x)$ értéke $\mathcal{N}(0,1+x^2)$. Hasonlóan, bármennyire függvényértékek gyűjteményének, $(f(x_1),\dots,f(x_n))$-nek, bemenetek bármely gyűjteményéhez $x_1,\dots,x_n$, együttes multivariát Gauss-eloszlása van. Ezért $f(x)$ Gauss-folyamat.

Röviden, $f(x)$ _véletlenszerű függvény_, vagy _eloszlás a függvények felett_. Betekintést nyerhetünk ebbe az eloszlásba azáltal, hogy ismételten mintákat veszünk $w_0, w_1$ értékekre, és vizualizáljuk a megfelelő $f(x)$ függvényeket, amelyek különböző meredekségű és különböző tengelymetszetű egyenesek, az alábbiak szerint:

```{.python .input}
def lin_func(x, n_sample):
    preds = np.zeros((n_sample, x.shape[0]))
    for ii in range(n_sample):
        w = np.random.normal(0, 1, 2)
        y = w[0] + w[1] * x
        preds[ii, :] = y
    return preds

x_points = np.linspace(-5, 5, 50)
outs = lin_func(x_points, 10)
lw_bd = -2 * np.sqrt((1 + x_points ** 2))
up_bd = 2 * np.sqrt((1 + x_points ** 2))

d2l.plt.fill_between(x_points, lw_bd, up_bd, alpha=0.25)
d2l.plt.plot(x_points, np.zeros(len(x_points)), linewidth=4, color='black')
d2l.plt.plot(x_points, outs.T)
d2l.plt.xlabel("x", fontsize=20)
d2l.plt.ylabel("f(x)", fontsize=20)
d2l.plt.show()
```

Ha $w_0$ és $w_1$ ehelyett $\mathcal{N}(0,\alpha^2)$-ből kerülnek mintázásra, hogyan képzeled, hogy $\alpha$ változtatása befolyásolja a függvények feletti eloszlást?

## A súlytértől a függvénytérig

A fenti ábrán láttuk, hogyan indukál egy modell paraméterei feletti eloszlás egy eloszlást a függvények felett. Bár sokszor vannak elképzeléseink a modellezni kívánt függvényekről — hogy simák, periodikusak, gyorsan változók-e stb. — viszonylag fáradságos a paraméterekről érvelni, amelyek nagyrészt értelmezhetetlenek. Szerencsére a Gauss-folyamatok egyszerű mechanizmust biztosítanak arra, hogy _közvetlenül_ a függvényekről érveljünk. Mivel a Gauss-eloszlást teljesen meghatározza az első két momentuma — az átlag és a kovariancia-mátrix —, a Gauss-folyamatot hasonlóan az átlagfüggvénye és a kovariancia-függvénye határozza meg.

A fenti példában az átlagfüggvény

$$m(x) = E[f(x)] = E[w_0 + w_1x] = E[w_0] + E[w_1]x = 0+0 = 0.$$

Hasonlóan, a kovariancia-függvény

$$k(x,x') = \textrm{Cov}(f(x),f(x')) = E[f(x)f(x')]-E[f(x)]E[f(x')] = E[w_0^2 + w_0w_1x' + w_1w_0x + w_1^2xx'] = 1 + xx'.$$

A függvények feletti eloszlásunk most közvetlenül megadható és mintázható, anélkül, hogy a paraméterek feletti eloszlásból kellene mintát venni. Például $f(x)$ mintázásához egyszerűen megalkothatjuk a multivariát Gauss-eloszlást az összes lekérdezni kívánt $x$ értékgyűjteményre, és ebből közvetlenül mintázhatunk. Hamarosan látni fogjuk, hogy ez a megfogalmazás milyen előnyös.

Először megjegyezzük, hogy lényegében ugyanaz a levezetés, amelyet a fenti egyszerű egyenesvonalú modellre alkalmaztunk, elvégezhető az átlag- és kovariancia-függvény meghatározásához _bármely_ $f(x) = w^{\top} \phi(x)$ alakú modellre, ahol $w \sim \mathcal{N}(u,S)$. Ebben az esetben az átlagfüggvény $m(x) = u^{\top}\phi(x)$, a kovariancia-függvény $k(x,x') = \phi(x)^{\top}S\phi(x')$. Mivel $\phi(x)$ tetszőleges nemlineáris bázisfüggvények vektorát reprezentálhatja, egy nagyon általános modellosztályt vizsgálunk, beleértve akár _végtelen_ paraméterszámú modelleket is.

## A radiális bázisfüggvény (RBF) kernel

A _radiális bázisfüggvény_ (RBF) kernel a Gauss-folyamatok és általában a kernel-gépek legnépszerűbb kovariancia-függvénye.
Ez a kernel alakja $k_{\textrm{RBF}}(x,x') = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right)$, ahol $a$ egy amplitúdóparaméter, és $\ell$ egy _hosszskála_ hiperparaméter.

Vezessük le ezt a kernelt a súlytérből kiindulva. Tekintsük a következő függvényt:

$$f(x) = \sum_{i=1}^J w_i \phi_i(x), w_i  \sim \mathcal{N}\left(0,\frac{\sigma^2}{J}\right), \phi_i(x) = \exp\left(-\frac{(x-c_i)^2}{2\ell^2 }\right).$$

$f(x)$ radiális bázisfüggvények összege, $\ell$ szélességgel, a $c_i$ pontok köré centrálva, ahogyan az alábbi ábrán látható.


Felismerjük, hogy $f(x)$ $w^{\top} \phi(x)$ alakú, ahol $w = (w_1,\dots,w_J)^{\top}$ és $\phi(x)$ egy vektor, amely az egyes radiális bázisfüggvényeket tartalmazza. Ennek a Gauss-folyamatnak a kovariancia-függvénye:

$$k(x,x') = \frac{\sigma^2}{J} \sum_{i=1}^{J} \phi_i(x)\phi_i(x').$$

Most vizsgáljuk meg, mi történik, ha a paraméterek (és bázisfüggvények) számát végtelenbe visszük. Legyen $c_J = \log J$, $c_1 = -\log J$, és $c_{i+1}-c_{i} = \Delta c = 2\frac{\log J}{J}$, és $J \to \infty$. A kovariancia-függvény Riemann-összeggé válik:

$$k(x,x') = \lim_{J \to \infty} \frac{\sigma^2}{J} \sum_{i=1}^{J} \phi_i(x)\phi_i(x') = \int_{c_0}^{c_\infty} \phi_c(x)\phi_c(x') dc.$$

$c_0 = -\infty$ és $c_\infty = \infty$ beállításával végtelen sok bázisfüggvényt szórunk el az egész valós egyenesen, mindegyik $\Delta c \to 0$ távolságra egymástól:

$$k(x,x') = \int_{-\infty}^{\infty} \exp(-\frac{(x-c)^2}{2\ell^2}) \exp(-\frac{(x'-c)^2}{2\ell^2 }) dc = \sqrt{\pi}\ell \sigma^2 \exp(-\frac{(x-x')^2}{2(\sqrt{2} \ell)^2}) \propto k_{\textrm{RBF}}(x,x').$$

Érdemes egy pillanatra elgondolkodni azon, amit itt tettünk. A függvénytéri reprezentációra áttérve levezetettük, hogyan lehet _végtelen_ paraméterszámú modellt reprezentálni véges számítással. Egy RBF kernellel rendelkező Gauss-folyamat _univerzális közelítő_, amely képes bármely folytonos függvényt tetszőleges pontossággal reprezentálni. A fenti levezetésből intuitívan láthatjuk, miért. Minden radiális bázisfüggvényt pontmasszává sűríthetünk, ha $\ell \to 0$, és bármekkora magasságot adhatunk minden egyes pontmasszának.

Tehát egy RBF kernellel rendelkező Gauss-folyamat végtelen paraméterszámú modell, sokkal rugalmasabb, mint bármely véges neurális hálózat. Talán az _túlparaméterezett_ neurális hálózatokról szóló felhajtás nem indokolt. Amint látni fogjuk, az RBF kernelű GP-k nem illeszkednek túl az adatokra, és valójában különösen meggyőző általánosítási teljesítményt mutatnak kis adathalmazokon. Ráadásul a :cite:`zhang2021understanding` cikkben szereplő példák, mint például a véletlenszerű címkékkel ellátott képek tökéletes illesztésének képessége, miközben strukturált problémákon jól általánosítanak, (tökéletesen reprodukálhatók Gauss-folyamatok segítségével) :cite:`wilson2020bayesian`. A neurális hálózatok nem annyira különböznek egymástól, mint ahogyan azt hiszük.

Tovább mélyíthetjük a Gauss-folyamatokkal kapcsolatos intuíciónkat RBF kernelekkel és hiperparaméterekkel, például a _hosszskálával_, ha közvetlenül mintázunk a függvények feletti eloszlásból. Mint korábban, ez egy egyszerű eljárást jelent:

1. Válasszuk ki az $x$ bemeneti pontokat, amelyeknél le szeretnénk kérdezni a GP-t: $x_1,\dots,x_n$.
2. Számítsuk ki $m(x_i)$, $i = 1,\dots,n$ és $k(x_i,x_j)$ értékeket $i,j = 1,\dots,n$ esetén, hogy megkapjuk a $\mu$ átlagvektort és a $K$ kovariancia-mátrixot, ahol $(f(x_1),\dots,f(x_n)) \sim \mathcal{N}(\mu, K)$.
3. Vegyünk mintát ebből a multivariát Gauss-eloszlásból, hogy megkapjuk a mintafüggvény-értékeket.
4. Vegyünk több mintát, hogy több mintafüggvényt vizualizáljunk azokon a pontokon.

Az alábbi ábrán szemléltetjük ezt az eljárást.

```{.python .input}
def rbfkernel(x1, x2, ls=4.):  #@save
    dist = distance_matrix(np.expand_dims(x1, 1), np.expand_dims(x2, 1))
    return np.exp(-(1. / ls / 2) * (dist ** 2))

x_points = np.linspace(0, 5, 50)
meanvec = np.zeros(len(x_points))
covmat = rbfkernel(x_points,x_points, 1)

prior_samples= np.random.multivariate_normal(meanvec, covmat, size=5);
d2l.plt.plot(x_points, prior_samples.T, alpha=0.5)
d2l.plt.show()
```

## A neurális hálózat kernel

A gépi tanulásban a Gauss-folyamatokkal kapcsolatos kutatást a neurális hálózatokra vonatkozó kutatás indította el. Radford Neal egyre nagyobb Bayesi neurális hálózatokat kutatott, és végül 1994-ben megmutatta (amelyet 1996-ban publikáltak, mivel ez az egyik leghírhedtebb NeurIPS elutasítás volt), hogy az ilyen, végtelen rejtett egységszámú hálózatok specifikus kernelfüggvényekkel rendelkező Gauss-folyamatokká válnak :cite:`neal1996bayesian`. Az iránt újra felkelt az érdeklődés, és olyan ötletek, mint a neurális érintő kernel, a neurális hálózatok általánosítási tulajdonságainak vizsgálatára szolgálnak :cite:`matthews2018gaussian` :cite:`novak2018bayesian`. A neurális hálózat kernelt a következőképpen vezethetjük le.

Tekintsünk egy $f(x)$ neurális hálózat függvényt egy rejtett réteggel:

$$f(x) = b + \sum_{i=1}^{J} v_i h(x; u_i).$$

$b$ egy torzítás, $v_i$ a rejtett-kimenet súlyok, $h$ bármely korlátozott rejtett egység transzferfüggvény, $u_i$ a bemenet-rejtett súlyok, és $J$ a rejtett egységek száma. Legyen $b$ és $v_i$ egymástól független, zéró átlaggal és rendre $\sigma_b^2$ és $\sigma_v^2/J$ varianciával, és az $u_i$ értékek legyenek egymástól független, azonos eloszlású értékek. Ekkor a centrális határeloszlás tétele alkalmazható annak megmutatásához, hogy bármely függvényértékgyűjtemény $f(x_1),\dots,f(x_n)$ együttes multivariát Gauss-eloszlással rendelkezik.

A megfelelő Gauss-folyamat átlag- és kovariancia-függvénye:

$$m(x) = E[f(x)] = 0$$

$$k(x,x') = \textrm{cov}[f(x),f(x')] = E[f(x)f(x')] = \sigma_b^2 + \frac{1}{J} \sum_{i=1}^{J} \sigma_v^2 E[h_i(x; u_i)h_i(x'; u_i)]$$

Egyes esetekben ezt a kovariancia-függvényt lényegében zárt formában értékelhetjük ki. Legyen $h(x; u) = \textrm{erf}(u_0 + \sum_{j=1}^{P} u_j x_j)$, ahol $\textrm{erf}(z) = \frac{2}{\sqrt{\pi}} \int_{0}^{z} e^{-t^2} dt$, és $u \sim \mathcal{N}(0,\Sigma)$. Ekkor $k(x,x') = \frac{2}{\pi} \textrm{sin}(\frac{2 \tilde{x}^{\top} \Sigma \tilde{x}'}{\sqrt{(1 + 2 \tilde{x}^{\top} \Sigma \tilde{x})(1 + 2 \tilde{x}'^{\top} \Sigma \tilde{x}')}})$.

Az RBF kernel _stacionárius_, azaz _eltolás-invariáns_, és ezért felírható $\tau = x-x'$ függvényeként. Intuitívan a stacionaritás azt jelenti, hogy a függvény magas szintű tulajdonságai, például a változási sebesség, nem változnak, ahogy mozgunk a bemeneti térben. A neurális hálózat kernel azonban _nem stacionárius_. Az alábbiakban megmutatjuk az ebből a kernelből vett minta Gauss-folyamat függvényeket. Láthatjuk, hogy a függvény kvalitatively eltérő a origó közelében.

## Összefoglalás


A Bayesi inferencia elvégzésének első lépése a prior megadása. Gauss-folyamatok felhasználhatók egy teljes prior megadásához a függvények felett. A modellezés hagyományos „súlytéri" szemléletéből kiindulva a függvények feletti priort úgy indukálhatjuk, hogy a modell funkcionális formájából indulunk ki, és eloszlást vezetünk be a paraméterei felett. Alternatívan közvetlenül a függvénytérben is megadhatunk prior eloszlást, amelynek tulajdonságait egy kernel szabályoz. A függvénytéri megközelítésnek számos előnye van. Olyan modelleket építhetünk, amelyek valójában végtelen számú paraméternek felelnek meg, de véges mennyiségű számítást használunk! Ráadásul, bár ezek a modellek rendkívüli rugalmassággal rendelkeznek, erős feltételezéseket is tesznek arról, hogy milyen típusú függvények valószínűek a priorban, ami viszonylag jó általánosításhoz vezet kis adathalmazokon.

A függvénytéri modellek feltételezéseit intuitívan a kernelek szabályozzák, amelyek sokszor a függvények magasabb szintű tulajdonságait kódolják, mint a simaság és a periodicitás. Sok kernel stacionárius, azaz eltolás-invariáns. A stacionárius kernellel rendelkező Gauss-folyamatból vett függvények nagyjából ugyanolyan magas szintű tulajdonságokkal rendelkeznek (például változási sebesség) attól függetlenül, hogy hol nézzük a bemeneti térben.

A Gauss-folyamatok viszonylag általános modellosztályt alkotnak, amely sok olyan modell példáját tartalmazza, amelyeket már ismerünk, beleértve a polinomokat, a Fourier-sorokat stb., amennyiben Gauss-prior szerepel a paramétereken. Tartalmazzák a végtelen paraméterszámú neurális hálózatokat is, még Gauss-eloszlás nélkül is a paramétereken. Ezt az összefüggést Radford Neal fedezte fel, ami arra indította a gépi tanulás kutatóit, hogy elhagyják a neurális hálózatokat, és a Gauss-folyamatok felé forduljanak.


## Feladatok

1. Rajzolj minta prior függvényeket egy GP-ből Ornstein-Uhlenbeck (OU) kernellel: $k_{\textrm{OU}}(x,x') = \exp\left(-\frac{1}{2\ell}||x - x'|\right)$. Ha ugyanolyan hosszskálát rögzítesz $\ell$-nek, mennyiben néznek ki ezek a függvények másképpen, mint egy RBF kernelű GP minta függvényei?

2. Hogyan befolyásolja az RBF kernel _amplitúdójának_ $a^2$ megváltoztatása a függvények feletti eloszlást?

3. Tegyük fel, hogy $u(x) = f(x) + 2g(x)$, ahol $f(x) \sim \mathcal{GP}(m_1,k_1)$ és $g(x) \sim \mathcal{GP}(m_2,k_2)$. Gauss-folyamat-e $u(x)$, és ha igen, mi az átlag- és kovariancia-függvénye?

4. Tegyük fel, hogy $g(x) = a(x)f(x)$, ahol $f(x) \sim \mathcal{GP}(0,k)$ és $a(x) = x^2$. Gauss-folyamat-e $g(x)$, és ha igen, mi az átlag- és kovariancia-függvénye? Mi a hatása $a(x)$-nek? Hogyan néznek ki a $g(x)$-ből vett minta függvények?

5. Tegyük fel, hogy $u(x) = f(x)g(x)$, ahol $f(x) \sim \mathcal{GP}(m_1,k_1)$ és $g(x) \sim \mathcal{GP}(m_2,k_2)$. Gauss-folyamat-e $u(x)$, és ha igen, mi az átlag- és kovariancia-függvénye?

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12116)
:end_tab:
