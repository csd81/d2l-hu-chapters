# Jelölés
:label:`chap_notation`

A könyvben végig az alábbi jelölési konvenciókat használjuk.
Fontos megjegyezni, hogy ezek közül egyes szimbólumok helykitöltők,
míg mások konkrét objektumokra utalnak. Általános ökölszabályként az
angol "a" névelő gyakran arra utal, hogy a szimbólum helykitöltő, és hogy
a hasonló formátumú szimbólumok ugyanilyen típusú más objektumokat is
jelölhetnek. Például az, hogy "$x$: egy skalár", azt jelenti, hogy a kisbetűk
általában skalár értékeket jelölnek, míg a
"$\mathbb{Z}$: az egész számok halmaza" kifejezés konkrétan a
$\mathbb{Z}$ szimbólumra utal.



## Numerikus objektumok

* $x$: egy skalár
* $\mathbf{x}$: egy vektor
* $\mathbf{X}$: egy mátrix
* $\mathsf{X}$: egy általános tenzor
* $\mathbf{I}$: az egységmátrix (valamilyen adott dimenzióban), azaz négyzetes mátrix, amelynek minden főátlóbeli eleme $1$, minden nem főátlóbeli eleme pedig $0$
* $x_i$, $[\mathbf{x}]_i$: a $\mathbf{x}$ vektor $i^\textrm{th}$ eleme
* $x_{ij}$, $x_{i,j}$,$[\mathbf{X}]_{ij}$, $[\mathbf{X}]_{i,j}$: a $\mathbf{X}$ mátrix $i$. sorának és $j$. oszlopának eleme



## Halmazelmélet


* $\mathcal{X}$: egy halmaz
* $\mathbb{Z}$: az egész számok halmaza
* $\mathbb{Z}^+$: a pozitív egész számok halmaza
* $\mathbb{R}$: a valós számok halmaza
* $\mathbb{R}^n$: az $n$ dimenziós valós vektorok halmaza
* $\mathbb{R}^{a\times b}$: az $a$ sorból és $b$ oszlopból álló valós mátrixok halmaza
* $|\mathcal{X}|$: a $\mathcal{X}$ halmaz számossága (elemeinek száma)
* $\mathcal{A}\cup\mathcal{B}$: az $\mathcal{A}$ és $\mathcal{B}$ halmaz uniója
* $\mathcal{A}\cap\mathcal{B}$: az $\mathcal{A}$ és $\mathcal{B}$ halmaz metszete
* $\mathcal{A}\setminus\mathcal{B}$: a $\mathcal{B}$ kivonása az $\mathcal{A}$ halmazból (csak azokat az elemeket tartalmazza, amelyek $\mathcal{A}$-ban benne vannak, de $\mathcal{B}$-ben nem)



## Függvények és operátorok


* $f(\cdot)$: egy függvény
* $\log(\cdot)$: a természetes logaritmus (alapja $e$)
* $\log_2(\cdot)$: 2 alapú logaritmus
* $\exp(\cdot)$: az exponenciális függvény
* $\mathbf{1}(\cdot)$: indikátorfüggvény; értéke $1$, ha a logikai argumentum igaz, különben $0$
* $\mathbf{1}_{\mathcal{X}}(z)$: halmaztagsági indikátorfüggvény; értéke $1$, ha a $z$ elem eleme a $\mathcal{X}$ halmaznak, különben $0$
* $\mathbf{(\cdot)}^\top$: egy vektor vagy mátrix transzponáltja
* $\mathbf{X}^{-1}$: a $\mathbf{X}$ mátrix inverze
* $\odot$: Hadamard- (elemenkénti) szorzat
* $[\cdot, \cdot]$: konkatenáció
* $\|\cdot\|_p$: $\ell_p$ norma
* $\|\cdot\|$: $\ell_2$ norma
* $\langle \mathbf{x}, \mathbf{y} \rangle$: a $\mathbf{x}$ és $\mathbf{y}$ vektorok belső (skaláris) szorzata
* $\sum$: összegzés egy elemgyűjtemény felett
* $\prod$: szorzás egy elemgyűjtemény felett
* $\stackrel{\textrm{def}}{=}$: olyan egyenlőség, amely a bal oldali szimbólum definícióját adja meg



## Kalkulus

* $\frac{dy}{dx}$: $y$ deriváltja $x$ szerint
* $\frac{\partial y}{\partial x}$: $y$ parciális deriváltja $x$ szerint
* $\nabla_{\mathbf{x}} y$: $y$ gradiense $\mathbf{x}$ szerint
* $\int_a^b f(x) \;dx$: az $f$ határozott integrálja $a$-tól $b$-ig $x$ szerint
* $\int f(x) \;dx$: az $f$ határozatlan integrálja $x$ szerint



## Valószínűségszámítás és információelmélet

* $X$: egy valószínűségi változó
* $P$: egy valószínűségi eloszlás
* $X \sim P$: az $X$ valószínűségi változó a $P$ eloszlást követi
* $P(X=x)$: annak az eseménynek a valószínűsége, hogy az $X$ valószínűségi változó az $x$ értéket veszi fel
* $P(X \mid Y)$: $X$ feltételes valószínűségi eloszlása $Y$ mellett
* $p(\cdot)$: a $P$ eloszláshoz tartozó valószínűségi sűrűségfüggvény (PDF)
* ${E}[X]$: az $X$ valószínűségi változó várható értéke
* $X \perp Y$: az $X$ és $Y$ valószínűségi változók függetlenek
* $X \perp Y \mid Z$: az $X$ és $Y$ valószínűségi változók feltételesen függetlenek $Z$ mellett
* $\sigma_X$: az $X$ valószínűségi változó szórása
* $\textrm{Var}(X)$: az $X$ valószínűségi változó varianciája, amely megegyezik $\sigma^2_X$-szel
* $\textrm{Cov}(X, Y)$: az $X$ és $Y$ valószínűségi változók kovarianciája
* $\rho(X, Y)$: az $X$ és $Y$ közötti Pearson-korrelációs együttható, értéke $\frac{\textrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$
* $H(X)$: az $X$ valószínűségi változó entrópiája
* $D_{\textrm{KL}}(P\|Q)$: a KL-divergencia (vagy relatív entrópia) a $Q$ eloszlástól a $P$ eloszlás felé



[Beszélgetések](https://discuss.d2l.ai/t/25)
