# Függelék: Matematika a mélytanuláshoz
:label:`chap_appendix_math`

**Brent Werness** (*Amazon*), **Rachel Hu** (*Amazon*), és a könyv szerzői


A modern mélytanulás egyik csodálatos tulajdonsága, hogy nagy részét megértheted és
alkalmazhatod anélkül, hogy teljes mértékben ismernéd az alatta rejlő matematikát.
Ez a terület érettségének jele. Ahogyan a legtöbb szoftverfejlesztőnek ma már nem kell
foglalkoznia a kiszámítható függvények elméletével, a mélytanulás gyakorlati felhasználóinak sem
kellene aggódniuk a maximum likelihood becslés elméleti alapjai miatt.

De még nem tartunk ott.

A gyakorlatban néha meg kell értened, hogy az architektúrális döntések hogyan
befolyásolják a gradiens áramlást, vagy hogy milyen implicit feltételezéseket teszel,
amikor egy adott veszteségfüggvénnyel tanítasz. Lehet, hogy tudnod kell, mit mér az
entrópia, és hogyan segíthet pontosan megérteni, mit jelent a bitek-per-karakter érték
a modelledben. Mindez mélyebb matematikai megértést igényel.

Ez a függelék azt a matematikai hátteret kívánja nyújtani, amely szükséges a modern
mélytanulás alapelméletének megértéséhez, de nem törekszik teljességre. A lineáris
algebra mélyebb vizsgálatával kezdjük. Kialakítjuk az összes általános lineáris algebrai
objektum és művelet geometriai értelmezését, amely lehetővé teszi számunkra, hogy
vizualizáljuk a különböző transzformációk hatásait adatainkra. Kulcseleme a
sajátérték-felbontás alapjainak kidolgozása.

Ezután a differenciálszámítás elméletét fejlesztjük addig a pontig, ahol teljes
mértékben megérthetjük, miért a negatív gradiens adja a legmeredekebb csökkenés irányát, és miért
olyan a visszaterjesztés formája, amilyen. Az integrálszámítást ezután olyan
mértékben tárgyaljuk, amennyire szükséges a következő témánk, a valószínűségszámítás
alátámasztásához.

A gyakorlatban felmerülő problémák gyakran nem biztosak, ezért szükségünk van egy
nyelvre a bizonytalan dolgok leírásához. Áttekintjük a valószínűségi változók elméletét
és a leggyakrabban előforduló eloszlásokat, hogy valószínűségi alapon tárgyalhassuk
a modelleket. Ez alapot nyújt a naiv Bayes osztályozóhoz, egy valószínűségi
osztályozási technikához.

A valószínűségszámítással szorosan összefügg a statisztika tanulmányozása. Bár a
statisztika túlságosan nagy terület ahhoz, hogy egy rövid szakaszban igazságot
tegyünk neki, bemutatjuk azokat az alapfogalmakat, amelyeknek minden gépi tanulási
gyakorlónak ismernie kell: különösen a becslők kiértékelését és összehasonlítását,
hipotézisvizsgálatok elvégzését és konfidenciaintervallumok felépítését.

Végül az információelmélet témájára térünk, amely az információ tárolásának és
továbbításának matematikai tanulmányozása. Ez adja azt az alapnyelvet, amellyel
mennyiségileg megvitathatjuk, hogy egy modell mennyi információt tartalmaz egy
vizsgált tartományról.

Együttesen ezek alkotják azokat az alapvető matematikai fogalmakat, amelyek szükségesek
ahhoz, hogy elindulj a mélytanulás mélyebb megértése felé vezető úton.

```toc
:maxdepth: 2

geometry-linear-algebraic-ops
eigendecomposition
single-variable-calculus
multivariable-calculus
integral-calculus
random-variables
maximum-likelihood
distributions
naive-bayes
statistics
information-theory
```
