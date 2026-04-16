# Optimalizálási algoritmusok
:label:`chap_optimization`

Ha eddig sorban olvastál a könyvben, már számos optimalizálási algoritmust használtál mélytanulási modellek betanításához.
Ezek az eszközök tették lehetővé, hogy folyamatosan frissítsük a modell paramétereit, és minimalizáljuk a veszteségfüggvény értékét a tanítóhalmazon. Valóban, aki megelégszik azzal, hogy az optimalizálást fekete dobozként kezelje – amelynek célja az objektív függvény minimalizálása egyszerű körülmények között –, az megelégedhet azzal a tudással, hogy léteznek ilyen eljárások (nevük például „SGD" és „Adam").

A jobb teljesítményhez azonban mélyebb ismeretekre van szükség.
Az optimalizálási algoritmusok kulcsfontosságúak a mélytanulásban.
Egyrészt egy összetett mélytanulási modell betanítása órákat, napokat vagy akár heteket is igénybe vehet.
Az optimalizálási algoritmus teljesítménye közvetlenül befolyásolja a modell tanítási hatékonyságát.
Másrészt a különböző optimalizálási algoritmusok elveinek és hiperparamétereik szerepének megértése lehetővé teszi számunkra, hogy célzottan hangoljuk a hiperparamétereket, és javítsuk a mélytanulási modellek teljesítményét.

Ebben a fejezetben mélyebben megvizsgáljuk a mélytanulásban leggyakrabban használt optimalizálási algoritmusokat.
A mélytanulásban felmerülő optimalizálási problémák szinte mindegyike *nemkonvex*.
Ennek ellenére a *konvex* problémák kontextusában kidolgozott algoritmusok tervezése és elemzése rendkívül tanulságosnak bizonyult.
Éppen ezért ez a fejezet tartalmaz egy bevezető áttekintést a konvex optimalizálásról, valamint egy egyszerű sztochasztikus gradient descent algoritmus bizonyítását konvex célfüggvényen.

```toc
:maxdepth: 2

optimization-intro
convexity
gd
sgd
minibatch-sgd
momentum
adagrad
rmsprop
adadelta
adam
lr-scheduler
```

