# Többrétegű perceptronok
:label:`chap_perceptrons`

Ebben a fejezetben bemutatjuk az első igazán *mély* hálózatodat.
A legegyszerűbb mély hálózatokat *többrétegű perceptronoknak* nevezzük,
amelyek több neuronrétegből állnak,
ahol minden réteg teljesen kapcsolódik az alatta lévő réteghez
(amelyből bemenetet kap)
és a felette lévő réteghez (amelyre hatással van).
Bár az automatikus differenciálás
jelentősen leegyszerűsíti a mély tanulási algoritmusok implementálását,
mélyen bele fogunk merülni abba, hogyan
számítják ki ezeket a gradienseket a mély hálózatokban.
Ezt követően készen állunk arra,
hogy megvitassuk a numerikus stabilitással és a paraméter-inicializálással kapcsolatos kérdéseket,
amelyek kulcsfontosságúak a mély hálózatok sikeres tanításához.
Amikor ilyen nagy kapacitású modelleket tanítunk, fennáll a túlillesztés veszélye. Ezért
újra áttekintjük a regularizációt és az általánosítást
mély hálózatok esetén.
Célunk, hogy szilárd alapot adjunk neked nemcsak a fogalmakhoz, hanem a mély hálózatok használatának gyakorlatához is.
A fejezet végén az eddig bemutatottakat egy valós problémára alkalmazzuk: házárak
előrejelzésére. A modelljeink számítási teljesítményével, skálázhatóságával és hatékonyságával kapcsolatos kérdéseket a következő fejezetekre halasztjuk.

```toc
:maxdepth: 2

mlp
mlp-implementation
backprop
numerical-stability-and-init
generalization-deep
dropout
kaggle-house-price
```

