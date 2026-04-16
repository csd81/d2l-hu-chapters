# Hiperparaméter-optimalizálás
:label:`chap_hyperopt`

**Aaron Klein** (*Amazon*), **Matthias Seeger** (*Amazon*), és **Cedric Archambeau** (*Amazon*)

Minden gépi tanulási modell teljesítménye a hiperparamétereitől függ.
Ezek vezérlik a tanulási algoritmust vagy az alapul szolgáló
statisztikai modell szerkezetét. Ugyanakkor nincs általános módszer a hiperparaméterek
megválasztására a gyakorlatban. Ehelyett a hiperparamétereket gyakran próba-hiba alapon állítják be,
vagy néha a gyakorlók az alapértelmezett értékeken hagyják őket, ami
az általánosítás romlásához vezet.

A hiperparaméter-optimalizálás rendszeres megközelítést nyújt ehhez a problémához, azáltal,
hogy optimalizálási feladatként fogalmazza meg: egy jó hiperparaméterkészletnek (legalább)
minimalizálnia kell a validációs hibát. A gépi tanulásban felmerülő legtöbb más
optimalizálási feladathoz képest a hiperparaméter-optimalizálás egy beágyazott feladat,
ahol minden iteráció egy gépi tanulási modell betanítását és érvényesítését igényli.

Ebben a fejezetben először bemutatjuk a hiperparaméter-optimalizálás alapjait.
Bemutatunk néhány újabb fejlesztést is, amelyek javítják a hiperparaméter-optimalizálás
általános hatékonyságát azáltal, hogy az eredeti célfüggvény olcsón kiértékelhető
közelítőit használják fel. A fejezet végére képes leszel korszerű
hiperparaméter-optimalizálási technikákat alkalmazni saját gépi tanulási algoritmusod
hiperparamétereinek optimalizálásához.

```toc
:maxdepth: 2

hyperopt-intro
hyperopt-api
rs-async.md
sh-intro
sh-async
```
