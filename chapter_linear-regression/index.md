# Lineáris neurális hálózatok regresszióhoz
:label:`chap_regression`

Mielőtt a neurális hálózatok mélyítésével foglalkoznánk,
hasznos lesz néhány sekély hálózatot megvalósítani,
amelyekben a bemenetek közvetlenül a kimenetekhez kapcsolódnak.
Ez több okból is fontos lesz.
Először is, ahelyett, hogy bonyolult architektúrákba bonyolódnánk,
a neurális hálózatok tanításának alapjaira összpontosíthatunk,
beleértve a kimeneti réteg parametrizálását, az adatok kezelését,
a veszteségfüggvény meghatározását és a modell betanítását.
Másodszor, ez a sekély hálózatosztály éppen
a lineáris modellek halmazát alkotja,
amely magában foglalja a statisztikai előrejelzés
számos klasszikus módszerét, köztük
a lineáris és a softmax regressziót.
Ezeknek a klasszikus eszközöknek a megértése kulcsfontosságú,
mivel széles körben alkalmazzák őket,
és gyakran alapvonalként kell majd felhasználnunk őket,
amikor bonyolultabb architektúrák használatát indokoljuk.
Ez a fejezet kizárólag a lineáris regresszióra koncentrál,
a következő pedig kibővíti modellezési repertoárunkat
a lineáris neurális hálózatok osztályozáshoz való kifejlesztésével.

```toc
:maxdepth: 2

linear-regression
oo-design
synthetic-regression-data
linear-regression-scratch
linear-regression-concise
generalization
weight-decay
```

