# Megerősítéses tanulás
:label:`chap_reinforcement_learning`


**Pratik Chaudhari** (*University of Pennsylvania and Amazon*), **Rasool Fakoor** (*Amazon*), and **Kavosh Asadi** (*Amazon*)

A megerősítéses tanulás (RL) egy olyan technikacsoport, amely lehetővé teszi számunkra, hogy olyan gépi tanulási rendszereket építsünk, amelyek szekvenciálisan hoznak döntéseket. Például egy online kiskereskedőtől vásárolt új ruhákat tartalmazó csomag döntések sorozata után érkezik meg az ajtóhoz: a kereskedő megtalálja a ruhákat az otthonhoz legközelebbi raktárban, becsomagolja azokat, szárazföldi vagy légi úton szállítja a dobozt, majd kiszállítja a városon belül. Számos változó befolyásolja a csomag kézbesítését közben — például hogy rendelkezésre álltak-e a ruhák a raktárban, mennyi ideig tartott a szállítás, megérkezett-e a városba a napi kiszállítóautó indulása előtt stb. A lényeg az, hogy minden egyes szakaszban ezek az általunk ritkán irányítható változók befolyásolják a jövőbeli eseménysorozat egészét — például ha késedelem adódott a raktárban a csomag pakolásánál, a kereskedőnek esetleg légi úton kell elküldenie a csomagot a szárazföldi szállítás helyett, hogy biztosítsa az időbeni kézbesítést. A megerősítéses tanulás módszerei lehetővé teszik, hogy a szekvenciális döntéshozatali problémák minden egyes szakaszában a megfelelő cselekvést hozzuk meg, hogy végül maximalizáljunk valamiféle hasznosságot — például a csomag időbeni kézbesítését.

Ilyen szekvenciális döntéshozatali problémák számos más területen is előfordulnak: például a [Go](https://en.wikipedia.org/wiki/Go_(game)) játék során az aktuális lépés meghatározza a következő lépéseket, és az ellenfél lépései azok a változók, amelyeket nem lehet irányítani — a lépések sorozata dönti el végül, hogy nyerünk-e; a Netflix által most ajánlott filmek meghatározzák, mit nézünk, a Netflix pedig nem tudja, tetszett-e nekünk a film, és végül a filmajánlások sorozata határozza meg, mennyire vagyunk elégedettek a Netflix-szel. A megerősítéses tanulást ma ezekre a problémákra hatékony megoldások kidolgozására alkalmazzák :cite:`mnih2013playing,Silver.Huang.Maddison.ea.2016`. A megerősítéses tanulás és a standard deep learning közötti fő különbség az, hogy a standard deep learningben egy betanított modell egy tesztadaton adott előrejelzése nem befolyásolja a jövőbeli tesztadatokon adott előrejelzéseket; a megerősítéses tanulásban viszont a jövőbeli pillanatokban hozott döntéseket (az RL-ben a döntéseket cselekvéseknek is hívják) befolyásolja, hogy a múltban milyen döntések születtek.

Ebben a fejezetben a megerősítéses tanulás alapjait dolgozzuk ki, és gyakorlati tapasztalatot szerzünk néhány népszerű megerősítéses tanulási módszer implementálásában. Először egy Markov-döntési folyamat (MDP) nevű fogalmat dolgozunk ki, amely lehetővé teszi az ilyen szekvenciális döntéshozatali problémák átgondolását. Az értékiteráció (Value Iteration) nevű algoritmus lesz az első betekintésünk a megerősítéses tanulási problémák megoldásába, feltéve, hogy tudjuk, hogyan viselkednek általában az MDP nem irányítható változói (az RL-ben ezeket a nem irányítható változókat környezetnek nevezzük). Az értékiteráció általánosabb változatát, a Q-tanulás nevű algoritmust alkalmazva megfelelő cselekvéseket tudunk végrehajtani még akkor is, ha nincs teljes ismeretünk a környezetről. Ezután megvizsgáljuk, hogyan használhatók mély hálózatok a megerősítéses tanulási problémákhoz egy szakértő cselekvéseinek utánzásával. Végül egy olyan megerősítéses tanulási módszert dolgozunk ki, amely mély hálózatot használ ismeretlen környezetekben való cselekvésre. Ezek a technikák képezik az alapját a mai fejlettebb RL-algoritmusoknak, amelyeket számos valós alkalmazásban használnak, és amelyekre a fejezetben rá fogunk mutatni.

![Megerősítéses tanulás felépítése](../img/RL_main.png)
:width:`400px`
:label:`fig_rl_big`

```toc
:maxdepth: 2

mdp
value-iter
qlearning
```
