# Terminológiai munkafolyamat

## Cél

**Globális konzisztencia.** Minden fejezet, függelék és megjegyzés szövege egyazon magyar terminológiát használja. Az olvasó soha nem találkozhat azzal, hogy ugyanazt a fogalmat az egyik fejezetben `gradienscsökkenés`-nek, a másikban `gradiens ereszkedés`-nek hívják.

Az egyetlen hiteles forrás: **`TERMINOLOGY.md`**

---

## Szabályok

1. **Ha egy fogalom szerepel a `TERMINOLOGY.md`-ben**, azt a táblázatban szereplő magyar alakban kell használni — kivételek nincsenek.
2. **Ha egy fogalom nincs a `TERMINOLOGY.md`-ben**, először fel kell venni, és csak utána szabad a fejezetbe írni.
3. **Kódblokkokban** (```` ``` ```` és `` ` `` között) az eredeti angol API-nevek és változónevek maradnak.
4. **Kódkommentekben** a szöveg a táblázat alapján magyarul írandó.
5. **Angol megjelenés prózában** csak akkor engedett, ha a `TERMINOLOGY.md` a `Megjegyzés` oszlopban explicit `megtartjuk angolul` jelölést tartalmaz.

---

## Új terminus felvétele

1. Ellenőrizd, hogy a fogalom nincs-e már a táblázatban.
2. Adj hozzá egy sort `TERMINOLOGY.md`-hez alfabetikus sorrendben.
3. Ha a döntés vitatható (két elfogadott alak létezik), add meg a `nem:` megszorítást is.
4. Commit `TERMINOLOGY.md` külön, a fejezet előtt — így a változás nyomon követhető.

---

## Fejezet ellenőrzése (manuális)

Mielőtt egy fejezet módosítása bekerül a repóba:

```bash
# Tiltott alakok keresése (példák)
grep -rn "gradiens ereszkedés" chapter_*/
grep -rn "tanulási sebesség" chapter_*/
grep -rn "figyelem mechanizmus" chapter_*/
grep -rn "minibatch[^-]" chapter_*/
grep -rn "torzítás" chapter_*/   # csak lineáris modell kontextusban ellenőrizd
```

Ha találat van, javítsd a `TERMINOLOGY.md` alapján, mielőtt pusholsz.

---

## Fejezetek ellenőrzési sorrendje

Új terminológiai pass elvégzésekor ebben a sorrendben haladj:

1. `chapter_appendix-mathematics-for-deep-learning/`
2. `chapter_optimization/`
3. `chapter_linear-regression/`
4. `chapter_linear-classification/`
5. `chapter_multilayer-perceptrons/`
6. `chapter_convolutional-neural-networks/`
7. `chapter_convolutional-modern/`
8. `chapter_recurrent-neural-networks/`
9. `chapter_recurrent-modern/`
10. `chapter_attention-mechanisms-and-transformers/`
11. `chapter_natural-language-processing-pretraining/`
12. `chapter_natural-language-processing-applications/`
13. `chapter_computer-vision/`
14. `chapter_builders-guide/`
15. `chapter_computational-performance/`
16. `chapter_recommender-systems/`
17. `chapter_reinforcement-learning/`
18. `chapter_gaussian-processes/`
19. `chapter_appendix-tools-for-deep-learning/`

---

## Elvégzett pass-ok (napló)

| Dátum | Fejezetek | Érintett fájlok | Fő változások |
|-------|-----------|-----------------|---------------|
| 2026-04-16 | Teljes repó | 71 | Alapszótár felépítése; gradienscsökkenés, figyelemmechanizmus, tanítóhalmaz stb. egységesítése |
| 2026-04-16 | Teljes repó | 29 | tanulási ráta, előreterjesztés, objektumdetektálás, autograd.md javítások |
| 2026-04-16 | conv, rnn, attention | 5 | padding/stride/CNN/RNN felvétele; kötegméret, dot product figyelem, title-case javítások |
| 2026-04-16 | Teljes repó | 71 | mini-batch egységesítés (49 fájl), batchnormalizáció, torzítás→eltolás (25 fájl), szókincskészlet javítás |
