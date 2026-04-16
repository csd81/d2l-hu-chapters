# d2l-hu-chapters

## Terminológiai döntések és glosszárium

### Terminológiai döntések
- **Broadcasting**: magyar szövegben egységesen *kiterjesztés* (zárójelben: Broadcasting) formában használjuk.
- **Bias (b paraméter)**: lineáris modellekben egységesen *eltolás*.
- **Input/Output**: szövegben egységesen *bemenet/kimenet*.
- **Scalar**: főnévként *skalár* (többes számban: *skalárok*), a *skaláris* alakot melléknévként használjuk.

### Glosszárium
- **kiterjesztés (Broadcasting)**: eltérő, de kompatibilis alakú tenzorok elemenkénti műveleteinek kiterjesztési mechanizmusa.
- **elemenkénti művelet**: olyan művelet, amely azonos pozíciójú elemekre alkalmaz operátort.
- **skalár**: egyetlen numerikus értéket reprezentáló matematikai objektum.
- **vektor**: rendezett skalárlista (1. rendű tenzor).
- **mátrix**: kétdimenziós, sorokból és oszlopokból álló objektum (2. rendű tenzor).
- **tenzor**: többdimenziós adatszerkezet, a skalár/vektor/mátrix általánosítása.
- **eltolás (bias)**: a lineáris modell konstans tagja, amely az előrejelzés alapértékét adja.
- **veszteségfüggvény**: a modell hibáját mérő függvény, amelyet optimalizálás során minimalizálunk.
