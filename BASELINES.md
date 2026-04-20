# Baselines — Door Detection

## v4_hn2 — floorplan_door_only4_hn2.pth (BASELINE ACTUELLE)
**Date:** 2026-04-18 (ré-évalué sur nouveau test set Roboflow)
**Test set:** 62 images, `test_set/` — Roboflow workspace `marcellos-workspace-ugf7g`
**Dataset classes:** `['2door', 'door', 'window']` → door = class 1
**Config:** score_thr=0.75, iou_thr=0.50, SAHI ON (slice=640, overlap=0.2)

| Métrique | Valeur |
|---|---|
| Precision | 0.9013 |
| Recall    | 0.9362 |
| F1        | **0.9184** |
| TP        | 411 |
| FP        | 45 |
| FN        | 28 |
| GT total  | 439 |

**Time:** 0.05 s/img

### Plans faibles (candidats hard negatives / annotation review)
- Design-3257 : F1=0.529 (12 FP, 4 FN) — plan chargé style atypique
- IIa_MN0605 : F1=0.632 (5 FN sur 11 GT) — recall faible
- plans2_1117101 : F1=0.667 (7 FP)
- IId_PN0603 : F1=0.727
- house_original : F1=0.741 (5 FP)

### Historique (ancien test set, 69 images, class filter différent)
Strict F1=0.8729 / Lenient+SAHI F1=0.8586 — déprécié, le nouveau test set est plus propre et mieux annoté.

### Règle
Tout modèle candidat (v5+, post-processing variants, C# port) doit battre
**F1=0.9184** sur le test set `test_set/` (62 images) pour être promu.