# Phase 1 Dataset Audit Report: Chronological Splits & Target Integrity

**Author:** NBA Prediction Model Engineering Team  
**Date:** November 6, 2026  
**Target Statistics (7):** Points (PTS), Rebounds (REB), Assists (AST), 3-Pointers (3PM/FG3M), Steals (STL), Blocks (BLK), Turnovers (TOV)  
**Split Strategy:** Walk-Forward Chronological (Non-random, zero game overlap)  

---

## 1. Executive Summary

Phase 1 has completed successfully. The historical feature matrix has been partitioned into **three strictly chronological, non-overlapping datasets**:

- **Training Set (2022–23 → 2024–25):** 72,387 player-game rows across 3,690 regular season games.
- **Validation Set (2025–26):** 24,605 player-game rows across 1,230 regular season games (full 82-game benchmark season).
- **Test Set (2026–27):** 1,646 player-game rows across 88 games (unseen early-season forward test).

> [!IMPORTANT]
> **Zero Data Leakage Confirmed:**
> - Date separation: Train ends `2025-04-13` < Val begins `2025-10-21` < Test begins `2026-10-20`.
> - Game ID intersection between splits is exactly **0**.
> - Target-game actual stats are strictly excluded from the pre-game feature vector.

---

## 2. Chronological Split Details

| Split | Seasons Included | Start Date | End Date | Player Rows | Unique Games | Unique Players |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Training** | 2022–23, 2023–24, 2024–25 | `2022-10-18` | `2025-04-13` | 72,387 | 3,690 | 691 |
| **Validation** | 2025–26 | `2025-10-21` | `2026-04-12` | 24,605 | 1,230 | 530 |
| **Test** | 2026–27 | `2026-10-20` | `2026-10-31` | 1,646 | 88 | 357 |

---

## 3. Target Statistics Integrity & Observations

All 7 target statistics are non-negative discrete counts. Every split contains valid, complete target observations:

| Target Stat | Train Mean ± Std | Val Mean ± Std | Test Mean ± Std | Missing Count (Train/Val/Test) |
| :--- | :---: | :---: | :---: | :---: |
| **PTS** | 11.57 ± 8.81 | 11.46 ± 8.51 | 11.12 ± 8.41 | 0 / 0 / 0 |
| **REB** | 4.41 ± 3.42 | 4.32 ± 3.28 | 4.31 ± 3.22 | 0 / 0 / 0 |
| **AST** | 2.65 ± 2.65 | 2.65 ± 2.58 | 2.57 ± 2.51 | 0 / 0 / 0 |
| **FG3M** | 1.31 ± 1.53 | 1.32 ± 1.51 | 1.27 ± 1.45 | 0 / 0 / 0 |
| **STL** | 0.77 ± 0.97 | 0.83 ± 1.00 | 0.87 ± 1.04 | 0 / 0 / 0 |
| **BLK** | 0.49 ± 0.85 | 0.48 ± 0.83 | 0.49 ± 0.85 | 0 / 0 / 0 |
| **TOV** | 1.34 ± 1.41 | 1.36 ± 1.40 | 1.48 ± 1.49 | 0 / 0 / 0 |

---

## 4. Leakage & Boundary Verification

| Check Description | Requirement | Result | Status |
| :--- | :--- | :---: | :---: |
| **Train-to-Val Date Separation** | Max Train Date < Min Val Date | `2025-04-13 < 2025-10-21` | **PASS** |
| **Val-to-Test Date Separation** | Max Val Date < Min Test Date | `2026-04-12 < 2026-10-20` | **PASS** |
| **Game Overlap (Train ∩ Val)** | Zero shared game IDs | 0 games shared | **PASS** |
| **Game Overlap (Val ∩ Test)** | Zero shared game IDs | 0 games shared | **PASS** |
| **Game Overlap (Train ∩ Test)** | Zero shared game IDs | 0 games shared | **PASS** |
| **Target Leakage Check** | Targets strictly excluded from feature vector $X$ | Verified | **PASS** |

---

## 5. Storage Location of Verified Datasets

- Training: `data/processed/train_players.parquet`
- Validation: `data/processed/val_players.parquet`
- Test: `data/processed/test_players.parquet`

Phase 1 is complete. We are cleared to proceed with Phase 2 (Ridge Baseline).