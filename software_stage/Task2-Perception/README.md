# RoboGambit – Task 2
## ArUco-Based Board State Estimation

A **vision-based perception pipeline** that reconstructs the current board state from an overhead image of the RoboGambit arena.

The pipeline detects **ArUco markers**, estimates the board pose via homography, and infers the **6×6 board configuration** — mapping each game piece to its correct board square.

---

## Our Approach — `perception.py`

Our implementation uses a **robust multi-pass ArUco detection** strategy combined with **homography-based coordinate transformation** and **Euclidean nearest-cell matching** to accurately reconstruct the board state.

### Key Design Decisions

1. **Multi-Pass Marker Detection (`detect_markers_robust`):**
   Instead of relying on a single detection pass, we run ArUco detection on three different image representations — original grayscale, CLAHE contrast-enhanced, and adaptive-threshold — and keep the pass that yields the most markers. This makes the pipeline resilient to varying lighting conditions.

2. **CLAHE Contrast Enhancement (`enhance_image`):**
   We apply Contrast Limited Adaptive Histogram Equalization (CLAHE) with a clip limit of 2.0 and 8×8 tile grid to handle uneven illumination across the board.

3. **Tuned ArUco Detector Parameters:**
   The detector uses sub-pixel corner refinement (`CORNER_REFINE_SUBPIX`) with 50 iterations, a wide adaptive threshold window range (3–30), and relaxed perimeter rate bounds (0.02–4.0) to maximize marker detection reliability.

4. **RANSAC Homography with Validation (`validate_homography`):**
   We compute the homography using `cv2.findHomography` with RANSAC (threshold 5.0) and validate it by computing mean reprojection error. A warning is issued if the error exceeds 50mm.

5. **Batch Coordinate Transformation (`pixel_to_world_batch`):**
   All piece pixel coordinates are converted to world coordinates in a single `cv2.perspectiveTransform` call for efficiency.

6. **Euclidean Nearest-Cell Matching (`place_piece_on_board`):**
   Each piece's world coordinate is matched to the nearest cell center using Euclidean distance. Pieces further than 60mm from any cell center are rejected as outliers. Collision warnings are printed when two pieces map to the same cell.

7. **Board State Output (`save_board_state`):**
   The final board state is saved to `board_output.txt` in three formats: raw NumPy array, piece-position list with chess notation (A1–F6), and a visual ASCII board.

---

## Files

```
Task2-Perception/
│
├── perception.py          # Complete perception pipeline (our solution)
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
└── input/
    ├── board_1.png
    ├── board_2.png
    ├── board_3.png
    └── ...
```

---

## Setup Instructions

### 1. Install Python

Ensure **Python 3.8 or later** is installed:

```bash
python --version
```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

Dependencies:
- **opencv-contrib-python** (ArUco detection + image processing)
- **numpy**

---

## Running the Code

```bash
python perception.py input/board_1.png
```

### Output

Two windows will appear:

1. **Detected Markers** — Input image with all detected ArUco markers drawn and labeled.
2. **Game Board** — Reconstructed 6×6 board with piece IDs displayed.

A file `board_output.txt` is also generated with the full board state in multiple formats.

### Console Output

The pipeline prints step-by-step progress:
- Number of markers detected and their IDs
- Corner markers found (IDs 21–24)
- Homography reprojection error
- Each piece's world coordinate and assigned board cell
- Final 6×6 board array

---

## Pipeline Architecture

```
Input Image
    │
    ▼
prepare_image()          ← Undistort + grayscale (DO NOT MODIFY)
    │
    ▼
detect_markers_robust()  ← 3-pass detection (original / CLAHE / adaptive threshold)
    │
    ▼
Extract corner markers   ← IDs 21–24 → pixel coordinates → world coordinates
    │
    ▼
cv2.findHomography()     ← RANSAC homography from corner correspondences
    │
    ▼
validate_homography()    ← Reprojection error check
    │
    ▼
pixel_to_world_batch()   ← Batch transform piece markers (IDs 1–10)
    │
    ▼
place_piece_on_board()   ← Euclidean nearest-cell matching (60mm threshold)
    │
    ▼
visualize_board()        ← Display result (DO NOT MODIFY)
save_board_state()       ← Write board_output.txt
```

---

## Marker IDs

### Corner Markers (Reference)

Used for homography computation:

| ID | World Coordinate (mm) |
|----|----------------------|
| 21 | (350, 350)           |
| 22 | (350, -350)          |
| 23 | (-350, -350)         |
| 24 | (-350, 350)          |

### Game Piece Markers

| ID | Piece         |
|----|---------------|
| 1  | White Pawn    |
| 2  | White Knight  |
| 3  | White Bishop  |
| 4  | White Queen   |
| 5  | White King    |
| 6  | Black Pawn    |
| 7  | Black Knight  |
| 8  | Black Bishop  |
| 9  | Black Queen   |
| 10 | Black King    |

---

## Board Representation

The board is a **6×6 NumPy int array**. Cell centers in world coordinates:

```
         Col 0   Col 1   Col 2   Col 3   Col 4   Col 5
          250     150      50     -50    -150    -250    (x, mm)
Row 0  (y=250)
Row 1  (y=150)
Row 2  (y= 50)
Row 3  (y=-50)
Row 4  (y=-150)
Row 5  (y=-250)
```

Example output:

```
[[0 0 0 0 0 0]
 [0 1 0 0 6 0]
 [0 0 0 0 0 0]
 [0 0 0 7 0 0]
 [0 0 0 0 0 0]
 [0 0 0 0 0 0]]
```

Where `0 = empty`, `1–10 = piece IDs`.

---

## Functions Implemented

| Function | Description |
|----------|-------------|
| `enhance_image()` | CLAHE contrast enhancement for challenging lighting |
| `detect_markers_robust()` | Multi-pass detection across 3 image representations |
| `pixel_to_world()` | Single-point homography transformation |
| `pixel_to_world_batch()` | Batch homography transformation for efficiency |
| `process_image()` | Full perception pipeline orchestration |
| `place_piece_on_board()` | Euclidean nearest-cell matching with 60mm threshold |
| `validate_homography()` | Reprojection error computation for quality check |
| `save_board_state()` | Export board to `board_output.txt` with chess notation |

## Functions NOT Modified

```
prepare_image()      — Camera undistortion + grayscale
visualize_board()    — 6×6 board visualization
main()               — CLI entry point
```

Camera calibration parameters are also unchanged.

---

