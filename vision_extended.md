# Vision Extended — Image Preparation Guide

## Overview

63 items (18 Thai, 45 Math) require stimulus images from the original NT exam PDFs.

## Directory structure

```
nt-tests/images/
  pages/        # Full pages extracted by script (intermediate)
  questions/    # Cropped per-question stimulus images (final)
```

## Naming convention

- Pages: `nt-tests/images/pages/{exam_id}_p{page:02d}.png`
- Questions: `nt-tests/images/questions/{exam_id}_q{question_id:02d}.png`

## Workflow

1. Run `extract_pages.py` to extract referenced PDF pages at 300 DPI
2. Manually crop each question's stimulus image from the extracted page
3. Save cropped images to `nt-tests/images/questions/`
4. Add `image_path` field to JSON data files

## Cropping guidelines

- Crop only the **stimulus image** (chart, map, diagram, infographic) — not the question text or choices (those are already in JSON)
- For shared pages (marked below), crop the relevant image area for each question separately
- For questions sharing the same stimulus (e.g. a map used by q11 and q12), save identical crops as separate files

## JSON schema

```json
{
  "image_path": "images/questions/nt_p3_th_2566_q11.png"
}
```

Path is relative to `nt-tests/`. Set to `null` for non-vision items.

---

## Checklist: Thai (18 items)

### nt_p3_th_2565 (5 items)

| File | PDF page | Shared | Cropped |
|------|----------|--------|---------|
| `nt_p3_th_2565_q09.png` | p.13 | | [ ] |
| `nt_p3_th_2565_q11.png` | p.15 | | [ ] |
| `nt_p3_th_2565_q12.png` | p.17 | | [ ] |
| `nt_p3_th_2565_q14.png` | p.20 | SHARED | [ ] |
| `nt_p3_th_2565_q15.png` | p.20, p.22 | SHARED | [ ] |

### nt_p3_th_2566 (7 items)

| File | PDF page | Shared | Cropped |
|------|----------|--------|---------|
| `nt_p3_th_2566_q09.png` | p.8 | | [ ] |
| `nt_p3_th_2566_q11.png` | p.10 | SHARED | [ ] |
| `nt_p3_th_2566_q12.png` | p.10 | SHARED | [ ] |
| `nt_p3_th_2566_q14.png` | p.11 | | [ ] |
| `nt_p3_th_2566_q15.png` | p.12 | | [ ] |
| `nt_p3_th_2566_q16.png` | p.13 | | [ ] |
| `nt_p3_th_2566_q19.png` | p.14 | | [ ] |

### nt_p3_th_2567 (6 items)

| File | PDF page | Shared | Cropped |
|------|----------|--------|---------|
| `nt_p3_th_2567_q06.png` | p.9 | | [ ] |
| `nt_p3_th_2567_q11.png` | p.15 | | [ ] |
| `nt_p3_th_2567_q12.png` | p.17 | | [ ] |
| `nt_p3_th_2567_q14.png` | p.21 | | [ ] |
| `nt_p3_th_2567_q16.png` | p.24 | | [ ] |
| `nt_p3_th_2567_q23.png` | p.31 | | [ ] |

---

## Checklist: Math (45 items)

### nt_p3_math_2565 (20 items)

| File | PDF page | Shared | Cropped |
|------|----------|--------|---------|
| `nt_p3_math_2565_q01.png` | p.3 | | [ ] |
| `nt_p3_math_2565_q02.png` | p.5 | | [ ] |
| `nt_p3_math_2565_q03.png` | p.7 | | [ ] |
| `nt_p3_math_2565_q04.png` | p.9 | | [ ] |
| `nt_p3_math_2565_q05.png` | p.11 | | [ ] |
| `nt_p3_math_2565_q07.png` | p.15 | | [ ] |
| `nt_p3_math_2565_q09.png` | p.20 | | [ ] |
| `nt_p3_math_2565_q12.png` | p.27 | | [ ] |
| `nt_p3_math_2565_q13.png` | p.29 | | [ ] |
| `nt_p3_math_2565_q14.png` | p.30 | | [ ] |
| `nt_p3_math_2565_q15.png` | p.32 | | [ ] |
| `nt_p3_math_2565_q16.png` | p.34 | | [ ] |
| `nt_p3_math_2565_q17.png` | p.36 | | [ ] |
| `nt_p3_math_2565_q19.png` | p.40 | | [ ] |
| `nt_p3_math_2565_q20.png` | p.42 | | [ ] |
| `nt_p3_math_2565_q21.png` | p.44 | | [ ] |
| `nt_p3_math_2565_q22.png` | p.47 | | [ ] |
| `nt_p3_math_2565_q23.png` | p.49 | | [ ] |
| `nt_p3_math_2565_q24.png` | p.51 | | [ ] |
| `nt_p3_math_2565_q25.png` | p.53 | | [ ] |

### nt_p3_math_2566 (16 items)

| File | PDF page | Shared | Cropped |
|------|----------|--------|---------|
| `nt_p3_math_2566_q02.png` | p.3 | | [ ] |
| `nt_p3_math_2566_q03.png` | p.4 | SHARED | [ ] |
| `nt_p3_math_2566_q04.png` | p.4 | SHARED | [ ] |
| `nt_p3_math_2566_q05.png` | p.5 | | [ ] |
| `nt_p3_math_2566_q11.png` | p.8 | SHARED | [ ] |
| `nt_p3_math_2566_q12.png` | p.8 | SHARED | [ ] |
| `nt_p3_math_2566_q13.png` | p.9 | | [ ] |
| `nt_p3_math_2566_q15.png` | p.10 | SHARED | [ ] |
| `nt_p3_math_2566_q16.png` | p.10 | SHARED | [ ] |
| `nt_p3_math_2566_q17.png` | p.11 | | [ ] |
| `nt_p3_math_2566_q19.png` | p.12 | SHARED | [ ] |
| `nt_p3_math_2566_q20.png` | p.12 | SHARED | [ ] |
| `nt_p3_math_2566_q21.png` | p.13 | SHARED | [ ] |
| `nt_p3_math_2566_q22.png` | p.13 | SHARED | [ ] |
| `nt_p3_math_2566_q23.png` | p.14 | SHARED | [ ] |
| `nt_p3_math_2566_q24.png` | p.14 | SHARED | [ ] |

### nt_p3_math_2567 (9 items)

| File | PDF page | Shared | Cropped |
|------|----------|--------|---------|
| `nt_p3_math_2567_q02.png` | p.5 | | [ ] |
| `nt_p3_math_2567_q11.png` | p.25 | | [ ] |
| `nt_p3_math_2567_q12.png` | p.27 | | [ ] |
| `nt_p3_math_2567_q14.png` | p.31 | | [ ] |
| `nt_p3_math_2567_q15.png` | p.33 | | [ ] |
| `nt_p3_math_2567_q19.png` | p.41 | | [ ] |
| `nt_p3_math_2567_q20.png` | p.44 | | [ ] |
| `nt_p3_math_2567_q21.png` | p.46 | | [ ] |
| `nt_p3_math_2567_q23.png` | p.50 | | [ ] |
