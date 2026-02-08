
from typing import List, Callable, Type
from aria.types import Grid
from .primitives import (
    shift, rotate_cw, rotate_ccw, flip_x, flip_y, color_replace,
    flood_fill, isolate_color, bitwise_or, detect_objects, label_grid,
    map_grid, fold_grid, hconcat, vconcat,
    filter_list, is_even, is_positive, logical_and, logical_or,
    add, sub, mul, div
)

# --- TYPED PRIMITIVES ---
# These are the blocks used by the Genetic Programming engine.
TYPED_PRIMITIVES = [
    (shift, [Grid, int, int], Grid),
    (rotate_cw, [Grid], Grid),
    (rotate_ccw, [Grid], Grid),
    (flip_x, [Grid], Grid),
    (flip_y, [Grid], Grid),
    (color_replace, [Grid, int, int], Grid),
    (flood_fill, [Grid, int, int, int], Grid),
    (isolate_color, [Grid, int], Grid),
    (bitwise_or, [Grid, Grid], Grid),
    (detect_objects, [Grid], List[Grid]),
    (label_grid, [Grid], List[Grid]),
    (map_grid, [List[Grid], Callable[[Grid], Grid]], List[Grid]),
    (fold_grid, [List[Grid], Callable[[Grid, Grid], Grid], Grid], Grid),
    (hconcat, [Grid, Grid], Grid),
    (vconcat, [Grid, Grid], Grid),
    (filter_list, [List[int], Callable[[int], bool]], List[int]),
    (is_even, [int], bool),
    (is_positive, [int], bool),
    (logical_and, [bool, bool], bool),
    (logical_or, [bool, bool], bool),
    # Math (Meta)
    (add, [float, float], float),
    (sub, [float, float], float),
    (mul, [float, float], float),
    (div, [float, float], float),
]