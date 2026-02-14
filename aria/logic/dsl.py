"""Domain-specific language exports for logic synthesis engines.

This module provides a stable compatibility surface for both legacy
(untype-aware) genetic engine and typed/MCTS engines.
"""

from typing import List, Callable

from aria.types import Grid
from .primitives import (
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_RED,
    COLOR_GREEN,
    COLOR_YELLOW,
    COLOR_GRAY,
    COLOR_MAGENTA,
    COLOR_ORANGE,
    COLOR_CYAN,
    COLOR_MAROON,
    shift,
    rotate_cw,
    rotate_ccw,
    flip_x,
    flip_y,
    color_replace,
    flood_fill,
    isolate_color,
    bitwise_or,
    if_color,
    detect_objects,
    label_grid,
    map_grid,
    fold_grid,
    hconcat,
    vconcat,
    subgrid,
    overlay,
    filter_list,
    is_even,
    is_positive,
    logical_and,
    logical_or,
    add,
    sub,
    mul,
    div,
)

# --- Legacy primitive set (for aria.logic.genetic.GeneticEngine) ---
# Keep this narrow and grid-focused so old engine behavior stays deterministic.
PRIMITIVES = [
    (shift, [Grid, int, int]),
    (rotate_cw, [Grid]),
    (rotate_ccw, [Grid]),
    (flip_x, [Grid]),
    (flip_y, [Grid]),
    (color_replace, [Grid, int, int]),
    (flood_fill, [Grid, int, int, int]),
    (subgrid, [Grid, int, int, int, int]),
    (overlay, [Grid, Grid, int, int]),
]

# --- Typed primitive set (for TypedGeneticEngine / MCTSSolver) ---
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
    (if_color, [Grid, int, Callable[[Grid], Grid], Callable[[Grid], Grid]], Grid),
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
    (add, [float, float], float),
    (sub, [float, float], float),
    (mul, [float, float], float),
    (div, [float, float], float),
]
