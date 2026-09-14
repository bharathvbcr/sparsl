# Brand assets

**Mark:** Compressed row walk · **Palette:** Emerald Sparse.

A 4×4 grid holding five stored entries, threaded in the order the kernels
actually read them — row-major over `col_indices`, skipping everything the
format never stored. The hollow cells are structural zeros: never allocated,
never fetched, never multiplied. Row 2 carries two entries where its neighbours
carry one, which is the density difference `RowKernel` resolves into a lane
tier. The mark and the code agree.

It is deliberately a sibling of [tessl](https://github.com/bharathvbcr/tessl)'s
mark rather than a variation on it: same plate geometry, same 128 viewBox, same
optical size — and the opposite content. tessl threads a Z-order walk through a
*full* grid, because every tile of a dense GEMM is visited. sparsl threads a
row-major walk through a *sparse* one, because most of the matrix does not
exist.

| | |
| --- | --- |
| Base | Carbon Green-Black `#050B09` |
| Surfaces | Deep Moss `#0A1310` |
| Stored entries | Emerald `#34D399` · structural zeros outlined in `#10B981` at 0.16 |
| The walk | Pale Mint `#DFFFF0` |
| Wordmark on dark | Mint Frost `#E6F7EF` · on light: base `#050B09` |

## Files

```
logo-mark.svg            vector source (128×128), the single owner of the art
png/logo-mark@{1024,512,256,128,64,32}.png    icon / avatar / favicon
png/logo@{2048,900}.png       wordmark lockup for light pages
png/logo-dark@{2048,900}.png  wordmark lockup for dark pages
build_logo.py            regenerates every PNG above
```

```bash
python3 build_logo.py
```

## Why the wordmark is PNG and not SVG

An SVG `<text>` element renders with whatever font the viewer has. GitHub serves
README SVGs through `<img>`, so the wordmark would reflow or fall back on any
machine without SF Mono. The lockups are composed in Pillow with the font baked
to pixels; the mark stays vector.

## Two renderer facts worth keeping

`qlmanage` is macOS's own renderer and the only one here that handles the glow
filter and clip path. It flattens onto opaque white with no alpha, so
`build_logo.py` rebuilds the alpha analytically from the known plate geometry
rather than flood-filling — a flood fill eats into the pale walk, which is
nearly white where it crosses a stored cell.

The glow is tuned for the icon ladder, not for the 1024 render. At
`stdDeviation` 3.2 — tessl's value, which suits its thinner path — the walk
bleeds into the stored cells and the whole mark turns to mush by 32 px. 1.4 is
the largest value that keeps the cells and the walk reading as separate objects
all the way down.
