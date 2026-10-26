# Light Field Foundry

Light Field Foundry develops Lytro captures into modern Portrait and Spatial AVIF images.

## Motivation

I owned a first-gen Lytro camera in the early 2010s. I took several ~~noisy,~~
~~low-resolution,~~ precious photos with that camera. Those photos were
(mostly) locked inside the Lytro desktop photo manager. Sure, photos can be
exported as JPEGs, but then there is no ability to change the focus or change
perspective. Apple is also dropping support for Intel apps, so the
Lytro-provided apps won't be around forever.

Separately, when Apple started to introduce features like Portrait Photos and
Spatial Photos, I immediately recognized that these were tricks my old Lytro
was capable of performing years earlier.

This project is my attempt to liberate those old Lytro photos by bringing them
into the modern era! I want to develop these photos while Lytro's software
still runs, so I can keep refocusing them and viewing them in 3D in
Apple Photos long after that software stops working.

I chose AVIF because it's an open, royalty-free image format that offers
efficient compression, including a lossless mode. It also lets me keep the
images, depth map, and metadata needed for Portrait and Spatial photos together
in one file.

For Portrait, that means an all-in-focus image plus a depth map, which
Apple Photos uses to control which parts of the photo appear in focus.
For Spatial, it means two slightly different views, one for each eye. Lytro's
Light Field Engine can already generate all of these from the original capture.

This Python package uses Lytro's TNT command-line tool to generate those images
and depth data, then packages them with metadata from the original capture and
the additional metadata Apple Photos needs for Portrait and Spatial features.

## Technical Details & Limitations

The Python package is `light-field-foundry`, with the `lff` command and
`light_field_foundry` import package (`python -m light_field_foundry`).

`lff image-out` uses an existing TNT installation to reconstruct an Extended Depth
of Field (EDOF) center view,
two independent stereo eyes, and floating-point depth. It delivers one verified
AVIF containing exact RGB16 sample-transform representations for all three
views, ordinary RGB12 fallbacks, and one native monochrome12 depth auxiliary.
The original TIFF ICC bytes, source metadata, render settings, and the
demonstrated Apple Portrait/Spatial compatibility bundle are embedded.

This release supports **first-generation F01 captures, orientation 1,
center U=V=0, and 0 < baseline ≤ 0.2**. Raw-only inputs default to full-frame geometry; a selected saved recipe can
supply crop/rotation for recomputation.
Exported `edofWarp` stacks also preserve their saved rotation, crop and focus
spread, using the original stack center/depth and matching raw captures for
independent stereo eyes. Other cameras, pan/zoom/tilt and animated rendering
remain unsupported.

## Setup

Use Python 3.12 or newer and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked
```

Supply these external dependencies yourself:

- A working macOS TNT installation with its companion libraries in place.
  Apple Silicon needs Rosetta for the supplied x86_64 TNT 2.0/LFE 21 build.
- Matching camera calibration. Pass the **parent** containing camera serial
  folders, rather than one serial folder, as `--calibration`.
- `avifenc` supporting lossless RGB16 sample transforms (`--depth 12,4`);
  `avifdec` supporting `--sato`, AOM and dav1d decoding. Depth is encoded
  directly with `imagecodecs` as lossless full-range monochrome12.
- Apple Swift command-line tools, for read-only ImageIO/AVDepthData validation.

Set `TNT_PATH` to the TNT **executable file**, or put `tnt` on `PATH`:

```sh
export TNT_PATH='/path/to/lytro-power-tools/lpt/resources/bin/tnt'
uv run lff doctor
```

An invalid `TNT_PATH` is an error. The program never downloads, patches, or
redistributes TNT, its libraries, or calibration. Selected calibration is copied
into temporary storage for read-only input isolation; TNT remains externally
installed. `--avifenc PATH` and
`--avifdec PATH` override `PATH` discovery for those tools. Nothing is fetched
silently by the CLI. Windows and Linux TNT execution are not validated.

`doctor` checks reported capabilities, then proves actual RGB16 and all 4096
depth-code round trips using both AOM and dav1d. It assembles a complete
synthetic stereo/depth graph and checks native Apple depth readback using the
default `0..4.316406011581421` range. A version string alone does not pass.
Conversion runs the same generated control before
processing a capture; a failed control is a dependency/environment failure.

## Choose independent LFP inputs

The input roles are independent. Capture-bound LFP sources must identify the
same raw image payload; filenames and picture names alone are not sufficient.
Desktop recipe-only `stack.view.lfp` files can have no capture frames or image
identity. Explicitly selected recipe-only controls are reusable like JSON
controls; a conflicting identity, when present, is still rejected. Original
public/private/master metadata comes from the raw source, with the selected
recipe and each role's master hash recorded separately.

| Option         | Responsibility                                              | Default                                                         |
|----------------|-------------------------------------------------------------|-----------------------------------------------------------------|
| `--lfp-raw`    | Embedded packed raw in an `.lfp` or `.lfr`                  | Positional raw, then read-only library lookup                   |
| `--lfp-recipe` | Saved static recipe from an LFP, including `stack.view.lfp` | Positional saved recipe, then TNT defaults                      |
| `--lfp-depth`  | Native `packedFloat` or `pngEncodedByte` depth              | Positional native stack depth, otherwise TNT recomputation      |
| `--lfp-edof`   | Canonical center of a native TIFF `edofWarp` acceleration   | Positional EDOF cache, otherwise native TNT EDOF reconstruction |

`--lfp-receipe` is accepted as an alias for `--lfp-recipe`. The existing
`--recipe` option still means a JSON override. CLI controls override that JSON,
which overrides the selected LFP recipe, which overrides TNT defaults.
The positional input remains a shorthand that supplies whichever roles it
contains; explicit role options take precedence.

For scattered raw files, no library or database is required:

```sh
uv run lff image-out \
  --lfp-raw "$HOME/Pictures/Lytro Fresh Export/img00954.lfr" \
  --calibration /path/to/cameras \
  --output /path/to/photo.avif \
  --perspective-u 0 --perspective-v 0 --baseline 0.2
```

An explicit raw file bypasses library raw lookup. Calibration is required
without `--library`; with a library it defaults to that library's `cameras`
directory. Every native call uses a temporary raw-only container retaining the
exact raw and metadata chunks. Processed cache references are cleared, with
empty arrays retained where TNT's schema requires them. Calibration is also
copied into temporary storage. Original input files are read-only.

Mix the roles as needed, for example:

```sh
uv run lff image-out \
  --lfp-raw /path/to/capture.lfr \
  --lfp-recipe /path/to/UUID/stack.view.lfp \
  --lfp-depth /path/to/UUID/stack.lfp \
  --lfp-edof /path/to/UUID/stack.lfp \
  --no-recompute \
  --calibration /path/to/cameras \
  --output /path/to/photo.avif \
  --perspective-u 0 --perspective-v 0 --baseline 0.2
```

`--recompute-depth` ignores selected/pre-existing depth; `--recompute-edof`
ignores selected/pre-existing EDOF. `--recompute` enables both. These switches
also override caches supplied through explicit role flags. The selected recipe
still applies; omit `--lfp-recipe` and the positional saved-view input when you
want TNT's default controls. Cache-free raw staging prevents TNT from silently
reusing accelerations embedded in a raw-containing input.

Use `--no-recompute-depth` or `--no-recompute-edof` to require an existing,
matching lossless asset for that role. `--no-recompute` requires both. Missing,
incompatible, corrupt or unsupported caches fail the process; the tool never
substitutes newly generated data for a required cache. Select the caches with
`--lfp-depth` / `--lfp-edof`, or use a positional native stack. The switches
govern depth/EDOF selection; stereo eyes are still rendered from raw.

| Policy                                      | Depth                  | EDOF                  |
|---------------------------------------------|------------------------|-----------------------|
| Use cache when selected, otherwise generate | default                | default               |
| Force fresh data                            | `--recompute-depth`    | `--recompute-edof`    |
| Require existing data                       | `--no-recompute-depth` | `--no-recompute-edof` |

Force and require options for the same role conflict. Independent policies can
be mixed, for example `--no-recompute-edof --recompute-depth`. An EDOF-cache
requirement conflicts with `--refocus`; use `--no-recompute-depth` alone when
requesting a refocused primary with existing depth.

Cached EDOF must be unsigned RGB16 TIFF with a provably lossless compression:
uncompressed, LZW, Deflate, PackBits, LZMA or Zstd. JPEG-family, JPEG2000,
JPEG-XL, WebP and unknown TIFF compression tags are rejected before decoding;
TIFF as a container alone does not establish losslessness. This applies to
all reused caches, including default automatic reuse. Depth accepts lossless
PNG (`pngEncodedByte`) or literal Float32 (`packedFloat`) data. An 8-bit depth
PNG retains its original precision when stored in the 12-bit AVIF auxiliary.

The primary is **EDOF by default**. The attached Demo stacks' native EDOF images
are byte-identical across changed focus, aperture and focus-spread exports;
those saved defocus values are for later view rendering. Native recomputation
uses `--lfp-out --unpack`, never an aperture-zero ordinary still as a substitute.
Only `--refocus` requests a limited-depth primary. An arbitrary still or
`edofWarpPreAdjust` acceleration is not accepted as `--lfp-edof`.

Cached EDOF finishing/geometry must match the selected recipe. Depth must share
the selected geometry and a consistent normalized mapping/aspect. Use the
corresponding recompute switch after changing a baked setting. EDOF and depth
roles can omit each other's assets; depth-only metadata consistency is distinct
from an actual paired color-image witness.

## Convert one capture

```sh
uv run lff image-out /path/to/raw.lfp \
  --calibration /path/to/cameras \
  --output /path/to/photo.avif \
  --perspective-u 0 --perspective-v 0 --baseline 0.2 \
  --focus 2.20561 --stereo-pivot 0
```

`.lfr` and raw-containing `.lfp` inputs are accepted directly. Exported
`edofWarp` stacks use the `--library` route below. The primary/center image defaults to TNT's native all-in-focus
EDOF stack view. The example focus controls the stereo eyes; choose the lambda
appropriate for the capture. Baseline is TNT's normalized viewpoint separation, not a
distance in millimeters. Left/right coordinates are U−B/2 and U+B/2.

## Convert an exported stack

```sh
uv run lff image-out '/path/to/picture-uuid/stack.lfp' \
  --library "$HOME/Pictures/Lytro Library.lytrolibrary" \
  --output /path/to/photo.avif \
  --perspective-u 0 --perspective-v 0 --baseline 0.2
```

`--library` reads the Desktop database with SQLite `mode=ro&immutable=1`,
looks up the stack parent directory's picture UUID, and verifies the located
raw's `imageRef` against the stack. It never opens Desktop or modifies the
library; outputs inside the library are refused. Raw and calibration files
from the library are copied to the temporary workspace before TNT runs; TNT
receives only those temporary copies. Keep Desktop closed during
lookup so its database and capture files remain a stable snapshot.
Calibration defaults to the library's `cameras` directory; `--calibration`
can override it. Alternatively, provide `--lfp-raw` directly and calibration
explicitly to avoid the library entirely.

The original exported center TIFF and its ICC/RGB16 samples are retained.
Each view retains its own original ICC: Desktop exports may use ProPhoto RGB
while fresh TNT stills use sRGB. The stereo eyes must share an ICC; the center
may differ. No profile is substituted and no RGB samples are color-converted.
Depth comes from the associated `pngEncodedByte` PNG (or `packedFloat` asset)
using its own declared lambda endpoints. An 8-bit source remains 8-bit in
precision even though it is stored in a 12-bit auxiliary. Saved rotation/crop
are already baked into those assets; TNT receives the same geometry and
saved focus spread when rendering the independent eyes. Binary undeclared
coverage at rotated black corners is baked onto black without changing RGB
samples or trimming the image. Other nonopaque channels are rejected.

Saved static recipe values are used; animation tracks are preserved in the
original recipe metadata but are not evaluated. Stack apertures up to 2 are
supported. Focus/aperture/pivot overrides affect the eyes (and `--refocus`
center). The default EDOF center remains the selected native stack image; changing its
baked finishing or geometry requires `--recompute-edof` (and
`--recompute-depth` when geometry changes), or a newly exported stack.

Optional `--recipe RECIPE.json`, `--focus`, `--stereo-pivot`, and `--aperture`
control rendering. Explicit CLI values take precedence over the requested
recipe, which takes precedence over TNT's exported source recipe. Zero is a
real value, distinct from omission. Omitted focus freezes the exported recipe's
focus (which can be zero); it does **not** invoke independent automatic focus
for each eye. Both eyes share focus, aperture, geometry and convergence. For raw inputs, the
EDOF center uses the actual `edofWarp` center TIFF, with its native ICC, from
TNT's `--unpack` output; its depth of field is independent of those eye controls.
Use `--refocus` to render the center with the chosen focus/aperture instead.
Ordinary TNT stills use `viewPivot`; the CLI explicitly puts the resolved
stereo pivot in both pivot fields. Effective still logs and the generated
depth LFP recipe are checked against the resolved settings.

`--disparity-min` and `--disparity-max` control Apple's relative-disparity
rendering range independently of TNT focus/aperture. Defaults are `0` and
`4.316406011581421` (approximately `4.316`), the gain-only range accepted in a
manual Photos refocusing comparison. For example, append
`--disparity-min 1.5 --disparity-max 5.5` to choose another range, or
`--disparity-max 1` to restore the earlier `0..1` interpretation. Endpoints must
be finite, nonnegative, strictly increasing, and support Float32 depth-code
recovery. These options change the declared native disparity interpretation and
its provenance; stored depth codes, source lambda endpoints, and color samples
are unchanged. They select a relative presentation scale, not distances in meters.

`--threads 1..32` defaults to 4; `--timeout SECONDS` defaults to 600 **per native
process**. `--include-private-metadata` explicitly embeds original private JSON;
it is omitted by default. Source public/master JSON bytes, source recipe and
effective per-view controls remain distinct in embedded XMP. No absolute tool,
capture, calibration or output paths are added to render provenance.

The output's parent directory must exist. Existing outputs are never replaced,
including publication races. The CLI reports validation results, file size and
SHA-256 to the terminal. The **only delivered file is the completed AVIF**:
there is no receipt, sidecar, log, retained intermediate or resume mode.
Each run uses a unique `TemporaryDirectory` in the system temporary location.
It contains all render, encode, validation and completed-AVIF intermediates and
is removed on success, failure and handled interruption. There is no startup
scan, owner marker, PID tracking or inode journal. After an uncatchable kill or
power loss, leftovers are left to the operating system's temporary-file policy;
macOS may purge these at startup but does not guarantee every-restart deletion.
See [Apple's temporary-directory guidance](https://developer.apple.com/library/archive/documentation/General/Conceptual/MOSXAppProgrammingGuide/AppRuntime/AppRuntime.html).

Publication links the verified file atomically to its destination. If the
destination is on another filesystem, one temporary publication copy is needed
beside it; that directory is removed when publication finishes or fails. A hard
kill during that brief copy can leave it behind. Originals and unrelated files
are preserved, and existing outputs are never replaced.

## Precision, color and Apple interpretation

The TIFF/PNG bridge is checked sample-for-sample and preserves the complete ICC
payload. RGB8 output, missing/invalid ICC and unresolved TIFF
color-description conflicts are rejected. Nonopaque alpha is rejected except
for the documented exported-stack binary black coverage. The fourth channel
of this TNT build has no alpha declaration. The raw route discards it only after every value is
65535; exported stacks also admit the documented binary black coverage.

The ICC remains authoritative. This release uses unspecified standardized
primaries/transfer (`2/2/0`) and Exif uncalibrated rather than identifying sRGB
from a profile label. Samples are never color-converted. Full-range identity
4:4:4 coding preserves RGB codes. Each view's final base, suffix and derived
item has that view's original ICC. Readers without sample-transform support see RGB12.

The installed libavif shifts the four suffix bits when using `12,4`. The CLI
independently encodes literal low4 values in RGB8, preserves the RGB12 base,
and normalizes the derived expression to `16 * base + suffix`. Both decoders
check every final RGB16 sample, every RGB12 fallback and every literal suffix
code by selecting the final container's actual item identities.

F01 depth lambda decreases toward nearer surfaces. Declared maximum lambda
maps to code 0 and declared minimum lambda to code 4095, with nearest-integer,
ties-to-even rounding. There is no observed-extrema stretch or 8-bit fallback.
The raw route preserves its native 330² depth raster; exported stacks retain
their declared native depth dimensions, including cropped 320² depth.
Invalid/sentinel floating samples fail.
Apple reads the selected relative-disparity range with `StoredFormat=L016`:
12-bit code `q` expands to `(q << 4) | (q >> 8)`, then maps from `0..65535` to
the declared float endpoints (default `0..4.316406011581421`). Automated native
checks compare the resulting Float32 samples against those endpoints and
recover every prepared depth code exactly.
Source lambda endpoints and this intentional quantization are embedded. Depth
and Spatial presentation defaults are not metric Lytro camera calibration.

Portrait/Spatial metadata follows the demonstrated reference recipe, including
the minimal generated MakerNote, exact opaque REND packet, full-frame focus
region and reference Spatial properties. Their origins are documented in
[resource provenance](src/light_field_foundry/resources/PROVENANCE.md). The full-frame
focus region is presentation metadata and makes no detected-subject claim.

Embedded provenance identifies the application as Light Field Foundry. Original
capture metadata and render provenance use the
`https://light-field-foundry.org/ns/1.0/` XMP namespace with the `lff` prefix.

**Photos recognition, focus/aperture UI behavior, stereo comfort and physical
eye order for a new file remain manual acceptance gates.** The CLI never
opens Photos, imports pictures, or accesses a Photos library. Automated native
depth support does not by itself prove those UI behaviors. The default
disparity range has manual Portrait refocusing acceptance for one F01 EDOF
capture; this is not general scene, camera, or Spatial acceptance.

## Development

LFP/LFR binary framing is defined in [`formats/lfp_framing.ksy`](formats/lfp_framing.ksy)
and read by a Kaitai-generated Python parser. Python retains the original JSON
and payload bytes and validates SHA-1 references, metadata, and asset semantics.
Chunks are parsed individually to avoid retaining an entire capture in memory.
The schema documents field offsets, named layout constants, the shared signature
structure and ASCII `LFP`/`LFM`/`LFC` tags, SHA-1 names, payloads, and padding.
Its lazy `chunks` view also describes the complete file sequence for inspection.
Format observations and historical field interpretations are distinguished from
the application's size and compatibility limits.

The generated Python is checked into source control with its `.ksy` schema and
SHA-256 manifest. `uv build` verifies that pair before packaging with `uv_build`;
missing or stale output fails with regeneration instructions. Normal source,
wheel, and editable builds need no compiler or Java and download no compiler.
The runtime is pinned exactly to **`kaitaistruct==0.11.0`** in both build and
runtime dependencies; `uv_build` is also pinned exactly.

Regeneration and CI use **Kaitai Struct compiler 0.11.0**, pinned in
`[tool.kaitai]`. Upstream names this exact release **0.11**, which is equivalent
to 0.11.0; newer patch releases and snapshots are rejected. Install the compiler
from the [official 0.11 release](https://github.com/kaitai-io/kaitai_struct_compiler/releases/tag/0.11)
and a Java runtime, then regenerate:

```sh
uv run python build_backend.py --generate
```

`KAITAI_STRUCT_COMPILER` can select an alternate compiler executable path. The
generator normalizes trailing whitespace for Git. Commit the schema, generated
parser, and `formats/lfp_framing.generated.json` together.

The [Kaitai workflow](.github/workflows/kaitai.yml) runs when schemas, generated
files, compiler configuration, or build tooling change. It downloads the pinned
compiler, verifies the archive's SHA-256, and runs this read-only comparison:

```sh
uv run python build_backend.py --check
```

This compiles the schema afresh and compares the actual generated Python, so
updating the manifest alone cannot make an edited parser pass CI. Installed
wheels only use the Python runtime; reading a photo never invokes the compiler.

The [code quality workflow](.github/workflows/quality.yml) runs Ruff lint,
Ruff formatting, and ty on every push to any branch and every pull request,
without path filters. Its separate `Ruff` and `Types` jobs use Python 3.12 and
the development tools pinned in `uv.lock`. It can also be run manually.

Ruff requires function argument and return annotations, and checks imports,
common bugs, security mistakes, pytest usage, and Python modernization.
Formatting uses a 100-character target and formats code examples in docstrings.
ty checks application code, tests, and the build backend with every diagnostic
enabled as an error, warnings treated as failures, and strict equality and
generic narrowing enabled. The generated Kaitai parser is exempt from lint,
formatting, and direct type checking; its exact compiler output is verified by
the separate Kaitai workflow. Tests allow assertions and imports inside test
cases. Other exceptions are documented beside the affected code. Dynamic
metadata and third-party libraries still have some `Any` types; passing these
checks does not mean every value is statically typed.

Run the same quality checks locally, then the tests and package build:

```sh
uv sync --locked --dev
uv run --frozen --no-sync ruff check .
uv run --frozen --no-sync ruff format --check .
uv run --frozen --no-sync ty check --error-on-warning
uv run --frozen --no-sync python build_backend.py --verify
uv run --frozen --no-sync pytest
uv build
```

Tests use generated pixel/container fixtures and mocked failure paths. Native
codec/native Apple contract checks run through `lff doctor`; external photographic
acceptance is documented separately. Personal images, calibration, vendor
binaries and decoded photographic fixtures are never test/release assets.
