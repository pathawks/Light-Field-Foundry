meta:
  id: lfp_framing
  title: Lytro LFP/LFR container framing (supported version 1 profile)
  file-extension: [lfp, lfr]
  endian: be
  ks-version: '0.11'
doc: |
  LFP/LFR is a container of named payloads. A file begins with a 16-byte
  package header, followed by chunks through end of file. Each chunk has a
  96-byte header, its declared payload, and zero padding to the next absolute
  16-byte file offset. No closing record or chunk-count field is present.

  All signatures have the same structure: a non-ASCII prefix byte (0x89), a
  three-character ASCII tag, and the control bytes CR LF SUB LF. The tags are
  LFP for the package, LFM for master metadata, and LFC for other content.
  Master/content signatures differ at byte offset 3: ASCII 'M' (0x4d) versus
  'C' (0x43). In a continuous hexadecimal display this can look like 'd0'
  versus '30' when the digits are grouped across byte boundaries. The shared
  prefix and suffix are encoded once by signature below.

  LFM contains the master UTF-8 JSON object describing the picture and linking
  to assets. LFC can contain public/private JSON metadata, packed sensor data,
  encoded images, depth, or other binary assets. The tag alone does not identify
  an LFC payload's encoding; JSON references and asset descriptors do that.
  For example, imageRef, metadataRef, and privateMetadataRef identify chunks by
  their stored names. A referenced asset may be absent from a processed export;
  unpacked exports can instead describe external assets using imagePath.

  This schema describes binary framing, leaves payloads as exact bytes, and
  declares the complete chunk sequence as a lazy instance. Accessing chunks
  materializes that sequence. The application instead reads Chunk objects one
  at a time from the same stream, verifies SHA-1 names, requires one master,
  and applies its chunk-count, JSON, image, and capture-compatibility checks.

  The accepted profile is version 1 with an empty package record. Size limits
  are caller policy, not intrinsic LFP constants. JSON structure varies between
  camera/Desktop versions and is outside this framing schema. The header-word
  interpretation and older chunk-length layout are documented on their fields;
  this is not a claim to cover every Lytro container variant.
doc-ref:
  - https://github.com/kmader/lytro-power-tools/blob/master/lpt/lfp/lfp.py
  - https://github.com/nrpatel/lfptools/blob/master/lfpsplitter.c
  - https://optics.miloush.net/lytro/TheFileFormat.aspx
params:
  - id: max_section_bytes
    type: u8
    doc: Caller-selected maximum payload size; checked before reading any payload.
  - id: file_size
    type: u8
    doc: Actual file size captured by the caller; bounds payloads and their padding.
seq:
  - id: magic
    type: signature
    valid:
      expr: _.tag == file_marker
    doc: Package signature at file offset 0; its ASCII tag must be LFP.
  - id: version
    type: u4
    valid: supported_version
    doc: Big-endian header word at file offset 8; this profile accepts version 1.
  - id: package_payload_length
    type: u4
    valid: empty_package_payload_length
    doc: |
      Big-endian word at file offset 12, zero in this profile. Early format
      descriptions interpret this as the initial package record's empty data
      length, so no reference field or payload follows it. The first chunk
      begins immediately at file_header_size. The Python container historically
      exposes this word as flags; no flag-bit meanings are established here.
instances:
  alignment:
    value: 16
    doc: Absolute file-offset alignment in bytes for the start/end of each chunk.
  reference_size:
    value: 80
    doc: Fixed-width chunk-name field in bytes, including its trailing NUL fill.
  signature_size:
    value: 8
    doc: One prefix byte, a three-byte ASCII tag, and four control bytes.
  marker_text_size:
    value: 3
    doc: Byte length of the ASCII tags LFP, LFM, and LFC.
  header_word_size:
    value: 4
    doc: Byte width of each package-header integer (Kaitai u4).
  payload_length_size:
    value: 8
    doc: Byte width of the chunk-length region read by Lytro Power Tools (Kaitai s8).
  file_header_size:
    value: signature_size + header_word_size + header_word_size
    doc: Package header size, 16 bytes; also the first chunk's absolute offset.
  chunk_header_size:
    value: signature_size + payload_length_size + reference_size
    doc: Chunk header size, 96 bytes; excludes payload and trailing alignment padding.
  supported_version:
    value: 1
    doc: Package version accepted by this schema's supported profile.
  empty_package_payload_length:
    value: 0
    doc: The supported package header carries no data of its own.
  padding_byte:
    value: 0
    doc: NUL byte used to fill alignment padding.
  file_marker:
    value: '"LFP"'
    doc: ASCII tag identifying the initial package header.
  master_marker:
    value: '"LFM"'
    doc: ASCII tag identifying the master JSON chunk; M denotes master metadata.
  content_marker:
    value: '"LFC"'
    doc: ASCII tag identifying a content/component chunk; C denotes other content.
  chunks:
    pos: file_header_size
    type: chunk
    repeat: eos
    doc: |
      Complete sequence of LFM/LFC chunks after the package header. This lazy
      view makes the file grammar explicit and supports format-inspection tools.
      Access it while the stream is open; it retains all parsed payloads. Large
      files should instead be consumed as individual Chunk objects.
types:
  signature:
    doc: Shared eight-byte signature layout for package, master, and content records.
    seq:
      - id: prefix
        contents: [0x89]
        doc: Non-ASCII signature prefix, shared by all three record kinds.
      - id: tag
        type: str
        size: _root.marker_text_size
        encoding: ASCII
        doc: Human-readable record tag; the enclosing record validates its allowed kind.
      - id: suffix
        contents: "\r\n\u001a\n"
        doc: Shared control-byte suffix, in order CR, LF, SUB (ASCII 26), LF.
  chunk:
    doc: |
      A master or content record beginning at an aligned absolute file offset.
      Relative layout: signature at 0, length at 8, reference at 16, payload at
      chunk_header_size (96). Only payload bytes contribute to len_payload and
      its SHA-1 name. Padding is outside both the length and the digest.
    seq:
      - id: marker
        type: signature
        valid:
          expr: _.tag == _root.master_marker or _.tag == _root.content_marker
        doc: Chunk signature; only the ASCII tags LFM and LFC are admitted here.
      - id: len_payload
        type: s8
        valid:
          expr: >-
            _ >= 0 and _ <= _root.max_section_bytes and
            (_io.pos + _root.reference_size + _ + _root.alignment - 1) /
            _root.alignment * _root.alignment <= _root.file_size
        doc: |
          Payload byte count, excluding the reference and padding. Lytro Power
          Tools reads this eight-byte region as one signed big-endian integer.
          Earlier readers treat it as a zero/reserved 32-bit word followed by
          an unsigned 32-bit length. Both layouts coincide within this reader's
          payload limit; the schema does not assign independent semantics to
          the high word. The rounded end must fit inside the original file.
      - id: reference
        size: _root.reference_size
        doc: |
          Fixed-width raw chunk name, preserved with its trailing NUL bytes.
          In the supported profile, remove the trailing NULs to obtain ASCII
          'sha1-' plus 40 lowercase hexadecimal digits (45 bytes, then 35 NULs).
          The digest covers the exact payload, including any JSON whitespace
          or binary suffix, and excludes the chunk header and alignment padding.
          The application validates syntax, digest, and uniqueness; these are
          reader requirements rather than constraints on all possible names.
      - id: payload
        size: len_payload
        doc: |
          Opaque stored bytes. A master chunk is a UTF-8 JSON object; other
          payloads may be JSON, binary, or a JSON prefix with a binary suffix.
          Re-encoding JSON can change the digest even if its parsed values are
          equal, so original bytes must be retained independently of any model.
      - id: padding
        type: u1
        repeat: expr
        repeat-expr: (_root.alignment - _io.pos % _root.alignment) % _root.alignment
        valid: _root.padding_byte
        doc: |
          Zero to alignment-1 NUL bytes reaching the next aligned file offset.
          An already aligned payload end has no padding; this is not a whole
          extra alignment block. The last chunk follows the same rule.
    instances:
      is_master:
        value: marker.tag == _root.master_marker
        doc: Whether this record contains the master JSON rather than another asset.
