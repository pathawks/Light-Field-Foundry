# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild
# type: ignore

import kaitaistruct
from kaitaistruct import KaitaiStruct, KaitaiStream, BytesIO


if getattr(kaitaistruct, 'API_VERSION', (0, 9)) < (0, 11):
    raise Exception("Incompatible Kaitai Struct Python API: 0.11 or later is required, but you have %s" % (kaitaistruct.__version__))

class LfpFraming(KaitaiStruct):
    """LFP/LFR is a container of named payloads. A file begins with a 16-byte
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

    .. seealso::
       Source - https://github.com/kmader/lytro-power-tools/blob/master/lpt/lfp/lfp.py


    .. seealso::
       Source - https://github.com/nrpatel/lfptools/blob/master/lfpsplitter.c


    .. seealso::
       Source - https://optics.miloush.net/lytro/TheFileFormat.aspx
    """
    def __init__(self, max_section_bytes, file_size, _io, _parent=None, _root=None):
        super(LfpFraming, self).__init__(_io)
        self._parent = _parent
        self._root = _root or self
        self.max_section_bytes = max_section_bytes
        self.file_size = file_size
        self._read()

    def _read(self):
        self.magic = LfpFraming.Signature(self._io, self, self._root)
        _ = self.magic
        if not _.tag == self.file_marker:
            raise kaitaistruct.ValidationExprError(self.magic, self._io, u"/seq/0")
        self.version = self._io.read_u4be()
        if not self.version == self.supported_version:
            raise kaitaistruct.ValidationNotEqualError(self.supported_version, self.version, self._io, u"/seq/1")
        self.package_payload_length = self._io.read_u4be()
        if not self.package_payload_length == self.empty_package_payload_length:
            raise kaitaistruct.ValidationNotEqualError(self.empty_package_payload_length, self.package_payload_length, self._io, u"/seq/2")


    def _fetch_instances(self):
        pass
        self.magic._fetch_instances()
        _ = self.chunks
        if hasattr(self, '_m_chunks'):
            pass
            for i in range(len(self._m_chunks)):
                pass
                self._m_chunks[i]._fetch_instances()



    class Chunk(KaitaiStruct):
        """A master or content record beginning at an aligned absolute file offset.
        Relative layout: signature at 0, length at 8, reference at 16, payload at
        chunk_header_size (96). Only payload bytes contribute to len_payload and
        its SHA-1 name. Padding is outside both the length and the digest.
        """
        def __init__(self, _io, _parent=None, _root=None):
            super(LfpFraming.Chunk, self).__init__(_io)
            self._parent = _parent
            self._root = _root
            self._read()

        def _read(self):
            self.marker = LfpFraming.Signature(self._io, self, self._root)
            _ = self.marker
            if not  ((_.tag == self._root.master_marker) or (_.tag == self._root.content_marker)) :
                raise kaitaistruct.ValidationExprError(self.marker, self._io, u"/types/chunk/seq/0")
            self.len_payload = self._io.read_s8be()
            _ = self.len_payload
            if not  ((_ >= 0) and (_ <= self._root.max_section_bytes) and ((((((self._io.pos() + self._root.reference_size) + _) + self._root.alignment) - 1) // self._root.alignment) * self._root.alignment <= self._root.file_size)) :
                raise kaitaistruct.ValidationExprError(self.len_payload, self._io, u"/types/chunk/seq/1")
            self.reference = self._io.read_bytes(self._root.reference_size)
            self.payload = self._io.read_bytes(self.len_payload)
            self.padding = []
            for i in range((self._root.alignment - self._io.pos() % self._root.alignment) % self._root.alignment):
                self.padding.append(self._io.read_u1())
                if not self.padding[i] == self._root.padding_byte:
                    raise kaitaistruct.ValidationNotEqualError(self._root.padding_byte, self.padding[i], self._io, u"/types/chunk/seq/4")



        def _fetch_instances(self):
            pass
            self.marker._fetch_instances()
            for i in range(len(self.padding)):
                pass


        @property
        def is_master(self):
            """Whether this record contains the master JSON rather than another asset."""
            if hasattr(self, '_m_is_master'):
                return self._m_is_master

            self._m_is_master = self.marker.tag == self._root.master_marker
            return getattr(self, '_m_is_master', None)


    class Signature(KaitaiStruct):
        """Shared eight-byte signature layout for package, master, and content records."""
        def __init__(self, _io, _parent=None, _root=None):
            super(LfpFraming.Signature, self).__init__(_io)
            self._parent = _parent
            self._root = _root
            self._read()

        def _read(self):
            self.prefix = self._io.read_bytes(1)
            if not self.prefix == b"\x89":
                raise kaitaistruct.ValidationNotEqualError(b"\x89", self.prefix, self._io, u"/types/signature/seq/0")
            self.tag = (self._io.read_bytes(self._root.marker_text_size)).decode(u"ASCII")
            self.suffix = self._io.read_bytes(4)
            if not self.suffix == b"\x0D\x0A\x1A\x0A":
                raise kaitaistruct.ValidationNotEqualError(b"\x0D\x0A\x1A\x0A", self.suffix, self._io, u"/types/signature/seq/2")


        def _fetch_instances(self):
            pass


    @property
    def alignment(self):
        """Absolute file-offset alignment in bytes for the start/end of each chunk."""
        if hasattr(self, '_m_alignment'):
            return self._m_alignment

        self._m_alignment = 16
        return getattr(self, '_m_alignment', None)

    @property
    def chunk_header_size(self):
        """Chunk header size, 96 bytes; excludes payload and trailing alignment padding."""
        if hasattr(self, '_m_chunk_header_size'):
            return self._m_chunk_header_size

        self._m_chunk_header_size = (self.signature_size + self.payload_length_size) + self.reference_size
        return getattr(self, '_m_chunk_header_size', None)

    @property
    def chunks(self):
        """Complete sequence of LFM/LFC chunks after the package header. This lazy
        view makes the file grammar explicit and supports format-inspection tools.
        Access it while the stream is open; it retains all parsed payloads. Large
        files should instead be consumed as individual Chunk objects.
        """
        if hasattr(self, '_m_chunks'):
            return self._m_chunks

        _pos = self._io.pos()
        self._io.seek(self.file_header_size)
        self._m_chunks = []
        i = 0
        while not self._io.is_eof():
            self._m_chunks.append(LfpFraming.Chunk(self._io, self, self._root))
            i += 1

        self._io.seek(_pos)
        return getattr(self, '_m_chunks', None)

    @property
    def content_marker(self):
        """ASCII tag identifying a content/component chunk; C denotes other content."""
        if hasattr(self, '_m_content_marker'):
            return self._m_content_marker

        self._m_content_marker = u"LFC"
        return getattr(self, '_m_content_marker', None)

    @property
    def empty_package_payload_length(self):
        """The supported package header carries no data of its own."""
        if hasattr(self, '_m_empty_package_payload_length'):
            return self._m_empty_package_payload_length

        self._m_empty_package_payload_length = 0
        return getattr(self, '_m_empty_package_payload_length', None)

    @property
    def file_header_size(self):
        """Package header size, 16 bytes; also the first chunk's absolute offset."""
        if hasattr(self, '_m_file_header_size'):
            return self._m_file_header_size

        self._m_file_header_size = (self.signature_size + self.header_word_size) + self.header_word_size
        return getattr(self, '_m_file_header_size', None)

    @property
    def file_marker(self):
        """ASCII tag identifying the initial package header."""
        if hasattr(self, '_m_file_marker'):
            return self._m_file_marker

        self._m_file_marker = u"LFP"
        return getattr(self, '_m_file_marker', None)

    @property
    def header_word_size(self):
        """Byte width of each package-header integer (Kaitai u4)."""
        if hasattr(self, '_m_header_word_size'):
            return self._m_header_word_size

        self._m_header_word_size = 4
        return getattr(self, '_m_header_word_size', None)

    @property
    def marker_text_size(self):
        """Byte length of the ASCII tags LFP, LFM, and LFC."""
        if hasattr(self, '_m_marker_text_size'):
            return self._m_marker_text_size

        self._m_marker_text_size = 3
        return getattr(self, '_m_marker_text_size', None)

    @property
    def master_marker(self):
        """ASCII tag identifying the master JSON chunk; M denotes master metadata."""
        if hasattr(self, '_m_master_marker'):
            return self._m_master_marker

        self._m_master_marker = u"LFM"
        return getattr(self, '_m_master_marker', None)

    @property
    def padding_byte(self):
        """NUL byte used to fill alignment padding."""
        if hasattr(self, '_m_padding_byte'):
            return self._m_padding_byte

        self._m_padding_byte = 0
        return getattr(self, '_m_padding_byte', None)

    @property
    def payload_length_size(self):
        """Byte width of the chunk-length region read by Lytro Power Tools (Kaitai s8)."""
        if hasattr(self, '_m_payload_length_size'):
            return self._m_payload_length_size

        self._m_payload_length_size = 8
        return getattr(self, '_m_payload_length_size', None)

    @property
    def reference_size(self):
        """Fixed-width chunk-name field in bytes, including its trailing NUL fill."""
        if hasattr(self, '_m_reference_size'):
            return self._m_reference_size

        self._m_reference_size = 80
        return getattr(self, '_m_reference_size', None)

    @property
    def signature_size(self):
        """One prefix byte, a three-byte ASCII tag, and four control bytes."""
        if hasattr(self, '_m_signature_size'):
            return self._m_signature_size

        self._m_signature_size = 8
        return getattr(self, '_m_signature_size', None)

    @property
    def supported_version(self):
        """Package version accepted by this schema's supported profile."""
        if hasattr(self, '_m_supported_version'):
            return self._m_supported_version

        self._m_supported_version = 1
        return getattr(self, '_m_supported_version', None)
