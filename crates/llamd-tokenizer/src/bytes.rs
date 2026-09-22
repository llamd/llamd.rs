//! The GPT-2 byte ↔ unicode mapping.
//!
//! PROVENANCE. Derived from `unicode_byte_to_utf8_map` in
//! `llama.cpp/src/unicode.cpp` (MIT, © 2023-2026 The ggml authors).
//!
//! WHY THIS EXISTS AT ALL. BPE operates on text, but a tokenizer must handle
//! arbitrary bytes — including invalid UTF-8. GPT-2's answer is to map all
//! 256 byte values to printable codepoints first: bytes that are already
//! printable map to themselves, and the remaining 68 are lifted into
//! U+0100..U+0143. That is why a GGUF vocabulary contains tokens like `Ġ`
//! (U+0120) — it is byte 0x20, a space.
//!
//! Getting this wrong does not produce an error. It produces different tokens
//! for every string containing a space, which is every string.

/// Byte → the codepoint that represents it in vocabulary text.
pub fn byte_to_cpt(b: u8) -> char {
    BYTE_TO_CPT[b as usize]
}

/// Codepoint → the byte it represents, if it is one of the 256.
pub fn cpt_to_byte(c: char) -> Option<u8> {
    let u = c as u32;
    if u < 0x200 {
        CPT_TO_BYTE[u as usize]
    } else {
        None
    }
}

/// True for the bytes GPT-2 considers directly printable.
fn is_printable_byte(ch: u32) -> bool {
    (0x21..=0x7E).contains(&ch) || (0xA1..=0xAC).contains(&ch) || (0xAE..=0xFF).contains(&ch)
}

static BYTE_TO_CPT: std::sync::LazyLock<[char; 256]> = std::sync::LazyLock::new(|| {
    let mut map = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        map[b as usize] = if is_printable_byte(b) {
            char::from_u32(b).expect("printable byte is a valid codepoint")
        } else {
            // The 68 non-printable bytes are lifted above U+0100, in byte
            // order. The counter must advance only for those, which is why
            // this cannot be folded into the branch above.
            let c = char::from_u32(256 + n).expect("lifted codepoint is valid");
            n += 1;
            c
        };
    }
    map
});

static CPT_TO_BYTE: std::sync::LazyLock<[Option<u8>; 0x200]> = std::sync::LazyLock::new(|| {
    let mut inv = [None; 0x200];
    for b in 0..256usize {
        inv[BYTE_TO_CPT[b] as usize] = Some(b as u8);
    }
    inv
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_becomes_g_with_a_stroke() {
        // Byte 0x20 is the famous `Ġ` seen throughout a GGUF vocabulary. If
        // this is wrong, every token containing a space is wrong.
        assert_eq!(byte_to_cpt(0x20), 'Ġ');
        assert_eq!(byte_to_cpt(0x20) as u32, 0x120);
        assert_eq!(cpt_to_byte('Ġ'), Some(0x20));
    }

    #[test]
    fn newline_and_tab_are_lifted() {
        assert_eq!(byte_to_cpt(b'\n') as u32, 0x10A);
        assert_eq!(byte_to_cpt(b'\t') as u32, 0x109);
    }

    #[test]
    fn printable_ascii_maps_to_itself() {
        for b in 0x21..=0x7Eu8 {
            assert_eq!(
                byte_to_cpt(b),
                b as char,
                "byte {b:#x} should map to itself"
            );
        }
    }

    #[test]
    fn the_mapping_is_a_bijection() {
        // 256 distinct bytes must give 256 distinct codepoints, and the
        // inverse must round-trip every one. A collision here would make two
        // different bytes tokenize identically.
        let mut seen = std::collections::HashSet::new();
        for b in 0..=255u8 {
            let c = byte_to_cpt(b);
            assert!(seen.insert(c), "collision: byte {b:#x} -> {c:?}");
            assert_eq!(cpt_to_byte(c), Some(b), "round-trip failed for {b:#x}");
        }
        assert_eq!(seen.len(), 256);
    }

    #[test]
    fn exactly_68_bytes_are_lifted() {
        // 94 printable ASCII + 12 + 82 latin-1 = 188 direct, 68 lifted.
        let lifted = (0..=255u8)
            .filter(|&b| byte_to_cpt(b) as u32 >= 0x100)
            .count();
        assert_eq!(lifted, 68);
    }
}
