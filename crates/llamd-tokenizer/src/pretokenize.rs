//! The qwen2 pre-tokenizer split.
//!
//! PROVENANCE. A reimplementation of `unicode_regex_split_custom_qwen2` in
//! `llama.cpp/src/unicode.cpp` (MIT, © 2023-2026 The ggml authors).
//!
//! WHY THIS IS HAND-WRITTEN RATHER THAN A REGEX. The reference regex is:
//!
//! ```text
//! (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//! |[^\r\n\p{L}\p{N}]?\p{L}+
//! |\p{N}
//! | ?[^\s\p{L}\p{N}]+[\r\n]*
//! |\s*[\r\n]+
//! |\s+(?!\S)
//! |\s+
//! ```
//!
//! The sixth alternative uses a negative lookahead, which Rust's `regex`
//! crate deliberately does not support — it is what guarantees linear-time
//! matching. llama.cpp hand-writes this same split (for speed); LLAMD does it
//! to avoid a backtracking-regex dependency in a hot path, and gets exact
//! control over the semantics as a side effect.
//!
//! THE SUBTLE RULE is alternative six. `\s+(?!\S)` before `\s+` means: in a
//! run of whitespace followed by text, all but the LAST space are taken as
//! one piece, and the last space attaches to the word that follows. That is
//! why " hello" tokenizes as one piece and not two. Getting it wrong changes
//! the tokenization of every indented line of code.

use unicode_general_category::{GeneralCategory as GC, get_general_category};

/// Is this `\p{L}` — a Unicode letter?
///
/// Deliberately the *category*, not `char::is_alphabetic`. The latter is the
/// Alphabetic property, which also includes `Nl` and `Other_Alphabetic` marks;
/// using it would disagree with the reference on scripts with combining marks.
fn is_letter(c: char) -> bool {
    matches!(
        get_general_category(c),
        GC::UppercaseLetter
            | GC::LowercaseLetter
            | GC::TitlecaseLetter
            | GC::ModifierLetter
            | GC::OtherLetter
    )
}

/// Is this `\p{N}` — a Unicode number?
fn is_number(c: char) -> bool {
    matches!(
        get_general_category(c),
        GC::DecimalNumber | GC::LetterNumber | GC::OtherNumber
    )
}

/// Is this `\s`?
fn is_space(c: char) -> bool {
    c.is_whitespace()
}

/// Split `text` into pre-token pieces, in order. Concatenating the result
/// reproduces the input exactly — a split that loses or duplicates a byte is
/// a bug the round-trip test catches.
pub fn split(text: &str) -> Vec<&str> {
    let cp: Vec<(usize, char)> = text.char_indices().collect();
    let n = cp.len();
    let at = |i: usize| -> Option<char> { cp.get(i).map(|&(_, c)| c) };
    // Byte offset of codepoint index `i`, or the end of the string.
    let off = |i: usize| -> usize { cp.get(i).map(|&(b, _)| b).unwrap_or(text.len()) };

    let mut bounds: Vec<usize> = Vec::new();
    let mut pos = 0usize;
    let mut prev = 0usize;
    macro_rules! emit {
        ($end:expr) => {{
            let end = $end;
            if end > prev {
                bounds.push(end);
                prev = end;
            }
        }};
    }

    while pos < n {
        let c = at(pos).unwrap();

        // 1. contractions: 's 't 're 've 'm 'll 'd, case-insensitive
        if c == '\'' && pos + 1 < n {
            let c1 = at(pos + 1).unwrap().to_ascii_lowercase();
            if matches!(c1, 's' | 't' | 'm' | 'd') {
                pos += 2;
                emit!(pos);
                continue;
            }
            if pos + 2 < n {
                let c2 = at(pos + 2).unwrap().to_ascii_lowercase();
                if (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')
                {
                    pos += 3;
                    emit!(pos);
                    continue;
                }
            }
        }

        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        //    an optional leading non-letter (not a newline), then letters
        if !(c == '\r' || c == '\n' || is_number(c))
            && (is_letter(c) || at(pos + 1).is_some_and(is_letter))
        {
            pos += 1;
            while at(pos).is_some_and(is_letter) {
                pos += 1;
            }
            emit!(pos);
            continue;
        }

        // 3. \p{N} — a SINGLE digit, not a run. Qwen splits numbers per digit.
        if is_number(c) {
            pos += 1;
            emit!(pos);
            continue;
        }

        // 4. ' ?[^\s\p{L}\p{N}]+[\r\n]*' — punctuation, with one leading space
        {
            let probe = if c == ' ' { at(pos + 1) } else { Some(c) };
            let is_punct =
                |c: Option<char>| c.is_some_and(|c| !is_space(c) && !is_letter(c) && !is_number(c));
            if is_punct(probe) {
                if c == ' ' {
                    pos += 1;
                }
                while is_punct(at(pos)) {
                    pos += 1;
                }
                while matches!(at(pos), Some('\r') | Some('\n')) {
                    pos += 1;
                }
                emit!(pos);
                continue;
            }
        }

        // 5/6/7 — the whitespace rules. Scan the run once, then decide.
        let mut ws = 0usize;
        let mut last_nl = 0usize;
        while at(pos + ws).is_some_and(is_space) {
            if matches!(at(pos + ws), Some('\r') | Some('\n')) {
                last_nl = pos + ws + 1;
            }
            ws += 1;
        }

        // 5. \s*[\r\n]+ — whitespace ending at the last newline
        if last_nl > 0 {
            pos = last_nl;
            emit!(pos);
            continue;
        }

        // 6. \s+(?!\S) — all but the last space, when text follows.
        //    This is the rule that makes " hello" a single piece.
        if ws > 1 && pos + ws < n {
            pos += ws - 1;
            emit!(pos);
            continue;
        }

        // 7. \s+
        if ws > 0 {
            pos += ws;
            emit!(pos);
            continue;
        }

        // Nothing matched: consume one codepoint so this cannot loop forever.
        // The reference does the same. A pre-tokenizer that hangs on one
        // unusual codepoint would be worse than one that splits it oddly.
        pos += 1;
        emit!(pos);
    }

    // Boundaries are codepoint indices; turn them into string slices. Because
    // every piece runs from the previous boundary to the next, concatenating
    // the result reproduces the input by construction.
    let mut out = Vec::with_capacity(bounds.len());
    let mut start = 0usize;
    for &end in &bounds {
        out.push(&text[off(start)..off(end)]);
        start = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_input() {
        // The split must be lossless. A dropped or duplicated byte here is a
        // silent corruption of the prompt.
        for s in [
            "hello world",
            "  indented\n\ttabbed\r\n",
            "it's a test, isn't it?",
            "fn main() { println!(\"hi\"); }",
            "日本語のテキスト",
            "emoji 🎉 and combining é",
            "",
            "   ",
            "\n\n\n",
            "12345",
        ] {
            let parts = split(s);
            assert_eq!(parts.concat(), s, "round-trip failed for {s:?}");
        }
    }

    #[test]
    fn leading_space_attaches_to_the_word() {
        // The classic GPT-2 behaviour, and rule 6's whole purpose.
        assert_eq!(split(" hello"), vec![" hello"]);
        assert_eq!(split("hello world"), vec!["hello", " world"]);
    }

    #[test]
    fn a_run_of_spaces_keeps_one_for_the_word() {
        // "    x" -> three spaces, then " x". This is what makes indented
        // code tokenize the way the model was trained on.
        assert_eq!(split("    x"), vec!["   ", " x"]);
    }

    #[test]
    fn trailing_whitespace_stays_whole() {
        // With nothing following, rule 6 does not apply and rule 7 takes all.
        assert_eq!(split("x    "), vec!["x", "    "]);
    }

    /// A DOCUMENTED BLIND SPOT, found by deliberately breaking this rule and
    /// watching the oracle test still pass.
    ///
    /// Making rule 3 greedy (`\p{N}+`) changes the SPLIT but not the final
    /// token ids, because BPE then merges the digits back together. So the
    /// end-to-end oracle comparison cannot see this rule at all, and only
    /// this unit test can.
    ///
    ///
    /// The lesson generalises: an end-to-end comparison proves the pipeline
    /// agrees, not that each stage does. Stages whose errors are absorbed
    /// downstream need their own tests, and knowing WHICH those are requires
    /// trying to break them.
    #[test]
    fn digits_split_individually() {
        // Rule 3 is `\p{N}`, not `\p{N}+`. Qwen tokenizes numbers per digit,
        // and a `+` here would change every number in every prompt.
        assert_eq!(split("2024"), vec!["2", "0", "2", "4"]);
    }

    #[test]
    fn contractions_are_one_piece() {
        assert_eq!(split("it's"), vec!["it", "'s"]);
        assert_eq!(split("IT'S"), vec!["IT", "'S"]);
        assert_eq!(split("we'll"), vec!["we", "'ll"]);
        assert_eq!(split("I'd"), vec!["I", "'d"]);
    }

    #[test]
    fn newlines_end_a_whitespace_run() {
        assert_eq!(split("a\n\nb"), vec!["a", "\n\n", "b"]);
        assert_eq!(split("a  \n b"), vec!["a", "  \n", " b"]);
    }

    #[test]
    fn punctuation_takes_one_leading_space() {
        assert_eq!(split("a, b"), vec!["a", ",", " b"]);
        assert_eq!(split("x ==> y"), vec!["x", " ==>", " y"]);
    }

    #[test]
    fn never_loops_on_unusual_input() {
        // A pre-tokenizer that hangs is worse than one that splits oddly.
        for s in ["\u{0}", "\u{FFFD}", "\u{200B}zero width"] {
            let parts = split(s);
            assert_eq!(parts.concat(), s);
        }
    }
}
