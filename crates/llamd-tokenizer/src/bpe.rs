//! Byte-pair encoding, and the vocabulary loaded from a GGUF.
//!
//! PROVENANCE. The merge order is derived from the bigram loop in
//! `llama.cpp/src/llama-vocab.cpp` (MIT, © 2023-2026 The ggml authors).
//!
//! THE MERGE RULE. Repeatedly join the adjacent pair with the LOWEST rank,
//! where rank is the pair's position in the merges list. Ties break toward
//! the leftmost pair. Both halves of that sentence matter: a different tie
//! rule produces different tokens on perhaps one string in a thousand, which
//! is frequent enough to corrupt real prompts and rare enough to survive
//! casual testing.

use llamd_core::{Error, Result};
use std::collections::HashMap;

use crate::{bytes, pretokenize};

/// What a token is for. Values are ggml's `llama_token_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

pub struct Vocab {
    /// id → token text, in the byte-encoded representation.
    pub tokens: Vec<String>,
    pub types: Vec<TokenType>,
    /// token text → id.
    index: HashMap<String, u32>,
    /// (left, right) → rank. Lower wins.
    merges: HashMap<(String, String), u32>,
    /// Tokens matched before any splitting: `<|im_start|>` and friends.
    special: Vec<(String, u32)>,
    pub eos: Option<u32>,
    pub bos: Option<u32>,
    pub add_bos: bool,
}

impl Vocab {
    /// Load the vocabulary from an open GGUF.
    pub fn from_gguf(g: &llamd_gguf::Gguf) -> Result<Self> {
        let model = g
            .get("tokenizer.ggml.model")
            .and_then(llamd_gguf::Value::as_str)
            .ok_or_else(|| Error::malformed("vocab", "tokenizer.ggml.model is absent"))?;
        if model != "gpt2" {
            return Err(Error::unsupported(
                "tokenizer model",
                format!("{model}; only gpt2-style BPE is implemented"),
            ));
        }
        // The pre-tokenizer decides how text is split before BPE. Refusing an
        // unknown one is deliberate: silently falling back to a different
        // split would produce plausible tokens that are simply not the
        // model's, and nothing downstream would notice.
        let pre = g
            .get("tokenizer.ggml.pre")
            .and_then(llamd_gguf::Value::as_str)
            .unwrap_or("default");
        if pre != "qwen2" {
            return Err(Error::unsupported(
                "pre-tokenizer",
                format!("{pre}; only qwen2 is implemented"),
            ));
        }

        let toks = g
            .get("tokenizer.ggml.tokens")
            .and_then(llamd_gguf::Value::as_array)
            .ok_or_else(|| Error::malformed("vocab", "tokenizer.ggml.tokens is absent"))?;
        let tokens: Vec<String> = toks
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();

        let types: Vec<TokenType> = g
            .get("tokenizer.ggml.token_type")
            .and_then(llamd_gguf::Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|v| match v.as_u64().unwrap_or(1) {
                        2 => TokenType::Unknown,
                        3 => TokenType::Control,
                        4 => TokenType::UserDefined,
                        5 => TokenType::Unused,
                        6 => TokenType::Byte,
                        _ => TokenType::Normal,
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![TokenType::Normal; tokens.len()]);

        let mut index = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            // First id wins: a duplicate later in the list must not shadow the
            // canonical one.
            index.entry(t.clone()).or_insert(i as u32);
        }

        let mut merges = HashMap::new();
        if let Some(ms) = g
            .get("tokenizer.ggml.merges")
            .and_then(llamd_gguf::Value::as_array)
        {
            for (rank, m) in ms.iter().enumerate() {
                let s = m.as_str().unwrap_or("");
                // Merges are "left right". Split on the FIRST space only: a
                // token may itself be a space, and splitting on all of them
                // silently drops those merges.
                if let Some((l, r)) = s.split_once(' ') {
                    merges.insert((l.to_string(), r.to_string()), rank as u32);
                }
            }
        }

        // Control and user-defined tokens are matched literally, before the
        // text is split at all. Longest first, so `<|im_start|>` is preferred
        // over any shorter token that prefixes it.
        let mut special: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| matches!(types[*i], TokenType::Control | TokenType::UserDefined))
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        // Longest first, so `<|im_start|>` wins over any token that prefixes it.
        special.sort_by_key(|(t, _)| std::cmp::Reverse(t.len()));

        let get_id = |k: &str| {
            g.get(k)
                .and_then(llamd_gguf::Value::as_u64)
                .map(|v| v as u32)
        };

        Ok(Vocab {
            tokens,
            types,
            index,
            merges,
            special,
            eos: get_id("tokenizer.ggml.eos_token_id"),
            bos: get_id("tokenizer.ggml.bos_token_id"),
            add_bos: g
                .get("tokenizer.ggml.add_bos_token")
                .map(|v| matches!(v, llamd_gguf::Value::Bool(true)))
                .unwrap_or(false),
        })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
    pub fn id_of(&self, text: &str) -> Option<u32> {
        self.index.get(text).copied()
    }
    pub fn text_of(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    /// Encode text to token ids.
    ///
    /// `parse_special` controls whether `<|im_start|>` and friends are
    /// recognised as single tokens. A chat template needs it; raw user text
    /// generally should not, or a prompt containing those characters could
    /// forge a role boundary.
    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if self.add_bos {
            if let Some(b) = self.bos {
                out.push(b);
            }
        }
        self.encode_fragment(text, parse_special, &mut out);
        out
    }

    fn encode_fragment(&self, text: &str, parse_special: bool, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        // Special tokens are matched first and split the text around them.
        if parse_special {
            if let Some((idx, tok, id)) = self.find_special(text) {
                self.encode_fragment(&text[..idx], parse_special, out);
                out.push(id);
                self.encode_fragment(&text[idx + tok.len()..], parse_special, out);
                return;
            }
        }
        for piece in pretokenize::split(text) {
            self.encode_piece(piece, out);
        }
    }

    /// Earliest occurrence of any special token, preferring the longest match.
    fn find_special(&self, text: &str) -> Option<(usize, &str, u32)> {
        let mut best: Option<(usize, &str, u32)> = None;
        for (tok, id) in &self.special {
            if tok.is_empty() {
                continue;
            }
            if let Some(i) = text.find(tok.as_str()) {
                // Earliest wins; at the same position the longer token wins,
                // which the sort already guarantees.
                if best.is_none_or(|(bi, _, _)| i < bi) {
                    best = Some((i, tok.as_str(), *id));
                }
            }
        }
        best
    }

    /// BPE one pre-token piece.
    fn encode_piece(&self, piece: &str, out: &mut Vec<u32>) {
        // Byte-encode first: BPE operates on the mapped representation, which
        // is what the vocabulary and merges are written in.
        let mut syms: Vec<String> = piece
            .as_bytes()
            .iter()
            .map(|&b| bytes::byte_to_cpt(b).to_string())
            .collect();
        if syms.is_empty() {
            return;
        }

        // Repeatedly merge the best-ranked adjacent pair. This is O(n²) in the
        // worst case and llama.cpp uses a heap; a pre-token piece is short, so
        // the simple version is used until a benchmark says otherwise. It is
        // also much easier to see that it is correct, which matters more here.
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                if let Some(&rank) = self.merges.get(&(syms[i].clone(), syms[i + 1].clone())) {
                    // Strictly-less keeps the LEFTMOST pair on a tie.
                    if best.is_none_or(|(_, br)| rank < br) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", syms[i], syms[i + 1]);
            syms.splice(i..i + 2, [merged]);
        }

        for s in syms {
            match self.id_of(&s) {
                Some(id) => out.push(id),
                None => {
                    // A symbol with no id means the merges produced something
                    // outside the vocabulary. Fall back to single characters
                    // rather than dropping text silently.
                    for ch in s.chars() {
                        if let Some(id) = self.id_of(&ch.to_string()) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    /// Decode ids back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut raw: Vec<u8> = Vec::new();
        for &id in ids {
            let Some(t) = self.text_of(id) else { continue };
            if matches!(
                self.types.get(id as usize),
                Some(TokenType::Control) | Some(TokenType::UserDefined)
            ) {
                raw.extend_from_slice(t.as_bytes());
                continue;
            }
            for ch in t.chars() {
                match bytes::cpt_to_byte(ch) {
                    Some(b) => raw.push(b),
                    None => {
                        let mut buf = [0u8; 4];
                        raw.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&raw).into_owned()
    }
}
