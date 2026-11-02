//! Byte counts as people write them.
//!
//! `K`/`M`/`G`/`T` are BINARY (×1024) with or without the `i`, because memory
//! is sold and reported in binary units and a budget that silently means
//! 10^9 on a "32G" card is 7% off — enough to overcommit VRAM. Decimal units
//! must be spelled `GB` / `MB` / `KB` / `TB` explicitly.

use llamd_core::{Error, Result};

pub fn parse_bytes(s: &str) -> Result<u64> {
    let t = s.trim();
    let split = t
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(t.len());
    let (num, unit) = t.split_at(split);
    let v: f64 = num
        .parse()
        .map_err(|_| Error::malformed("size", format!("{s:?} has no number")))?;
    let mul: f64 = match unit.trim() {
        "" | "B" | "b" => 1.0,
        "K" | "k" | "KiB" | "Ki" => 1024.0,
        "M" | "MiB" | "Mi" => 1024f64.powi(2),
        "G" | "GiB" | "Gi" => 1024f64.powi(3),
        "T" | "TiB" | "Ti" => 1024f64.powi(4),
        "KB" | "kB" => 1e3,
        "MB" => 1e6,
        "GB" => 1e9,
        "TB" => 1e12,
        other => {
            return Err(Error::malformed(
                "size",
                format!("{s:?}: unknown unit {other:?}"),
            ));
        }
    };
    let bytes = v * mul;
    if !bytes.is_finite() || bytes < 0.0 || bytes > u64::MAX as f64 {
        return Err(Error::malformed("size", format!("{s:?} is out of range")));
    }
    Ok(bytes.round() as u64)
}

/// Binary units, two decimals: `31.75 GiB`.
pub fn fmt_bytes(b: u64) -> String {
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{b} B")
    } else {
        format!("{v:.2} {}", U[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn units_are_binary_unless_spelled_decimal() {
        assert_eq!(parse_bytes("32G").unwrap(), 32 << 30);
        assert_eq!(parse_bytes("32GiB").unwrap(), 32 << 30);
        assert_eq!(parse_bytes("32GB").unwrap(), 32_000_000_000);
        assert_eq!(parse_bytes("512M").unwrap(), 512 << 20);
        assert_eq!(parse_bytes("1.5G").unwrap(), 3 << 29);
        assert_eq!(parse_bytes("4096").unwrap(), 4096);
    }

    #[test]
    fn nonsense_is_refused() {
        for s in ["", "G", "12Q", "-1G", "1e400G"] {
            assert!(parse_bytes(s).is_err(), "{s:?}");
        }
    }

    #[test]
    fn formatting_round_trips_the_unit() {
        assert_eq!(fmt_bytes(512), "512 B");
        assert_eq!(fmt_bytes(32 << 30), "32.00 GiB");
        assert_eq!(fmt_bytes(1536 << 20), "1.50 GiB");
    }
}
