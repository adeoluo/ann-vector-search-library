//! Distance functions. The `f64` versions sum in the same order as the Python
//! implementation, one term at a time, so results match it bit for bit. The `f32`
//! versions keep independent partial sums in lanes that the compiler maps to SIMD
//! registers; that reorders the additions, so they are fast but not bit-identical.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Squared Euclidean distance (ranks like Euclidean, no square root).
    Euclidean,
    /// `1 - cosine similarity`; 1.0 when either vector is zero.
    Cosine,
}

impl Metric {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "euclidean" => Ok(Self::Euclidean),
            "cosine" => Ok(Self::Cosine),
            other => Err(format!("unknown metric: {other:?}")),
        }
    }

    #[inline]
    pub fn distance(self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            Self::Euclidean => a.iter().zip(b).fold(0.0, |acc, (x, y)| {
                let d = x - y;
                acc + d * d
            }),
            Self::Cosine => {
                let (mut dot, mut na, mut nb) = (0.0, 0.0, 0.0);
                for (x, y) in a.iter().zip(b) {
                    dot += x * y;
                    na += x * x;
                    nb += y * y;
                }
                if na == 0.0 || nb == 0.0 {
                    1.0
                } else {
                    1.0 - dot / (na.sqrt() * nb.sqrt())
                }
            }
        }
    }
}

/// Partial sums kept side by side: four 128-bit registers of `f32`.
const LANES: usize = 16;

#[inline]
fn squared_euclidean_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<LANES>(), b.as_chunks::<LANES>());
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..LANES {
            let d = x[i] - y[i];
            acc[i] += d * d;
        }
    }
    let mut total: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        let d = x - y;
        total += d * d;
    }
    total
}

#[inline]
fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    const W: usize = 8;
    let (mut dot, mut na, mut nb) = ([0.0f32; W], [0.0f32; W], [0.0f32; W]);
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<W>(), b.as_chunks::<W>());
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..W {
            dot[i] += x[i] * y[i];
            na[i] += x[i] * x[i];
            nb[i] += y[i] * y[i];
        }
    }
    // Finish in f64 so the final 1 - cos keeps its precision near zero distance.
    let (mut d, mut a2, mut b2) = (
        dot.iter().map(|&v| v as f64).sum::<f64>(),
        na.iter().map(|&v| v as f64).sum::<f64>(),
        nb.iter().map(|&v| v as f64).sum::<f64>(),
    );
    for (&x, &y) in ra.iter().zip(rb) {
        d += (x * y) as f64;
        a2 += (x * x) as f64;
        b2 += (y * y) as f64;
    }
    if a2 == 0.0 || b2 == 0.0 {
        1.0
    } else {
        1.0 - d / (a2.sqrt() * b2.sqrt())
    }
}

/// A vector element type the index can store.
pub trait Element: Copy + Send + Sync + 'static {
    fn from_f64(x: f64) -> Self;
    fn distance(metric: Metric, a: &[Self], b: &[Self]) -> f64;
}

impl Element for f64 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline]
    fn distance(metric: Metric, a: &[f64], b: &[f64]) -> f64 {
        metric.distance(a, b)
    }
}

impl Element for f32 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    #[inline]
    fn distance(metric: Metric, a: &[f32], b: &[f32]) -> f64 {
        match metric {
            Metric::Euclidean => squared_euclidean_f32(a, b) as f64,
            Metric::Cosine => cosine_f32(a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert_eq!(Metric::Euclidean.distance(&[0.0, 0.0], &[3.0, 4.0]), 25.0);
        assert_eq!(Metric::Cosine.distance(&[1.0, 0.0], &[0.0, 1.0]), 1.0);
        assert_eq!(Metric::Cosine.distance(&[0.0, 0.0], &[1.0, 1.0]), 1.0);
        assert!(Metric::Cosine.distance(&[1.0, 2.0], &[2.0, 4.0]).abs() < 1e-12);
        assert!(Metric::parse("dot").is_err());
    }

    #[test]
    fn f32_lanes_agree_with_f64() {
        for dim in [1, 7, 16, 33, 128, 769] {
            let a: Vec<f64> = (0..dim)
                .map(|i| ((i * 37 % 101) as f64 - 50.0) / 7.0)
                .collect();
            let b: Vec<f64> = (0..dim)
                .map(|i| ((i * 53 % 97) as f64 - 40.0) / 9.0)
                .collect();
            let (af, bf): (Vec<f32>, Vec<f32>) = (
                a.iter().map(|&x| x as f32).collect(),
                b.iter().map(|&x| x as f32).collect(),
            );
            for metric in [Metric::Euclidean, Metric::Cosine] {
                let exact = metric.distance(&a, &b);
                let fast = f32::distance(metric, &af, &bf);
                assert!(
                    (exact - fast).abs() <= 1e-5 * exact.abs().max(1.0),
                    "{dim} {metric:?}"
                );
            }
        }
        assert_eq!(f32::distance(Metric::Cosine, &[0.0, 0.0], &[1.0, 1.0]), 1.0);
    }
}
