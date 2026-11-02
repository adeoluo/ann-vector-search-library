//! Distance functions, summed in the same order as the Python implementation.

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
}
