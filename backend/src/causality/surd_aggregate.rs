//! Aggregate SURD decomposition (redundant, unique, synergistic, leak).
//!
//! Follows the reference SURD algorithm of Martínez-Sánchez et al. (2024) and the structure of
//! `deep_causality_algorithms::surd::surd_states`. We do not call the library function: at commit
//! 6cc7f6f it returns wrong values when p(T) is not uniform (single source, p(T) = (0.7, 0.3):
//! 0.110 bits reported vs 0.192 bits mutual information), files unique information as redundant
//! when another source has zero specific information (it skips zero increments before updating
//! the redundant set), fails with `ShapeMismatch` when target and source state counts differ, and
//! panics when a source state never occurs (reported upstream: https://github.com/deepcausality-rs/deep_causality/issues/878). This port agrees with a numpy transcription of the
//! reference code (scripts/surd_reference.py) to within 5e-13 bits (see
//! `tests::dump_for_reference`).

use std::collections::BTreeMap;

const EPS: f64 = 1e-14;

pub struct SurdAggregate {
    /// Keyed by source set (1-based source indices). Singletons are unique information.
    pub redundant: BTreeMap<Vec<usize>, f64>,
    pub synergistic: BTreeMap<Vec<usize>, f64>,
    pub mutual_info: BTreeMap<Vec<usize>, f64>,
    /// H(target | all sources) / H(target), as in the library.
    pub info_leak_fraction: f64,
    pub target_entropy_bits: f64,
}

/// `p` is a joint distribution over [target, source_1..source_n], row-major with `shape`.
pub fn surd_aggregate(p: &[f64], shape: &[usize]) -> SurdAggregate {
    let total: f64 = p.iter().sum();
    let p: Vec<f64> = p.iter().map(|v| v / total).collect();
    let n_vars = shape.len() - 1;
    let n_t = shape[0];
    let sources: Vec<usize> = (1..=n_vars).collect();

    let p_t = marginal(&p, shape, &[0]);
    let combs: Vec<Vec<usize>> = (1..=n_vars)
        .flat_map(|k| combinations(&sources, k))
        .collect();

    // Specific information I_s(t; S_j) for every combination j and target state t.
    let specific: BTreeMap<Vec<usize>, Vec<f64>> = combs
        .iter()
        .map(|j| {
            let keep_tj: Vec<usize> = std::iter::once(0).chain(j.iter().copied()).collect();
            let p_tj = marginal(&p, shape, &keep_tj);
            let p_j = marginal(&p, shape, j);
            let cells_j = p_j.len();
            let is: Vec<f64> = (0..n_t)
                .map(|t| {
                    (0..cells_j)
                        .map(|s| {
                            let joint = p_tj[t * cells_j + s];
                            if joint <= 0.0 || p_t[t] <= 0.0 || p_j[s] <= 0.0 {
                                return 0.0;
                            }
                            (joint / p_t[t]) * ((joint / p_j[s]).log2() - p_t[t].log2())
                        })
                        .sum()
                })
                .collect();
            (j.clone(), is)
        })
        .collect();

    let mutual_info = specific
        .iter()
        .map(|(j, is)| (j.clone(), is.iter().zip(&p_t).map(|(i, pt)| i * pt).sum()))
        .collect();

    let mut redundant: BTreeMap<Vec<usize>, f64> = BTreeMap::new();
    let mut synergistic: BTreeMap<Vec<usize>, f64> = BTreeMap::new();
    for t in 0..n_t {
        let values: Vec<f64> = combs.iter().map(|c| specific[c][t]).collect();
        let order = arg_sort(&values);
        let lab: Vec<Vec<usize>> = order.iter().map(|&i| combs[i].clone()).collect();
        let mut sorted: Vec<f64> = order.iter().map(|&i| values[i]).collect();
        let lens: Vec<usize> = lab.iter().map(Vec::len).collect();

        // A combination of order l+1 only counts if it beats every combination of order l.
        for l in 1..n_vars {
            let max_prev = sorted
                .iter()
                .zip(&lens)
                .filter(|(_, &n)| n == l)
                .map(|(&v, _)| v)
                .fold(f64::NEG_INFINITY, f64::max);
            if max_prev.is_finite() {
                sorted
                    .iter_mut()
                    .zip(&lens)
                    .filter(|(_, &n)| n == l + 1)
                    .for_each(|(v, _)| {
                        if *v < max_prev {
                            *v = 0.0;
                        }
                    });
            }
        }

        let reorder = arg_sort(&sorted);
        let final_vals: Vec<f64> = reorder.iter().map(|&i| sorted[i]).collect();
        let final_lab: Vec<Vec<usize>> = reorder.iter().map(|&i| lab[i].clone()).collect();
        let increments: Vec<f64> = final_vals
            .iter()
            .scan(0.0, |prev, &v| {
                let d = v - *prev;
                *prev = v;
                Some(d)
            })
            .collect();

        let mut red_vars = sources.clone();
        for (ll, di) in final_lab.iter().zip(&increments) {
            let info = di * p_t[t];
            // No early `continue` on zero increments (DeepCausality has one): a source with zero
            // specific information must still leave the redundant set, as in the reference code;
            // otherwise the next source's unique information is filed as redundant.
            if ll.len() > 1 {
                *synergistic.entry(ll.clone()).or_insert(0.0) += info;
            } else {
                *redundant.entry(red_vars.clone()).or_insert(0.0) += info;
                red_vars.retain(|&v| v != ll[0]);
            }
        }
    }

    let h_t = entropy(&p_t);
    let h_t_given_s = entropy(&p) - entropy(&marginal(&p, shape, &sources));
    SurdAggregate {
        redundant,
        synergistic,
        mutual_info,
        info_leak_fraction: if h_t > EPS {
            (h_t_given_s / h_t).clamp(0.0, 1.0)
        } else {
            0.0
        },
        target_entropy_bits: h_t,
    }
}

/// Marginal over `keep` axes (in the given order), row-major.
fn marginal(p: &[f64], shape: &[usize], keep: &[usize]) -> Vec<f64> {
    let kept: Vec<usize> = keep.iter().map(|&a| shape[a]).collect();
    let mut out = vec![0.0; kept.iter().product()];
    let mut idx = vec![0usize; shape.len()];
    for &v in p {
        let k = keep
            .iter()
            .zip(&kept)
            .fold(0, |acc, (&a, &dim)| acc * dim + idx[a]);
        out[k] += v;
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

fn entropy(p: &[f64]) -> f64 {
    -p.iter()
        .filter(|&&v| v > 0.0)
        .map(|v| v * v.log2())
        .sum::<f64>()
}

fn arg_sort(v: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    idx
}

fn combinations(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    (0..items.len())
        .flat_map(|i| {
            combinations(&items[i + 1..], k - 1)
                .into_iter()
                .map(move |mut rest| {
                    rest.insert(0, items[i]);
                    rest
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn joint(cells: &[(usize, usize, usize)], shape: &[usize]) -> Vec<f64> {
        // cells: (t, s1, s2), equal mass
        let mut p = vec![0.0; shape.iter().product()];
        for &(t, a, b) in cells {
            p[(t * shape[1] + a) * shape[2] + b] += 1.0;
        }
        p
    }

    #[test]
    fn xor_is_pure_synergy() {
        // T = S1 xor S2, uniform sources: 1 bit, all synergistic.
        let p = joint(&[(0, 0, 0), (1, 0, 1), (1, 1, 0), (0, 1, 1)], &[2, 2, 2]);
        let r = surd_aggregate(&p, &[2, 2, 2]);
        assert!((r.synergistic[&vec![1, 2]] - 1.0).abs() < 1e-9);
        assert!(r.redundant.values().all(|v| v.abs() < 1e-9));
    }

    #[test]
    fn duplicated_source_is_pure_redundancy() {
        // T = S1 = S2: 1 bit, shared by both sources.
        let p = joint(&[(0, 0, 0), (1, 1, 1)], &[2, 2, 2]);
        let r = surd_aggregate(&p, &[2, 2, 2]);
        assert!((r.redundant[&vec![1, 2]] - 1.0).abs() < 1e-9);
        assert!(r.redundant.get(&vec![1]).map_or(true, |v| v.abs() < 1e-9));
        assert!(r.synergistic.values().all(|v| v.abs() < 1e-9));
    }

    #[test]
    fn independent_noise_leaves_unique_information_to_the_driver() {
        // T = S1; S2 independent coin: 1 bit unique to S1, nothing else, no leak.
        let p = joint(&[(0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1)], &[2, 2, 2]);
        let r = surd_aggregate(&p, &[2, 2, 2]);
        assert!((r.redundant[&vec![1]] - 1.0).abs() < 1e-9);
        assert!(r.info_leak_fraction < 1e-9);
        assert!(r.synergistic.values().all(|v| v.abs() < 1e-9));
    }

    /// Cross-check against the reference SURD (numpy, keepdims broadcasting):
    ///   cargo test -p icu_sepsis_backend dump_for_reference -- --ignored --nocapture \
    ///     | grep -E '^CASE|^  PORT' > /tmp/surd_dump.txt
    ///   python3 scripts/surd_reference.py /tmp/surd_dump.txt
    /// Last run (2026-11-23): max |port - reference| <= 4.8e-13 bits on all five tables.
    #[test]
    #[ignore]
    fn dump_for_reference() {
        let cases: Vec<(Vec<usize>, u64)> = vec![
            (vec![2, 2, 2], 1),
            (vec![2, 3, 3], 7),
            (vec![3, 3, 3], 11),
            (vec![2, 2, 2, 2], 3),
            (vec![2, 3, 2, 3], 5),
        ];
        for (shape, seed) in cases {
            let n: usize = shape.iter().product();
            let mut state = seed;
            let p: Vec<f64> = (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    0.05 + ((state >> 33) % 1000) as f64 / 1000.0
                })
                .collect();
            let r = surd_aggregate(&p, &shape);
            let fmt = |m: &BTreeMap<Vec<usize>, f64>| {
                m.iter()
                    .map(|(k, v)| format!("{:?}:{:.12}", k, v))
                    .collect::<Vec<_>>()
                    .join(";")
            };
            println!("CASE shape={:?} p={:?}", shape, p);
            println!("  PORT R={} S={}", fmt(&r.redundant), fmt(&r.synergistic));
        }
    }
}
