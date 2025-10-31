use crate::*;
use nalgebra::DMatrix;
use std::collections::HashMap;

/// Cluster assignment for stocks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterAssignment {
    pub stock_to_cluster: HashMap<String, usize>,
    pub cluster_to_stocks: HashMap<usize, Vec<String>>,
    pub num_clusters: usize,
}

/// Statistical Industry Classification using Kakushadze & Yu (2016) methodology
pub struct StatisticalClustering {
    pub lookback: usize,
    pub max_iterations: usize,
    pub num_samplings: usize, // For aggregation
}

impl StatisticalClustering {
    pub fn new(lookback: usize, num_samplings: usize) -> Self {
        Self {
            lookback,
            max_iterations: 100,
            num_samplings,
        }
    }

    /// Main clustering function - implements Kakushadze & Yu methodology
    /// Clusters normalized returns R_is / σ_i^2
    pub fn cluster_stocks(
        &self,
        train_data: &HashMap<String, PriceData>,
        num_clusters: usize,
    ) -> ClusterAssignment {
        // Step 1: Normalize returns following Kakushadze & Yu (2016) Eq. 6-8
        let normalized_returns = self.normalize_returns(train_data);

        // Step 2: Run k-means multiple times and aggregate (Section 3.3.3)
        let aggregated_clusters = if self.num_samplings > 1 {
            self.aggregate_multiple_samplings(&normalized_returns, num_clusters)
        } else {
            self.single_kmeans_clustering(&normalized_returns, num_clusters)
        };

        aggregated_clusters
    }

    /// Normalize returns following Kakushadze & Yu (2016) Eq. 6-8
    /// Returns R_is / (σ_i * u_i) where u_i handles low-volatility stocks
    fn normalize_returns(&self, train_data: &HashMap<String, PriceData>) -> DMatrix<f64> {
        let symbols: Vec<String> = train_data.keys().cloned().collect();
        let n = symbols.len();
        let d = self.lookback;

        let mut returns_matrix = DMatrix::zeros(n, d);

        // Calculate returns matrix
        for (i, symbol) in symbols.iter().enumerate() {
            let data = &train_data[symbol];
            let rets = &data.returns[..d.min(data.returns.len())];
            for (j, &ret) in rets.iter().enumerate() {
                returns_matrix[(i, j)] = ret;
            }
        }

        // Calculate standard deviations
        let mut sigmas = Vec::with_capacity(n);
        for i in 0..n {
            let row = returns_matrix.row(i);
            let mean = row.sum() / d as f64;
            let variance = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (d - 1) as f64;
            sigmas.push(variance.sqrt());
        }

        // Calculate u_i following Eq. 7-8
        let log_sigmas: Vec<f64> = sigmas.iter().map(|s| s.ln()).collect();
        let median_log_sigma = self.median(&log_sigmas);
        let mad_log_sigma = self.mad(&log_sigmas, median_log_sigma);
        let v = (median_log_sigma - 3.0 * mad_log_sigma).exp();

        let mut u_vec = Vec::with_capacity(n);
        for &sigma in &sigmas {
            let u = (sigma / v).max(1.0); // u_i = max(σ_i/v, 1)
            u_vec.push(u);
        }

        // Normalize: R_is / (σ_i * u_i)
        let mut normalized = DMatrix::zeros(n, d);
        for i in 0..n {
            for j in 0..d {
                normalized[(i, j)] = returns_matrix[(i, j)] / (sigmas[i] * u_vec[i]);
            }
        }

        normalized
    }

    /// Single k-means clustering
    fn single_kmeans_clustering(&self, data: &DMatrix<f64>, k: usize) -> ClusterAssignment {
        let n = data.nrows();
        let d = data.ncols();

        // Initialize centers randomly
        let mut centers = self.initialize_centers(data, k);
        let mut assignments = vec![0; n];

        // K-means iterations
        for _ in 0..self.max_iterations {
            let mut changed = false;

            // Assignment step
            for i in 0..n {
                let point = data.row(i);
                let mut min_dist = f64::INFINITY;
                let mut best_cluster = 0;

                for c in 0..k {
                    let center = centers.row(c);
                    let dist = self.euclidean_distance(&point.into(), &center.into());
                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = c;
                    }
                }

                if assignments[i] != best_cluster {
                    assignments[i] = best_cluster;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update step
            for c in 0..k {
                let cluster_points: Vec<usize> = assignments
                    .iter()
                    .enumerate()
                    .filter(|(_, cluster)| **cluster == c)
                    .map(|(idx, _)| idx)
                    .collect();

                if cluster_points.is_empty() {
                    continue;
                }

                // Calculate new center
                for j in 0..d {
                    let sum: f64 = cluster_points.iter().map(|&idx| data[(idx, j)]).sum();
                    centers[(c, j)] = sum / cluster_points.len() as f64;
                }
            }
        }

        self.create_cluster_assignment(assignments, k)
    }

    /// Aggregate multiple k-means samplings (Kakushadze & Yu Section 3.3.3)
    fn aggregate_multiple_samplings(&self, data: &DMatrix<f64>, k: usize) -> ClusterAssignment {
        let n = data.nrows();
        let mut occurrence_matrix = DMatrix::zeros(n, k);
        // let mut all_centers = Vec::new();

        // Run multiple samplings
        for _ in 0..self.num_samplings {
            let assignment = self.single_kmeans_clustering(data, k);

            // Accumulate occurrences
            for (stock_idx, &cluster_id) in assignment.stock_to_cluster.values().enumerate() {
                occurrence_matrix[(stock_idx, cluster_id)] += 1.0;
            }

            // Store centers (would need to extract from clustering)
            // This is simplified - full implementation would track centers
        }

        // Determine final assignments based on max occurrences
        let mut final_assignments = vec![0; n];
        for i in 0..n {
            let mut max_count = 0.0;
            let mut best_cluster = 0;

            for c in 0..k {
                if occurrence_matrix[(i, c)] > max_count {
                    max_count = occurrence_matrix[(i, c)];
                    best_cluster = c;
                }
            }

            final_assignments[i] = best_cluster;
        }

        self.create_cluster_assignment(final_assignments, k)
    }

    /// Initialize centers using k-means++ strategy
    fn initialize_centers(&self, data: &DMatrix<f64>, k: usize) -> DMatrix<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let n = data.nrows();
        let d = data.ncols();
        let mut centers = DMatrix::zeros(k, d);

        // Pick first center randomly
        let first_idx = rng.gen_range(0..n);
        centers.row_mut(0).copy_from(&data.row(first_idx));

        // Pick remaining centers using k-means++
        for c in 1..k {
            let mut distances = vec![f64::INFINITY; n];

            for i in 0..n {
                let point = data.row(i);
                for j in 0..c {
                    let center = centers.row(j);
                    let dist = self.euclidean_distance(&point.into(), &center.into());
                    distances[i] = distances[i].min(dist);
                }
            }

            // Sample proportional to squared distance
            let sum_sq_dist: f64 = distances.iter().map(|d| d * d).sum();
            let mut rand_val = rng.r#gen::<f64>() * sum_sq_dist;

            let mut selected_idx = 0;
            for (i, &dist) in distances.iter().enumerate() {
                rand_val -= dist * dist;
                if rand_val <= 0.0 {
                    selected_idx = i;
                    break;
                }
            }

            centers.row_mut(c).copy_from(&data.row(selected_idx));
        }

        centers
    }

    fn euclidean_distance(
        &self,
        a: &nalgebra::RowDVector<f64>,
        b: &nalgebra::RowDVector<f64>,
    ) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn create_cluster_assignment(&self, assignments: Vec<usize>, k: usize) -> ClusterAssignment {
        let mut stock_to_cluster = HashMap::new();
        let mut cluster_to_stocks: HashMap<usize, Vec<String>> = HashMap::new();

        for c in 0..k {
            cluster_to_stocks.insert(c, Vec::new());
        }

        // This is simplified - full implementation would map back to stock symbols
        for (idx, &cluster) in assignments.iter().enumerate() {
            let symbol = format!("STOCK_{}", idx); // Placeholder
            stock_to_cluster.insert(symbol.clone(), cluster);
            cluster_to_stocks.get_mut(&cluster).unwrap().push(symbol);
        }

        ClusterAssignment {
            stock_to_cluster,
            cluster_to_stocks,
            num_clusters: k,
        }
    }

    fn median(&self, values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }
    }

    fn mad(&self, values: &[f64], median: f64) -> f64 {
        let deviations: Vec<f64> = values.iter().map(|v| (v - median).abs()).collect();
        self.median(&deviations)
    }

    /// Determine optimal number of clusters using eRank (Kakushadze & Yu Section 5.1)
    pub fn determine_num_clusters(&self, train_data: &HashMap<String, PriceData>) -> usize {
        // Calculate correlation matrix eigenvalues
        let normalized_returns = self.normalize_returns(train_data);
        let n = normalized_returns.nrows();

        // Calculate correlation matrix
        let mut corr_matrix = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let row_i = normalized_returns.row(i);
                let row_j = normalized_returns.row(j);
                let corr = self.correlation(&row_i.into(), &row_j.into());
                corr_matrix[(i, j)] = corr;
                corr_matrix[(j, i)] = corr;
            }
        }

        // Compute eigenvalues
        let eigen = corr_matrix.symmetric_eigen();
        let eigenvalues = eigen.eigenvalues;

        // Calculate eRank (Eq. 25-27 in paper)
        let positive_eigenvalues: Vec<f64> = eigenvalues
            .iter()
            .filter(|&&lambda| lambda > 1e-10)
            .cloned()
            .collect();

        let sum_eigenvalues: f64 = positive_eigenvalues.iter().sum();
        let probabilities: Vec<f64> = positive_eigenvalues
            .iter()
            .map(|&lambda| lambda / sum_eigenvalues)
            .collect();

        // Shannon entropy
        let entropy: f64 = probabilities.iter().map(|&p| -p * p.ln()).sum();

        // eRank = exp(entropy)
        let erank = entropy.exp();

        // Round to get cluster number (Eq. 28)
        erank.round() as usize
    }

    fn correlation(&self, a: &nalgebra::RowDVector<f64>, b: &nalgebra::RowDVector<f64>) -> f64 {
        let n = a.len() as f64;
        let mean_a = a.sum() / n;
        let mean_b = b.sum() / n;

        let cov: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - mean_a) * (y - mean_b))
            .sum::<f64>()
            / n;

        let std_a = (a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / n).sqrt();
        let std_b = (b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / n).sqrt();

        if std_a == 0.0 || std_b == 0.0 {
            0.0
        } else {
            cov / (std_a * std_b)
        }
    }
}
