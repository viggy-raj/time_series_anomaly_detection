# Explainable Deep Learning for Multivariate Time-Series Anomaly Detection Using an LSTM Autoencoder and Integrated Gradients

**[Student Name]**  
[Department / College Name]  
[City, India]  
[Email]

---

## Abstract

Anomaly detection in multivariate time-series data is a critical challenge in real-world monitoring systems, where early identification of abnormal behaviour can prevent service disruptions and cascading system failures. This paper presents a reproducible deep-learning pipeline for anomaly detection on the Pooled Server Metrics (PSM) public dataset, which comprises 25-dimensional server telemetry recorded over time. An encoder–decoder Long Short-Term Memory (LSTM) autoencoder is trained exclusively on normal data; at inference time, windows whose mean-squared reconstruction error exceeds the 99th-percentile training threshold are classified as anomalous. To address the opacity of the detection decision, Integrated Gradients (IG)—a path-based attribution method satisfying formal axiomatic properties—is applied to quantify the contribution of each input feature to the anomaly score. Gradient saliency maps further highlight temporal regions within each flagged window that elevated the reconstruction error. All code, preprocessing steps, hyperparameters, and evaluation scripts are made available, and the complete pipeline can be reproduced with a single command. Preliminary results from a five-epoch smoke test yield an AUROC of 0.7325 and an F1 score of 0.3769; these figures are reported as initial indicators only. A full 50-epoch training run with formal baseline comparisons is required before substantive performance claims can be made. This work serves as a documented, reproducible foundation for college-level research in explainable anomaly detection on public time-series benchmarks.

**Keywords:** multivariate time-series anomaly detection, LSTM autoencoder, reconstruction error, Integrated Gradients, explainability, PSM dataset, deep learning.

---

## I. Introduction

The proliferation of interconnected monitoring systems—spanning server infrastructure, industrial control systems, and IoT networks—has made automated anomaly detection an essential component of modern operations management [11]. These systems continuously produce high-dimensional time-series data, where anomalies often manifest as subtle, correlated deviations across multiple channels simultaneously, making manual inspection impractical at scale.

Deep learning approaches to anomaly detection have demonstrated considerable promise in recent years [11]. Among them, autoencoder-based methods have gained particular traction because they can be trained on unlabelled normal data without requiring labelled examples of anomalies [8]. The core intuition is that a model trained exclusively on normal data will fail to accurately reconstruct anomalous inputs, yielding a high reconstruction error that serves as an anomaly signal.

Despite their effectiveness, deep anomaly detection models are frequently criticised for their lack of interpretability [3], [10]. In operational settings, an alert that flags a time window as anomalous without indicating which features or timesteps drove the decision provides limited actionable information to system operators. Explainable artificial intelligence (XAI) methods can bridge this gap by attributing predictions to specific input dimensions, thereby supporting human reasoning and diagnostics.

This paper describes the design and preliminary evaluation of a reproducible pipeline for explainable multivariate time-series anomaly detection. The specific contributions are as follows:

1. A PyTorch implementation of an encoder–decoder LSTM autoencoder trained on the PSM dataset in a normal-data-only setting.
2. A threshold-based detection rule defined at the 99th percentile of training reconstruction errors.
3. Integration of Integrated Gradients [3] for per-feature attribution of anomaly scores, and gradient saliency maps [7] for temporal attribution within flagged windows.
4. A self-contained, reproducible codebase with documented commands for full pipeline re-execution.
5. Preliminary quantitative results and an explicit, honest account of their limitations.

The remainder of this paper is organised as follows. Section II reviews related work. Section III describes the methodology. Section IV details the experimental setup. Sections V and VI present preliminary results and explainability findings. Section VII discusses limitations and threats to validity. Section VIII concludes with directions for future work.

---

## II. Related Work

### A. LSTM-Based Anomaly Detection

Long Short-Term Memory networks [5] were among the first deep architectures applied to time-series anomaly detection. Malhotra et al. [6] demonstrated that prediction errors from multi-layer LSTMs trained on normal sequences could serve as anomaly scores, establishing the foundational approach later extended by reconstruction-error methods. Hundman et al. [1] refined this paradigm by pairing LSTM prediction models with a non-parametric dynamic thresholding scheme for spacecraft telemetry, highlighting that fixed-percentile thresholds can be insufficient for non-stationary signals—a limitation directly relevant to the present work.

Autoencoder architectures offer an alternative framing: instead of predicting future values, they compress and then reconstruct the input window, with high reconstruction error indicating a potential anomaly. This formulation has been widely adopted [8], [11] because it supports training on unlabelled normal data without requiring manually annotated anomaly examples.

### B. Multivariate Anomaly Detection

OmniAnomaly [2] introduced stochastic recurrent networks with variational inference to capture both deterministic and stochastic temporal dependencies in multivariate time series. USAD [8] proposed dual adversarially-trained autoencoders to amplify reconstruction errors for anomalous inputs while suppressing them for normal inputs. Abdulaal et al. [4] introduced the PSM dataset alongside a synchronisation-aware model, providing a standardised public benchmark with ground-truth anomaly labels.

Schmidl et al. [9] conducted a large-scale comparative evaluation of anomaly detection algorithms across multiple benchmark datasets and evaluation protocols. Their findings document the substantial sensitivity of reported performance metrics to threshold selection, window labelling conventions, and post-processing steps such as point-adjustment, underscoring the importance of transparent experimental reporting—an objective this paper attempts to uphold.

### C. Explainable AI for Time-Series

Attribution methods for deep neural networks have been developed primarily in the computer vision and NLP domains [10]. Simonyan et al. [7] pioneered gradient-based saliency maps, computing the gradient of the model output with respect to the input to identify influential regions. Lundberg and Lee [10] introduced SHAP (SHapley Additive exPlanations), unifying several prior attribution methods under the game-theoretic framework of Shapley values.

Sundararajan et al. [3] proposed Integrated Gradients (IG), which integrates the gradient of the model output along a linear path from a baseline input to the actual input. IG satisfies two formal axioms—*sensitivity* and *implementation invariance*—that are not guaranteed by simple gradient methods. For time-series models with scalar outputs, IG provides a natural mechanism to attribute the anomaly score to individual (timestep, feature) pairs, making it well-suited to this task.

---

## III. Methodology

### A. Problem Formulation

Let **X** ∈ ℝ^{T×F} denote a multivariate time-series window of T timesteps and F features. A dataset of N training windows {**X**₁, …, **X**_N}, drawn from normal operating conditions, is used to fit a model M. For each test window **X**, the model produces a scalar anomaly score E(**X**). A window is classified as anomalous when:

```
ŷ = 1   if E(X) > τ
ŷ = 0   otherwise
```

where τ is a threshold determined from the training error distribution.

### B. LSTM Autoencoder Architecture

The detection model is an encoder–decoder LSTM autoencoder implemented in PyTorch. The encoder maps the input window to a fixed-dimensional latent representation; the decoder reconstructs the original window from this representation.

**Encoder:**
- Layer 1: LSTM (input dim F → hidden dim H), two stacked layers, processes all T timesteps and returns hidden states.
- Layer 2: LSTM (H → latent dim L), two stacked layers, extracts the latent vector from the final timestep.

**Decoder:**
- Layer 3: LSTM (L → H), two stacked layers, expands the latent vector to H dimensions over T timesteps.
- Layer 4: LSTM (H → H), two stacked layers, processes the expanded sequence.
- Output projection: Linear (H → F), applied at each timestep to recover input dimensionality.

In the present implementation: T = 60, F = 25, H = 64, L = 32, yielding approximately 162,393 trainable parameters.

**[Figure 2 Placeholder — Architecture Diagram]**
*Textual description: Input sequence (T=60, F=25) → Encoder LSTM (64 hidden) → Encoder LSTM (32 latent) → Bottleneck representation → Decoder LSTM (64 hidden) → Decoder LSTM (64 hidden) → Linear projection → Reconstructed sequence (T=60, F=25). A block-diagram figure should be inserted in the final version.*

### C. Reconstruction Error and Anomaly Scoring

The anomaly score for window **X** with reconstruction **X̂** produced by the autoencoder is the mean-squared error (MSE) over all timesteps and features:

```
E(X) = (1 / (T × F)) × Σ_{t=1}^{T} Σ_{f=1}^{F} (x_{t,f} − x̂_{t,f})²
```

This scalar score is computed for every test window. The anomaly threshold τ is set at the 99th percentile of E values computed over all training windows after training:

```
τ = Percentile( { E(Xᵢ) : i = 1, …, N_train }, 99 )
```

This choice reflects a principled, data-driven cutoff: under ideal training conditions, normal windows should produce reconstruction errors below this level, while anomalous windows—whose patterns were never seen during training—should exceed it.

### D. Integrated Gradients for Feature Attribution

To explain why a specific window receives a high anomaly score, Integrated Gradients (IG) [3] is applied. Let F_θ(**X**) = E(**X**) denote the differentiable scalar anomaly scoring function (reconstruction error implemented as a differentiable PyTorch wrapper around the autoencoder). IG computes the attribution of each input element (timestep t, feature f) as:

```
IG_{t,f}(X) = (x_{t,f} − x'_{t,f}) × (1/K) × Σ_{k=1}^{K} [∂F_θ(X' + (k/K)(X − X')) / ∂x_{t,f}]
```

where **X'** is the baseline input (set to the all-zeros vector, representing the absence of signal), K = 50 is the number of Riemann integration steps, and the summation approximates the path integral from baseline to input. The per-feature importance score is then obtained by averaging the absolute IG values over all T timesteps:

```
Imp(f) = (1/T) × Σ_{t=1}^{T} |IG_{t,f}(X)|
```

**Implementation note:** The explainability module resides in `explainability/shap_explainer.py`. Despite the filename, the implemented method is Integrated Gradients—not SHAP. SHAP's `GradientExplainer` was originally attempted but produced a tensor-dimension incompatibility with the LSTM model's scalar output. Integrated Gradients was selected as a principled alternative that satisfies the same axiomatic properties of completeness and sensitivity [3].

### E. Gradient Saliency Maps

Gradient saliency maps [7] provide complementary temporal attribution. For a given window **X**, the saliency at each (timestep, feature) position is:

```
S_{t,f} = |∂E(X) / ∂x_{t,f}|
```

This quantity identifies which (timestep, feature) pairs are most sensitive to small perturbations, highlighting temporal sub-regions within a flagged window that most strongly influence the reconstruction error. Saliency maps are visualised as heatmaps with features on the y-axis and timesteps on the x-axis.

### F. Pipeline Overview

The full pipeline is orchestrated by `main.py` and executes sequentially through seven stages:

1. **Download**: PSM training and test CSV files are retrieved from the `eBay/RANSynCoders` GitHub repository using a streaming HTTP request.
2. **EDA**: Exploratory visualisations—sample time-series plots, label distributions, feature correlation heatmap, and feature histograms—are generated and saved.
3. **Preprocessing**: Min-Max normalisation (scaler fitted on training data only), followed by sliding-window segmentation (T = 60, stride = 1). The training set is further split 90/10 for training and validation.
4. **Training**: LSTM autoencoder trained on normal training windows using the Adam optimiser with early stopping and model checkpointing.
5. **Evaluation**: Reconstruction errors computed for all test windows; precision, recall, F1 score, AUROC, and average precision evaluated against ground-truth window labels.
6. **Explainability**: IG and gradient saliency maps generated for the five test windows with the highest reconstruction errors.
7. **Reporting**: Summary report, metrics JSON/CSV, and all plots saved to `outputs/`.

---

## IV. Experimental Setup

### A. Dataset

The Pooled Server Metrics (PSM) dataset [4] is a publicly available multivariate time-series benchmark collected from server infrastructure. It contains 25 server performance metrics sampled at one-minute intervals. Ground-truth anomaly labels are provided for the test split only, so the training regime is effectively normal-data-only—consistent with a semi-supervised anomaly detection setting.

**Table I: PSM Dataset Summary**

| Split | Timesteps | Features | Anomalous Timesteps | Anomaly Rate |
|-------|-----------|----------|---------------------|--------------|
| Train | 132,481   | 25       | Not applicable      | Not applicable |
| Test  | 87,841    | 25       | 24,381              | 27.8%        |

*Note: Training data consists exclusively of normal timesteps; anomaly labels are not provided or used during training.*

**[Figure 1 Placeholder]**
*Caption: Four representative PSM features over time, with the train split shown on the left and the test split on the right. Anomalous test regions are shaded. See `outputs/plots/eda/sample_timeseries.png`.*

### B. Exploratory Data Analysis

Initial exploration revealed 4,195 null values in the training set, addressed via forward-fill imputation during preprocessing, and zero null values in the test set. Feature distributions are approximately unimodal for most channels, though several exhibit right-skewed tails. The correlation heatmap suggests moderate inter-feature correlation, consistent with the expectation that server performance metrics share common underlying drivers (e.g., load spikes affecting CPU, memory, and I/O concurrently). The anomaly prevalence in the test set—27.8%—is relatively high for an anomaly detection benchmark and is likely attributable to extended incident episodes that span many consecutive timesteps.

### C. Preprocessing

Raw CSV files are loaded and the timestamp column is discarded. Min-Max normalisation is applied column-wise, with the scaler fitted exclusively on the training split to prevent data leakage. Sliding windows of length T = 60 with stride 1 are constructed, yielding 119,180 training windows (of which 13,242 are reserved for validation), and 87,782 test windows. A test window is labelled as anomalous if any constituent timestep carries a positive ground-truth label; this window labelling rule is implemented consistently throughout evaluation.

### D. Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Window size (T) | 60 timesteps |
| Stride | 1 |
| Hidden size (H) | 64 |
| Latent size (L) | 32 |
| Optimiser | Adam |
| Learning rate | 0.001 |
| Weight decay | 1×10⁻⁵ |
| Batch size | 64 |
| Maximum epochs | 50 (5 in smoke test) |
| Early stopping patience | 10 epochs |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Anomaly threshold | 99th percentile of training errors |
| Random seed | 42 |

All random seeds (Python built-in, NumPy, PyTorch) were fixed to 42. The model was trained on CPU. A GPU can be selected via `python main.py --device cuda`.

---

## V. Results and Discussion

> **Important caveat:** All quantitative results reported in this section derive from a five-epoch smoke test and must be treated as preliminary indicators only. The model loss showed a decreasing trend during this run, suggesting that continued training may improve detection performance. A complete 50-epoch training run with early stopping, followed by formal comparison with published baselines, is required before any substantive or generalisable performance claims can be made.

### A. Training Dynamics

**[Figure 3 Placeholder]**
*Caption: Training loss and validation loss curves over 5 epochs. Both curves decrease monotonically, and no divergence between training and validation loss is visible, suggesting the model had not begun to overfit at this stage. See `outputs/plots/train_loss.png`.*

After 5 epochs, the training loss reached 0.001287 and the best recorded validation loss was 0.001194. The close tracking of validation loss relative to training loss and the absence of divergence indicate that the model was still in an improving phase and had not saturated. These observations are consistent with the expectation that additional training epochs will further reduce reconstruction error on normal windows and sharpen the separation between normal and anomalous errors.

### B. Anomaly Detection Metrics

**Table II: Preliminary Detection Results — 5-Epoch Smoke Test Only**

| Precision | Recall | F1 Score | AUROC  | Avg. Precision | Threshold (τ) |
|-----------|--------|----------|--------|----------------|---------------|
| 0.3993    | 0.3569 | 0.3769   | 0.7325 | 0.4614         | 0.006464      |

*These results are preliminary and should not be interpreted as the final performance of the proposed approach. See Section VII for a comprehensive list of limitations.*

**[Figure 4 Placeholder]**
*Caption: Reconstruction error over test timesteps (converted to per-window mean), with the anomaly threshold τ = 0.006464 marked as a horizontal line and ground-truth anomaly regions shaded in red. See `outputs/plots/anomaly_overlay.png`.*

**[Figure 5 Placeholder]**
*Caption: ROC curve for the preliminary 5-epoch model. The shaded area corresponds to a preliminary AUROC of 0.7325. See `outputs/plots/roc_curve.png`.*

**[Figure 6 Placeholder]**
*Caption: Precision-recall curve for the preliminary 5-epoch model. The area under the curve (average precision) is 0.4614. See `outputs/plots/pr_curve.png`.*

### C. Discussion

The preliminary AUROC of 0.7325 indicates that the reconstruction-error signal provides a non-trivial ordering of normal versus anomalous windows even after very limited training. AUROC is threshold-free and therefore less sensitive to the specific choice of the 99th-percentile cutoff than the precision, recall, and F1 values; its preliminary level suggests that the model has already begun to learn a reconstruction profile that separates the two classes.

The F1 score of 0.3769 at the fixed threshold reflects the combined effects of incomplete training, the particular threshold choice, and the inherent difficulty of the PSM benchmark. The relatively low recall (0.3569) is consistent with a conservative threshold that suppresses false positives at the cost of missing a substantial fraction of anomalous windows—a pattern expected when the anomaly decision boundary has not yet been well-calibrated through sufficient training.

As a minimal sanity check, a naive classifier that labels every window as anomalous would achieve recall of 1.0 and precision of approximately 0.278 (the base rate). The preliminary precision of 0.3993 exceeds this trivial baseline, but this observation alone is not sufficient to claim meaningful anomaly discrimination at this stage of training.

It is also important to note that reported metrics across published PSM studies vary considerably depending on evaluation protocol, window labelling rules, and the application of point-adjustment procedures [9]. Direct comparison with numbers from prior work is therefore not possible without replicating those protocols precisely, which is left to future work.

---

## VI. Explainability Analysis

### A. Integrated Gradients Feature Attribution

IG was applied to the five test windows with the highest reconstruction errors—the windows for which the model exhibited the strongest anomaly signal. For each window, a 50-step Riemann approximation of the path integral was computed from the zero-vector baseline, and the mean absolute attribution was aggregated over all T timesteps to yield a per-feature importance score.

**Table III: Top Five IG Feature Attributions (Aggregated over Five Highest-Error Anomaly Windows)**

| Rank | Feature    | Mean Absolute Attribution |
|------|------------|--------------------------|
| 1    | feature_3  | 0.001128                 |
| 2    | feature_1  | 0.000342                 |
| 3    | feature_0  | 0.000253                 |
| 4    | feature_11 | 0.000064                 |
| 5    | feature_2  | 0.000054                 |

**[Figure 7 Placeholder]**
*Caption: Global IG feature importance summary. Horizontal bars show the mean absolute IG attribution for all 25 PSM features across the five selected anomaly windows, sorted in descending order. feature_3 is the dominant contributor by a substantial margin. See `outputs/plots/ig_summary.png`.*

**Interpretation:** In the analyzed sample, feature_3 received an attribution score approximately 3.3 times that of feature_1, the next-highest contributor. Features 1, 0, 11, and 2 received successively lower attributions. The remaining 20 features contributed relatively little to the anomaly score in these specific instances. This initial result suggests that the anomaly signal is concentrated in a small subset of features rather than distributed uniformly across all 25 channels.

**Important caveat:** These attributions reflect the gradient-based sensitivity of the trained model to each feature in the specific windows analyzed. They indicate which inputs the model relied upon most in computing its reconstruction error—they do not constitute verified causal root-cause identifications. Without ground-truth causal annotations for the selected anomaly events, stronger interpretive claims cannot be supported. Furthermore, the IG values depend on the current model state, which is preliminary after only five epochs; attribution rankings may shift as the model reaches convergence.

### B. Temporal Saliency Maps

**[Figure 8 Placeholder]**
*Caption: IG temporal attribution heatmap (features × timesteps) averaged across the five analyzed anomaly windows. Colour intensity indicates the mean signed IG attribution; warmer colours correspond to features and timesteps that contributed positively to the anomaly score. See `outputs/plots/ig_heatmap.png`.*

**[Figure 9 Placeholder]**
*Caption: Gradient saliency heatmap for anomaly window #1. Each cell (timestep, feature) shows the absolute gradient of the reconstruction error with respect to that input position. Brighter regions indicate greater local sensitivity. See `outputs/plots/saliency_01.png`.*

The gradient saliency maps reveal that elevated saliency values are concentrated in specific temporal sub-regions within the 60-timestep windows rather than distributed uniformly. This qualitative observation is consistent with the interpretation that anomalies in this dataset manifest as localised deviations rather than sustained global shifts, though this hypothesis requires validation against ground-truth anomaly onset timestamps, which are not available at the window level in the PSM release.

### C. Comparison of IG and Saliency

IG and gradient saliency are related but distinct attribution methods. Gradient saliency reflects the local gradient at the actual input point, while IG integrates gradients along the entire interpolated path from baseline to input, producing attributions that account for the cumulative contribution of each feature rather than only its instantaneous local sensitivity. In practice, IG tends to produce more stable attributions less susceptible to gradient saturation at extreme input values [3]. Both methods are gradient-based and therefore sensitive to the model's current learned representations; their reliability and consistency will improve as training proceeds toward convergence.

---

## VII. Limitations and Threats to Validity

The following limitations must be addressed before this work can constitute a finalized research study.

**Table IV: Limitations and Required Extensions**

| Limitation | Description |
|------------|-------------|
| Insufficient training duration | The model was trained for 5 epochs only (smoke test). Final metrics require the full 50-epoch run with early stopping. |
| No published baseline comparisons | No established anomaly detection methods were re-implemented under matched preprocessing and evaluation conditions for direct comparison. |
| Fixed threshold, no sensitivity analysis | Only the 99th-percentile threshold was assessed. Sensitivity of precision, recall, and F1 to the threshold choice was not studied. |
| Window labelling rule not fully justified | The rule for assigning a binary anomaly label to a window from timestep-level labels should be explicitly stated and its effect on metrics examined. |
| No point-adjustment evaluation | Point-adjustment scoring, which is used in some published PSM results, was not applied or discussed, making cross-study comparison difficult. |
| Explanation faithfulness not evaluated | No feature-deletion or perturbation experiments were conducted to verify that features with high IG scores produce measurable changes in the anomaly score when removed or modified. |
| Window size and hyperparameter sensitivity not studied | The window length T = 60 and architecture hyperparameters were not tuned; their effect on detection and explanation quality is unknown. |
| Single dataset | Results are limited to PSM. Generalisability to other benchmarks (e.g., SMAP, MSL, SWaT) has not been assessed. |
| Anonymised feature names | PSM features are labelled feature_0 through feature_24. Without semantic annotations, IG attributions cannot be translated into operationally meaningful domain terms. |
| No causal claims | IG identifies which inputs the model is sensitive to, not verified root causes of anomalies. Causal claims require controlled experiments or ground-truth causal annotations. |
| No user study | No human evaluation of explanation usefulness or actionability was conducted. |

---

## VIII. Conclusion and Future Work

This paper presented a reproducible, fully documented pipeline for explainable multivariate time-series anomaly detection using an encoder–decoder LSTM autoencoder on the publicly available PSM dataset. The model was trained exclusively on normal data in a semi-supervised setting and evaluated on a labelled test set using reconstruction-error-based anomaly scoring. Integrated Gradients and gradient saliency maps were applied to provide feature-level and temporal attribution of individual anomaly predictions, implemented without reliance on SHAP following an identified tensor-dimension incompatibility with the LSTM architecture.

Preliminary results from a five-epoch smoke test yielded a preliminary AUROC of 0.7325 and a preliminary F1 score of 0.3769 at a 99th-percentile threshold. The IG analysis identified feature_3 as the most influential input across the five highest-error anomaly windows, with features 1, 0, 11, and 2 contributing at successively lower levels. These results are reported honestly as initial, preliminary findings, not as final performance claims.

**Future work** should address the following directions:

1. **Full training run**: Execute the complete 50-epoch pipeline with early stopping and report final metrics under identical conditions to the smoke test.
2. **Baseline comparisons**: Reproduce at least one established anomaly detection method (e.g., OmniAnomaly [2] or USAD [8]) under matched preprocessing and evaluation conditions.
3. **Threshold sensitivity analysis**: Evaluate precision, recall, and F1 across a range of threshold percentiles and consider data-driven threshold selection methods [1].
4. **Point-adjustment and event-based evaluation**: Implement and explicitly justify point-adjustment scoring rules to facilitate comparison with published results.
5. **Quantitative explanation evaluation**: Apply feature-removal tests and perturbation studies to assess the faithfulness of IG attributions.
6. **Multi-dataset evaluation**: Extend the pipeline to SMAP, MSL, and SWaT benchmarks to assess generalisability.
7. **Semantic feature annotation**: Investigate whether anonymised PSM features can be associated with known server metric semantics to enable operationally meaningful explanation.
8. **Human evaluation**: Conduct a user study with system operators to assess whether IG and saliency explanations are actionable in practice.

---

## References

[1] K. Hundman, V. Ladd, C. Carter, T. Faber, and I. Hodge, "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding," in *Proc. 24th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, London, UK, 2018. [VERIFY: complete author list, exact title spelling, page numbers]

[2] Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, and D. Pei, "Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network," in *Proc. 25th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, Anchorage, AK, USA, 2019. [VERIFY: page numbers]

[3] M. Sundararajan, A. Taly, and Q. Yan, "Axiomatic Attribution for Deep Networks," in *Proc. 34th Int. Conf. Machine Learning (ICML)*, Sydney, Australia, vol. 70, pp. 3319–3328, 2017.

[4] A. Abdulaal, Z. Liu, and T. Lancewicki, "Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization," in *Proc. 27th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, Virtual, 2021. [VERIFY: full title and page numbers]

[5] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, Nov. 1997.

[6] P. Malhotra, L. Vig, G. Shroff, and P. Agarwal, "Long Short Term Memory Networks for Anomaly Detection in Time Series," in *Proc. 23rd European Symp. Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN)*, Bruges, Belgium, 2015. [VERIFY: exact page numbers]

[7] K. Simonyan, A. Vedaldi, and A. Zisserman, "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps," in *Proc. Int. Conf. Learning Representations Workshop (ICLR Workshop)*, Banff, Canada, 2014.

[8] J. Audibert, P. Michiardi, F. Guyard, S. Marti, and M. A. Zuluaga, "USAD: UnSupervised Anomaly Detection on Multivariate Time Series," in *Proc. 26th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, Virtual, 2020. [VERIFY: exact page numbers]

[9] S. Schmidl, P. Wenig, and T. Papenbrock, "Anomaly Detection in Time Series: A Comprehensive Evaluation," *Proc. VLDB Endow.*, vol. 15, no. 9, pp. 1779–1797, 2022. [VERIFY: volume, issue, page numbers]

[10] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Proc. 31st Conf. Neural Information Processing Systems (NeurIPS)*, Long Beach, CA, USA, 2017.

[11] G. Pang, C. Shen, L. Cao, and A. V. D. Hengel, "Deep Learning for Anomaly Detection: A Review," *ACM Comput. Surv.*, vol. 54, no. 2, Art. 38, Mar. 2022. [VERIFY: article number and exact year]

---

## Appendix A: Reproducibility Checklist

| Step | Command |
|------|---------|
| Install all dependencies | `pip install -r requirements.txt` |
| Download PSM data and run full pipeline | `python main.py` |
| Run without re-downloading data | `python main.py --skip-download` |
| Evaluate using a previously saved checkpoint | `python main.py --skip-download --skip-train` |
| Use GPU acceleration (if available) | `python main.py --device cuda` |

All random seeds (Python built-in, NumPy, PyTorch) are fixed to 42 via `config.py` to ensure deterministic behaviour across runs. Output artifacts are saved to `outputs/` and include the model checkpoint (`model.pth`), evaluation metrics (`metrics.json`, `metrics.csv`), IG feature importances (`shap_feature_importance.json`), all evaluation and explainability plots (PNG), and a human-readable summary report (`reports/summary_report.txt`).

---

## B. Items Required Before Submission

The following items must be completed or verified before this draft is submitted for evaluation.

1. **Complete the 50-epoch training run**: Execute `python main.py --skip-download` and replace all preliminary metrics in Tables II and III with the final values obtained after full training.

2. **Baseline comparison**: Re-implement or re-run at least one published baseline method (e.g., USAD [8] or OmniAnomaly [2]) under matched preprocessing conditions. Add results to a new comparison table (see Results Update Template below).

3. **Verify all [VERIFY]-marked references**: Check author names, exact titles, page numbers, and year against the official ACM DL, IEEE Xplore, and VLDB records for references [1], [2], [4], [6], [8], [9], and [11].

4. **Insert actual figure files**: Replace all nine figure placeholders with the generated PNG files from `outputs/plots/`. Ensure figures are 300 DPI or higher for print submission.

5. **Create Figure 2 (architecture diagram)**: Prepare a proper block diagram of the LSTM Autoencoder encoder-decoder structure showing input/output dimensions at each layer (T=60, F=25, H=64, L=32).

6. **Clarify and justify window labelling rule**: Formally state and justify the rule used to assign a binary label to each window from timestep-level labels (e.g., "a window is labelled anomalous if any of its constituent timesteps is labelled anomalous in the ground truth"). Discuss sensitivity to this choice.

7. **Point-adjustment decision**: Decide whether point-adjustment will be applied to align with published protocols. Document and justify the decision explicitly in Section V.

8. **Hyperparameter verification**: Cross-check all values in the training configuration table against the current contents of `config.py` and `models/trainer.py` to confirm accuracy.

9. **Plagiarism check**: Run the submitted draft through an institutional plagiarism detection tool (Turnitin, iThenticate, or equivalent) before final submission.

10. **IEEE template conversion**: Convert this Markdown draft to IEEE two-column conference format using the official IEEE LaTeX template (`IEEEtran`) or the Microsoft Word template. Verify compliance with the target venue's page limit and formatting guidelines.

11. **Author details**: Insert the real student name, department, institution, city, and email address in place of all placeholder text in the author block.

12. **Abstract word count**: Confirm the abstract word count falls within the target venue's specified limits (typically 150–250 words for IEEE conference papers).

---

## C. Results Update Template

*Once the 50-epoch training run is complete and any baseline experiments have been conducted, copy the template below, fill in the blanks, and use the values to replace the preliminary entries in Tables II and III in the paper draft.*

```
=== FINAL 50-EPOCH RESULTS (PSM Dataset) ===

Training:
  Final Train Loss (Epoch 50 or last)  : ___________
  Best Validation Loss                 : ___________
  Epochs Trained (early stopped at)    : ___ of 50
  Total Training Time                  : ___________ seconds

Detection Metrics (Test Set, 99th-Percentile Threshold):
  Threshold (τ)                        : ___________
  Precision                            : ___________
  Recall                               : ___________
  F1 Score                             : ___________
  AUROC                                : ___________
  Average Precision                    : ___________
  True Anomaly Windows                 : ___________ / 87,782
  Predicted Anomaly Windows            : ___________ / 87,782

=== BASELINE COMPARISON (insert if conducted) ===

Method                | Precision | Recall | F1     | AUROC
----------------------|-----------|--------|--------|-------
LSTM-AE (Ours, 50ep)  | ______    | ______  | ______ | ______
[Baseline 1 name]     | ______    | ______  | ______ | ______
[Baseline 2 name]     | ______    | ______  | ______ | ______

Note: Confirm all methods use identical preprocessing and evaluation
protocols before inserting values into Table II.

=== FINAL IG ATTRIBUTIONS (from 50-epoch model) ===

Rank | Feature     | Mean |IG| Attribution
-----|-------------|---------------------
1    | ___________  | ___________
2    | ___________  | ___________
3    | ___________  | ___________
4    | ___________  | ___________
5    | ___________  | ___________
```
