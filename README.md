# Can a quantum kernel catch cyberattacks better than a classical one?

Hospitals store a lot of sensitive patient data, so attacks on their networks are a real risk. For this challenge I tested whether a quantum kernel can help detect those attacks compared with a standard classical approach. The short answer: it performs about the same, and I explain below why I think that is still a useful result.

## The problem
An intrusion detection system has two jobs. It should catch as many attacks as possible (recall), and it should not raise false alarms all the time (false-alarm rate), because security staff stop trusting a system that gives too many false alarms. I wanted to see if swapping one part of a classical model for a quantum one changes either of those.

## The data
I used NSL-KDD (the KDDTrain+ file), a well-known public benchmark for network intrusion detection. It has 125,973 network connections, each described by 41 features and labeled either "normal" or as a specific attack. I grouped all attack types into one "attack" class.

How I prepared the data (on both models):
- Removed the 3 text columns, which left 38 numeric features.
- Took a balanced sample of 300 normal and 300 attack records.
- Split it 70% training and 30% testing.
- Standardized the features, reduced them to 4 with PCA, and rescaled them to the range
  0 to pi (the quantum circuit needs that).
- All of this was fitted on the training data only, so nothing leaks from the test set.

## The classical baseline
A support vector machine (SVM) with a standard RBF kernel, using the 4 PCA features. I tuned it with 3 fold cross-validation over C and gamma, optimizing the F1 score.

## The quantum version
I kept the exact same SVM and only replaced its kernel. The quantum kernel works like this:
each data point is encoded into a 4-qubit circuit (an IQP-style feature map from PennyLane), and the similarity between two points is the overlap between their quantum states. This puts the data in a 16-dimensional quantum space, which gives a different idea of "similar" than the RBF kernel does. The idea is that this might separate attacks from normal traffic better. I ran it on a simulator, and used the same cross-validation and the same C values as the baseline.

## Results
I repeated the experiment with 3 random seeds, so the numbers below are the mean ± standard deviation.

| Metric | Classical RBF SVM | Quantum kernel SVM |
|---|---|---|
| Accuracy | 0.933 ± 0.019 | 0.935 ± 0.026 |
| Precision | 0.961 ± 0.006 | 0.968 ± 0.015 |
| Recall | 0.904 ± 0.042 | 0.900 ± 0.040 |
| F1 | 0.931 ± 0.022 | 0.933 ± 0.028 |
| ROC-AUC | 0.976 ± 0.018 | 0.982 ± 0.020 |
| False-alarm rate | 0.037 ± 0.006 | 0.030 ± 0.013 |
| Training time (s) | 0.32 | 2.52 |

Quantum resources used:
- 4 qubits
- Circuit depth of 1 as reported by PennyLane for the feature-map block
- 0 trainable circuit parameters (the circuit is fixed, only the SVM is trained)
- 163,590 kernel entries that would need to be computed on real quantum hardware

The raw results are in `results_raw.csv` and `summary.csv`.

## What I take from this
- The two models are very close. The quantum kernel is slightly ahead on accuracy, precision, F1, ROC-AUC and false-alarm rate, and slightly behind on recall. But each of those gaps is smaller than the variation between seeds, so I can't honestly say the quantum model is better. What I can say is that it matches the classical baseline.

- The most interesting hint is the lower false-alarm rate (0.030 vs 0.037). In a hospital that would mean fewer pointless alerts, and it came without losing accuracy. But with only 3 seeds and overlapping error bars, I see it as something worth testing further, not as an advantage.

- The cost is real. Training took about 8 times longer on the simulator, and the quantum kernel needs a circuit run for every pair of data points, so it grows quickly with more data. Also, the circuit has no trainable parameters, so any benefit comes from the kernel itself and not from a bigger model.

## Limitations
- Everything ran on a simulator, with no hardware noise. Real quantum devices would also need many repeated measurements per kernel entry.
- I only used 600 samples and 4 features, because simulating more qubits gets slow.
- NSL-KDD is not hospital data.
- NSL-KDD is general network traffic, so I used it as a stand-in for a hospital network. Results on real hospital traffic could differ.
- A big share of the attack records are one attack type (neptune, a denial-of-service attack), which is easier to detect than a realistic mix.
- The test set is balanced 50/50, which is not how real traffic looks.
- I did not tune the quantum feature map. A different one could change the results.

## References
## References
- Quantum Computing Initiative, "Introduction to Practical Quantum Computing" , Section 5.4 on quantum feature maps and SVMs.
- Tavallaee et al., "A Detailed Analysis of the KDD CUP 99 Data Set", 2009 (NSL-KDD dataset).
- Havlicek et al., "Supervised learning with quantum-enhanced feature spaces", Nature, 2019.
- PennyLane: https://pennylane.ai
