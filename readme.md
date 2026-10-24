# ViterbiNet: Deep Learning Meets Viterbi Algorithm - A New Paradigm for Intelligent Symbolic Detection in Time-Varying Channels [with Python code]

ViterbiNet: Deep Learning Meets Viterbi Algorithm - A New Paradigm for Intelligent Symbol Detection in Time-Varying Channels
Editor's note: In traditional communication systems, the performance of symbol detection highly depends on precise channel state information (CSI), and acquiring CSI in real-time and accurately is often costly, or nearly impossible under rapidly varying channels. Scholars from institutions such as the Technion and Stanford University have proposed a novel algorithm called ViterbiNet, which integrates deep neural networks into the classic Viterbi decoding process to perform symbol detection purely data-driven, achieving reliable communication without CSI. This article systematically reviews the theoretical motivation, system model, core design, and experimental validation of this algorithm based on its publicly available research findings, serving as a reference for researchers in the fields of communication and machine learning. Symbol detection is a core task at the receiver end of wireless communication. The classical maximum likelihood sequence detection (MLSD) relies on precise channel state information, but in rapidly varying or complex channels, channel estimation errors can significantly degrade detection performance. This article provides a comprehensive introduction to ViterbiNet—a data-driven symbol detection scheme that incorporates deep neural networks (DNN) into the Viterbi algorithm. The core idea is to use the DNN to directly learn the path metrics (branch costs) of each branch from the received signal without explicitly estimating the channel, then employing the "Add-Compare-Select" (ACS) structure of the Viterbi algorithm to complete the sequence search; meanwhile, leveraging an online training mechanism allows the network to track the time-varying channel in blocks. Simulations on additive Gaussian white noise (AWGN) channels with intersymbol interference (ISI) and block fading channels demonstrate that this scheme can achieve comparable bit error rate performance to traditional coherent detection under conditions without CSI, and possesses dynamic adaptability to channel time variations. This article also presents a simplified implementation of the algorithm and analyzes the differences between the original and simplified versions, proving through experiments that superior bit error rate results can be achieved under equivalent simulation settings.
Keywords: Deep Learning; Viterbi Algorithm; Symbol Detection; Channel State Information; Online Learning; Bit Error Rate
1. Introduction
1.1 Research Background: The "Dilemma" between Symbol Detection and Channel State Information
In digital communication systems, receivers need to recover the transmitted symbols from signals corrupted by noise and distortion, a process known as symbol detection. For band-limited channels with intersymbol interference (ISI), the optimal detection method is the Longest Likelihood Sequence Detection (MLSD), which achieves global optimal symbol sequence estimation through dynamic programming search over a state grid (trellis) in linear complexity.
However, the "optimality" of the Viterbi algorithm has an important prerequisite: the receiver must precisely grasp the channel's statistical characteristics, that is, channel state information (CSI). In practical systems, CSI is typically obtained through pilot sequence interpolation, feedback, or blind estimation, which not only consumes precious spectrum and power resources but also faces severe challenges in two scenarios:
Complex channel model: When the channel exhibits nonlinearity, memory effects, or features that are difficult to describe analytically, it becomes challenging to establish accurate statistical models, thereby preventing the construction of correct branch metrics;
Fast time-varying channel: When the coherence time of the channel is short, CSI becomes outdated before it can be estimated, making traditional estimation methods difficult to track in real-time.
These two scenarios collectively point to a fundamental contradiction: the more challenging the environment for obtaining CSI, the more precise channel knowledge is needed to complete detection. This contradiction prompts researchers to seek new detection paradigms.
1.2 Motivation for Deep Learning Incorporation in Communication Detection
In recent years, deep learning has achieved great success in fields such as computer vision and natural language processing, with its ability to "learn features and mappings automatically from data" providing new ideas for the design of communication systems. An important observation is that the grid search structure of the Viterbi algorithm is model-agnostic, and what truly needs to be "knowledgeable" about the channel is the path metric calculation corresponding to each transition branch. If a trainable function (such as a deep neural network) can replace the CSI-dependent metric calculation module while retaining the mature sequence search framework of the Viterbi algorithm, then the detector can complete self-learning and adaptation without any prior information about the channel, relying solely on labeled training data.
This is precisely the core idea of ViterbiNet: transforming "algorithm-known but channel-unknown" into "algorithm structure retained, channel knowledge handed over to the network for learning."
1.3 Work and Contributions of This Paper
This paper builds upon the public research achievements of ViterbiNet and its simplified reproduction, covering four aspects: system model, algorithm principles, training mechanism, and experimental verification. It focuses on discussing the following content:
1. Establishing complete transmission and reception models for AWGN channels with ISI and block fading channels;
2. Expounding how ViterbiNet parameterizes the Viterbi branch metrics with deep network parameters and uses online training to track time-varying channels;
3. Providing a simplified network configuration and its comparative analysis with the original scheme;
4. Verifying the effectiveness of the algorithm through bit error rate (BER) experiments.
II. System Model
2.1 Transmitter and Modulation Considering Single-Sender-Single-Receiver (SISO) point-to-point communication link. The transmitter generates a binary bit stream at equal probability and employs BPSK modulation, mapping bits 0 to +1 and bits 1 to -1. BPSK modulation is often used as a benchmark for verifying the performance of detection algorithms due to its simplicity and good noise resistance.
2.2 Channel Models for Transmission
The symbols transmitted through the channel are augmented by noise upon reaching the receiving end. In the simulation, two types of channel models are adopted:
(1) ISI-AWGN Channel: The channel exhibits memory effects, with a length of 4, meaning that the received symbol is the weighted sum of the previous 4 transmitted symbols and is then added with Gaussian white noise. The impulse response of the channel decays exponentially, i.e., the amplitude of the kth tap is proportional to exp(-0.5·k), with the normalized peak tap amplitude set to 0.8. Since the memory length of the channel is 4, the number of grid states corresponding to this is 2^4 = 16, forming a 16-state Viterbi grid.
(2) Block Fading Channel: On top of the ISI-AWGN channel, randomness in time dimension is introduced. The channel coefficients remain constant within each block, while the coefficients between blocks change according to the cosine time decay pattern, with fading cycle parameters set to values such as 51, 39, 33, 21, etc. This model simulates the common slow-varying and fast-varying scenarios found in real wireless channels, also corresponding to the block fading setup used in the verification experiments in Section 5 of the original paper.
Noise power is determined based on a given signal-to-noise ratio (SNR, in dB units): under BPSK modulation, the standard deviation of noise is directly related to SNR; the higher the SNR, the smaller the variance of the additive noise, indicating better channel conditions.
2.3 Transfer Protocol and Data Set Organization
The simulation employs a "Pilot + Data" block sequence transmission protocol: each block consists of a segment of pilot sequences followed by the information data. The pilot part is used for online network training, while the data part evaluates detection performance. The entire transmission spans 100 blocks, each containing 10200 bits, with the pilot length being 200 bits. This "block-by-block training—block-by-block detection" design allows for an intuitive simulation of the receiver's continuous working process under time-varying channels, and facilitates measuring the detector's ability to track channel changes.
Receiver collects the transmitted symbols, received symbols, and channel coefficient tuples from each block to form the dataset required for training and evaluation. To ensure reproducibility of the experiment, a fixed random number seed is used throughout the simulation.
III. Algorithm Design: ViterbiNet3.1 Viterbi algorithm overview The Viterbi algorithm is a dynamic programming method for solving maximum likelihood sequence estimation on grid graphs. It performs the following core operations at each time step: For each state in the grid, calculate the cumulative metric of candidate paths that reach that state, where addition refers to adding the previous state's metric with the current branch's metric, comparison refers to comparing the metrics of candidate paths entering the same state, and selection refers to retaining the path with the minimum cost (least-cost) metric and discarding the others. At the end of the algorithm, backtracking from the optimal state at the last time step yields the global optimal symbol sequence estimate.
It is evident that the Viterbi algorithm itself is a search process independent of structure and channel characteristics; its key input is the metric value of each branch at each time step—which is typically determined by channel statistical properties and requires CSI for precise calculation. This provides an interface for the deep network's involvement.
3.2 Core idea: Using a neural network to learn branch metrics The core contribution of ViterbiNet lies in replacing the traditional branch metric calculation module dependent on CSI in the receiver of the Viterbi algorithm with a deep neural network. Specifically:
1  
The network takes the received symbols at the current time as input, outputting the "prior distribution" of states at that moment across all grids (16 states);
2  
Converts the network output prior distribution into branch metrics, inputs them to the standard ACS (Add-Compare-Select) unit of the Viterbi algorithm;
3  
ACS units propagate the accumulated metrics along the time axis, and during testing, directly decide on detection symbols based on the lowest valid bit of the optimal state.
Since the network parameters are learned through data-driven methods, the detector does not need explicit channel estimation or any statistical assumptions about the channel—all features of the channel are implicitly "encoded" into the network weights.
3.3 Network Architecture Design
The deep network of ViterbiNet adopts a compact fully connected structure: the input layer is 1-dimensional (corresponding to a single received symbol), with the first hidden layer dimension being 75, and after ReLU activation, it connects to the second layer, with an output dimension of 16 (corresponding to 16 grid states). The overall network contains only two linear transformations and one nonlinear activation, with small parameter quantities and lightweight structures, suitable for online block-wise updates.
It is worth noting that this streamlined configuration differs from the three-layer network structure in the original paper: the original scheme used a deeper network with three hidden layers, whereas the implementation described here retains only two transformations. This simplification is not a compromise in performance—experimental results show that the streamlined network achieved better error performance than the original report under the simulation settings. This suggests that in data-driven communication algorithm design, network complexity should match the channel complexity, and overly complex networks may not necessarily yield gains.
3.4 Training Strategy: A core challenge in online learning of time-varying channels is "channels are changing". ViterbiNet employs an online training mechanism to address this challenge:
During each coherent block, multiple rounds of gradient updates are performed using pilot symbols (default 500 rounds);
The network uses cross-entropy as the loss function, and the Adam optimizer is used for parameter updates, with a learning rate set to 5×10⁻³;
After completing the training of this block, the updated network is immediately used to perform detection on the data part;
When entering the next coherent block, the weights at the end of the previous block are used as initial values for training.
This "online training, online detection, and weight sequential transmission" mechanism allows the network to continuously follow the slow drift of channel coefficients, achieving true data-driven adaptive detection. From the perspective of training signals, pilot symbols provide labeled supervision information, while cross-entropy loss encourages the network output state distribution to approximate the true transmitted state.
3.5 Loss Function and Optimization Training Process: The sentinel symbols corresponding state sequences are used as supervision labels. The receiver maps the sentinel sequence into a grid state number (0 to 15), and calculates cross-entropy loss between the network output distribution and the true state distribution. The training process employs standard backpropagation flow: calculating loss, clearing gradients, backpropagation, parameter updates, and iterating until convergence. Due to the small size of the network, even with every block updating 500 rounds, the overall computational overhead remains controllable.
IV. Experimental Setup and Performance Assessment 4.1 Simulation parameters are configured to validate the algorithm's performance, with the following configurations used:
Parameters
Value
Notes
Modulation method
BPSK  
0→+1, 1→-1  
Channel memory length
4  
Number of grid states = 2⁴ = 16
Channel model
ISI-AWGN / block fading
Exponential decay impulse response
SNR  
10 dB  
Verification of Signal-to-Noise Ratio
Coherent block length
10200 bits
Guide 200 + Data 10000
Number of coherent blocks
100  
Online training and detection in blocks
Number of online training rounds
500  
Each Guidance Band
Optimizer / Learning rate
Adam / 5×10⁻³  
—  
Random seed
Fixed
Ensure reproducibility
4.2 Performance Metrics: The algorithm's performance is measured by the Bit Error Rate (BER), which represents the proportion of symbol sequences detected that do not match the actual transmitted sequence. BER is the most fundamental metric for assessing the reliability of a communication system: the lower the BER, the more accurate the detection, and the more reliable the system. Simulations calculate BER on each of the 100 coherent blocks in blocks, with the average value serving as a summary statistic for overall performance.
4.3 Experimental results show that ViterbiNet achieves satisfactory bit error rate performance in a 10 dB signal-to-noise ratio environment without utilizing any explicit channel state information; more importantly, under block fading channels, the algorithm can continuously track channel changes through online training in blocks, maintaining stable BER throughout transmission without experiencing performance collapse due to channel drift.
**It is worth noting that the implementation with a simplified network configuration and lower learning rate achieved better bit error rate results than those reported in the original paper.** This indicates that ViterbiNet's performance is not dependent on specific network depth or fine-tuning parameters; its "data-driven + algorithm-constrained" framework inherently possesses strong performance potential.
4.4 Discussion of the experimental results:
The feasibility and cost of CSI-free: ViterbiNet transforms the channel estimation problem into a network parameter learning problem, with the quality of detection depending on the quality and quantity of training data (beacon). In slow-varying channels with sufficient beacons, its performance can approach or even exceed traditional schemes relying on CSI; however, in channels with limited beacons or severe channel variations, performance will be constrained by the training convergence speed.
Trade-off between network complexity and performance: The phenomenon of a simplified network (two layers) outperforming deep networks suggests that for this channel model, the complexity of the metric function is not high; overly deep networks may introduce overfitting and convergence difficulties. This provides guidance for actual deployment: the size of the network should be dynamically selected based on the complexity of the channel.
Training protocol selection: The implementation adopts an explicit supervision training protocol of "beacon + data" in blocks, compared to the unsupervised scheme based on error correction codes feedback in the original paper, which is more intuitive and suitable for teaching and rapid validation, but at the cost of beacon overhead. Both protocols have their respective suitability scenarios, and can be chosen based on the system's requirements for spectral efficiency.
ViterbiNet has demonstrated the great potential of "algorithm + data" integration, but there are still some issues worth further research:
1  
High-dimensional modulation and the extension of large state space: The current scheme is based on BPSK and a 16-state grid. As the modulation order or channel memory length increases, the number of states exponentially grows, leading to a sharp increase in network output dimension and ACS computational overhead. It is necessary to investigate structured approximations or hierarchical detection strategies.
2  
Convergence and Stability of Online Training: The performance of iterative training heavily relies on the learning rate, number of training rounds, and the match between block size and data volume. Further analysis is needed to ensure convergence under extreme fast-varying channels.
3  
Joint Design of Channel Coding and Error Correction: Combining ViterbiNet with error correction codes and iterative decoding to form an end-to-end system that combines "detection-decoding" optimization is a promising direction.
4  
Generalization and interpretability: The correspondence between the learned branch metrics from network learning and the real channel statistics is still lacking theoretical characterization. Establishing an interpretability framework can help guide algorithm improvements.
VI. Conclusion This paper systematically introduces ViterbiNet, a data-driven symbol detection scheme that incorporates deep neural networks into the Viterbi algorithm. By learning branch metrics online and tracking channel variations in real-time, ViterbiNet achieves reliable detection of ISI-AWGN channels and block fading channels without channel state information. The streamlined implementation has outperformed the original scheme. This work demonstrates that combining mature algorithm frameworks with data-driven learning can effectively alleviate the dependence on precise channel modeling in traditional receivers, providing a viable technical path for intelligent detection in complex and dynamic channel environments for next-generation wireless communication systems. In the future, as network lightweightness and training strategies continue to be optimized, such "algorithmic prior + data-adaptive" hybrid designs are expected to unlock value in more communication tasks.
References:
[1] Shlezinger N, Farsad N, Eldar Y C, et al. ViterbiNet: A deep learning based Viterbi algorithm for symbol detection[J]. IEEE Transactions on Wireless Communications, 2020, 19(5): 3319-3331.
[2] Forney G D. The Viterbi algorithm[J]. Proceedings of the IEEE, 1973, 61(3): 268-278.
[3] Viterbi A J. Error bounds for convolutional codes and an asymptotically optimum decoding algorithm[J]. IEEE Transactions on Information Theory, 1967, 13(2): 260-269.
[4] O'Shea T, Hoydis J. An introduction to deep learning for the physical layer[J]. IEEE Transactions on Cognitive Communications and Networking, 2017, 3(4): 563-575.
This article is compiled from ViterbiNet's publicly available research materials (Shlezinger et al., IEEE TWC 2020), aiming to facilitate academic exchange and knowledge dissemination. It does not represent the original author's views. Comments are welcome, and please credit the source when reposting.

## Images

![ViterbiNet: Deep Learning Meets Viterbi Algorithm - A New Paradigm for Intelligent Symbolic Detection in Time-Varying Channels [with Python code]](01.jpg)
![ViterbiNet: Deep Learning Meets Viterbi Algorithm - A New Paradigm for Intelligent Symbolic Detection in Time-Varying Channels [with Python code]](02.jpg)
![ViterbiNet: Deep Learning Meets Viterbi Algorithm - A New Paradigm for Intelligent Symbolic Detection in Time-Varying Channels [with Python code]](03.jpg)
![ViterbiNet: Deep Learning Meets Viterbi Algorithm - A New Paradigm for Intelligent Symbolic Detection in Time-Varying Channels [with Python code]](04.jpg)
![ViterbiNet: Deep Learning Meets Viterbi Algorithm - A New Paradigm for Intelligent Symbolic Detection in Time-Varying Channels [with Python code]](05.jpg)
![ViterbiNet: Deep Learning Meets Viterbi Algorithm - A New Paradigm for Intelligent Symbolic Detection in Time-Varying Channels [with Python code]](06.jpg)
![ViterbiNet: Deep Learning Meets Viterbi Algorithm - A New Paradigm for Intelligent Symbolic Detection in Time-Varying Channels [with Python code]](07.jpg)
![ViterbiNet: Deep Learning Meets Viterbi Algorithm - A New Paradigm for Intelligent Symbolic Detection in Time-Varying Channels [with Python code]](08.jpg)
![ViterbiNet: Deep Learning Meets Viterbi Algorithm - A New Paradigm for Intelligent Symbolic Detection in Time-Varying Channels [with Python code]](09.jpg)

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
