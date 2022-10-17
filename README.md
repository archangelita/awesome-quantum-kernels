# Quantum Kernels [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

*A curated list of quantum-accelerated kernels.*

### 101

**What are quantum kernels?**

*Like their classical counterparts, quantum kernels are similarity measures between pairs of data samples encoded in a high-dimensional feature space. The distinction is that quantum kernels use quantum circuits as feature maps. This allows for the exploitation of an exponentially large quantum state space during kernel matrix calculation without having to store the exponentially large vector representations of that space in computer memory.*

**Are quantum kernels better than normal kernels?**

*Not necessarily. We know that a quantum advantage can only be obtained if the quantum kernel circuit is hard to estimate classically [2]. In general, finding a good quantum kernel for a given dataset can be challenging.*

**How do I choose a quantum kernel?**

*Designing a quantum kernel is a process that strongly depends on the learning problem at hand. In general, we cannot suggest an optimal feature mapping with no prior knowledge of the learning problem.*

*Sometimes structure in the data can inform the selection of a kernel [1], [3]; whereas other times kernels may be chosen in an ad hoc manner. Another option is to train a parametrized quantum kernel on a labeled dataset using Qiskit.*

*As quoted by Qiskit. Here you can find the [article.](https://medium.com/qiskit/training-quantum-kernels-for-machine-learning-using-qiskit-617f6e4ed9ac)*

### Kernels In Quantum Context

In computing, a compute kernel is a routine compiled for high throughput accelerators. Compute kernels roughly correspond to inner loops when implementing algorithms in traditional languages (except there is no implied sequential operation), or to code passed to internal iterators. Software layers or APIs allows software to use certain types of devices for general purpose processing operations. This accessibility makes it easier for specialists in programming to use devices which can be GPUs or QPUs. Gives direct access to compute devices' virtual instruction set and computational elements, for the execution of compute kernels.

### Why Does This Repository Stands For?

This repository is a collection of kernels for both mathematical and software libraries for compute functionality.
Also contains fundamental resources to dive into this area.

# Table of Content
| <!-- -->                         | <!-- -->                         |
| -------------------------------- | -------------------------------- |
| [Fundamentals](#qc-fundamentals) | [Communities](#qc-communities) |
| [Conferences, Events/Talks, and Hackathon](#conferences-events-talks-hackathon) | [News](#qc-news)
| [Books](#qc-books) | [Tools](#qc-tools)
| [Quantum Kernels](#q-kernels) | ❤️‍🔥

<a name="qc-fundamentals"></a>
# Fundamentals

## Nice-to-have Fundamental Prerequisites

- Algorithms
- Basics of Quantum Mechanics
- Boolean Algebra
- Complex Numbers
- CUDA Kernels
- Linear Algebra
- Machine Learning
- Probability
- Statistics

## Fundamental Learning Materials

- [Understanding Quantum Computers](https://www.futurelearn.com/courses/intro-to-quantum-computing)
- [What is Quantum Computer?](https://www.technologyreview.com/2019/01/29/66141/what-is-quantum-computing/)
- [How Do Quantum Computers Work?](https://www.sciencealert.com/quantum-computers)
- [Quantum Computing: A Gentle Introduction](http://mmrc.amss.cas.cn/tlb/201702/W020170224608150244118.pdf)
- [Quantum Information Science I](https://www.edx.org/course/quantum-information-science-i-part-1)
- [Quantum Information Science Lecture Notes by MIT](https://ocw.mit.edu/courses/media-arts-and-sciences/mas-865j-quantum-information-science-spring-2006/lecture-notes/)
- [Quantum Machine Learning](https://www.edx.org/course/quantum-machine-learning)
- [Shtetl-Optimized](https://www.scottaaronson.com/blog/)
- [Elements of Quantum Computer Programming](https://cs269q.stanford.edu/syllabus.html?fbclid=IwAR09_JNstMi4WVU4oMHDpWR6xWaSISlrYPjWTUTnhcRdEQhzpoOTRgQN8LI)
- [John Preskill's Notes on Quantum Computation](http://www.theory.caltech.edu/~preskill/ph219/index.html#lecture)
- [CNOT: Introduction to Quantum Computing](https://cnot.io/get_started/)
- [Qiskit Tutorials](https://github.com/Qiskit/qiskit-tutorials)
- [Quantum Algorithm Zoo](http://quantumalgorithmzoo.org)
- [Introduction to Quantum Computing by Saint Petersburg University](https://www.coursera.org/learn/quantum-computing-algorithms)
- [Physical Basics of Quantum Computing by Saint Petersburg University](https://www.coursera.org/learn/physical-basis-quantum-computing)
- []()

## Application Areas

- Artificial Intelligence/ML
- Computational Physics
- Climate Models
- Cyber-security & Cryptography
- Drug Discovery
- Espionage
- Finance
- Modelling
- Optimization
- Quantum Chemistry
- Research
- Simulations
- Technology Design & Development
- Weather Forecasting

<a name="qc-communities"></a>
# Communities
1. [QuantumComputing StackExchange](https://quantumcomputing.stackexchange.com) - Q&A platform for quantum computing
2. [IBM Q Community](https://qiskit.org/advocates) - A global program that provides support to the individuals who actively contribute to the Qiskit Community
3. [Qiskit Slack](https://qiskit.slack.com) - Slack channel for Qiskit community
4. [Subreddit for Quantum Computing](https://www.reddit.com/r/QuantumComputing/) - Subreddit for discussion of quantum computing topics
5. [QWorld](https://qworld.lu.lv) - QWorld is a global network of individuals, groups, and communities collaborating on education and implementation of quantum technologies and research activities.
   - [QTurkey](http://qworld.lu.lv/index.php/qturkey/)
   - [QLatvia](http://qworld.lu.lv/index.php/qlatvia/)
   - [QHungary](http://qworld.lu.lv/index.php/qhungary/)
   - [QBalkan](http://qworld.lu.lv/index.php/qbalkan/)
   - [QPoland](http://qworld.lu.lv/index.php/qpoland/)
   - [QRussia](http://qworld.lu.lv/index.php/qrussia/)
   - [QSlovakia](http://qworld.lu.lv/index.php/qslovakia/)
7. [Q# Community](https://qsharp.community) - This group collects + maintains projects related to the Q# programming language by a community of folks who are excited about quantum programming!
8. [PennyLane Forum](https://discuss.pennylane.ai) - Forum for quantum machine learning discussions
9. [Strawberry Fields Community](https://u.strawberryfields.ai/slack) - Slack channel for Strawberry Fields library
10. [Tensorflow Quantum Community](https://stackoverflow.com/questions/tagged/tensorflow-quantum) - Q&A for Tensorflow Quantum
11. [Quantum Information and Quantum Computer Scientists of the World Unite](https://www.facebook.com/groups/qinfo.scientists.unite/) - Facebook group for Quantum Computer Scientists
12. [Quantum Inferiority](https://matrix.to/#/#quantum_inferiority:chat.weho.st) - Chat rooms for quantum programming
13. [QuantumGrad Discord](https://discord.gg/vxpwmsVaHD)

<a name="conferences-events-talks-hackathon"></a>
# Conferences, Events/Talks & Hackathons

## Conferences

- [Complete list from World Academy of Science, Engineering and Technology](https://waset.org/quantum-computing-conferences)
- [IEEE Quantum](https://quantum.ieee.org/conferences)
- [Quantum Computing Reports](https://quantumcomputingreport.com/conferences/)

## Events

### Quantum Flagship Events
- [Upcoming Events](https://qt.eu/about-quantum-flagship/events/)

### Cambridge Quantum Computing Events
- [Complete List](https://cambridgequantum.com/events/)

### D-Wave Systems Events
- [Upcoming Events](https://www.dwavesys.com/company/news/events)

## Hackathons
- [QHack](https://qhack.ai/)
- [UCLQ](https://www.ucl.ac.uk/quantum/innovation/hackathons)
- [QOSF Quantum Futures](https://qosf.org/quantum-hackathon/)
- QWorld Quantum Programming & Quantum Gamejam Hackathons

## Others
- [ETH Zürich QSIT Seminars](https://video.ethz.ch/speakers/qsitseminars.html)
- [Le Lab Quantique Meetups](https://lelabquantique.com/)
- [Quantum Science Seminars](https://quantumscienceseminar.com/)
- [Virtual Amo Seminars](https://sites.google.com/stanford.edu/virtual-amo-seminar/home?authuser=0)
- [Quantum Research Seminars Toronto](https://twitter.com/qrstoronto)
- [Quantum Information Seminar Series](https://www.youtube.com/playlist?list=PLOFEBzvs-Vvr0uEoGFo08n4-WrM_8fft2)
- [Machine Learning for Quantum Simulation Virtual Conference](https://www.youtube.com/playlist?list=PLWAzLum_3a1_S98TvxMahoQJgeEoGVObp)

<a name="qc-news"></a>
# QC News

- [MIT News](https://news.mit.edu/topic/quantum-computing)
- [Science Daily](https://www.sciencedaily.com/news/matter_energy/quantum_computing/)
- [Phys.org](https://phys.org/tags/quantum+computing/)
- [Wired](https://www.wired.com/tag/quantum-computing/)
- [The Quantum Daily](https://thequantumdaily.com)

<a name="qc-books"></a>
# QC/QP Books

## Quantum Computing

- The Amazing World of Quantum Computing (Rajendra K. Bera, 2020)
- Quantum Computing for Everyone (Chris Bernhardt, 2019)
- Quantum Computing: An Applied Approach (Jack Hidary, 2019)
- Dancing with Qubits: How quantum computing works and how it can change the world (Robert S. Sutor, 2019)
- Mathematics of Quantum Computing: An Introduction (Wolfgang Scherer,2019)
- Quantum Computing Since Democritus (Scott Aaronson, 2013)
- Computing with Quantum Cats: From Colossus to Qubits (John Gribbin, 2013)
- Quantum Computing for Computer Scientists (Noson S. Yanofsky, Mirco A. Mannucci, 2013)
- Mathematics of Quantum Computation and Quantum Technology (Louis Kauffman, Samuel J. Lomonaco, 2007)
- Quantum Computing Explained (David McMahon, 2007)

## Quantum Programming

- Programming Quantum Computers: Essential Algorithms and Code Samples (Eric R. Johnston, Nic Harrigan, Mercedes Gimeno-Segovia, 2019)
- Practical Quantum Computing for Developers: Programming Quantum Rigs in the Cloud using Python, Quantum Assembly Language and IBM QExperience (Vladimir Silva, 2018)
- Foundations of Quantum Programming (Mingsheng Ying, 2016)

<a name="qc-tools"></a>
# Tools

## Cloud Platforms
- IBM Q Experience by IBM
- Azure Quantum by Microsoft
- Amazon Braket by Amazon Web Services
- Quantum Playground by Google
- Black Opal by Q-CTRL
- Orquestra by Zapata
- Xanadu Quantum Cloud by Xanadu
- Forest by Rigetti Computing
- Quantum in the Cloud by The University of Bristol
- Quantum in the Cloud by Tsinghua University
- Quantum Inspire by Qutech

## Programming Languages

- Python
- Q#
- Quantum Computation Language
- Silq
- Q Language
- QML
- OpenQASM
- Q|SI>
- QMASM
- Julia

## Development Tools/Libraries

- [Qiskit](https://qiskit.org) - Qiskit is an open source SDK for working with quantum computers at the level of pulses, circuits and algorithms
- [cuQuantum SDK](https://developer.nvidia.com/cuquantum-sdk) - NVIDIA cuQuantum is an SDK of optimized libraries and tools for accelerating quantum computing workflows. Using NVIDIA GPU Tensor Core GPUs, developers can use cuQuantum to speed up quantum circuit simulations based on state vector and tensor network methods by orders of magnitude.
- [Qiskit.js](https://github.com/qiskit-community/qiskit-js) - IBM’s quantum information software kit for JavaScript.
- [PennyLane](https://pennylane.ai) - A cross-platform Python library for quantum machine learning, automatic differentiation, and optimization of hybrid quantum-classical computations
- [Tensorflow Quantum](https://www.tensorflow.org/quantum) - TensorFlow Quantum (TFQ) is a Python framework for quantum machine learning.
- [Strawberry Fields](https://strawberryfields.ai) - A cross-platform Python library for simulating and executing programs on quantum photonic hardware.
- [Microsoft Quantum Development Kit](https://www.microsoft.com/en-us/quantum/development-kit) - The Quantum Development Kit is the development kit for Azure Quantum.
- [Ocean SDK](https://docs.ocean.dwavesys.com/en/latest/overview/install.html) - Ocean software is a suite of tools D-Wave Systems provides on the D-Wave GitHub repository for solving hard problems with quantum computers.
- [Rigetti Forest](https://github.com/rigetti/pyquil) - The Rigetti Forest suite consists of a quantum instruction language called Quil, an open source Python library for construction Quil programs called pyQuil, a library of quantum programs called Grove, and a simulation environment called QVM standing for Quantum Virtual Machine. 
- [Cirq](https://github.com/quantumlib/cirq) - A python framework for creating, editing, and invoking Noisy Intermediate Scale Quantum (NISQ) circuits.
- [ProjectQ](https://projectq.ch) - ProjectQ is an open-source software framework for quantum computing implemented in Python.
- [Yao.jl](https://yaoquantum.org) - Yao is an extensible, efficient open-source framework for quantum algorithm design.
- [Quantum++](https://github.com/softwareQinc/qpp) - High-performance general purpose quantum simulator (can simulate d-dimensional qudits).
- [Qbsolv](https://github.com/dwavesystems/qbsolv) - QUBO solver with D-Wave or classical tabu solver backend.
- [QRL](https://github.com/theQRL/QRL/) - Quantum Resistant Ledger utilizing hash-based one-time merkle tree signature scheme instead of ECDSA.
- [Qlab](https://github.com/BBN-Q/Qlab) - Measurement and control software for superconducting qubits.
- [BLACK-STONE](https://github.com/thephoeron/black-stone) - Specification and Implementation of Quantum Common Lisp, for gate-model quantum computers
- [Paddle Quantum](https://github.com/PaddlePaddle/Quantum) - Baidu's python toolkit for quantum machine learning.
- [Orquestra](https://www.zapatacomputing.com/orquestra/) - Zapata Computing's unified quantum operating environment, allowing for quantum-enabled workflows.
- [NISQAI](https://github.com/quantumai-lib/nisqai) - A Python toolkit for quantum neural networks.
- [staq](https://github.com/softwareQinc/staq) - staq is a modern C++17 library for the synthesis, transformation, optimization and compilation of quantum circuits.
- [Quantum Programming Studio](https://quantum-circuit.com)- The Quantum Programming Studio is a web based graphical user interface designed to allow users to construct quantum algorithms and obtain results by simulating directly in browser or by executing on real quantum computers
- [Orquestra Workflow Tool by Zapata Computing](https://orquestra.io) - Quantum orchestration suit

<a name="q-kernels"></a>
# Quantum Kernels

## Quantum

- [Training and evaluating quantum kernels](https://pennylane.ai/qml/demos/tutorial_kernels_module.html)
- [Kernel-based training of quantum models with scikit-learn](https://pennylane.ai/qml/demos/tutorial_kernel_based_training.html)
- [QuantumKernel Implementations](https://github.com/qiskit-community/prototype-quantum-kernel-training/tree/main/docs/how_tos)
- [Quantum Kernel Training for Machine Learning Applications](https://github.com/qiskit-community/prototype-quantum-kernel-training/blob/main/docs/tutorials/kernel_optimization_using_qkt.ipynb)

## Software

- [Qiskit Quantum Kernel Toolkit](https://github.com/qiskit-community/prototype-quantum-kernel-training)
- [Qiskit QKT Prototype - Optimizing kernels using weighted kernel alignment](https://github.com/qiskit-community/prototype-quantum-kernel-training/blob/main/docs/background/svm_weighted_kernel_alignment.ipynb)
- [Qiskit QuantumKernel Base Class](https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.kernels.QuantumKernel.html)
- [Qiskit QuantumKernelTrainer Class](https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.kernels.algorithms.QuantumKernelTrainer.html#qiskit_machine_learning.kernels.algorithms.QuantumKernelTrainer)
- [Qiskit PegasosQSVC Class](https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.algorithms.PegasosQSVC.html)
- [Qiskit QSVC Class](https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.algorithms.QSVC.html)
- [Qiskit QSVR Class](https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.algorithms.QSVR.html)
- [Quantum Circuit Simulation Accelerator with NVIDIA cuStateVec](https://developer.nvidia.com/blog/accelerating-quantum-circuit-simulation-with-nvidia-custatevec/)
- [Efficient Quantum Neural Network Training with Probabilistic Gradient Pruning](https://www.youtube.com/watch?v=Z85TddJqi6c)

## Papers

- [Covariant quantum kernels for data with group structure](https://arxiv.org/abs/2105.03406)
- [Supervised learning with quantum-enhanced feature spaces](https://www.nature.com/articles/s41586-019-0980-2)
- [Optimal quantum kernels for small data classification](https://arxiv.org/pdf/2203.13848.pdf)
- [The Inductive Bias of Quantum Kernels](https://arxiv.org/pdf/2106.03747v2.pdf)
- [Importance of kernel bandwidth in quantum machine learning](https://arxiv.org/pdf/2111.05451.pdf)
- [QOC: Quantum On-Chip Training with Parameter Shift and Gradient Pruning](https://arxiv.org/abs/2202.13239)
- [Deterministic and random features for large-scale quantum kernel machine](https://arxiv.org/pdf/2209.01958.pdf)
