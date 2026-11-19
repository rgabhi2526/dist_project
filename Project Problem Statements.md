Distributed Systems: Monsoon 2026

Project Problem Statements
------------------------------------------------------------------------------------------------------------------------------
With the arrival of multicore and distributed systems, parallel and concurrent programming have
become very important to harness the power of these processors. Multithreading and distributed
computing are common ways to achieve parallel programming. A distributed system is a model in
which components of a software system are shared among multiple computers. Even though the
components are spread out across multiple computers, they are run as one system. This is done in
order to improve efficiency and performance. There are several issues and open problems in this
area that will be given as topics for DS projects:

1. Distributed Algorithms

The field  of Distributed  Systems  has been around for  some time. With the rise of connectivity
through  the  Internet,  this  paradigm  has  become  even  more  useful.  Here  are  some  interesting
problems in this area.

- Blockchains: Blockchain is an encrypted, distributed ledger (or database) shared across multiple
computers or nodes that are part of a community or system. Each node holds an identical copy of
the ledger, which is usually represented as a chain of blocks. Each block in the chain is a logical
sequence of business (or financial) transactions. Each block encloses the hash of its immediate
previous block, thereby guaranteeing the immutability of the ledger.

Blockchain  is  completely  decentralized  and  can  be  accessed  by  anyone  with  an  internet
connection. As a result, blockchain technology has become very popular with cryptocurrencies
such  as  Bitcoin,  Ethereum,  etc.  However,  blockchain  technology  is  not  only  used  for
cryptocurrencies.  Several  businesses  have  also  adopted  this  innovative  technology  and  have
already  achieved  significant  benefits,  some  of  which  include  greater  transparency,  enhanced
security, improved traceability, increased efficiency, and speed of transactions as a result reduced
costs. Blockchains have become so popular that several State Governments (such as Telangana
records.
State  Govt)

for  maintaining

and  medical

using

land

this

are

1  Study and comparison of directed acyclic graphs based blockchains (BlockDags): The
normal structure of blockchains is chains, as the name suggests. Apart from chains, other
structures  have  been  explored  in  the  literature,  such  as  using  Directed  Acyclic  Graphs
(DAG) for blockchains called BlockDags. So in this project, we will study and compare
the various structures for Blockchains.
Reference:
[1] Kahmann, F., Honecker, F., Dreyer, J., Fischer, M. and Tönjes, R., 2023. Performance comparison of
directed acyclic graph-based distributed ledgers and blockchain platforms. Computers, 12(12), p.257.

[2] Wang, Q., Yu, J., Chen, S. and Xiang, Y., 2023. SoK: DAG-based blockchain systems. ACM Computing
Surveys, 55(12), pp.1-38.
[3]  Živi,  N.,  E.  Kadušić,  and  K.  Kadušić.  "Directed  Acyclic  Graph  as  Tangle:  an  IoT  Alternative  to
Blockchains." In 2019 27th Telecommunications Forum (TELFOR), pp. 1-3. IEEE, 2019.
[4]  Pervez,  Huma,  Muhammad  Muneeb,  Muhammad  Usama  Irfan,  and  Irfan  Ul  Haq.  "A  comparative
analysis of DAG-based blockchain architectures." In 2018 12th International Conference on Open Source
Systems and Technologies (ICOSST), pp. 27-34. IEEE, 2018.
[5]  Benčić,  Federico  Matteo,  and  Ivana  Podnar  Žarko.  "Distributed  ledger  technology:  Blockchain
compared to directed acyclic graph." In 2018 IEEE 38th International Conference on Distributed Computing
Systems (ICDCS), pp. 1569-1570. IEEE, 2018.
[6] Kotilevets, I. D., I. A. Ivanova, I. O. Romanov, S. G. Magomedov, V. V. Nikonov, and S. A. Pavelev.
"Implementation  of  directed  acyclic  graph  in  blockchain  network  to  improve  security  and  speed  of
transactions." IFAC-PapersOnLine 51, no. 30 (2018): 693-696.
[7] Gorczyca, Andrew, and Audrey Decker. "Distributed ledger witness selection in bounded width directed
acyclic graphs." In 2019 IEEE International Conference on Blockchain and Cryptocurrency (ICBC), pp.
124-127. IEEE, 2019.
[8] Ferraro, Pietro, Christopher King, and Robert Shorten. "On the stability of unverified transactions in a
DAG-based Distributed Ledger." IEEE Transactions on Automatic Control 65, no. 9 (2019): 3772-3783.
[9] Swaroopa Reddy, B., and G. V. V. Sharma. "Scalable Consensus Protocols for PoW based Blockchain
and blockDAG." arXiv e-prints (2020): arXiv-2010.

2  Efficient  Smart  Contract  Execution:  Modern  blockchain  systems  often  interpose  an
additional software layer between clients and the blockchain. Client requests are directed
to  scripts,  called  smart  contracts,  that  perform  the  logic  needed  to  provide  a  complex
service,  such  as  managing  state,  enforcing  governance,  or  checking  credentials.  In  this
project, we plan to devise a new framework for the efficient execution of smart contracts
on blockchains.

A.  Efficient  smart  contract  execution  in  Hyperledger  Sawtooth  blockchain:
Hyperledger is a multi-project open source collaborative effort hosted by the Linux
Foundation,  created  to  advance  cross-industry  blockchain  technologies.  Its  two
popular blockchain platforms are Fabric and Sawtooth. So, this is a specific project
in  which  we  will  explore  and  develop  mechanisms  for  parallelizing  various
components of these platforms.
References:
[1] Piduguralla, M., Chakraborty, S., Anjana, P.S. and Peri, S., 2023. An efficient framework for
execution of smart contracts in hyperledger sawtooth. arXiv preprint arXiv:2302.08452.
[2] Shi, Z., Zhou, H., Hu, Y., Jayachander, S., De Laat, C. and Zhao, Z., 2019, June. Operating
permissioned  blockchain  in  clouds:  A  performance  study  of  hyperledger  sawtooth.  In  2019 18th
International Symposium on Parallel and Distributed Computing (ISPDC) (pp. 50-57). IEEE.
[3] Woznica, A. and Kedziora, M., 2022. Performance and scalability evaluation of a permissioned
Blockchain  based  on  the  Hyperledger  Fabric,  Sawtooth  and  Iroha.  Computer  Science  and
Information Systems, 19(2), pp.659-678.
[4] Benahmed, S., Pidikseev, I., Hussain, R., Lee, J., Kazmi, S.A., Oracevic, A. and Hussain, F.,
2019,  September.  A  comparative  analysis  of  distributed  ledger  technologies  for  smart  contract
development. In 2019 IEEE 30th Annual International Symposium on Personal, Indoor and Mobile
Radio Communications (PIMRC) (pp. 1-6). IEEE.

B.  Efficient smart contract execution in Tezos blockchain: The effort here is similar
to the above project (parallelizing Tezos Blockchain). But one has to work on Tezos
blockchain  -  www.tezos.com.  For  this  project,  you  will  have  to  understand  the
Tezos platform well to develop parallelization techniques for it.
References:
[1]  Baheti  S,  Anjana  PS,  Peri  S,  Simmhan  Y.  DiPETrans:  A  framework  for  distributed  parallel
execution  of  transactions  of  blocks  in  blockchains.  Concurrency  Computat  Pract  Exper.  2022;
34(10):e6804. doi:10.1002/cpe.6804
[2] Wang, Y., Li, J., Liu, W. and Tan, A., 2021. Efficient concurrent execution of smart contracts in
blockchain sharding. Security and Communication Networks, 2021(1), p.6688168.
[3]  Anjana,  P.S.,  Kumari,  S.,  Peri,  S.,  Rathor,  S.  and  Somani,  A.,  2019,  February.  An  efficient
framework  for  optimistic  concurrent  execution  of  smart  contracts.  In  2019  27th  Euromicro
International Conference on Parallel, Distributed and Network-Based Processing (PDP) (pp. 83-
92). IEEE.

C.  Bitcoin Mining: Bitcoin is the most popular cryptocurrency using blockchains. In
this project, we will understand the existing blockchain and algorithms for mining
Bitcoin.
References:
[1] Benkoczi, R., Gaur, D., Nagy, N., Nagy, M. and Hossain, S., 2022. Quantum bitcoin mining.
Entropy, 24(3), p.323.
[2]  Dev,  J.A.,  2014,  May.  Bitcoin  mining  acceleration  and  performance  quantification.  In  2014
IEEE 27th Canadian conference on electrical and computer engineering (CCECE) (pp. 1-6). IEEE.
[3] Naik, R.P. and Courtois, N.T., 2013. Optimising the sha256 hashing algorithm for faster and
more  efficient  bitcoin  mining.  MSc  Information Security  Department  of  Computer  Science  UCL,
pp.1-65.

3  Efficient distributed graph algorithms: In this project, one must come up with efficient
ways of building distributed graph structures, such as a distributed spanning tree. Here, the
nodes of the tree are spread across a geographical network. In fact all the problems studied
in the context of parallel programming can be extended here: (1) Minimum Spanning Trees
(2) Graph Coloring (3) Betweenness Centrality, (4) Page rank, (5) Shortest Paths, etc.
References:
[1]  Pasham,  S.D.,  2022.  Graph-Based  Algorithms  for  Optimizing  Data  Flow  in  Distributed  Cloud
Architectures. International Journal of Acta Informatica, 1(1), pp.67-95.
[2] Pasham, S.D., 2024. Scalable Graph-Based Algorithms for Real-Time Analysis of Big Data in Social
Networks. The Metascience, 2(1), pp.92-129.
[3]D. Nguyen and S. S. Kulkarni, "Benefits of Stabilization versus Rollback in Self-Stabilizing Graph-Based
Applications  on  Eventually  Consistent  Key-Value  Stores,"  2020  International  Symposium  on  Reliable
Distributed Systems (SRDS), Shanghai, China, 2020, pp. 11-20,  Dissertation-PDF  DOI
[4] Nguyen, D., Charapko, A., Kulkarni, S.S. et al. Using weaker consistency models with monitoring and
recovery for improving performance of key-value stores. J Braz Comput Soc 25, 10 (2019). DOI  PDF

4  Federated  learning  over  unreliable  distributed  networks:  Federated  Learning  (FL)
enables multiple distributed devices or edge nodes to collaboratively train AI/ML models
without sharing raw data, thereby preserving privacy. However, practical deployment of

FL in  real-world  distributed environments  faces significant  challenges due to  unreliable
networks, intermittent connectivity, node failures, communication delays, heterogeneous
devices, and limited bandwidth.
This  project  aims  to  design  a  robust  and  communication-efficient  federated  learning
framework  for  unreliable  distributed  networks.  The  proposed  system  will  incorporate
adaptive  client  selection,  asynchronous  model  aggregation,  fault-tolerant  update
mechanisms,  and  network-aware  communication  strategies  to  maintain  model  accuracy
even under unstable connectivity conditions. AI/ML-based prediction techniques can also
be  explored  to  estimate  network  reliability  and  dynamically  optimize  participation  of
clients during training.
The framework can be evaluated on distributed HPC clusters using metrics such as model
convergence,  communication  overhead,  fault  tolerance,  energy  efficiency,  and  training
latency.
References:

o  FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping
o  Fault-Tolerant Decentralized Distributed Asynchronous Federated Learning with

Adaptive Termination Detection

o  Reliable Federated Learning for Mobile Networks

5  Distributed reinforcement learning (RL) for  multi-agent coordination:  Multi-Agent
Systems (MAS) are increasingly used in applications such as autonomous vehicles, drone
swarms, smart grids, robotics, and distributed IoT networks, where multiple agents must
cooperate  and  make  decisions  in  dynamic  environments.  Traditional  centralized
reinforcement  learning  approaches  often  suffer  from  scalability  limitations,  high
communication overhead, and single points of failure.

This  project  proposes a distributed reinforcement learning (RL) framework that enables
multiple agents to collaboratively learn optimal coordination strategies while operating in
decentralized  environments.  The  framework  will  allow  agents  to  learn  locally  through
interactions with the environment and periodically exchange knowledge or policies with
neighboring agents to improve global performance.

The research will focus on designing scalable and communication-efficient RL algorithms
that support decentralized decision-making, adaptive coordination, and fault tolerance in
heterogeneous  distributed  systems.  Techniques  such  as  multi-agent  deep  reinforcement
learning
federated  RL,  and  graph-based
learning,
communication  models  can  be  explored  to  improve  cooperation  among  agents  under
dynamic network conditions.

(MADRL),  asynchronous

The  proposed  framework  can  be  implemented  and  evaluated  in  simulated  distributed
environments  using  metrics  such  as  convergence  speed,  scalability,  coordination
efficiency,  communication  cost,  energy  consumption,  and  robustness  against  node  or

communication failures.
References:
Distributed Cooperative Multi-Agent Reinforcement Learning with Directed Coordination
Graph

6  Design  a  Distributed  computing  framework  for  efficient  AI/ML  models:  Modern
AI/ML models require massive computational resources and large-scale data processing,
making centralized training increasingly inefficient and expensive. This project proposes
the design of a scalable distributed computing framework that accelerates AI/ML model
training and inference across multiple computing nodes, GPUs, or edge devices.

The framework will focus on efficient workload distribution, parallel processing, resource-
aware  scheduling,  and  communication  optimization  to  improve  performance  of  AI/ML
applications.  The  system  may  integrate  techniques  such  as  data  parallelism,  model
parallelism, distributed caching, adaptive load balancing, and GPU-aware task scheduling
to minimize training time and maximize hardware utilization.

The  proposed  framework  can  support  deep  learning,  federated  learning,  and  large-scale
data  analytics  workloads  in  heterogeneous  environments  such  as  HPC  clusters,  cloud
platforms, and edge systems. Performance evaluation can be carried out using metrics like
scalability,  throughput,  latency,  energy efficiency, communication overhead, and model
convergence time.
References:
Ray: A Distributed Framework for Emerging AI Applications

7  An  efficient  distributed  scheduling  algorithm:  In  the  OS  course,  you  studied  several
scheduling algorithms. All these algorithms assumed that the processes/threads are on a
single machine. Suppose these are not on a single machine, but are spread geographically.
Then, will the same scheduling algorithms work?

8  Self-stabilizing Clustering Algorithm for Maintaining Low Stretch Trees: A special
variety of spanning trees in graphs aims to preserve the shortest distances between each
pair of nodes in the underlying graph. Trees with low distortion of distances are called low-
stretch  trees.  Our  work  here  involves  the  construction  and  maintenance  of  low-stretch
spanning  trees  using  low-diameter  decomposition  techniques  in  distributed  graphs.  We
would further like to make this distributed construction fault-tolerant by using techniques
from self-stabilization. This is a technique by which an algorithm is made robust against
transient faults that infrequently occur in the system.

9  Fault-tolerance  in  a  minimum-max  stretch  spanning  tree:  We  will  look  at  the
construction and fault-tolerance of a variant of low stretch trees known as the minimum-

max stretch spanning tree. This is an optimization problem where we aim to minimize the
maximum stretch of all the pairs of nodes in the spanning tree to be constructed. Further,
we  will  use  the  concept  of  distributed  swap  edges  for  the  fault  tolerance  of  the  tree  so
obtained, in the wake of system perturbations.

10  Causally consistent CRDTs using Hybrid Logical Clock: In this work, we will focus on
developing  causally  consistent  (nested)-CRDTs  (Conflict-free  Replicated  Data  Types)
using HLC (Hybrid Logical Clock).
CRDTs permit data to converge without conflict on different nodes of a distributed system.
Operational Transform (OT) is another algorithm used by Google to converge the data in
collaborative access to shared data files. OT is mostly used for text, and CRDT is more
general and can handle more advanced structures. Both approaches are similar in that they
provide eventual consistency. The difference is in how they do it. One way of looking at it
is: OT does it by changing operations. Operations are sent over the wire, and concurrent
operations  are  transformed  once  they  are  received.  CRDTs  do  it  by  changing  the  state.
Operations are made on the local CRDT. Its state is sent over the wire and is merged with
the state of a copy. It does not matter how many times or in what order merges are made -
all copies converge.
We  will  look  at  causally  consistent  CRDTs.  The  expected  outcome  is  to  identify  the
significant  problems  caused  by  failures  in  the  system  while  satisfying  the  causality  in
partial replication and propose a working solution to overcome such problems. Later, after
the course completion, we will explore causally consistent nested CRDTs, nested CRDTs
consisting of multiple objects nested within each other, for example, maps and counter data
types nested together. For Nested CRDTs: We will work on the Graph data type at the top
level; within each graph node (vertex), there can be a list or map or counter CRDT.

11  An  efficient  practical  interoperability  protocol  for  blockchains:  Modern  distributed
data management systems face a new challenge - how autonomous, mutually distrusting
parties  work  together  securely  and  effectively?  Blockchain  Interoperability  is  a  way  to
allow state transitions across different blockchain networks that is necessary to achieve the
adoption  of  blockchains  in  many  real  world  applications.  Existing  interoperability
protocols focus mostly on the interchange of atomic tokens between blockchains. However,
the  scope  of  interoperability  of  blockchain  extends  beyond  just  token  exchange.  In  this
project,  we  will  explore  the  atomic  swap  of  smart  contract  states  between  different
blockchains and focus on developing an efficient methodology.

2. Concurrent Programming: The problems for concurrent programming have also been
mentioned here. This is because several of these problems can be modified to distributed settings.
So, these problems give you more options to choose from.

1.  Parallel Graph Algorithms: Large-scale Graph Analytics
Description: This topic explores the power of multicore systems in parallelizing graph algorithms.
Graphs are useful in a wide variety of settings. In this topic, we can explore how to efficiently
parallelize  them  while  keeping  concurrency  in  mind.  Two  particular  algorithms  of  interest  are
parallel  graph  coloring  and  the  Minimum  Spanning  Tree  algorithm.  There  have  been  many
algorithms proposed in the literature to solve these problems. But unfortunately, either many of
them are inefficient or incorrect (due to synchronization issues).

One idea can be to design a parallel graph-partitioning technique that serves as a one-time pre-
processing step for lock-free PageRank computation.  The idea is  to  divide lock-free PageRank
computation into two phases. In the first phase, we partition the graph into groups that ensure the
minimum  communication  between  any  two  partitions.  In  the  second  phase,  we  compute  the
PageRank values by effectively utilizing the hardware resources with minimal cache misses.

Apart from these problems, we can explore other algorithms which are as follows:
a.  Minimum Spanning Trees
b.  Graph Coloring
c.  Betweenness Centrality,
d.  Page rank,
e.  Shortest Paths etc.

References:

A.  https://arxiv.org/pdf/1404.4797.pdf
B.  https://userweb.cs.txstate.edu/~mb92/papers/lcpc10.pdf
C.  https://research.nvidia.com/sites/default/files/pubs/2016-03_Parallel-Spectral-Graph/nvr-

2016-001.pdf

D.  https://patterns.eecs.berkeley.edu/?page_id=571
E.  https://www.osti.gov/servlets/purl/4179
F.  https://www.mathsci.ai/publication/Parallel-Computing-2000.pdf
G.  https://ieeexplore.ieee.org/abstract/document/7859409
H.  A simple and practical concurrent non-blocking unbounded graph with reachability

queries

I.  Non-blocking dynamic unbounded graphs with worst-case amortized bounds

2.  Lock Free Concurrency Control

Description: An important issue that comes with a multithreaded system is synchronization of
shared memory and concurrency control. Traditionally locks have been used for concurrency
control. But it was realized that programming with locks can be inefficient, cumbersome and
error  prone.  Hence,  alternative  paradigms  for  concurrency  control  have  become  important.
Some approaches for efficient concurrency control are:
a.  Lock-free Data Structures. The idea of this project is to build lock-free data-structures
(such as queues, stacks) that can yet support multiple threads. For instance, this project can

be to build a lock-free queue that supports simultaneous invocation of enq, deq operations
by different threads and yet be correct.

References:
A.  Ruslan  Nikolaev,  “A  Scalable,  Portable,  and  Memory-Efficient  Lock-Free  FIFO  Queue.”

DISC’19,
PDF: https://drops.dagstuhl.de/opus/volltexte/2019/11335/pdf/LIPIcs-DISC-2019-28.pdf
Github: https://github.com/rusnikola/lfqueue
ArXiv: https://arxiv.org/pdf/1908.04511.pdf

B.  Ladan-Mozes,  Edya,  and  Nir  Shavit.  "An  optimistic  approach  to  lock-free  FIFO  queues."

DISC’04.
PDF: http://people.csail.mit.edu/shanir/publications/FIFO_Queues.pdf

3.  Developing a concurrent Merkel tree

Description:  Merkel  trees  are  a  significant  part  of  blockchain  architecture  and  act  as
fingerprints  for  the  block  in  the  blockchain.  They  are  hash  trees,  where  each  leaf  node  is
labelled with the cryptographic hash of a data block, and each non-leaf node is labelled with
the cryptographic hash of its child nodes label. In this project we will be studying the operations
performed on a Merkel tree with an aim to develop concurrent algorithms for these operations.
This would lead to concurrent execution of smart contracts in blockchains.

References:

A.  Flangas,  Andrew,  Autumn  A.  Cuellar,  Michael  Reyes  and  Frederick  C.  Harris.
“Parallelized C Implementation of a Merkle Tree.” Advances in Intelligent Systems and
Computing (2021)

B.  Kalidhindi, Janakirama, Alex Kazorian, Aneesh Khera and C. Senthil Pari. “Angela : A

Sparse , Distributed , and Highly Concurrent Merkle Tree.” (2018).

4.  Parallelizing Divide & Conquer algorithms:

Description:  In this problem you can  consider any standard divide  and  conquer  algorithm.
Specifically, consider the non-recursive version of the problem. And then consider developing
an efficient parallel version of that problem. For simplicity, you can consider one of these two
problems:

a.  Merge sort algorithm or
b.  Finding a convex hull given a set of points.

5.  Constant Time Snapshot

Description: If you want the state of a data structure at any given time, you should take a
snapshot. Naive snapshot algorithms can return inconsistent results. The snapshot algorithm
that you will study in the PCP reference text works only when the size of the data structure
remains fixed.

– Recently, a paper does constant time lock-free snapshots for any dynamically changing
data structure. Impressive, isn’t it? They do this by maintaining versions of each data
element. A version v of any data element x maintains the value of x at some time t. So, when
a snapshot routine is called, the snapshot thread will iterate to the correct version of any data
element by choosing which version to go to based on when the snapshot was called.

– For example, let a data element y have a value 10 at time 5 and a value 20 at time 7. If a
snapshot was called at time 6, the snapshot thread would return the value 10 since at time 6,
the value of y was 10. If the snapshot were instead called at time 9, it would return the value
20!

– Clearly, deleted nodes cannot be physically removed from the data structure right away.
There could be a chance that an old snapshot has been called before a node gets deleted and
that snapshot is still going on, perhaps because that thread is slow. So when do we physically
clean up the data structure?!

– This project deals with clever ways to perform efficient garbage collection for this
algorithm.

– What can you expect to learn from this? Briefly, massive learnings in lock-free data
structure design and implementation, lock-free garbage collection in C++, designing
memory-efficient lock-free constant-time snapshots, multi-threaded debugging and logging.

References:

A.  Bundled references: an abstraction for highly-concurrent linearizable range queries
B.  Constant-time snapshots with applications to concurrent data structures
C.  EEMARQ: Efficient Lock-Free Range Queries with Memory Reclamation

6.  Non-scalability of lock-free data structures

– During our research in the development and design of fancy lock-free data structures, we
noticed a rather peculiar problem.

– Let’s say you are using a multi-threaded environment, with say, 50 threads. Now, let’s say
you have some lock-free data structure, a set. You perform various CRUD operations on the
set. Let’s say, the throughput of this set for 1 thread is 1 million operations per second.

– Now, you create 50 threads, where each thread has its own independent set. We run the
same CRUD operations on each thread in parallel. What do you think the throughput should
be? 50 million, right? NO!! The throughput remains around 1 million! Very very strange,
isn’t it?

– This project deals with a thorough investigation of the non-scalability of independent lock-
free data structures.

– What can you expect to learn from this? Briefly, massive learnings in lock-free data
structure implementations, cache level performance and optimisations, RAM level
performance and optimisations, multi-threaded debugging and logging. You will also learn to
profile memory/cache accesses for a highly multi-threaded program, which in turn, you will
investigate and analyse to optimise every memory access.

7.  Irregular-Memory-Accesses based Loop-Fusion Generalization to Sparse-Polyhedral-

Framework

Description: The scope of this work extends to analyzing the loop-dependencies when memory
accesses are irregular. We have a variation of the loop-fusion pattern which is giving potential
performance and convergence improvements for the PageRank algorithm. In this work, we target
to generalize our fusion pattern to the existing Inspector-Executor model.
Inspector-Executor model analyzes the data dependencies when memory accesses are irregular.
Requirements:

I.  Good understanding of Computational-Geometry and Parallelization concepts
II.  Reasonable experience in Posix/OpenMP multithreaded interface
III.  Good understanding in runtime analysis of programs

References:

A.  http://www.cs.utah.edu/~payalgn/cs6967/pictures/prj1/Design-poster-v2.pdf
B.  https://ruc.udc.es/dspace/bitstream/handle/2183/22436/M.Arenaz_An_Inspector-

Executor_Algorithm_for_Irregular_Assignment_Parallelization_2004.pdf?sequence=2&i
sAllowed=y

C.  http://www.cs.uoregon.edu/research/paraducks/papers/icpp13/icpp13_inspector.pdf
D.  https://ieeexplore.ieee.org/abstract/document/9529750
E.  https://ieeexplore.ieee.org/abstract/document/9460492
F.  https://www.semanticscholar.org/paper/An-Inspector-Executor-Algorithm-for-Irregular-

Arenaz-Touri%C3%B1o/bae9dcbdb5fd75fc8c34b2c6b8ef4de96dde3aa1

G.  https://dl.acm.org/doi/abs/10.1145/2464996.2467282
H.  https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.572.818&rep=rep1&type=pdf
I.  Some articles published by this author on Sparse-Polyhedral-Framework is interesting
https://dblp.org/pid/06/29.html

8.  Scalability in Sequential-Quadratic-Programming
Description:  SQP  is  an  iterative  technique  for  solving  nonlinear  programming  problems.  The
existing  C Library for solving SQP is suffering from scalability issues. For small problems, the
library  is  taking  less  time,  whereas  for  larger  problems,  the  time  complexity  is  increasing

exponentially. The scope of this project is to change the existing design of the SQP program in
such  a  way  that  the  redundant  operations  are  eliminated,  the  code  is  pipelined  easily,  set  of
independent instructions can be executed in parallel.

Requirements:

I.  Good understanding of Non-Linear programming
II.  Expertise in C programming language
III.  Reasonable experience in Posix/OpenMP multithreading interface
IV.  Brief understanding of Sparse Matrix operations

References:

A.  https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#:~:text=on%20each%20iteratio

n.-,SLSQP,-Specified%20in%20NLopt

B.  http://degenerateconic.com/uploads/2018/03/DFVLR_FB_88_28.pdf
C.  https://github.com/stevengj/nlopt/tree/master/src/algs/slsqp

9.  Complexity Estimation of Gauss-Seidel-Multivariate-Linear Stochastic-Gradient-

Descent Lock-free algorithm

Description: The Gauss-Seidel technique is proven to show improvement in the convergence of
iterative  algorithms.  The  scope  of  this  work  extends  to  the  Design  and  Analysis  of  the  lock-
freedom property on the Linear Optimization Technique. This analysis will be helpful in solving
nonlinear  programming  problems.  The  main  objective  of  this  work  is  to  understand  the  time
complexity of the Gauss-Seidel method using concurrent data structures.

Requirements:

I.  Good understanding in Non-Linear-programming and Approximation-techniques
II.  Expertise in C++ programming language and the STL Library
III.  Reasonable experience in Posix/OpenMP multithreaded interface and concurrent data-

structures

IV.  Brief understanding of Sparse Matrix operations

References:

A.  https://arxiv.org/pdf/1106.5730.pdf
B.  https://docs.google.com/presentation/d/1SUOVua4SML1NYabD7lFPxKpBw5jsQlao/edit

#slide=id.p1

10. Parallelizing Singular Value Decomposition
Description: Singular value decomposition (SVD) is a fundamental operation in linear algebra.
SVD  is  regularly  used  in  several  applications,  especially  for  low-rank  approximation  and
dimensionality reduction.

SVD  computation  is  expensive  in  general.  However,  efficient  randomized  approximation
algorithms  exist,  especially  when  the  target  dimensions  are  much  smaller  than  the  input
dimensions. Such a setting is called truncated SVD, where the target dimension is smaller than the
input dimension. The goal is to build a highly efficient, scalable, and parallel implementation for
truncated SVD that can scale well to millions of vectors with input dimensions close to 100,000
or more. Such dataset volumes are regularly analyzed in social networks, genomics, etc.

There are some publicly available libraries (see References) for efficient computation of truncated
SVD.  However,  they  have  limited  scalability  on  usual  desktop  configurations.  These  could  be
candidates  for  benchmarking  your  implementation.  If  there  are  other  efficient  implementations
available, you can benchmark against them as well.

References:

A.  https://scikit-

learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
B.  https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
C.  https://towardsdatascience.com/turbocharging-svd-with-jax-749ae12f93af
D.  https://pytorch.org/docs/1.8.0/linalg.html
E.  https://arxiv.org/abs/0909.4061
F.  https://onlinelibrary.wiley.com/doi/abs/10.1002/nla.2246
G.  https://arxiv.org/abs/1412.3510
H.  https://www.sciencedirect.com/science/article/pii/S1063520310000242
I.  https://research.facebook.com/blog/2014/09/fast-randomized-svd/

11. Wait-free multi-threaded implementation of sFCM image segmentation
Description:  Image  segmentation  has  proven  its  vital  importance  in  many  image  processing
applications. It converts the input image into several disjoint regions. For efficient segmentation,
many techniques have been proposed in the literature, such as edge, threshold, region, watershed,
clustering,  partial  differential  equation,  and  neural  network-based.  Further,  to  improve  the
segmentation effectiveness,  several  attempts have been made using a fuzzy set,  active contour,
level set, and k-means clustering. Fuzzy c-means (FCM) clustering is a prevalent method of image
segmentation for regions of interest (ROI) information from magnetic resonance imaging (MRI)
images.  However,  the  FCM  algorithm  does  not  fully  utilize  the  spatial  (neighbourhood  pixel)
information for segmentation. The inclusion of special information in the FCM (sFCM) renders
several  advantages for apparent reasons, such as the algorithm becomes less sensitive to noise,
helps to reduce spurious blobs, takes away noisy spots, and tends to produce more homogeneous
regions in the segmented image. However, scientists suggest that performance is also one of the
core problems in using sFCM.
There have been various implementations in literature to improve the performance of standard
sFCM algorithms using GPGPUs. However, there is no much work in improving the

performance for multi-core systems. In this work, we will focus on developing a multithreaded
wait-free implementation of the sFCM algorithm.

References:

A.  Chuang KS, Tzeng HL, Chen S, Wu J, Chen TJ. Fuzzy c-means clustering with spatial
information  for image segmentation. computerized medical  imaging  and graphics. 2006
Jan 1;30(1):9-15.
PDF:
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.96.8124&rep=rep1&type=pdf

B.  Ravi A., Suvarna A., D’Souza A., Ram Mohana Reddy G., Megha (2013) A Parallel Fuzzy
C  Means  Algorithm  for  Brain  Tumor  Segmentation  on  Multiple  MRI  Images.  In
Proceedings of International Conference on Advances in Computing, vol 174. Springer,
New Delhi.

C.  Cherradi  B,  Ali  NA,  Abbassi  AE,  Bouattane  O,  Youssfi  M.  CPU  and  GPU  Behaviour
Modelling Versus Sequential and Parallel Bias Field Correction Fuzzy C-means Algorithm
Implementations. Contemp Eng Sci. 2017;10:423-38.
PDF:  http://www.m-hikari.com/ces/ces2017/ces9-12-2017/p/cherradiCES9-12-2017.pdf

12. Smart Validator for Transaction Validation in Blockchain

Blockchain is a digital ledger that records transactions in a secure and unchangeable way. It's
like a chain of connected blocks where each block contains data, and once data is added, it's
virtually  impossible  to  alter,  making  it  great  for  secure  online  transactions  and  data
management.
Within  the  blockchain,  transactions  must  undergo  validation  by  nodes  before  receiving
confirmation. This validation process guarantees the transaction's authenticity and inclusion in
the unalterable chain of blocks, ensuring its security and resistance to tampering.
The  term  Smart  Validator  in  blockchain  refers  to  the  use  of  intelligent  techniques  or
mechanisms for verifying and validating transactions or data within the blockchain network.
The Smart Validator utilizes information from miners to efficiently shard transactions. This
process reduces the complexity of transaction sharding, resulting in faster validation.
The  Smart  Validator  plays  a  pivotal  role  in  optimizing  the  validation  process  within  a
blockchain network. It achieves this by efficiently sharding the transactions by utilizing the
information from miners. This results in improved transaction throughput and overall network
performance.
The  primary  objective  of  this  project  is  to  design  and  implement  a  Smart  Validator  for
blockchain systems to enhance the efficiency, speed, and accuracy of transaction validation
processes.
Requirements- Basic knowledge of Blockchain, C++, and Threads

13. Parallel Conflict Resolution Algorithm for Blockchain Transaction Validation

Description: Blockchain technology has gained significant prominence in various industries for
its secure and decentralized nature. However, the validation of transactions within a blockchain
network  remains  a  critical  challenge.  This  project  aims  to  develop  a  parallel  algorithm  for
validating  blockchain  transactions  efficiently.  The  primary  objective  is  to  identify  conflicts
between transactions and group them in an order that ensures independence among transactions
within each group.

Objectives:

1.  Conflict Identification:

-  Develop algorithms to identify conflicts between transactions within a blockchain.

2.  Transaction Grouping:

-  Group transactions based on the identified conflicts.
-  Ensure that transactions within each group are independent of each other.
-  Maintaining a specific order of grouping.

3.  Data Item Tracking:

-
-

 Create and maintain a list of data items accessed by transactions within each group.
 Ensure efficient data item tracking for conflict resolution.

Requirements- Basic knowledge of Blockchain, C++, and Threads

14. Developing a Parallel Fast Fourier Transformation Algorithm

Description:  In  this  course  project,  we  will  look  into  the  world  of  Fourier  transformations,
specifically the Fast Fourier Transformation (FFT) algorithm.

The Discrete Fourier Transformation (DFT) is a specialized form of Fourier transformation that
transforms  a  time-based  sequence  into  a  frequency-based  sequence.  However,  performing  a
straightforward  DFT  computation  can  be  computationally  intensive,  with  a  time  complexity  of
O(N^2).

In  1955,  Cooley  and  Tukey  introduced  the  Fast  Fourier  Transform  (FFT)  algorithm,  which
significantly improves the efficiency of DFT calculations by achieving a time complexity of O(N
log N). The FFT employs a divide-and-conquer approach to compute DFT values and can also be
used to evaluate polynomials of degree n at the nth complex roots of unity, enabling polynomial
multiplication in O(N log N) time.

For this course project, your task is to create a parallelized version of the FFT algorithm using
pthreads or C++ 11 threads. Additionally, you will investigate and identify other existing parallel
algorithms  or  libraries  related  to  FFT  computations.  Subsequently,  you  will  conduct  a
comprehensive  performance  analysis,  comparing  your  parallel  implementation  with  both
sequential counterparts and alternative parallel solutions. This project will provide you with a deep
understanding of parallel computing and its impact on FFT algorithm efficiency.

Requirements:- Knowledge of the FFT algorithm and threading in C/C++.

15. Parallelization  of  the  BLAST  algorithm  for  faster  processing,  OR  Moderation  of  the

Diamond BLAST algorithm to allow regular saving of the output data

Introduction: The Basic Local Alignment Search Tool (BLAST) is a widely used bioinformatics
program  and  algorithm  for  comparing  biological  sequences,  such  as  DNA,  RNA,  or  protein
sequences, against a vast database of sequences to find regions of similarity or homology. BLAST
is  a  fundamental  tool  in  genomics,  molecular  biology,  and  bioinformatics  and  is  employed  for
various purposes, including gene identification, sequence annotation, and evolutionary analysis. It
works  based  on  the  Smith–Waterman  algorithm,  which  is  a  rigorous  dynamic  programming
method for finding optimal local alignments.
Parallelization  of  BLAST  is  used  to  accelerate  searching  for  similarities  between  biological
sequences, such as DNA or protein sequences, in large databases. Parallelization distributes the
computational workload across multiple processors, nodes, or cores, allowing BLAST to run faster
and  analyze  larger  datasets  efficiently.  The  BLAST  algorithm  has  already  been  complemented
with the DIAMOND tool for parallel processing; however, with our large dataset, it still needs to
be increased. Another issue faced while working with DIAMOND is that it produces the output in
the  end,  and  in  this  manner,  the  RAM  size  keeps  increasing,  and  in  the  long  run  (with  our
resources),  it  gets  terminated.  To  decrease  the  RAM  load,  we  would  like  to  moderate  the
DIAMOND algorithm so that it can keep saving the output similar to BLAST. This project will
explore two aspects:

1)  How  can  we  change  the  BLAST  algorithm  to  perform  in  a  better  parallel  processing

manner?

2)  How can we moderate the Diamond BLAST technique to allow regular saving of the output

data?

References:

A.  https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/
B.  https://blast.ncbi.nlm.nih.gov/Blast.cgi
C.  https://blast.ncbi.nlm.nih.gov/doc/blast-help/
D.  https://github.com/bbuchfink/diamond

16. GPU based Parallel algorithm for image processing for a phone device

Description: Modern smartphone applications increasingly rely on real-time image processing
for  tasks  such  as  enhancement,  object  detection,  augmented  reality,  and  computer  vision.
However, traditional CPU-based implementations often fail to meet performance and energy-
efficiency requirements due to the computational intensity of image processing algorithms.

This  project  aims  to  design  and  implement  GPU-based  parallel  algorithms  for  image
processing  on  phone  devices,  leveraging  the  massive  parallelism  of  mobile  GPUs.  By
exploiting  data-level  parallelism  inherent  in  image  operations,  the  project  focuses  on
accelerating key image processing tasks while minimizing latency and power consumption.
The  proposed  solution  will  utilize  mobile  GPU  programming  frameworks  (such  as
CUDA/OpenCL/Vulkan  Compute  or  equivalent)  to  achieve  significant  performance
improvements over CPU-based approaches, enabling efficient and scalable image processing
for next-generation mobile applications.

17. Studying the state-of-the-art papers and implementing an unimplemented application:

1.  Study Ligra/Ligra+ and implement the capability to add/remove the vertex.
2.  Study Stinger and implement SSSP.
3.  Study GraphOne and implement SSSP.
4.  Study Debra and compare the performance of version linked list with and without

garbage collection.

5.  Implement the LF Hash (PODC’14).
6.  Implement the WF Hash (PODC’14).
7.  Implement the 5 exercises mentioned here: https://gitlab.mpi-sws.org/iris/tutorial-popl21/

and explain each of them thoroughly. Additionally, modify the 5th exercise to
parallel_multiplication_division.

8.  After discussing with me, select one of the many algorithms in

https://arxiv.org/pdf/1712.05406.pdf thesis. Implement and do a thorough analysis. Also,
one application like full snapshot, partial snapshot etc.

9.  Explore whether this can be implemented in a parallel setting, if yes, inform me and
proceed. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10177392
(git repo in the paper)

10. Refer https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10177434

(their site https://anydsl.github.io/anyq/ ).
Explain and compare MSQ (lock free) and YMCQ (wait free) performance.

a.  [OPEN] You can refer https://www.ipdps.org/archives.htm and select papers that
align with our course, discuss with me and proceed. Select the papers with code.


