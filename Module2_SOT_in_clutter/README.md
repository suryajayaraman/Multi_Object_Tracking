## Module2

## Single object tracking in clutter
1. Inclusion of object detection (Pd) in hypothesis
2. Clutter models
3. Bernoulli, Binomial and Poission distribution
4. Exact Motion and measurement models for SOT in clutter
5. Cons of actual model and ways to approximate solution
6. NN  - Algorithm, pros and cons
7. PDA - Algorithm, pros and cons
8. GSF - Algorithm, pros and cons
9. Practical approach - Gating, pruning and merging methods

### Simulation setup

1. Objective is to track a single object using measurements from sensor readings.

2. Coordinated turn motion model (object assumed to undergo constant angular 
   velocity motion)

3. Bearing - range measurement model (~ radar sensor readings)

4. Constant Detection probability (P_D) and clutter rate intensity within the
   sensor field of view
 
The NN, PDA and GSF algorithms are compared in SOT case with coordinated

### Comparison results

#### Success case

In simple scenarios with higher probability of detection and less clutter rate,
there is little difference between the three algorithms in terms of tracking 
accuracy. 

![Success_case_SOT](SOT_animation/SOT_Success_cases_comparison.png)


#### Success case

In scenarios where the Nearest Neighbour algorithm assumes clutter as possible
measurement and loses track, it is very difficult for the algorithm to converge
again. Highlighted in the results below.
 
![Failure_case_SOT](SOT_animation/SOT_Failure_cases_comparison.png)
