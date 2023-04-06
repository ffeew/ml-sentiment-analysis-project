## Second-Order HMM Model

The second-order model will require $N$, the number of tags, $M$, the number of words in the system, $A = \{a_{ij}\}$, the transition probability distribution, $B = \{b_{j}(k)\}$, the emission probability distribution, and $\pi = \{{\pi_i}\}$, the probability that the model starts in state $i$.

<br>

**How to train your model under the new assumptions?**

Since transition probabilities must now include 2 previous states, $q(y_i|y_{i-2}, y_{i-1})$, the Viterbi Algorithm for Second-Order HMM is:
> Initialisation 
$\pi(-1, t) =  
\begin{cases}
1, \text{if t = START} \\
0, otherwise.
\end{cases}$
$\pi(0, u) =  
\begin{cases}
1, \text{if u = START} \\
0, otherwise.
\end{cases}$
For j = 0, ..., n - 1:
$\quad\text{For each } v \in \Gamma :$
$\quad\quad\pi(j + 1, v) = max_{t, u\in\Gamma}\{\pi(j - 1,t) \times \pi(j, u)\times a_{t,u,v}\times b_v(x_{j+1})\}$
Final Step
$\pi(n+1,\text{STOP}) = max_{t,u\in\Gamma}\{\pi(n - 1, t)\times\pi(n,u)\times a_{t,u,\text{STOP}}\}$
Backtracking
$y_n^* = \text{argmax}_{t, u\in\Gamma}\{\pi(n - 1, t)\times\pi(n,u)\times a_{t, u,\text{STOP}}\}$
For j = n - 1, ..., 1:
$\quad y_j^* = \text{argmax}_{t, u\in\Gamma}\{\pi(j - 1,t)\times\pi(j,u)\times a_{t,u,y_{j+1}^*}\}$

<br>

**How to learn the new transition parameters?**

The new transition parameters can be learned by the equation from this [paper](https://aclanthology.org/P99-1023.pdf):
> $q(y_i|y_{i-2},y_{i-1}) = k_3\cdot\frac{N_3}{C_2} + (1-k_3)k_2\cdot\frac{N_2}{C_1}+(1-k_3)(1-k_2)\cdot\frac{N_1}{C_0}$
where
$N_1$ = number of times $y_i$ occurs
$N_2$ = number of times sequence $y_{i-1}y_i$ occurs
$N_3$ = number of times sequence $y_{i-2}y_{i-1}y_i$ occurs
$C_0$ = total number of tags that appear
$C_1$ = number of times $y_{i-1}$ occurs
$C_2$ = number of times sequence $y_{i-2}y_{i-1}$ occurs
$k_2 = \frac{log(N_2+1) + 1}{log(N_2+1)+2}$
$k_3 = \frac{log(N_3+1) + 1}{log(N_3+1)+2}$

<br>

**Any changes to the emission parameters?**

Since the emission parameters rely only on the current hidden state, there is no change in the emission parameters from the First-Order HMM to the Second-Order HMM. Hence, we can use the function from part 1 to estimate these emission parameters.

<br>

**What will be the time complexity of the new Viterbi algorithm?**

The time complexity of the new Viterbi algorithm per part of the algorithm is $O(n)$ for the for loop from j = 0 to n - 1, $O(|\Gamma|)$ for the for loop over each possible tag $v \in \Gamma$, $O(|\Gamma|)$ for the argument giving the maximum value for $u\in\Gamma$, and $O(|\Gamma|)$ for the argument giving the maximum value for $t\in\Gamma$. 

Hence, the overall time complexity is $O(n|\Gamma^3|)$.