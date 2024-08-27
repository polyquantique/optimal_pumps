This repo is used to find the optimal parameters of the pump laser to get high purity quantum light by SPDC process. This project is based on Jax.
To use Jax, Jaxlib as well as Jax need to be downloaded. A way to download Jax for windows is to use the [unstable build](https://github.com/cloudhan/jax-windows-builder). For Linux and MacOS users, please refer to installation section [here](https://github.com/google/jax).

# Theory of high-gain spontaneous parametric down-conversion

The numerical optimization model is from the paper by _Quesada et al_ **Phys. Rev. A**, (2020). The dynamics describing SPDC can be seen as 
```math
\frac{\partial }{\partial z}a_S(z, \omega) = \text{i}\omega \Delta k_S a_S(z, \omega) + \frac{\text{i}\gamma (z)}{\sqrt{2\pi}}\int \beta (\omega + \omega ')a_I^\dagger (z, \omega ')d\omega ',
```
```math
\frac{\partial }{\partial z}a_I^\dagger(z, \omega) = -\text{i}\omega \Delta k_I a_I ^\dagger(z, \omega) - \frac{\text{i}\gamma ^* (z)}{\sqrt{2\pi}}\int \beta ^* (\omega + \omega ')a_S(z, \omega),
```
where $\Delta k_j = \frac{1}{v_j} - \frac{1}{v_p}$, where $v_j$ is the group velocity of the mode $j \in \{S, I\}$, $\omega$ the frequency, $\beta(\omega)$ the slowly varying envelope of the pump in frequency domain, $\gamma (z)$ a tophat function associated to second order nonlinearity and $a_j (z, \omega)$ the annihilation operator for the mode $j$. The equations can be described by discretization into normal modes as

```math
\frac{\partial}{\partial z}\begin{bmatrix}
\bar{a}_S(z, \omega)\\
\bar{a}_I^\dagger(z, \omega)
\end{bmatrix}=
\text{i}\begin{bmatrix}
\Delta k_S \omega & \frac{\text{i}\gamma}{\sqrt{2\pi}}\beta \\
-\frac{\text{i}\gamma ^*}{\sqrt{2\pi}}\beta ^\dagger & -\Delta k_I \omega
\end{bmatrix}
\begin{bmatrix}
\bar{a}_S(z, \omega')\\
\bar{a}_I^\dagger(z, \omega')
\end{bmatrix} = \text{i}A\begin{bmatrix}
\bar{a}_S(z, \omega')\\
\bar{a}_I^\dagger(z, \omega')
\end{bmatrix} ,
```
