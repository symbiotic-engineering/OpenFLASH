# Final fitted formulas
> This file is generated from `convergence_results.ipynb`.

The number of terms needed $N^{i_m}$ needed in region $i_m$ to achieve an
error $\epsilon$ is modeled as $N^{i_m} = \beta^{i_m}\epsilon^{-1/\alpha^{i_m}}$.
There is one such formula for added mass (subscript $A$) and one for radiation damping (subscript $B$),
so the recommended number of terms for region $i_m$ is $\max(N^{i_m}_A, N^{i_m}_B)$.

### Results

Added mass in the innermost (m=1) region was fitted using the following models:

$$\begin{aligned}
\alpha^{i_1}_{A} &= c_1 + \begin{cases}
    c_2 \ln(\frac{h-d_1}{a_1} + c_3), &\frac{h-d_2}{h-d_1} < 1\\
    c_4 \ln(\frac{h-d_1}{a_1} + c_5), &\frac{h-d_2}{h-d_1} \geq 1
\end{cases}\\
\beta^{i_1}_{A} &= \frac{h-d_1}{a_1} \cdot \bigg(\begin{cases}
    \frac{c_6}{1+e^{c_7(1-\frac{h-d_2}{h-d_1})}} + c_8, &\frac{h-d_2}{h-d_1} < 1\\
    c_9, &\frac{h-d_2}{h-d_1} \geq 1
\end{cases}\bigg)
\end{aligned}$$

and found the following values:

$$\begin{aligned}
\alpha^{i_1}_{A} &= 0.287 + \begin{cases}
    0.434 \ln(\frac{h-d_1}{a_1} + 7.44), &\frac{h-d_2}{h-d_1} < 1\\
    0.3 \ln(\frac{h-d_1}{a_1} + 81), &\frac{h-d_2}{h-d_1} \geq 1
\end{cases}\\
\beta^{i_1}_{A} &= \frac{h-d_1}{a_1} \cdot \bigg(\begin{cases}
    \frac{0.172}{1+e^{6.3(1-\frac{h-d_2}{h-d_1})}} + 0.0356, &\frac{h-d_2}{h-d_1} < 1\\
    0.259, &\frac{h-d_2}{h-d_1} \geq 1
\end{cases}\bigg)
\end{aligned}$$

Added mass in the middle regions ($2\leq m < M$) was fitted using the following models:

$$\begin{aligned}
\alpha^{i_m}_{A} &= \bigg(\begin{cases}
    c_1, &\frac{h-d_{m+1}}{h-d_m} < 1\\
    c_2, &\frac{h-d_{m+1}}{h-d_m} \geq 1
\end{cases}\bigg) \cdot 
\bigg(\begin{cases}
    c_3, &\frac{h-d_{m-1}}{h-d_m} < 1\\
    c_4, &\frac{h-d_{m-1}}{h-d_m} \geq 1
\end{cases}\bigg) \cdot
\ln(\frac{h-d_m}{a_{m+1}-a_m}) + c_5\\

\beta^{i_m}_{A} &= \frac{h-d_m}{a_{m+1}-a_m} \cdot \bigg(\begin{cases}
    \frac{c_6}{1+e^{c_7(1-\frac{h-d_2}{h-d_1})}} + c_8, &\frac{h-d_{m+1}}{h-d_m} < 1\\
    c_{9}, &\frac{h-d_{m+1}}{h-d_m} \geq 1
\end{cases}\bigg)\cdot 
\bigg(\begin{cases}
    c_{10}, &\frac{h-d_{m-1}}{h-d_m} < 1\\
    c_{11}, &\frac{h-d_{m-1}}{h-d_m} \geq 1
\end{cases}\bigg)
\end{aligned}$$

and found the following values:

$$\begin{aligned}
\alpha^{i_m}_{A} &= \bigg(\begin{cases}
    1.27, &\frac{h-d_{m+1}}{h-d_m} < 1\\
    0.627, &\frac{h-d_{m+1}}{h-d_m} \geq 1
\end{cases}\bigg) \cdot 
\bigg(\begin{cases}
    0.133, &\frac{h-d_{m-1}}{h-d_m} < 1\\
    0.204, &\frac{h-d_{m-1}}{h-d_m} \geq 1
\end{cases}\bigg) \cdot
\ln(\frac{h-d_m}{a_m - a_{m-1}}) + 1.21\\

\beta^{i_m}_{A} &= \frac{h-d_m}{a_m-a_{m-1}} \cdot \bigg(\begin{cases}
    \frac{0.35}{1+e^{7.04(1-\frac{h-d_{m+1}}{h-d_m})}} + 0.113, &\frac{h-d_{m+1}}{h-d_m} < 1\\
    0.335, &\frac{h-d_{m+1}}{h-d_m} \geq 1
\end{cases}\bigg)\cdot 
\bigg(\begin{cases}
    0.378, &\frac{h-d_{m-1}}{h-d_m} < 1\\
    0.41, &\frac{h-d_{m-1}}{h-d_m} \geq 1
\end{cases}\bigg)
\end{aligned}$$

Added mass in the outermost region ($m=M$) was fitted using the following models:

$$\begin{aligned}
\alpha^{i_M}_{A} &= \bigg(\begin{cases}
    c_1, &\frac{h-d_{M-1}}{h-d_M} < 1\\
    c_2, &\frac{h-d_{M-1}}{h-d_M} \geq 1
\end{cases}\bigg) \cdot
(\ln(\frac{h}{h-d_M} + c_3) + c_4) \cdot 
(\ln(\frac{h-d_M}{a_M-a_{M-1}} + c_5) + c_6)\\

\beta^{i_M}_{A} &= \frac{h-d_M}{a_M-a_{M-1}} \cdot
\bigg(\begin{cases}
    c_7, &\frac{h-d_{M-1}}{h-d_M} < 1\\
    c_8, &\frac{h-d_{M-1}}{h-d_M} \geq 1
\end{cases}\bigg) \cdot
(\ln(\frac{h}{h-d_M} + c9) + c_10) \cdot
(\frac{a_M-a_{M-1}}{a_{M-1}} + c_11)
\end{aligned}$$

and found the following values:

$$\begin{aligned}
\alpha^{i_M}_{A} &= \bigg(\begin{cases}
    0.0184, &\frac{h-d_{M-1}}{h-d_M} < 1\\
    0.0185, &\frac{h-d_{M-1}}{h-d_M} \geq 1
\end{cases}\bigg) \cdot
(\ln(\frac{h}{h-d_M} -0.312) + 8.65) \cdot 
(\ln(\frac{h-d_M}{a_M-a_{M-1}} + 8.85) + 6.53)\\

\beta^{i_M}_{A} &= \frac{h-d_M}{a_M-a_{M-1}} \cdot
\bigg(\begin{cases}
    0.000444, &\frac{h-d_{M-1}}{h-d_M} < 1\\
    0.000563, &\frac{h-d_{M-1}}{h-d_M} \geq 1
\end{cases}\bigg) \cdot
(\ln(\frac{h}{h-d_M} -0.887) + 26.9) \cdot
(\frac{a_M-a_{M-1}}{a_{M-1}} + 13.1)
\end{aligned}$$

Radiation damping in the outermost region ($m=M$) was fitted using the following models:

$$\begin{aligned}
\alpha^{i_M}_{B} &= c_1 \cdot
(\ln(\frac{h}{h-d_M} + c_2 ) + c_3) \cdot
(e^{-c_4 \cdot \lambda_0^e h} + c_5) \\

\beta^{i_M}_{B} &= 
(c_6\frac{h-d_M}{a_M-a_{M-1}} + c_7) \cdot
\bigg(\begin{cases}
    \frac{\lambda_0^e h}{\sqrt{5}}, &\lambda_0^e h < 5\\
    \sqrt{\lambda_0^e h}, &\lambda_0^e h \geq 5
\end{cases}\bigg) \cdot
(e^{-c_8 \cdot \frac{h}{h-d_M}})
\end{aligned}$$

and found the following values:

$$\begin{aligned}
\alpha^{i_M}_{B} &= 0.0977 \cdot
(\ln(\frac{h}{h-d_M} + 0.208 ) + 2.4) \cdot
(e^{-0.0302 \cdot \lambda_0^e h} + 4.35) \\

\beta^{i_M}_{B} &= 
(0.0375\frac{h-d_M}{a_M-a_{M-1}} + 0.00911) \cdot
\bigg(\begin{cases}
    \frac{\lambda_0^e h}{\sqrt{5}}, &\lambda_0^e h < 5\\
    \sqrt{\lambda_0^e h}, &\lambda_0^e h \geq 5
\end{cases}\bigg) \cdot
(e^{-0.548 \cdot \frac{h}{h-d_M}})
\end{aligned}$$


