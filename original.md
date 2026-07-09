\icmltitlerunning{Every Step a Thought: Implicit Visual Reasoning in  Diffusion Language Models}

\begin{document}

\twocolumn[
  \icmltitle{Every Step a Thought: Implicit Visual Reasoning in  Diffusion \\ Language Models}

  % It is OKAY to include author information, even for blind submissions: the
  % style file will automatically remove it for you unless you've provided
  % the [accepted] option to the icml2026 package.

  % List of affiliations: The first argument should be a (short) identifier you
  % will use later to specify author affiliations Academic affiliations
  % should list Department, University, City, Region, Country Industry
  % affiliations should list Company, City, Region, Country

  % You can specify symbols, otherwise they are numbered in order. Ideally, you
  % should not use this facility. Affiliations will be numbered in order of
  % appearance and this is the preferred way.
  \icmlsetsymbol{equal}{*}

% -----------------------------------------
  \begin{icmlauthorlist}
    \icmlauthor{Zitong Wang}{sysu}
    \icmlauthor{Haohao Xu}{tianjinsch}
    \icmlauthor{Zijun Shen}{nanjingsch}
    \icmlauthor{Firstname4 Lastname4}{sch}
    \icmlauthor{Firstname5 Lastname5}{yyy}
    \icmlauthor{Firstname6 Lastname6}{sch,yyy,comp}
    \icmlauthor{Firstname7 Lastname7}{comp}
    %\icmlauthor{}{sch}
    \icmlauthor{Firstname8 Lastname8}{sch}
    \icmlauthor{Firstname8 Lastname8}{yyy,comp}
    %\icmlauthor{}{sch}
    %\icmlauthor{}{sch}
  \end{icmlauthorlist}

  \icmlaffiliation{sysu}{Sun yat-sen Universiy, Zhuhai, China}
  \icmlaffiliation{comp}{Company Name, Location, Country}
  \icmlaffiliation{tianjinsch}{Tianjin University, Tianjin, China}

  \icmlcorrespondingauthor{Firstname1 Lastname1}{first1.last1@xxx.edu}
  \icmlcorrespondingauthor{Haohao Xu}{first2.last2@www.uk}

  % You may provide any keywords that you find helpful for describing your
  % paper; these are used to populate the "keywords" metadata in the PDF but
  % will not be shown in the document
  \icmlkeywords{Machine Learning, ICML}

  \vskip 0.3in
]

% this must go after the closing bracket ] following \twocolumn[ ...

% This command actually creates the footnote in the first column listing the
% affiliations and the copyright notice. The command takes one argument, which
% is text to display at the start of the footnote. The \icmlEqualContribution
% command is standard text for equal contribution. Remove it (just {}) if you
% do not need this facility.

% Use ONE of the following lines. DO NOT remove the command.
% If you have no special notice, KEEP empty braces:
\printAffiliationsAndNotice{}  % no special notice (required even if empty)
% Or, if applicable, use the standard equal contribution text:
% \printAffiliationsAndNotice{\icmlEqualContribution}

\begin{abstract}
Multimodal diffusion language models perform generation through an iterative denoising process, providing an effective framework for visual reasoning. However, existing inference treats each denoising step independently and ignores cross-step information, while many visual reasoning improvements require extra training or external modules, increasing cost and complexity. In this work, We show that intermediate denoising steps contain visual reasoning signals, and accumulating them across steps improves prediction stability. We propose a training-free framework that improves visual reasoning by accumulating information across denoising steps. We formulate step-wise generation as a recursive estimation problem with step-dependent uncertainty and introduce an uncertainty-aware fusion mechanism that recursively aggregates step-wise logits. Our theoretical analysis shows that the proposed framework is equivalent to the closed-form solution of a global weighted least-squares objective, ensuring a monotonic reduction in reasoning uncertainty. Extensive experiments demonstrate that our framework significantly enhances the visual reasoning performance of multimodal diffusion models.
\end{abstract}

\section{Introduction}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.9\linewidth]{fig/intro.pdf}
    \caption{\textbf{Comparison of reasoning mechanism across generation paradigms.} AR models perform explicit CoT reasoning via token sequences; traditional DLMs independently utilize per-step logits; our approach accumulates logits across denoising steps with a recursive structure, enabling implicit reasoning during inference.}
    \label{fig:intro}
\end{figure}

Diffusion Language Models (DLMs) generate sequences through an iterative denoising process, where masked tokens are progressively refined toward coherent outputs. Recent work has shown that this framework can be extended to multimodal settings by representing text and visual inputs in a shared discrete token space \citep{li2022diffusionlmimprovescontrollabletext,austin2023structureddenoisingdiffusionmodels,arriola2025block,team2024chameleon}. In particular, mask-based multimodal DLMs recover masked text tokens conditioned on both visual inputs and surrounding context, enabling parallel generation and strong vision–language alignment \citep{you2025llada,yang2025mmada,yu2025dimple,wang2025fudoki,xin2025lumina,li2025lavida}. % 看这里有没有lavida，没有的话帮我加引用

Visual reasoning in vision–language models has traditionally been addressed through explicit mechanisms, including chain-of-thought prompting, reasoning pipelines, or external tools \citep{wei2022chain,yao2023tree,suris2023vipergpt,wu2024controlmllm}. However, they introduce additional computation and rely on sequential decoding. Recent work therefore explores latent reasoning, where reasoning is performed within model states instead of text generation \citep{hao2024training,schone2025implicit,shen2025codi}.  Importantly, diffusion models provide a natural setting for latent reasoning, as their generation process consists of multiple denoising steps that progressively refine internal representations \citep{ye2024diffusion}. Besides, most existing research on multimodal diffusion language models focuses on improving multimodal reasoning through reinforcement learning, new training objectives, or improved noise scheduling \citep{yang2025mmada,wang2025revolutionizing,li2025lavida,li2025lavida,shi2025muddit,tian2025mmada}. During inference, logits produced at each denoising step are used only to determine the next masked state and are then discarded. As a result, the denoising process does not utilize intermediate logits as reasoning signals, which limits the model’s ability to aggregate visual information across steps.
% 这里的引用要加：Lavida-O, Lavida, muddit, MMaDA
% 加：MMaDA-Parallel: Multimodal Large Diffusion Language Models for Thinking-Aware Editing and Generation


Despite this potential, it remains unclear how reasoning is expressed during diffusion inference. In particular, two questions remain unexplored:
\textit{(i) whether intermediate denoising steps in multimodal DLMs already encode meaningful visual reasoning signals}, and 
\textit{(ii) whether these signals should be combined across steps rather than used independently}. In this work, we analyze the inference behavior of MDLMs and find that logits produced at intermediate denoising steps are already closer to the answers, even far from the final step. However, these step-wise predictions vary noticeably across adjacent steps, making single-step decisions unstable. This suggests that more reliable predictions can be obtained by combining information from multiple denoising steps rather than relying on any single step.

Motivated by this view, we propose \textbf{R}ecursive\textbf{ V}isual \textbf{L}ogit \textbf{F}usion (\textbf{RVLF}), a training-free inference-time method for multimodal DLMs. As illustrated in \autoref{fig:intro}, RVLF maintains a running fused logit state for each masked position and recursively updates it using logits from  denoising steps. It allows visual evidence to be accumulated across the denoising trajectory while reducing step-wise noise. Besides, OUT Theoretical analysis demonstrates that our framework is equivalent to the closed-form solution of a global weighted least-squares objective.



We evaluate RVLF on entensive visual reasoning benchmarks. Experimental results show consistent improvements over standard diffusion inference and existing test-time scaling methods, while maintaining high inference efficiency. These results indicate that implicit visual reasoning is  present during diffusion denoising, and that properly aggregating intermediate signals is an effective way to enhance reasoning in multimodal DLMs.


Our contributions are summarized as follows:
\begin{itemize}[nosep]
 \item We show that intermediate denoising steps in multimodal DLMs contain useful but noisy visual reasoning signals, and that accumulating these signals across steps leads to more stable predictions.

     \item We propose Recursive Visual Logit Fusion, a simple and training-free inference-time method that aggregates step-wise logits using uncertainty-aware recursive updates to enable stable implicit visual reasoning.

    \item Experiments on extensive benchmarks demonstrate that the proposed method consistently improves performance with minimal impact on inference speed.
\end{itemize}








\section{Related Work}
\noindent \textbf{Diffusion Language Models.} Diffusion models, originally developed for continuous data modeling, have been successfully generalized to discrete sequence domains~\citep{austin2023structureddenoisingdiffusionmodels}, giving rise to Diffusion Language Models (DLMs)~\cite{li2022diffusionlmimprovescontrollabletext,arriola2025block}. In this work, we focus on mask-based discrete diffusion models, characterized by a forward process that progressively replaces tokens with a special [MASK] token and a reverse process designed to reconstruct the original tokens~\cite{you2025llada,yu2025dimple,yang2025mmada}. Recent advancements have explored the joint modeling of multimodal data—such as text and vision—within a unified discrete token space, employing a consistent masked modeling objective for training~\cite{wang2025fudoki,li2025dual,xin2025lumina,team2024chameleon}. While training methodologies for these models are relatively well-established, research into the inference phase remains limited. Existing multimodal discrete diffusion models typically rely on fixed generation or iterative denoising trajectories~\cite{song2025seed,tian2025mmada}, lacking the design of inference-time reasoning (or scaling) mechanisms to enhance performance during the test phase.

\noindent \textbf{Latent Reasoning.} Explicit reasoning methods like CoT perform reasoning in token space~\cite{wei2022chain,snell2024scaling,yao2023tree,xu2025llava}, which introduce significant computational overhead and are limited in capturing complex, spatially structured visual abstractions. So to pursue efficiency and expressiveness, more and more works focus on latent reasoning, which directly operates the model’s internal states and eliminates the need for external intervention~\cite{hao2024training,deng2024explicit,schone2025implicit,shen2025codi}. And compared to autoregressive models, DLMs naturally facilitate multi-step latent reasoning through their denoising process~\cite{ye2024diffusion}. However, most methods design dedicated training
frameworks that optimize latent representations to enhance the reasoning ability~\cite{yang2025mmada,wang2025revolutionizing}. These methods often incur prohibitive computational costs, face challenges in reward design, and result in less generalizable and flexible reasoning capabilities. To avoid this, we propose a training-free enhancement to the inference-time stepwise generation of multimodal DLMs, which achieves improved performance on visual grounding reasoning.

\noindent \textbf{Visual Reasoning.} Visual reasoning is typically achieved through training-phase mechanisms, such as vision-language alignment objectives, structured visual representations, or auxiliary visual components~\cite{man2025argus,liu2025visionreasoner,zheng2025deepeyes,hu2024visual,huang2025visualtoolagent,zhang2025thyme,chen2024enhancing}. While these approaches enable effective integration of visual information, they often couple reasoning capabilities with specific training configurations~\cite{chern2025thinking}. Some studies attempt to enhance visual consistency during the generation process. However, these efforts often rely heavily on introducing complex external tools or modules for correction~\cite{zhao2025unsupervised,liu2025vlm,suris2023vipergpt,wu2024controlmllm}. To propose a general and simple solution, our approach uniquely enhances the stepwise generation process of multimodal DLMs without any additional tools and successfully extends visual reasoning into the DLM framework.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\section{Motivation and Preliminary}

\begin{figure}[t]
    \centering
    \includegraphics[width=1\linewidth]{fig/snr_v2_1_main_comparison.pdf}
    \caption{\textbf{LPGT Trajectories.} Evolution of LPGT over denoising progress for single-step (control) vs. cumulative (experiment) logits: (a) example trajectories; (b) cross-sample mean ± std.}
    \label{fig:moti_1}
\end{figure}

Multimodal Diffusion Language Models (MDLMs) generate text through iterative denoising. At each step $t$, the model predicts logits $\ell_t$ to sample the next latent state $x_{t-1}$:
\begin{equation}
    x_{t-1} \leftarrow \text{Sample}(p_{\theta}(\cdot | x_t, \ell_t)).
\end{equation}
Standard inference treats logits as single-step predictions that are only used to determine which tokens to reveal next. Under this view, intermediate logits are often regarded as unstable transition states rather than meaningful representations. 

In this work, we examine whether intermediate denoising steps encode meaningful visual reasoning signals for masked tokens, and whether such signals should be accumulated across steps rather than treated independently.

\begin{tcolorbox}[
    colback=gray!20,
    colframe=black,
    boxrule=1pt,
    arc=2pt,
    width=\columnwidth,
    boxsep=4pt,
    left=6pt, right=6pt
]
\textbf{\textit{RQ1:}} \textit{Does each denoising step perform meaningful visual reasoning for masked tokens?}
\end{tcolorbox}

To address RQ1, we analyze the denoising trajectories of a fixed masked position corresponding to a known ground-truth (GT) answer token. At denoising step $t$, the model produces a vocabulary logit vector $\ell_t \in \mathbb{R}^{|\mathcal{V}|}$. We quantify the model’s preference for the GT token using the Logit Preference for Ground Truth (LPGT), defined as:
\begin{equation}
    \mathrm{LPGT}(t) = \ell_t[y_{\mathrm{gt}}] - \frac{1}{|\mathcal{V}|}\sum_{v=1}^{|\mathcal{V}|}\ell_t[v].
\end{equation}
\begin{figure}[t]
    \centering
    \includegraphics[width=1\linewidth]{fig/snr_v2_3_jitter.pdf}
    \caption{\textbf{Stability and Jitter Analysis.} \textit{(a)-(b)} Comparison of step-wise changes between accumulated and single-step strategies; accumulation effectively suppresses sharp fluctuations. \textit{(c)-(d)} Quantitative distribution of the Jitter metric shows a shift toward zero, achieving a $94.7\%$ average reduction.}
    \label{fig:moti_3}
\end{figure}
\begin{figure}[t]
    \centering
    \includegraphics[width=1\linewidth]{fig/snr_v2_4_position_trajectories.pdf}
    \caption{Sample Trajectory Visualization. Visualizations reveal that step-local estimates suffer from noise and occasional spikes; cross-step accumulation filters this transient noise to yield consistent preference trajectories.}
    \label{fig:fig4}
\end{figure}
We conduct a controlled analysis on CVBench~\cite{tong2024cambrian}, index the trajectories by normalized denoising progress $\tau\in[0,1]$, and evaluate only the answer-token positions. As shown in \autoref{fig:moti_1}, LPGT maintains a non-degenerate magnitude across a broad range of intermediate $\tau$, rather than emerging only at the last few steps. Therefore, intermediate logits encode information conditioned on the visual input and textual context, rather than being dominated by random noise. 

However, despite containing valid signals, these single-step trajectories exhibit significant volatility between steps. To quantify this, we define a \textit{jitter} as the mean absolute difference between LPGT values at adjacent steps:
\begin{equation}
    \mathrm{Jitter}
=
\frac{1}{T-1}
\sum_{t=1}^{T-1}
\left|
\mathrm{LPGT}(t+1) - \mathrm{LPGT}(t)
\right|.
\end{equation}

\begin{tcolorbox}[
    colback=gray!20,
    colframe=black,
    boxrule=1pt,
    arc=2pt,
    width=\columnwidth,
    boxsep=4pt,
    left=6pt, right=6pt
]
\textbf{\textit{RQ2:}} \textit{Should single-step reasoning signals be accumulated across denoising steps to reduce noise and improve stability?}
\end{tcolorbox}

To isolate the effect of accumulation, we set:
\begin{itemize}[nosep,left=0pt]
    \item The control group: computes LPGT with logits $\ell_t$.
    \item The experimental group: substitutes $\bar{\ell}_{1:t}=\frac{1}{t}\sum_{i=1}^{t}\ell_i$ into the LPGT computation.
\end{itemize}


As shown in \autoref{fig:moti_3} and \autoref{fig:fig4}, results demonstrate that accumulation yields smoother trajectories and shifts the Jitter distribution toward lower values, achieving a $94.7\%$ reduction. To localize the divergence, we define the bin-wise improvement:
\begin{equation}
\Delta(\tau)=\mu_{\mathrm{exp}}(\tau)-\mu_{\mathrm{ctrl}}(\tau),
\end{equation}
where $\mu(\tau)$ denotes the cross-sample mean LPGT within a $\tau$ bin. As shown in \autoref{fig:moti_heatmap}, these accumulation gains exhibit heterogeneity across samples and timesteps. To be brief, we find that while intermediate logits are not purely noise (RQ1), single-step estimates suffer from instability; cross-step accumulation effectively mitigates this volatility (RQ2), thereby motivating the recursive visual logit fusion in our inference approach.


% Note: Ensure you include the correct heatmap figure file here, or reference Fig 6 if defined elsewhere.


















%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%---------------- METHOD -----------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%

\section{Methodology}







\subsection{Modeling Visual Reasoning Trajectories}
Given an image $I$ and a text context $x$, a diffusion language model generates tokens by running $T$ denoising steps. We focus on the reasoning trajectory for a \textit{single masked position}. At denoising step $t$, the model produces a vocabulary logit vector $\ell_t \in \mathbb{R}^{|\mathcal{V}|}$.

A common inference view is to treat $\ell_t$ as a step-local prediction used for sampling and token transfer at step $t$. In contrast, we interpret each denoising step as an implicit reasoning step that incorporates visual evidence and the unmasked context. Under this view, the sequence $\{\ell_t\}_{t=1}^{T}$ forms a visual reasoning trajectory, where the model incorporates visual evidence to refine its decision.

\begin{figure}[t]
    \centering
    % Placeholder for the Heatmap Figure
    \includegraphics[width=1\linewidth]{fig/snr_v2_6_heatmap.pdf} 
    \caption{\textbf{Heatmap of Improvement $\Delta(\tau)$.} The heatmap illustrates the heterogeneity of improvements across samples and timesteps.}
    \label{fig:moti_heatmap}
\end{figure}


\begin{figure*}[t]
    \centering
    \includegraphics[width=1\linewidth]{fig/main_icml_dlm-compressed.pdf}
    \caption{Overview of Recursive Visual Logit Fusion. We view the diffusion denoising process as a trajectory of visual reasoning. At each step $t$, the model produces noisy logits $\ell_t$ characterized by uncertainty $\sigma_t^2.$ Instead of independent sampling, we maintain a running latent state $h_t$ updated. The framework accumulates visual evidence over time, monotonically reducing uncertainty $P_{t}$ to yield stable and consistent predictions}
    \label{fig:main}
\end{figure*}
We model the step-wise logits as noisy observations of a latent visually consistent decision with step-dependent noise:
\begin{equation}
\ell_t = \text{Latent Decision} + \varepsilon_t, \quad
\begin{cases}
\mathbb{E}[\varepsilon_t] = 0 \\
\mathrm{Var}(\varepsilon_t) = \sigma_t^2
\end{cases}
\end{equation}
where $\sigma_t^2$ quantifies the uncertainty of visual reasoning at step $t$, estimated by Shannon entropy of the prediction distribution:
\begin{equation}\sigma_t^2=\alpha\cdot\mathcal{H}(\operatorname{softmax}(\ell_t))+\epsilon,\end{equation}
where \(\alpha\) is a scaling factor, and \(\epsilon\) is a small constant for numerical stability.

% --- 3.1 节结束 ---




\subsection{Recursive Logit Fusion}
To allow implicit reasoning across denoising steps, we maintain two variables: 
(1) \textbf{a running logits state} $h_t \in \mathbb{R}^{|\mathcal{V}|}$ as an estimate of the latent decision after incorporating logits up to step $t$; 
(2) \textbf{a scalar uncertainty} $P_t$, used as an upper bound on the MSE of $h_t$. 

The recursion is initialized by setting $h_0 = \ell_0$ and  hyperparameter $P_0$. At denoising step $t$, we update $(h_{t-1},P_{t-1})$ using the new observation $\ell_t$.

First, we define the step-dependent fusion weight $K_t$:
\begin{equation}
K_t = \frac{P_{t-1}}{P_{t-1} + \sigma_t^2},
\end{equation}
which weighs the reliability of the prior reasoning against the current visual uncertainty.

In step $t$, we update $h_t$ and $P_t$ as follows:
\begin{equation}
h_t = h_{t-1} + K_t(\ell_t - h_{t-1}), \quad 
P_t = (1 - K_t)P_{t-1}
\end{equation}
During inference, we use the fused logits $h_t$ in place of $\ell_t$ when computing sampling probabilities. Once a position is unmasked, we discard its state $(h_t, P_t)$, indicating that the visual reasoning process for that token has converged.

\subsection{Theoretical Analysis}
\paragraph{Monotonic uncertainty reduction.}
The uncertainty update admits a closed-form precision accumulation:
\begin{equation}
P_t = \frac{P_{t-1}\sigma_t^2}{P_{t-1}+\sigma_t^2} = \left(P_{t-1}^{-1} + \sigma_t^{-2}\right)^{-1} \leq P_{t-1}
\end{equation}
showing that $P_t$ is monotonically non-increasing. This property implies that the recursive fusion accumulates visual reasoning evidence step by step, ensuring that the uncertainty in understanding the image decreases continuously along the reasoning trajectory.

\paragraph{Global Objective Interpretation.}
The recursive update above admits a closed-form interpretation that clarifies the global objective implicitly optimized during inference. From the precision accumulation form, the running state $h_t$ can be written as
\begin{equation}
h_t=
\frac{P_0^{-1} h_0 + \sum_{k=1}^t \sigma_k^{-2}\ell_k}
{P_0^{-1} + \sum_{k=1}^t \sigma_k^{-2}},
\end{equation}
which shows that the fused logits correspond to a precision-weighted average of all step-wise logits, where steps with lower visual uncertainty contribute more strongly to the final decision.

In Appendix A, we prove that this closed-form solution is equivalent to the unique global minimizer of the following weighted least-squares objective:
\begin{equation}
\begin{split}
h_t=\arg\min&_{z\in\mathbb{R}^{|\mathcal{V}|}}\mathcal{J}_t(z), \\
\mathcal{J}_t(z)=\frac{1}{2P_0}\|z-h_0\|_2^2+&\sum_{k=1}^t\frac{1}{2\sigma_k^2}\|z-\ell_k\|_2^2
\end{split}
\end{equation}
The objective treats the early state as a stabilizing prior while interpreting logits from subsequent steps as evidence with heterogeneous reliability, whose influence on the final decision is adaptively regulated by their visual uncertainty.


During inference, we use the fused logits $h_t$ in place of $\ell_t$ when computing sampling probabilities. Once a position $M$ is unmasked, its state $(h_t, P_t)$ is discarded, indicating that the visual reasoning process for that token has converged.




\begin{algorithm}[tb]
\caption{Recursive Visual Logit Fusion Inference}
\label{alg:logit-fusion}
\begin{algorithmic}[1]
\REQUIRE Image $I$; Text context $x$; Total steps $T$; Initial uncertainty $P_0$.
\ENSURE Generated token sequence.

\STATE Initialize mask $M \in \{1\}^{N}$ \COMMENT{1 for masked positions}
\STATE Initialize running state $h \leftarrow \ell_0$, uncertainty $P \leftarrow P_0 \cdot \mathbf{1}$

\FOR{$t \gets 1$ to $T$}
    \STATE \textcolor{gray}{\textit{\# 1. Visual Observation}}
    \STATE $\ell_t, \sigma_t^2 \gets \mathrm{Model}(I, x, M, t)$ \COMMENT{Get logits and step-wise uncertainty}
    
    \STATE \textcolor{gray}{\textit{\# 2. Recursive State Update}}
    \STATE $K_t \gets P / (P + \sigma_t^2)$ \COMMENT{Fusion weight}
    
    \STATE $h_{\mathrm{new}} \gets h + K_t \odot (\ell_t - h_{t-1})$ 
    \COMMENT{Update logits state}
    \STATE $P_{\mathrm{new}} \gets (1 - K_t) \odot P$ \COMMENT{Reduce uncertainty}
    
    \STATE $h \gets \mathrm{where}(M, h_{\mathrm{new}}, h)$ \COMMENT{Update only masked positions}
    \STATE $P \gets \mathrm{where}(M, P_{\mathrm{new}}, P)$
    
    \STATE \textcolor{gray}{\textit{\# 3. Sampling and Mask Update}}
    \STATE $x_{\mathrm{pred}}, M_{\mathrm{next}} \gets \mathrm{Sample}(\mathrm{softmax}(h), M)$ \COMMENT{Sample using fused state $h$}
    \STATE \textcolor{gray}{\textit{\# Identify tokens that just converged in this step}}
    \STATE $\Delta M \gets M - M_{\mathrm{next}}$ \COMMENT{Positions where $M$ changes}
    \STATE $h[\Delta M], P[\Delta M] \gets \text{null/reset}$ \COMMENT{Discard reasoning state for unmasked tokens}
    
    \STATE $M \gets M_{\mathrm{next}}$
    
    \IF{$M$ is all zeros}
        \STATE \textbf{break}
    \ENDIF
\ENDFOR
\end{algorithmic}
\end{algorithm}














\section{Experiments}
\begin{table*}[t]
\centering
\caption{Main performance comparison across multiple benchmarks. Bold indicates the \textbf{best} result and the row with light blue background highlights our proposed method.}
\label{tab:main_results}
% 缩小列间距以适应百分比符号
\setlength{\tabcolsep}{3pt} 
\begin{tabular}{l cc cc ccccc c}
\toprule
\multirow{Method} & \multicolumn{2}{c}{CV-Bench-2D} & \multicolumn{2}{c}{CV-Bench-3D} & \multicolumn{5}{c}{GQA} & MMVP \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-10} \cmidrule(lr){11-11}
& ADE20k & COCO & Depth & Distance & Choose & Compare & Logical & Query & Verify & Accuracy \\
\midrule
% --- SOTA / Baselines ---
GPT-4V & - & - & - & - & - & - & - & - & - & 38.7 \\
LLaVA-1.5 & - & - & - & - & - & - & - & - & - & 24.7 \\
Gemini Pro  & - & - & - & - & - & - & - & - & - & 40.7 \\
\midrule
% --- MMaDA Block ---
% --- Vanilla 行 (x100) ---
MMaDA & \textbf{49.4} & 54.2 &  {54.2} & 57.5 &  {72.63} &  {56.54} &  {67.17} &  {37.05} & \textbf{67.23} & \textbf{55.0} \\
% --- COT 行 (x100) ---
+ COT & 44.5 &  {56.0} & 47.3 &  {57.6} & 71.84 & 54.84 & 64.94 & 36.40 & 65.74 &  {52.7} \\
+ CCOT & - & - & - & - & 72.02 & 55.51 & 64.39 & 36.96 & 66.19 & 52.7 \\
% --- Ours 行 ---
\rowcolor{blue!7} 
+ \textbf{Ours} & 
\begin{tabular}[t]{@{}c@{}} {45.3} \\ \textcolor{red}{(-4.1\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}}\textbf{58.1} \\ \textcolor[rgb]{0,0.6,0}{(+3.9\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}}\textbf{55.6} \\ \textcolor[rgb]{0,0.6,0}{(+1.4\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}}\textbf{58.2} \\ \textcolor[rgb]{0,0.6,0}{(+0.7\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}}\textbf{72.90} \\ \textcolor[rgb]{0,0.6,0}{(+0.27\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}}\textbf{59.93} \\ \textcolor[rgb]{0,0.6,0}{(+3.39\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}}\textbf{68.00} \\ \textcolor[rgb]{0,0.6,0}{(+0.83\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}}\textbf{37.39} \\ \textcolor[rgb]{0,0.6,0}{(+0.34\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}} {66.56} \\ \textcolor{red}{(-0.67\%)}\end{tabular} & 
\textbf{55.0} \\
\midrule
% --- LaVida Block ---
LaVida & 65.3 & 66.1 &  {65.3} & 70.3 &  {82.99} & {65.03} &  {72.10} &  {45.04} & \textbf{73.07} & {-} \\
+ COT & \textbf{66.5} &  \textbf{67.3} & 66.5 &  {69.8} & 80.60 & 65.87 & 73.04 & 42.99 & 71.97 &  {-} \\
+ CCOT & 64.3 & 63.1 & 66.6 & 67.5 & 82.63 & 66.55 & 69.60 & 43.14 & 72.19 & - \\
\rowcolor{blue!7} 
+ \textbf{Ours} & 
\begin{tabular}[t]{@{}c@{}} {65.8} \\ \textcolor[rgb]{0,0.6,0}{(+0.5\%)}\end{tabular} &
\begin{tabular}[t]{@{}c@{}} 66.3 \\ \textcolor[rgb]{0,0.6,0}{(+0.2\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}}\textbf{67.7} \\ \textcolor[rgb]{0,0.6,0}{(+3.4\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}}\textbf{72.2} \\ \textcolor[rgb]{0,0.6,0}{(+1.9\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}}\textbf{85.56} \\ \textcolor[rgb]{0,0.6,0}{(+2.57\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}}\textbf{68.25} \\ \textcolor[rgb]{0,0.6,0}{(+3.22\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}}\textbf{73.93} \\ \textcolor[rgb]{0,0.6,0}{(+1.82\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}}\textbf{46.50} \\ \textcolor[rgb]{0,0.6,0}{(+1.46\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}} {71.30} \\ \textcolor{red}{(-1.77\%)}\end{tabular} & 
{-} \\
\bottomrule
\end{tabular}
\end{table*}

\begin{table}[h]
\small
\centering
\caption{Performance comparison on HallusionBench, ScienQA, and MMstar benchmarks. Bold indicates the \textbf{best} result and the row with light blue background highlights our proposed method.}
\label{tab:optimized_results}
% 缩小列间距以适应内容
\setlength{\tabcolsep}{2pt} 
\begin{tabular}{l ccc cc}
\toprule
\multirow{Method} & \multicolumn{3}{c}{HallusionBench} & ScienQA & MMstar \\
\cmidrule(lr){2-4} \cmidrule(lr){5-5} \cmidrule(lr){6-6} 
 & aAcc & fAcc & qAcc & Acc & Acc \\
\midrule
% --- SOTA / Baselines ---
GPT-4V & 65.28 & 39.88 & 28.79 & 81.43 & - \\
Gemini & 36.85 & 8.67 & 7.69 & 80.63 & - \\
LLaVA-1.5 & 46.94 & 24.86 & 10.55 & 68.72 & - \\
\midrule
% --- MMaDA Block ---
% MMaDA Base: 41.28, 13.29, 12.31, 57.03, 35.33
MMaDA & 41.28 & 13.29 &  {12.31} &  {57.03} & 35.33 \\
% COT
+ COT &  {41.54} & \textbf{14.45} & \textbf{13.63} & 53.51 & 34.67 \\
% CCOT
+ CCOT & 41.11 & 13.01 & 10.99 & 56.35 &  {35.67} \\
% Ours
% Comparisons to MMaDA:
% aAcc: 45.64 - 41.28 = +4.36
% fAcc: 14.45 - 13.29 = +1.16
% qAcc: 12.09 - 12.31 = -0.22
% ScienQA: 57.27 - 57.03 = +0.24
% MMstar: 36.67 - 35.33 = +1.34
\rowcolor{blue!7} 
+ \textbf{Ours} & 
\begin{tabular}[t]{@{}c@{}}\textbf{45.64} \\ \textcolor[rgb]{0,0.6,0}{(+4.36\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}}\textbf{14.45} \\ \textcolor[rgb]{0,0.6,0}{(+1.16\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}}12.09 \\ \textcolor{red}{(-0.22\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}}\textbf{57.27} \\ \textcolor[rgb]{0,0.6,0}{(+0.24\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}}\textbf{36.67} \\ \textcolor[rgb]{0,0.6,0}{(+1.34\%)}\end{tabular} \\
\midrule
% --- LaVida Block ---
% LaVida Base: 54.03, 16.67, 15.39, 80.17
LaVida &  {54.03} & \textbf{16.67} & 15.39 &  {80.17} & - \\
% COT
+ COT & 53.01 & 15.03 & 15.60 & 76.37 & - \\
% CCOT
+ CCOT & 35.06 & 15.90 &  {16.26} & 78.73 & - \\
% Ours
% Comparisons to LaVida:
% aAcc: 54.92 - 54.03 = +0.89
% fAcc: 16.19 - 16.67 = -0.48
% qAcc: 16.48 - 15.39 = +1.09
% ScienQA: 81.47 - 80.17 = +1.30
\rowcolor{blue!7} 
+ \textbf{Ours} & 
\begin{tabular}[t]{@{}c@{}}\textbf{54.92} \\ \textcolor[rgb]{0,0.6,0}{(+0.89\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}} {16.19} \\ \textcolor{red}{(-0.48\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}}\textbf{16.48} \\ \textcolor[rgb]{0,0.6,0}{(+1.09\%)}\end{tabular} & 
\begin{tabular}[t]{@{}c@{}}\textbf{81.47} \\ \textcolor[rgb]{0,0.6,0}{(+1.30\%)}\end{tabular} & 
- \\
\bottomrule
\end{tabular}
\end{table}




















Our experiments consist of the following three parts:
\begin{enumerate}[nosep]
\item We evaluate our method on extensive benchmarks with \textit{MMaDA} and\textit{ LaVida}. The evaluation covers two complementary dimensions: (i) perceptual-intensive reasoning tasks, and (ii) semantic-intensive reasoning.
\item We conduct ablation studies on \textit{MMaDA} to analyze the sensitivity of key hyper-parameters. We also analyze the impact of different components in our method.
\item As our method introduces additional computation during inference, we compare generation speed with the baseline model and CoT decoding.
\end{enumerate}
\subsection{Experiment Setup}
\paragraph{Baselines.} We select the leading multimodal diffusion language model MMada~\cite{yang2025mmada} and Lavida~\cite{li2025lavida} as our baseline, and employ two test-time scaling methods CCoT~\cite{mitra2024compositional} and CoT~\cite{wei2022chain} for comparison. Notably, to the best of our knowledge, this represents the first study dedicated to enhancing test-time inference within the DLM framework. We also compare those DLM-based frameworks with auto-aggressive architecture, including LLaVA-1.5~\cite{liu2024improved}, Gemini Vision Pro~\cite{team2023gemini}and GPT4V~\cite{achiam2023gpt}.

\paragraph{Evaluation Benchmarks.} We evaluate our method on the tasks across extensive benchmarks of two dimensions: 1) perceptually-intensive rea-
soning tasks with GQA~\cite{hudson2019gqanewdatasetrealworld}, CV-Bench~\cite{tong2024cambrian1}, Mmvp~\cite{tong2024eyes}; 2) semantic-intensive reasoning tasks with Science QA~\cite{lu2022learn}, HallusionBench~\cite{guan2024hallusionbench}, MMStar~\cite{chen2024we}.

\paragraph{Implementation Details.} For the MMaDA diffusion process, we configure the model with 128 diffusion steps, 64 block length and 64 maximum sequence length. We use VLMEvalKit~\cite{duan2024vlmevalkit} as the toolkit for all experimental evaluations. All experiments are executed on NVIDIA RTX 5090 GPUs.

\subsection{Main Results}
\label{sec:main_results}

\autoref{tab:main_results} reports the quantitative comparison against state-of-the-art baselines. Our method consistently outperforms base DLMs and existing scaling strategies across most benchmarks. Specifically, RVLF shows substantial gains on both perceptual-intensive tasks (e.g., GQA, CV-Bench) and semantic-intensive datasets (e.g., ScienceQA), suggesting that recursive logit fusion effectively grounds fine-grained visual details while establishing the stable visual context required for high-level reasoning.

In contrast, standard CoT yields negligible improvements or even degradation compared to vanilla baselines. We attribute the limited performance of CoT to the architectural mismatch between autoregressive and diffusion language models. CoT relies on sequential token dependencies to construct logical chains, a mechanism inherent to the left-to-right generation of AR models. In contrast, DLMs generate tokens via parallel denoising. Consequently, imposing a sequential reasoning chain might introduce structural noise. This indicates that NAR architectures like DLM necessitate specialized strategies specifically for test-time inference enhancement, rather than direct adaptation of AR-based prompting. Unlike CoT, RVLF functions in the continuous logit space, aligning with the probabilistic nature of the diffusion process. By fusing logits based on estimated uncertainty, our method preserves the parallel generation mechanism of DLMs. The results confirm that modeling the visual reasoning trajectory is a more robust approach for test-time inference in DLMs than enforcing a textual reasoning chain.


%==============





\begin{figure*}[h!]
    \centering
    \includegraphics[width=1\linewidth]{fig/abla1.png}
    \caption{Sensitivity of model accuracy to key hyperparameters}
    \label{fig:hyper}
\end{figure*}
\subsection{Ablation Study}
We perform two ablation studys with MMaDA. Hyperparameter sensitivity is evaluated on a subset of GQA, while component ablations are conducted on a subset of CVBench-2D, respectively, to analyze parameter robustness and architectural contributions under identical inference settings.

\subsubsection{Hyperparameter Sensitivity Analysis}
We examine the sensitivity of three hyperparameters: 
\begin{enumerate}[nosep,left=0pt]
    \item {\textit{Initial uncertainty}} $P_{0}$, which controls the strength of the prior uncertainty in  $\boldsymbol{h}_{\mathbf{0}}$. We evaluate      $P_0\in\{30,50,75\}$, covering weak to relatively strong prior uncertainty settings.
     \item {\textit{Noise scale}} $\alpha$, which scales the step-wise uncertainty ${ \sigma }^{ 2 }_{{ t }}$. We vary the noise scale over
     $\{2,4,7,10\}$ to assess robustness under different levels of observation noise.;
     \item {\textit{Time window}} $[{t}_{{{min}}} , {t}_{{{max}}}]$, including window length and start-end points. We test multiple fusion start points $t_{\min}\in\{1,5,10\}$ and end points $t_\mathrm{max}\in\{32,64,128\}$, resulting in different fusion ranges $\Delta t=t_\mathrm{max}-t_\mathrm{min}$. We set:
\end{enumerate}
\begin{equation}\mathrm{Acc}(t_{\min},\Delta t)=\mathbb{E}_{\mathrm{noise}\_\mathrm{scale},P_0}
\begin{bmatrix}
\mathrm{Accuracy}
\end{bmatrix}\end{equation}
\paragraph{Result Analysis of $P_0$ and Noise scale.}While accuracy shows a mild increasing trend as these parameters grow, the overall variation remains small. This indicates that the proposed fusion mechanism is largely insensitive to the exact choice of $P_0$ and noise scale, suggesting stable behavior across a wide range of uncertainty settings.

\begin{table}[h!]
\centering
\small
\caption{Accuracy as a function of the fusion start step $t_{\min}$ and the effective fusion range $\Delta t$. }
\begin{tabular}{c c c}
\toprule
$t_{\min}$ & $\Delta t = t_{\max} - t_{\min}$ & Accuracy (\%) \\
\midrule
1  & 31  & \textbf{82.73} \\
1  & 63  & \textbf{82.73} \\
1  & 127 & \textbf{82.73} \\
\midrule
5  & 59  & 80.00 \\
5  & 123 & 80.00 \\
\midrule
10 & 54  & 78.18 \\
\bottomrule
\end{tabular}

\label{tab:tmin_delta}
\end{table}
\begin{figure}[h!]
    \centering
    \includegraphics[width=1\linewidth]{fig/inference_speed_chart_bold.pdf}
    \caption{Token inference time (tokens/s) }
    \label{fig:inference_speed}
\end{figure}
\paragraph{Result Analysis of $\Delta t$ and $t_{min}$.}
The accuracy is primarily determined by the fusion start step $t_{\text{min}}$, while being insensitive to the fusion range $\Delta t$. When fusion starts early ($t_{\text{min}}=1$), performance remains stable across different fusion spans, consistently achieving the highest accuracy $(82.73\%)$. In contrast, delaying the fusion start to later steps ($t_{\text{min }}5$ or $10$) leads to a clear performance drop which cannot be recovered by increasing the fusion range. Therefore, starting visual evidence integration at early stages is more important than the duration of fusion, supporting our theoretical analysis that early denoising steps contain noisy but complementary visual information.

\subsubsection{component ablations}
We ablate:  \textbf{1)} \textbf{\textit{The discrepancy term}} ($\ell_t - h_{t-1}$) in the stepwise-update, resulting $h_t = h_{t-1}$; \textbf{2)} \textbf{\textit{Uncertainty estimation.}} We set $\sigma_t^2 = H(\mathrm{softmax}(\ell_t))$; \textbf{3) \textit{Forgetting strategy.}} We replace $P_t^{-1} = P_{t-1}^{-1} + \sigma_t^{-2}$
with either a \textit{sliding window} over the last $k$ steps ($k \in \{3,5\}$) or an \textit{exponential decay}
$P_t \leftarrow \gamma P_{t-1}$ with $\gamma$ = 0.8, and ablate alternative definitions by fixing $\sigma_t^2$ to this entropy-based form throughout inference.
\begin{table}[h]
    \centering
    \caption{Ablation Results on CVBench-2D}
    \label{tab:experiment_accuracy}
    \begin{tabular}{@{}l r@{}}
        \toprule
        Configuration & Accuracy \\
        \midrule
        Baseline  & 0.5103 \\
        w/o discrepancy term ($\ell_t - h_{t-1}$) & 0.5087 \\
        w/o full steps (sliding window $k=3$) & 0.5166 \\
        w/o full steps (sliding window $k=5$) & 0.5126 \\

        use $P_t \leftarrow \gamma P_{t-1}$ ($\gamma=0.8$) & \underline{0.5197} \\
        \rowcolor{blue!7}
        Full model & \textbf{0.5228} \\
        \bottomrule
    \end{tabular}
    \label{}
\end{table}

\autoref{tab:experiment_accuracy} shows that the full model achieves the highest accuracy of 52.28\%. Removing the discrepancy term drops performance below the baseline, confirming the necessity of dynamically correcting the latent state with immediate observations. Furthermore, full-history accumulation outperforms both sliding window and exponential decay strategies. Results demonstrate the importance of each component.


\subsection{Analysis of inference time}
We evaluate the computational overhead of our method on GQA, CVBench and MMstar with MMaDA. As illustrated in Fig~\ref{fig:inference_speed}, our method maintains a high inference speed of approximately 43 tokens/s. Compared to the Baseline (48 tokens/s), the slight reduction in speed is due to the additional entropy calculation and state updates at each step. However, since these operations are performed on the vocabulary logits rather than high-dimensional hidden states, they introduce \textit{negligible} computational overhead compared to the heavy matrix multiplications in the backbone model. This confirms that our method improves reasoning reliability with minimal cost to efficiency.





\section{Advantages, Limitations, and Future Work}
RVLF treats the denoising trajectory as a structured reasoning process, providing a training-free, theoretically grounded method that enhances stability and performance with negligible latency. While currently optimized for mask-based DLMs, future research will extend this mechanism to uniform multimodal frameworks and investigate inter-step dependencies to further refine the dynamics and interpretability of implicit reasoning trajectories.


\section{Conclusion}
We introduced Recursive Visual Logit Fusion (RVLF) to stabilize multimodal DLM inference by transforming noisy denoising trajectories into consistent reasoning paths via uncertainty-aware recursive updates. RVLF is a training-free and mathematically rigorous approach that consistently outperforms standard inference and existing scaling methods across extensive benchmarks. Our results demonstrate that the iterative diffusion process serves as a powerful substrate for implicit reasoning, where proper state aggregation is key to unlocking its full potential.