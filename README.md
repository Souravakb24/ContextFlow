# RAG
A full Engineering Research on RAG for best custom usecase.


# Document Parser

We have evaluated two open-source document parser frameworks: **Docling** ([GitHub](https://github.com/docling-project/docling)) and **Unstructured** ([GitHub](https://github.com/Unstructured-IO/unstructured)).

Let's compare them using everything both are using.

---

## Equations

> ![From TurboQuant paper layout](images/turbo_quant_equation.png)
> Example taken from paper [TurboQuant](https://arxiv.org/pdf/2504.19874)

---

> **Unstructured Output:**
>
> Lemma 1 (coordinate distribution of random point on hypersphere). For any positive integer $d$, if $x \in S^{d-1}$ is a random variable uniformly distributed over the unit hypersphere, then for any $j \in [d]$ the coordinate $x_j$ follows the following (scaled/shifted) Beta distribution:
>
> $$ _ ri), Vr -T((d—1)/2) x; ~ fx(x): 2) 3)? $$
>
> In high dimensions this beta distribution converges to the normal distribution $f_X(\cdot) \to \mathcal{N}(0, 1/d)$.
>
> **Proof.** $f_X(x)$ equals the ratio of the area of a sphere with radius $\sqrt{1 - x^2}$ in dimension $d-1$ to the volume of a unit sphere in dimension $d$ scaled down by $1/\sqrt{1-x^2}$ (by Pythagorean theorem). Therefore,
>
> $$ Qn (4-1)/2 (1— g?)(d-2)/2 eee. _ -P(d/2) fx(x) = Dea Uv l= ed) 2292 $$
---

> **Docling Output:**
>
> Lemma 1 (coordinate distribution of random point on hypersphere). For any positive integer $d$, if $x \in S^{d-1}$ is a random variable uniformly distributed over the unit hypersphere, then for any $j \in [d]$ the coordinate $x_j$ follows the following (scaled/shifted) Beta distribution:
>
> $$x_{j} \sim f_{X}(x) := \frac{\Gamma(d/2)}{\sqrt{\pi} \cdot \Gamma((d-1)/2)} \left(1 - x^{2}\right)^{(d-3)/2}$$
>
> $$u_{i} \cdot u_{j} = f_{X}(x) := \frac{\Gamma(d/2)}{\sqrt{\pi} \cdot \Gamma((d-1)/2)} \left(1 - x^{2}\right)^{(d-3)/2}$$
>
> In high dimensions this beta distribution converges to the normal distribution $f_X(\cdot) \to \mathcal{N}(0, 1/d)$.
>
> **Proof.** $f_X(x)$ equals the ratio of the area of a sphere with radius $\sqrt{1 - x^2}$ in dimension $d-1$ to the volume of a unit sphere in dimension $d$ scaled down by $1/\sqrt{1-x^2}$ (by Pythagorean theorem). Therefore,
>
> $$f_{X}(x) = \frac{\dfrac{2\pi^{(d-1)/2}}{\Gamma((d-1)/2)} \cdot (1 - x^{2})^{(d-2)/2}}{\dfrac{2\pi^{d/2}}{\Gamma(d/2)}} \cdot \frac{1}{\sqrt{1 - x^{2}}} = \frac{\Gamma(d/2)}{\sqrt{\pi} \cdot \Gamma((d-1)/2)} \left(1 - x^{2}\right)^{(d-3)/2}$$


Unstructured couls not able to parse the equation as it uses redering the equation from text only. (No specific Equation Recognition model is used)
Docling uses VLM model for equation recognition from image to text. [CodeFormulaV2](https://huggingface.co/docling-project/CodeFormulaV2)
