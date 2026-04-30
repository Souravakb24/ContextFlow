# RAG
A full Engineering Research on RAG for best custom usecase.


#Document Parser
we have evaluated two oepnsource Document parser framework Docling([https://docling-project.github.io/docling/](https://github.com/docling-project/docling)) and Unstructured(https://github.com/Unstructured-IO/unstructured).

Lets first compare using aeverything what both are using.

## Equations: 
![From TurboQuant paper layout](images/turbo_quant_equation.png)
keep ina box{
UNstrutured Output:
Lemma 1 (coordinate distribution of random point on hypersphere). For any positive integer d if x € S&! is a random variable uniformly distributed over the unit hypersphere, then for any j € {d] the coordinate x; follows the following (scaled/shifted) Beta distribution:


$$
_ ri), Vr -T((d—1)/2) x; ~ fx(x): 2) 3)?
$$


In high dimensions this beta distribtion converges to the normal distribution fx(-) > N(0,1/d).


Proof. fx(a) equals the ratio of the area of a sphere with radius V1 — x? in dimension d— 1 to the volume of a unit sphere in dimension d scaled down by 1/V1— x? (by Pythagorean theorem). Therefore,


$$
Qn (4-1)/2 (1— g?)(d-2)/2 eee. _ -P(d/2) fx(x) = Dea Uv l= ed) 2292
$$
}

keep in a box{
DOcling output:
Lemma 1 (coordinate distribution of random point on hypersphere) . For any positive integer d if x ∈ S d -1 is a random variable uniformly distributed over the unit hypersphere, then for any j ∈ [ d ] the coordinate x j follows the following (scaled/shifted) Beta distribution:

$$x _ { j } \sim f _ { X } ( x ) \colon = \frac { \Gamma ( d / 2 ) } { \sqrt { \pi } \cdot \Gamma ( ( d - 1 ) / 2 ) } \left ( 1 - x ^ { 2 } \right ) ^ { ( d - 3 ) / 2 } . \\ \\ u _ { i } \cdot u _ { j } = f _ { X } ( x ) \colon = \frac { \Gamma ( d / 2 ) } { \sqrt { \pi } \cdot \Gamma ( ( d - 1 ) / 2 ) } \left ( 1 - x ^ { 2 } \right ) ^ { ( d - 3 ) / 2 } .$$

In high dimensions this beta distribtion converges to the normal distribution f X ( · ) →N (0 , 1 /d ) .

Proof. f X ( x ) equals the ratio of the area of a sphere with radius √ 1 -x 2 in dimension d -1 to the volume of a unit sphere in dimension d scaled down by 1 / √ 1 -x 2 (by Pythagorean theorem). Therefore,

$$f _ { X } ( x ) = \frac { \frac { 2 \pi ^ { ( d - 1 ) / 2 } } { \Gamma ( ( d - 1 ) / 2 ) } \cdot ( 1 - x ^ { 2 } ) ^ { ( d - 2 ) / 2 } } { \frac { 2 \pi ^ { d / 2 } } { \Gamma ( d / 2 ) } } \cdot 1 / \sqrt { 1 - x ^ { 2 } } = \frac { \Gamma ( d / 2 ) } { \sqrt { \pi } \cdot \Gamma ( ( d - 1 ) / 2 ) } \left ( 1 - x ^ { 2 } \right ) ^ { ( d - 3 ) / 2 } .$$
}
