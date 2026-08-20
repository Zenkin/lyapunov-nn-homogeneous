# Example 1

This directory contains the authors' public implementations of the first
numerical example in the article. The example studies a mechanical system with
bounded dry friction, nonlinear drag, and a cubic restoring term. It admits an
`r=(1,2)` homogeneous approximation at infinity.

Two implementations are provided:

- [`original`](original): the construction presented in the article, using the
  pointwise minimum from equation (16);
- [`improved`](improved): a subsequent author-developed modification using a
  positive-definite inner candidate and smooth level-set gluing.

The two folders use the same system parameters, equilibrium correction,
homogeneous outer candidate, deterministic seed, and level-angle grid. This
keeps the comparison focused on the changed inner architecture and gluing
rule. The improved version is an extension of the numerical method and is not
presented as the exact algorithm printed in the article.
