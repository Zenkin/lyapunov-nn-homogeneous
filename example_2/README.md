# Example 2: local stabilization

The second numerical example considers stabilization of the upper equilibrium
of a nonlinear pendulum. The implementations are separated according to how
the linear model printed in the article is treated.

| Version | Status |
| --- | --- |
| [`article_version`](article_version) | Literal implementation of the equations printed in the article, including its displayed matrix `A` |

The article does not report all numerical parameters needed to rerun the
training. Each implementation therefore distinguishes values taken from the
article from later, explicitly documented implementation choices.
