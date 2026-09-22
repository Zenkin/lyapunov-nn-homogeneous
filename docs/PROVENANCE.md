# Branch provenance

The September 2026 consolidation brings both numerical examples and their
recorded results onto `main`. It preserves the existing scientific source
files, numerical records, figures, and Git history from the latest branch.
The presentation changes add navigation, reproduction instructions, a shared
dependency entry point, and continuous integration.

## Branch tips before consolidation

| Branch | Commit | Contents |
| --- | --- | --- |
| `main` | [`8e6491778ef721765d1487d80a12b04475056afe`](https://github.com/Zenkin/lyapunov-nn-homogeneous/commit/8e6491778ef721765d1487d80a12b04475056afe) | Both Example 1 implementations |
| `example-2-article-version` | [`a76a17390b4282ca7dab37546c8e4c9b91f6c186`](https://github.com/Zenkin/lyapunov-nn-homogeneous/commit/a76a17390b4282ca7dab37546c8e4c9b91f6c186) | Adds the article implementation of Example 2 |
| `example-2-corrected-matrix-only` | [`d87c896ecb739ce6214a46855508f25620046ef8`](https://github.com/Zenkin/lyapunov-nn-homogeneous/commit/d87c896ecb739ce6214a46855508f25620046ef8) | Contains all five implementations, numerical audits, saved figures, and publication tables |

These tips form one ancestry chain. The latest branch already contains every
commit from both earlier branches, including intermediate corrections and
extensions. Consolidation therefore requires no selection between conflicting
scientific implementations. Earlier versions remain accessible through the
commit links above; the two existing experiment branches are retained.

These identifiers record repository states, not a claim about which commit
was used to submit the manuscript. No submission tag or paper DOI is inferred.

## Article implementations and subsequent work

`example_1/original` and `example_2/article_version` expose the constructions
described in the article, with additional numerical choices documented in
their respective READMEs and audits. They are not archival reconstructions of
every original numerical run.

`example_1/improved`, `example_2/corrected_matrix`, and `example_2/improved`
contain explicitly identified subsequent work. Their presence on `main` does
not relabel them as the methods printed in the submitted article. The
publication summary continues to describe the improved runs only.
