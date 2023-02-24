from typing import FrozenSet


symmetric_tag = object()
diagonal_tag = object()
unit_diagonal_tag = object()
lower_triangular_tag = object()
upper_triangular_tag = object()
positive_semidefinite_tag = object()
negative_semidefinite_tag = object()
nonsingular_tag = object()


transpose_tags_rules = []


for tag in (
    symmetric_tag,
    unit_diagonal_tag,
    diagonal_tag,
    positive_semidefinite_tag,
    negative_semidefinite_tag,
    nonsingular_tag,
):

    @transpose_tags_rules.append
    def _(tags: FrozenSet[object], tag=tag):
        if tag in tags:
            return tag


@transpose_tags_rules.append
def _(tags: FrozenSet[object]):
    if lower_triangular_tag in tags:
        return upper_triangular_tag


@transpose_tags_rules.append
def _(tags: FrozenSet[object]):
    if upper_triangular_tag in tags:
        return lower_triangular_tag


def transpose_tags(tags: FrozenSet[object]):
    if symmetric_tag in tags:
        return tags
    new_tags = []
    for rule in transpose_tags_rules:
        out = rule(tags)
        if out is not None:
            new_tags.append(out)
    return frozenset(new_tags)
