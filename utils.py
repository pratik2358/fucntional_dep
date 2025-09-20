from itertools import combinations
import random
import pandas as pd
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
from typing import Dict, Iterable, List, Tuple

def compute_closure(attributes, fds) -> set:
    """
    Compute the closure of a set of attributes under a set of functional dependencies
    ---------------------------------------------------------------------------------
    attributes: a set of attributes
    fds: a list of functional dependencies (contains tuples of two sets. First set implies the second set)
    """
    closure = set(attributes)
    changed = True
    while changed:
        changed = False
        for fd in fds:
            if fd[0].issubset(closure) and not fd[1].issubset(closure):
                closure.update(fd[1])
                changed = True
    return closure

def compute_all_closures(attributes, fds) -> dict:
    """
    Compute the closure of all possible subsets of a set of attributes
    ------------------------------------------------------------------
    attributes: a set of attributes
    fds: a list of functional dependencies (contains tuples of two sets. First set implies the second set)
    """
    all_closures = {}
    for r in range(1, len(attributes) + 1):
        for subset in combinations(attributes, r):
            subset_closure = compute_closure(set(subset), fds)
            all_closures[tuple(subset)] = subset_closure
    return all_closures

# def compute_candidate_keys(closure_set, attributes) -> list:
#     """
#     Compute the candidate keys of a set of attributes
#     -------------------------------------------------
#     closure_set: a dictionary of all closures
#     attributes: a set of attributes
#     """
#     super_keys = []
#     for i in closure_set:
#         if set(closure_set[i]) == set(attributes):
#             super_keys.append(i)
#     candidate_keys = []
#     for j in tqdm(super_keys):
#         flag = False
#         for i in super_keys:
#             if set(i) != set(j):
#                 if set(i).issubset(set(j)):
#                     flag = True
#         if flag == False:
#             candidate_keys.append(j)
#     return candidate_keys

def compute_candidate_keys(closure_set: Dict[Iterable[str], Iterable[str]],
                           attributes: Iterable[str]) -> List[Tuple[str, ...]]:
    """
    closure_set: dict mapping attribute-set (iterable) -> closure (iterable)
    attributes:  iterable of all attributes in the relation
    returns:     list of candidate keys as tuples (sorted for determinism)
    """
    # --- bitmask setup ---
    attrs = tuple(sorted(set(attributes)))
    bit_of = {a: 1 << i for i, a in enumerate(attrs)}
    ALL = (1 << len(attrs)) - 1

    def set_to_bits(xs: Iterable[str]) -> int:
        b = 0
        for x in xs:
            b |= bit_of[x]
        return b

    def bits_to_tuple(b: int) -> Tuple[str, ...]:
        return tuple(a for a in attrs if (b >> attrs.index(a)) & 1)

    # Collect superkeys (those whose closure == all attributes), as bitmasks
    superkeys_bits: List[int] = []
    for X, cl in closure_set.items():
        if set_to_bits(cl) == ALL:
            superkeys_bits.append(set_to_bits(X))

    # Sort by cardinality (popcount) so we consider small sets first.
    superkeys_bits.sort(key=lambda b: b.bit_count())

    # Single pass: keep only minimal ones.
    candidates: List[int] = []
    for b in superkeys_bits:
        # If any existing candidate is a subset of b, then b is not minimal → skip.
        # (subset check via bitmask: c ⊆ b  iff  (b & c) == c)
        if any((b & c) == c for c in candidates):
            continue
        candidates.append(b)

    # Convert back to stable, readable tuples
    result = []
    for b in candidates:
        key = tuple(a for a, i in bit_of.items() if (b & i))
        result.append(tuple(sorted(key)))  # sort for readability/determinism
    return result

def find_prime_attributes(candidate_keys) -> set:
    """
    Find the prime attributes of a set of candidate keys
    ----------------------------------------------------
    candidate_keys: a list of candidate keys
    """
    prime_attributes = set()
    for key in candidate_keys:
        prime_attributes.update(key)
    return prime_attributes

def compute_single_covers(attributes, fds) -> dict:
    """
    Compute the closure of each attribute in a set of attributes
    ------------------------------------------------------------
    attributes: a set of attributes
    fds: a list of functional dependencies (contains tuples of two sets. First set implies the second set)
    """
    all_closures = {}
    for a in attributes:
        subset_closure = compute_closure(a, fds)
        all_closures[a] = subset_closure
    return all_closures

def project_dependency(fds, R_hat) -> list:
    """
    Project a set of functional dependencies onto R_hat using the closure-based
    algorithm, then minimize (LHS + redundancy) via minimal_cover.

    fds: list[(set, set)]   e.g., [({'A'}, {'B','C'}), ({'B'}, {'C'})]
    R_hat: set
    returns: list[(set, set)] a minimized cover over R_hat
    """
    R_hat = set(R_hat)
    if not R_hat:
        return []

    # 1) Generate projected unit FDs via closures over all non-empty Y ⊆ R_hat
    projected_unit = set()  # {(frozenset(Y), frozenset({a}))}
    attrs = list(R_hat)
    for r in range(1, len(attrs) + 1):
        for Y_tuple in combinations(attrs, r):
            Y = set(Y_tuple)
            T = compute_closure(Y, fds)          # Y+ w.r.t. original Σ
            H = (T & R_hat) - Y                  # keep only nontrivial heads in R'
            for a in H:
                projected_unit.add((frozenset(Y), frozenset({a})))

    # 2) Convert to list[(set, set)] and minimize using your pipeline
    proj_fds = [(set(L), set(R)) for (L, R) in projected_unit]

    # Deterministic minimal cover (avoid randomness in tie-breaking)
    minimized = minimal_cover(proj_fds, p=0.0)

    return minimized
## Minimal cover computation

def decompose_fds(fds) -> list:
    """Decompose each FD so that the RHS contains only one attribute.
    For example, the FD {A} -> {B, C} will be decomposed into {A} -> {B} and {A} -> {C}.
    ------------------------------------------------------------------------------------
    fds: a list of functional dependencies (contains tuples of two sets. First set implies the second set)
    """
    decomposed_fds = []
    for lhs, rhs in fds:
        for attr in rhs:
            decomposed_fds.append((lhs, {attr}))
    return decomposed_fds

def remove_trivial_dependencies(fds) -> list:
    """Remove trivial FDs of the form A -> A.
    -----------------------------------------
    fds: a list of functional dependencies (contains tuples of two sets. First set implies the second set)
    """
    return [(lhs, rhs) for lhs, rhs in fds if lhs != rhs]

def remove_redundant_dependencies(fds) -> list:
    """Remove redundant FDs by checking if we can infer a FD from others.
    ---------------------------------------------------------------------
    fds: a list of functional dependencies (contains tuples of two sets. First set implies the second set)
    """
    fds_ = fds.copy()
    len_fds_1 = len(fds_)
    len_fds_2 = 0
    while len_fds_1>len_fds_2:
        len_fds_1 = len(fds_)
        for i, (lhs, rhs) in enumerate(fds_):
            remaining_fds = fds_[:i] + fds_[i+1:]
            closure_lhs = compute_closure(lhs, remaining_fds)
            if rhs.issubset(closure_lhs):
                fds_.remove((lhs, rhs))
        len_fds_2 = len(fds_)
    return fds_

def merge_fds(fds) -> list:
    """Merge FDs with the same LHS back together.
    --------------------------------------------
    fds: a list of functional dependencies (contains tuples of two sets. First set implies the second set)
    """
    merged_fds = {}
    for lhs, rhs in fds:
        lhs = tuple(lhs)
        if lhs in merged_fds:
            merged_fds[lhs].update(rhs)
        else:
            merged_fds[lhs] = set(rhs)
    
    return [(set(lhs), rhs) for lhs, rhs in merged_fds.items()]

def powerset(iterable):
    """Generate all non-empty proper subsets of a set."""
    s = list(iterable)
    combs = [[i for i in combinations(s, r)] for r in range(1, len(s)+1)]
    return [x for xs in combs for x in xs]

def remove_superfluous_lhs(fds, p):
    """
    Simplify the LHS by checking if any proper subset of the LHS can imply the RHS.
    --------------------------------------------------------------------------------
    fds: a list of functional dependencies (contains tuples of two sets. First set implies the second set)
    p: probability of choosing a random minimal lhs
    """
    minimal_fds = []
    for lhs, rhs in fds:
        minimal_lhs = lhs
        min_sub = 10000
        minimals = []
        for subset in powerset(lhs):
            if len(subset) <= min_sub:
                if rhs.issubset(compute_closure(set(subset), fds)):
                    minimal_lhs = set(subset)
                    min_sub = len(subset)
                    minimals.append(minimal_lhs)
        if len(minimals)>1 and random.randint(0, 10) <= p*10:
            minimal_lhs = set(random.choice(minimals))
        elif len(minimals)==1:
            minimal_lhs = minimals[0]
            
        minimal_fds.append((minimal_lhs, rhs))
    return minimal_fds

def minimal_cover(fds, p = 0.5) -> list:
    """Find the minimal cover of a set of FDs.
    -----------------------------------------
    attributes: a set of attributes
    fds: a list of functional dependencies (contains tuples of two sets. First set implies the second set)
    """
    # Step 1: Decompose the RHS
    decomposed_fds = decompose_fds(fds)

    # Step 2: Simplify LHS
    simplified_fds = remove_superfluous_lhs(decomposed_fds, p)

    # Step 3: Remove trivial dependencies (A -> A)
    simplified_fds = remove_trivial_dependencies(simplified_fds)

    # Step 4: Remove redundant FDs
    simplified_fds = remove_redundant_dependencies(simplified_fds)
    
    # Step 5: Recollect FDs with the same LHS
    minimal_fds = merge_fds(simplified_fds)
    
    return minimal_fds

def _attr_bitmask(attrs, attr_to_bit):
    """Helper to build a bitmask from an iterable of attribute names."""
    m = 0
    for a in attrs:
        m |= 1 << attr_to_bit[a]
    return m

def _bit_to_attrs(bitmask, bit_to_attr):
    """Inverse of _attr_bitmask."""
    res = []
    i = 0
    while bitmask:
        if bitmask & 1:
            res.append(bit_to_attr[i])
        bitmask >>= 1
        i += 1
    return res

def _partition(df, cols):
    """
    Build a partition (list of frozenset row indices) for a set of columns.
    Two rows are in the same block iff they have identical values on 'cols'.
    """
    if not cols:
        # Single block containing all row indices
        return [frozenset(range(len(df)))]
    groups = defaultdict(list)
    view = df[list(cols)].itertuples(index=False, name=None)
    for i, key in enumerate(view):
        groups[key].append(i)
    return [frozenset(g) for g in groups.values()]

def _partition_cardinality(part):
    """Number of distinct value-combinations = number of blocks in the partition."""
    return len(part)

def _refine_partition(part_left, part_right, nrows):
    """
    Compute partition of the union of attribute sets if we already know
    partitions of the two sets (Armstrong refinement). Equivalent to chasing
    equalities: blocks become intersections.
    """
    # Map row -> block id for each partition
    left_pos = [None]*nrows
    right_pos = [None]*nrows
    for bid, block in enumerate(part_left):
        for r in block:
            left_pos[r] = bid
    for bid, block in enumerate(part_right):
        for r in block:
            right_pos[r] = bid

    # Intersection blocks
    inter = defaultdict(list)
    for r in range(nrows):
        inter[(left_pos[r], right_pos[r])].append(r)
    return [frozenset(v) for v in inter.values()]

def discover_fds_with_chase(df: pd.DataFrame, max_lhs=None):
    """
    Discover a minimal cover of FDs X -> A from a pandas DataFrame using a chase-style
    partition refinement. Returns a list of (lhs_tuple, rhs_attr) with lhs sorted.

    Parameters
    ----------
    df : pd.DataFrame
        Input table (duplicates allowed; duplicates don't affect FDs).
    max_lhs : int | None
        Optional cap on the size of LHS to control runtime on wide tables.

    Notes
    -----
    - X -> A holds iff #blocks(X) == #blocks(X ∪ {A})
    - We build partitions level-wise and reuse refinements to avoid recomputation.
    - We prune supersets using discovered minimal LHSs.
    """
    cols = list(df.columns)
    n = len(cols)
    nrows = len(df)
    if n == 0:
        return []

    # Bit encodings for fast subset operations
    attr_to_bit = {a:i for i, a in enumerate(cols)}
    bit_to_attr = {i:a for a, i in attr_to_bit.items()}

    # Cache partitions by bitmask
    part_cache: dict[int, list[frozenset[int]]] = {}

    # Single-attribute partitions
    for a in cols:
        b = 1 << attr_to_bit[a]
        part_cache[b] = _partition(df, [a])

    # Empty set partition
    part_cache[0] = _partition(df, [])  # one block of all rows

    # Utility to get partition from cache, refining if needed
    def get_partition(bitmask: int) -> list[frozenset[int]]:
        if bitmask in part_cache:
            return part_cache[bitmask]
        # Split into two non-empty parts to refine
        # Use last set bit as singleton to refine incrementally
        b = bitmask & -bitmask               # least significant set bit
        rest = bitmask ^ b
        p_left = get_partition(rest)
        p_right = get_partition(b)
        part = _refine_partition(p_left, p_right, nrows)
        part_cache[bitmask] = part
        return part

    # Candidates: for each RHS attribute A, find minimal X ⊆ R\{A} such that X -> A
    # We do a BFS over subset sizes, with pruning by discovered minimal LHSs.
    fds = []  # (tuple(lhs_names), rhs_name)
    for rhs in cols:
        rhs_bit = 1 << attr_to_bit[rhs]
        attrs_wo_rhs = [a for a in cols if a != rhs]

        # Known minimal LHSs for this rhs (as bitmasks), to prune supersets
        minimal_lhss: list[int] = []

        # Level-wise exploration
        max_k = (max_lhs if max_lhs is not None else len(attrs_wo_rhs))
        for k in range(0, max_k + 1):
            level_candidates = []
            for comb in combinations(attrs_wo_rhs, k):
                bm = _attr_bitmask(comb, attr_to_bit)

                # Prune if it has a known minimal subset already
                skip = False
                for m in minimal_lhss:
                    if m & bm == m:  # m ⊆ bm
                        skip = True
                        break
                if skip:
                    continue
                level_candidates.append(bm)

            if not level_candidates:
                continue

            # Test candidates with partition cardinalities (chase of equalities)
            for bm in level_candidates:
                pX = get_partition(bm)
                pXA = get_partition(bm | rhs_bit)
                if _partition_cardinality(pX) == _partition_cardinality(pXA):
                    # Found X -> rhs; try to minimize X (standard left-reduction)
                    # Remove extraneous attributes greedily
                    X = bm
                    for a in _bit_to_attrs(bm, bit_to_attr):
                        abit = 1 << attr_to_bit[a]
                        if X & abit:
                            X2 = X ^ abit
                            pX2 = get_partition(X2)
                            pX2A = get_partition(X2 | rhs_bit)
                            if _partition_cardinality(pX2) == _partition_cardinality(pX2A):
                                X = X2
                    minimal_lhss.append(X)
                    fds.append((
                        tuple(sorted(_bit_to_attrs(X, bit_to_attr))),
                        rhs
                    ))
            # If we already found the empty LHS (i.e., ∅ -> rhs), nothing smaller exists
            if any(m == 0 for m in minimal_lhss):
                break

    # Remove redundant FDs across RHS with transitive minimization:
    # Compute a canonical minimal cover (simple pass).
    # Build dict rhs -> list of LHS bitmasks, then remove supersets.
    per_rhs = defaultdict(list)
    for lhs, r in fds:
        per_rhs[r].append(_attr_bitmask(lhs, attr_to_bit))
    minimal_cover = []
    for r, lhs_list in per_rhs.items():
        # Remove any LHS that is a superset of another LHS for same RHS
        lhs_list = sorted(set(lhs_list), key=lambda x: (bin(x).count("1"), x))
        keep = []
        for i, x in enumerate(lhs_list):
            if any((y & x) == y for j, y in enumerate(lhs_list) if j != i):
                # x has a proper subset y in the set; drop x
                continue
            keep.append(x)
        for bm in keep:
            minimal_cover.append((tuple(sorted(_bit_to_attrs(bm, bit_to_attr))), r))

    # Sort nicely
    minimal_cover.sort(key=lambda t: (t[1], len(t[0]), t[0]))
    return minimal_cover

def group_fds(fds):
    """
    Group functional dependencies by LHS.
    
    Parameters
    ----------
    fds : list of (tuple(str), str)
        List of FDs where each FD is (lhs_tuple, rhs).
    
    Returns
    -------
    dict
        Mapping {lhs_tuple: set of rhs attributes}
    """
    grouped = defaultdict(set)
    for lhs, rhs in fds:
        grouped[tuple(lhs)].add(rhs)
    fds = []
    for lhs, rhss in grouped.items():
        fds.append(tuple([set(lhs), set(rhss)]))

    return fds
