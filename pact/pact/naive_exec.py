"""
Naive plan execution engine using pandas.
TODO: specify expected input
"""
import sys
import warnings
import math
from pact.operation import Operation
import multiprocess as mp
from gmpy2 import mpz


def _slice_query(df, slicer):
    """
    Intervals are always open upwards: [low, hi)
    """
    conjuncts = []
    dfcols = set(df.columns)
    for k, v in slicer.items():
        if k not in dfcols:
            continue
        low, hi = v
        if low is not None:
            conjuncts.append(df[k] >= low)
        if hi is not None:
            conjuncts.append(df[k] < hi)

    res = conjuncts[0]
    for x in conjuncts[1:]:
        res &= x
    return res


def _expect_sum_overflow(series):
    # should query dataype in more detail here
    if series.dtype == 'O':
        return False
    max_c_bits = math.log(series.max(), 2)
    if max_c_bits + math.log(len(series), 2) >= 62.8:  # crude float precision safety
        return True


def _expect_mul_overflow(series1, series2):
    if series1.dtype == 'O' or series2.dtype == 'O':
        return False
    bits1 = math.log(series1.max(), 2)
    bits2 = math.log(series2.max(), 2)
    if bits1 + bits2 >= 62.8:  # crude float precision safety
        return True


def naive_pandas_plan_exec(plan, base,
                           vlabel_dfs=None,
                           debug=False,
                           sliced_eval=None,
                           graceful_bigint=True):
    basedf = base.value_counts(['s', 't']).rename('count').reset_index()
    state = {Operation.BASERELNAME: basedf}

    if vlabel_dfs is not None:
        for label, labeldf in vlabel_dfs.items():
            internal_base = Operation.LABELREL_PREFIX + label
            state[internal_base] = labeldf

    slice_keys = set()

    if sliced_eval is not None:
        slice_keys = set(sliced_eval.keys())

    for op in plan:
        if debug:
            print('DEBUG', op, file=sys.stderr)
        kind = op.kind
        if op.key is not None:
            key = list(op.key)

        if kind == Operation.RENAME:
            A = state[op.A]
            newA = A.rename(columns=op.rename_key)

            A_cols = set(newA.columns)
            if slice_keys.intersection(A_cols) != set():
                slice_query = _slice_query(newA, sliced_eval)
                newA = newA[slice_query]

            state[op.new_name] = newA

        elif kind == Operation.COUNT_EXT:
            A = state[op.A]
            index = key + ['count']

            if _expect_sum_overflow(A['count']):
                if debug:
                    print('DEBUG', 'degrading to gmpy2.mpz because of expected overflow.',
                          file=sys.stderr)
                if graceful_bigint:
                    A['count'] = A['count'].astype('object') * mpz(1)
                else:
                    raise RuntimeError('Expected int64 overflow and graceful_bigint disabled')

            extcount = A[index].groupby(key).sum().reset_index()
            state[op.new_name] = extcount

        elif kind == Operation.SUM_COUNT:
            A, B = state[op.A], state[op.B]
            keycount = B.rename(columns={'count': 'extcount'})
            Aprime = A.join(keycount.set_index(key), on=key, how='inner')

            if _expect_mul_overflow(Aprime['count'], Aprime['extcount']):
                if debug:
                    print('DEBUG', 'degrading to gmpy2.mpz type because of expected overflow.',
                          file=sys.stderr)
                if graceful_bigint:
                    newc1 = Aprime['count'].astype('object') * mpz(1)
                    newc2 = Aprime['extcount'].astype('object') * mpz(1)
                    newcount = newc1 * newc2
            else:
                newcount = Aprime['count'] * Aprime['extcount']
            Aprime['count'] = newcount
            state[op.new_name] = Aprime.drop('extcount', axis=1)

        elif kind == Operation.JOIN or kind == Operation.SEMIJOIN:
            A, B = state[op.A], state[op.B]
            Bnocount = B.drop('count', axis=1) if 'count' in B.columns else B
            if kind == Operation.JOIN and key == []:
                new = A.merge(Bnocount, how='cross')
            else:
                new = A.join(Bnocount.set_index(key), on=key, how='inner')
            state[op.new_name] = new

        elif kind == Operation.PROJECT:
            A = state[op.A]
            index = key + ['count']
            # we want to get rid of the multiples introduced by the intermediate nodes
            # so we take the max of the count (which should be 1 after the joins).
            # This is wrong if we project after doing some counting on this table
            # operations should be named more specifically to avoid confusion (+ doc/spec needed)
            state[op.new_name] = A[index].groupby(key).max().reset_index()

        else:
            raise RuntimeError(f'Unknown operation kind {kind}')

        if len(state[op.new_name]) == 0:
            return state, True

    return state, False


def _undir_df_degree_thres(df, thres):
    # undirected graph is symmetric so it doesn't matter where we count degrees
    degrees = df['s'].value_counts()
    good = degrees[degrees >= thres].index  # noqa:  W504
    return df.query('s in @good and t in @good')


def _star_shortcut(hostdf, star_k):
    degrees = hostdf['s'].value_counts()
    return sum((int(d)**star_k for d in degrees))


def _safe_sum_finalstate(state):
    finalcount = state['node$0']['count']
    if _expect_sum_overflow(finalcount):
        finalcount = finalcount.astype('object')
    return int(finalcount.sum())


def sliced_pandas_homcount(pattern, host, vlabel_dfs, slicer, debug=False):
    if not pattern.is_directed and pattern.star is not None:
        if slicer == dict():
            return _star_shortcut(host, pattern.star)
        else:
            warnings.warn('Slicer set for star pattern. Setting no slicer would allow for much\
 more efficient computation')

    if not hasattr(pattern, 'plan'):
        raise RuntimeError('No plan for', pattern.id)


    if (not pattern.is_directed and hasattr(pattern, 'clique') and
        pattern.clique is not None and pattern.clique > 2):
        host = _undir_df_degree_thres(host, pattern.clique - 1)

    x, empty = naive_pandas_plan_exec(pattern.plan,
                                      host,
                                      vlabel_dfs,
                                      debug=debug,
                                      sliced_eval=slicer)
    homs = _safe_sum_finalstate(x) if not empty else 0

    return homs


def naive_pandas_homcount(pattern, host, vlabel_dfs=None, debug=False):
    return sliced_pandas_homcount(pattern, host, vlabel_dfs, slicer={}, debug=debug)


# very simplistic for now, should allow for more variables and possible be automated in the future
def sliced_multithread_exec_helper(pattern, host,
                                   slice_var,
                                   interval_size,
                                   threads=2,
                                   debug=False):
    def _helper(interval):
        lo, hi = interval
        slicer = {slice_var: (lo, hi)}
        return sliced_pandas_homcount(pattern, host, vlabel_dfs=None,
                                      slicer=slicer, debug=debug)

    top = host.max().max()
    steps = list(range(0, top, interval_size)) + [None]
    intervals = zip(steps, steps[1:])

    if debug:
        print('DEBUG', list(zip(steps, steps[1:])), file=sys.stderr)

    pool = mp.Pool(threads)
    slice_counts = pool.imap_unordered(_helper, intervals)
    return sum(slice_counts)
