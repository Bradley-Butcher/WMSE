def score_file(data: pd.DataFrame, savepath: Path, penalty_mult: float = 0.1, use_bic: bool = False):
    lines = []
    lines += [str(len(data.columns))]
    for variable in data.columns:
        scores = drce(data.columns.get_loc(variable), data.values, penalty_mult, use_bic=use_bic)
        lines += [f"{variable} {len(scores)}"]
        for p_set, score in scores:
            if len(p_set) > 0:
                lines += [f"{score - 1000000} {len(p_set)} {' '.join([str(data.columns[p]) for p in p_set])}"]
            else:
                lines += [f"{score - 1000000} 0"]
    with open(savepath, "w") as f:
        f.write('\n'.join(lines) + '\n')