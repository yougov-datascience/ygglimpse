#-----------------------------------------
# Wrapper function for applying Glimpse 
# predictions to a Dataframe that contains
# a column of OE responses.
#-----------------------------------------
import pandas as pd
from scripts.local_infer import Glimpse

def run_batch_df(
    df: pd.DataFrame,
    text_col: str,
    args,
    prob_col: str = "prob_machine_generated",
    token_col: str = "n_tokens",
    crit_col: str = "glimpse_criterion"
    ):
    
    detector = Glimpse(args)

    probs = []
    tokens = []
    crits = []

    for text in df[text_col]:
        if pd.isna(text) or not str(text).strip():
            probs.append(np.nan)
            tokens.append(0)
            crits.append(np.nan)
            continue

        prob, crit, ntokens = detector.compute_prob(text)

        probs.append(round(prob, 2))
        tokens.append(ntokens)
        crits.append(crit)

    out = df.copy()
    out[prob_col] = probs
    out[token_col] = tokens
    out[crit_col] = crits

    return out

def glimpse_fulltext(
    df: pd.DataFrame,
    text_col: str,
    args,
    prob_col: str = "prob_machine_generated_full",
    token_col: str = "n_tokens_full",
    crit_col: str = "glimpse_criterion_full"
    ):
    
    detector = Glimpse(args)

    probs = []
    tokens = []
    crits = []

    for text in df[text_col]:
        if pd.isna(text) or not str(text).strip():
            probs.append(np.nan)
            tokens.append(0)
            crits.append(np.nan)
            continue

        prob, crit, ntokens = detector.compute_prob(text)

        probs.append(round(prob, 2))
        tokens.append(ntokens)
        crits.append(crit)

    out = df.copy()
    out[prob_col] = probs
    out[token_col] = tokens
    out[crit_col] = crits

    return out

args = argparse.Namespace(
    scoring_model_name="davinci-002",
    api_base="https://api.openai.com/v1",
    api_key=os.environ["OPENAI_API_KEY"],
    api_version="2023-09-15-preview",
    estimator="geometric",
    prompt="prompt3",
    rank_size=1000,
    top_k=5,
)





