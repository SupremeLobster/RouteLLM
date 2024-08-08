import argparse
import json

import yaml
from datasets import Dataset, load_dataset
from pandarallel import pandarallel
from tqdm import tqdm
import os

from huggingface_hub import login

from routellm.controller import Controller
from routellm.routers.routers import ROUTER_CLS
import pandas as pd

os.environ["OPENAI_API_KEY"] = "b75c6627bded4f8dbe42825aaa5a1528"
os.environ["AZURE_API_KEY"] = "b75c6627bded4f8dbe42825aaa5a1528"
os.environ["AZURE_API_BASE"] = "https://oai-dxclz-dev-oaicat-01.openai.azure.com"
# Very important to get this right, otherwise it throws "404 Resource not found" error
os.environ["AZURE_API_VERSION"] = "2024-02-01"

if __name__ == "__main__":
    login(token="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")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        # "--battles_dataset", type=str, default="lmsys/lmsys-arena-human-preference-55k"
        "--battles_dataset", type=str, default="SupremeLobster/gpt4_judge_battles_catalan"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--routers",
        nargs="+",
        type=str,
        default=["random"],
        choices=list(ROUTER_CLS.keys()),
    )
    parser.add_argument("--strong-model-pct", type=float)
    parser.add_argument(
        "--task", type=str, choices=["generate", "calibrate"], default="calibrate"
    )
    args = parser.parse_args()

    if args.task == "generate":
        pandarallel.initialize(progress_bar=True)
        battles_df_aux = load_dataset(args.battles_dataset, split="train").to_pandas()

        battles = []
        for i, (_, row) in enumerate(battles_df_aux.iterrows()):
            battles.append(row.to_dict()["train"])
        battles_df = pd.DataFrame(battles, columns=battles[0].keys())

        # controller = Controller(
        #     routers=args.routers,
        #     config=yaml.safe_load(open(args.config, "r")) if args.config else None,
        #     # This is not needed since we only calculate the win rate
        #     routed_pair=None,
        #     progress_bar=True,
        # )
        controller = Controller(
            routers=args.routers,
            config=yaml.safe_load(open(args.config, "r")) if args.config else None,
            strong_model="azure/GPT-4o-2",
            weak_model="azure/3022-DSO-chat",
            api_base="https://oai-dxclz-dev-oaicat-01.openai.azure.com",
            api_key=os.environ["AZURE_API_KEY"],
            # This is not needed since we only calculate the win rate
            # routed_pair=None,
            progress_bar=True,
        )

        for router in args.routers:
            # win_rates = controller.batch_calculate_win_rate(
            #     battles_df["prompt_catalan"].apply(lambda x: json.loads(x)[0]), router
            # )
            win_rates = controller.batch_calculate_win_rate(
                battles_df["prompt_catalan"], router
            )
            battles_df[str(router)] = win_rates
            Dataset.from_pandas(battles_df).push_to_hub(
                "SupremeLobster/gpt4_judge_battles_catalan-thresholds"
            )
    elif args.task == "calibrate":
        thresholds_df = load_dataset(
            "SupremeLobster/gpt4_judge_battles_catalan-thresholds", split="train"
        ).to_pandas()
        for router in args.routers:
            threshold = thresholds_df[router].quantile(q=1 - args.strong_model_pct)
            print(
                f"For {args.strong_model_pct * 100}% strong model calls for {router}, threshold = {round(threshold, 5)}"
            )
