"""
CLLM llm_gen copy: self-contained under octopus (no dependency on CLLM folder).
"""
import re
import json
import pandas as pd
from copy import deepcopy
import os
import openai


def _render_generator_template(generator_template: str, *, data: str, format_instructions: str) -> str:
    if generator_template is None:
        return ""
    template = str(generator_template)
    fmt = "" if format_instructions is None else str(format_instructions)
    return template.replace("{data}", data).replace("{format_instructions}", fmt)


def llm_gen(
    prompt,
    generator_template,
    format_instructions,
    example_df,
    llm_serving,
    api_details,
    n_samples=100,
    temperature=0.75,
    max_tokens=4000,
    model="gpt-4o",
    n_processes=10,
    ic_samples=20,
):
    init = True
    not_sufficient = True
    df_list = []
    df_llm = pd.DataFrame()
    for i in range(500):
        try:
            example_df = example_df.sample(n=ic_samples, replace=True).reset_index(drop=True)
            small_data = str(example_df.to_dict(orient="records"))
            user_prompt = _render_generator_template(
                generator_template, data=small_data, format_instructions=format_instructions,
            )
            try:
                already = int(df_llm.shape[0]) if df_llm is not None else 0
            except Exception:
                already = 0
            remaining_needed = max(1, int(n_samples) - already)
            denom = max(1, int(n_processes))
            per_completion = (remaining_needed + denom - 1) // denom
            batch_size = min(50, max(10, int(per_completion)))
            user_prompt += f"\n\nGenerate exactly {batch_size} samples."
            user_prompt += "\nOutput MUST be a valid JSON array and NOTHING else."
            if "y" in example_df.columns:
                try:
                    vc = example_df["y"].value_counts()
                    if not vc.empty and float(vc.sum()) > 0:
                        labels = list(vc.index)
                        total = float(vc.sum())
                        targets = {}
                        remain = int(batch_size)
                        for i, lab in enumerate(labels):
                            if i == len(labels) - 1:
                                targets[lab] = remain
                            else:
                                t = int(round(batch_size * (float(vc.get(lab, 0)) / total)))
                                t = max(1, t)
                                max_allowed = max(1, remain - (len(labels) - i - 1))
                                t = min(t, max_allowed)
                                targets[lab] = t
                                remain -= t
                        user_prompt += f"\nTarget y label counts (approximately): {targets}."
                except Exception:
                    pass

            if llm_serving == "together":
                formatted_messages = [
                    {"role": "system", "content": "You are a tabular synthetic data generation model."},
                    {"role": "user", "content": user_prompt},
                ]
                from openai import OpenAI
                base_url = api_details.get("api_base", "https://api.together.xyz/v1")
                client = OpenAI(api_key=api_details["api_key"], base_url=base_url)
                response = client.chat.completions.create(
                    model=api_details.get("model", model),
                    messages=formatted_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=0.95,
                    n=n_processes,
                )

            if llm_serving == "vllm":
                from openai import OpenAI
                openai_api_key = "EMPTY"
                openai_api_base = "http://localhost:8000/v1"

            if llm_serving == "azure_openai":
                openai.api_type = "azure"
                openai.api_base = api_details["api_base"]
                openai.api_version = api_details["api_version"]
                openai.api_key = api_details["api_key"]

            if llm_serving == "openai":
                from openai import OpenAI
                openai.api_key = api_details["api_key"]

            if llm_serving != "vllm":
                messages = [
                    {"role": "system", "content": "You are a tabular synthetic data generation model."},
                    {"role": "user", "content": user_prompt},
                ]
            else:
                prompt = "".join(user_prompt.split("\n")[1:])
                messages = [
                    {"role": "system", "content": "You are a synthetic data generator."},
                    {"role": "user", "content": f"{prompt}"},
                ]

            if llm_serving == "azure_openai":
                response = openai.ChatCompletion.create(
                    engine=model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, top_p=0.95, n=n_processes,
                    frequency_penalty=0, presence_penalty=0, stop=None,
                )

            if llm_serving == "openai":
                from openai import OpenAI
                base_url = api_details.get("api_base")
                client = OpenAI(api_key=api_details["api_key"], base_url=base_url)
                response = client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, top_p=0.95, n=n_processes,
                    frequency_penalty=0, presence_penalty=0, stop=None,
                )

            if llm_serving == "vllm":
                client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
                response = client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature,
                    top_p=0.95, n=n_processes, frequency_penalty=0, presence_penalty=0,
                    max_tokens=max_tokens, stop=None,
                )

            start_id = 0
            n_choices = len(getattr(response, "choices", []) or [])
            for idx in range(min(n_processes, max(1, n_choices))):
                try:
                    data = response.choices[idx].message.content
                    dict_strings = re.findall(r"\{[^{}]*\}", data)
                    dicts = [json.loads(ds) for ds in dict_strings]
                except Exception:
                    continue
                if llm_serving == "vllm":
                    df_tmp = deepcopy(pd.DataFrame(dicts))
                    df_tmp = df_tmp[~df_tmp.apply(
                        lambda row: any(
                            isinstance(cell, str) and cell in ["integer", "float", "numeric", "categorical"]
                            for cell in row
                        ), axis=1,
                    )]
                    df_list.append(df_tmp)
                else:
                    if start_id == 0:
                        df_tmp = deepcopy(pd.DataFrame(dicts))
                        df_tmp = df_tmp[~df_tmp.apply(
                            lambda row: any(
                                isinstance(cell, str) and cell in ["integer", "float", "numeric", "categorical"]
                                for cell in row
                            ), axis=1,
                        )]
                    else:
                        df_check = pd.DataFrame(dicts)
                        df_check = df_check[~df_check.apply(
                            lambda row: any(
                                isinstance(cell, str) and cell in ["integer", "float", "numeric", "categorical"]
                                for cell in row
                            ), axis=1,
                        )]
                        df_tmp = pd.concat([df_tmp, df_check], ignore_index=True)
                start_id += 1
            if llm_serving == "vllm" and df_list:
                df_tmp = df_list[0]
                for df_check in df_list[1:]:
                    df_tmp = pd.concat([df_tmp, df_check], ignore_index=True)
            else:
                if "df_tmp" in locals():
                    df_list.append(df_tmp)
            if "df_tmp" in locals() and not df_tmp.empty:
                if init:
                    df_llm = deepcopy(df_tmp)
                    init = False
                else:
                    df_llm = pd.concat([df_llm, df_tmp], ignore_index=True)
            n_gen = df_llm.shape[0]
            print("Current = ", n_gen, df_llm.shape)
            if n_gen >= n_samples:
                print("Done...", n_gen, df_llm.shape)
                not_sufficient = False
                break
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(e)
            continue
    return df_llm
