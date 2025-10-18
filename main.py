import json
import pandas as pd
from collections import defaultdict

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# === Licensing features ===
def load_license_features(path="data/license_histories.csv"):
    df = pd.read_csv(path)
    
    # Нормализация статусов (используем поле notes)
    status_map = {
        "successful": 3.0,
        "successfully": 3.0,
        "negotiating": 2.0,
        "pending": 1.0,
        "failed": 0.0,
        "terminated": 0.0,
        "unknown": 0.5
    }
    df["status_norm"] = df["notes"].str.lower().map(status_map).fillna(0.5)

    # Свод по licensor и licensee с агрегацией
    licensor_features = (
        df.groupby("licensor")
        .agg(
            license_status_score_mean=("status_norm", "mean"),
            license_count=("license_id", "count"),
            success_ratio=("status_norm", lambda x: (x == 3.0).sum() / len(x)),
            total_licenses_as_licensor=("license_id", "count")
        )
        .reset_index()
        .rename(columns={"licensor": "entity_id"})
    )
    
    licensee_features = (
        df.groupby("licensee")
        .agg(
            license_status_score_mean=("status_norm", "mean"),
            license_count=("license_id", "count"),
            success_ratio=("status_norm", lambda x: (x == 3.0).sum() / len(x)),
            total_licenses_as_licensee=("license_id", "count")
        )
        .reset_index()
        .rename(columns={"licensee": "entity_id"})
    )
    
    # Объединяем features для licensor и licensee
    entity_features = pd.concat([licensor_features, licensee_features], ignore_index=True)
    
    # Агрегируем дубликаты (если entity был и licensor и licensee)
    entity_features = (
        entity_features.groupby("entity_id")
        .agg({
            "license_status_score_mean": "mean",
            "license_count": "sum", 
            "success_ratio": "mean",
            "total_licenses_as_licensor": "sum",
            "total_licenses_as_licensee": "sum"
        })
        .reset_index()
    )
    
    # Добавляем общее количество лицензий
    entity_features["total_licenses"] = (
        entity_features["total_licenses_as_licensor"].fillna(0) + 
        entity_features["total_licenses_as_licensee"].fillna(0)
    )

    # Последний статус по году
    if "year" in df.columns:
        latest_licensor = (
            df.sort_values("year")
            .groupby("licensor")
            .tail(1)[["licensor", "notes"]]
            .rename(columns={"licensor": "entity_id", "notes": "latest_license_status"})
        )
        
        latest_licensee = (
            df.sort_values("year")
            .groupby("licensee")
            .tail(1)[["licensee", "notes"]]
            .rename(columns={"licensee": "entity_id", "notes": "latest_license_status"})
        )
        
        latest = pd.concat([latest_licensor, latest_licensee], ignore_index=True)
        latest = latest.groupby("entity_id").tail(1)  # Берем последний статус
        
        entity_features = pd.merge(entity_features, latest, on="entity_id", how="left")

    print(f"✅ Loaded {len(entity_features)} license feature records")
    return entity_features.set_index("entity_id").to_dict(orient="index")

# === Market signal features ===
def load_market_features(path="data/market_signals.csv"):
    df = pd.read_csv(path)
    
    # Нормализация типов market signals
    type_map = {
        "Product launch": 4.0,      # Самый высокий приоритет - продукт на рынке
        "M&A": 3.5,                 # Высокий приоритет - слияния и поглощения
        "Partnership": 3.0,         # Средне-высокий - партнерства
        "Funding": 2.5,             # Средний - финансирование
        "Market report excerpt": 2.0, # Низкий - отчеты
    }
    df["type_score"] = df["type"].map(type_map).fillna(1.0)
    
    # Нормализация источников (надежность)
    source_map = {
        "GlobalDataSim": 3.0,
        "DealScope": 2.8,
        "EvaluateSim": 2.5,
        "CrunchSim": 2.2,
        "VentureWatch": 2.0,
    }
    df["source_score"] = df["source"].map(source_map).fillna(1.5)
    
    # Комбинированный скор важности
    df["importance_score"] = (df["type_score"] + df["source_score"]) / 2
    
    # Агрегация по signal_id
    market_features = df.set_index("signal_id")[
        ["type", "source", "type_score", "source_score", "importance_score"]
    ].to_dict(orient="index")
    
    print(f"✅ Loaded {len(market_features)} market signal feature records")
    return market_features

# === Clinical trial features ===
def load_trial_features(path="data/clinical_trials.csv"):
    df = pd.read_csv(path)

    # Normalize status
    status_map = {
        "Completed": 3.0,
        "Active, not recruiting": 2.5,
        "Recruiting": 2.0,
        "Enrolling by invitation": 1.5,
        "Not yet recruiting": 1.0,
        "Suspended": 0.0,
        "Terminated": 0.0,
        "Withdrawn": 0.0,
        "Unknown status": 0.5
    }
    df["status_score"] = df["status"].map(status_map).fillna(0.5)

    # Normalize phase
    phase_map = {
        "Phase 4": 4.0,
        "Phase 3": 3.0,
        "Phase 2": 2.0,
        "Phase 1": 1.0,
        "Early Phase 1": 0.5,
        "N/A": 0.0
    }
    df["phase_score"] = df["phase"].map(phase_map).fillna(0.0)

    # Compute maturity score
    df["maturity_score"] = (df["status_score"] + df["phase_score"]) / 2

    # Aggregate per trial_id
    trial_features = df.set_index("trial_id")[
        ["status", "phase", "maturity_score"]
    ].to_dict(orient="index")

    print(f"✅ Loaded {len(trial_features)} clinical trial feature records")
    return trial_features

def load_evidence_sources():
    sources = [
        ("data/patents.csv", "patent", "pat", ["title", "abstract"]),
        ("data/papers.csv", "paper", "paper", ["title", "abstract"]),
        ("data/market_signals.csv", "market", "ms", ["type", "description"]),
        ("data/clinical_trials.csv", "trial", "ct", ["title", "description"]),
        ("data/entities.csv", "entity", "ent", ["name", "description", "industry"]),
        ("data/internal_disclosures.csv", "disclosure", "disc", ["title", "summary"])
    ]

    # Load feature enrichments
    license_features = load_license_features()
    trial_features = load_trial_features()
    market_features = load_market_features()

    evidence_dict = {}
    for path, typ, prefix, fields in sources:
        df = pd.read_csv(path)
        for i, row in df.iterrows():
            id_map = {
                "disc": "disclosure_id",
                "ms": "signal_id",
                "pat": "patent_id",
                "paper": "paper_id",
                "ct": "trial_id",
                "ent": "entity_id",
            }
            id_field = id_map.get(prefix, "id")
            ref_id = row.get(id_field, f"{prefix}_{str(i)}")

            text_parts = [str(row.get(f, "")) for f in fields if f in row]
            text = " ".join(text_parts).strip()
            meta = {c: row[c] for c in row.index if c not in fields and not pd.isna(row[c])}

            # Merge extra features
            if prefix == "ent" and ref_id in license_features:
                meta.update(license_features[ref_id])
            if prefix == "ct" and ref_id in trial_features:
                meta.update(trial_features[ref_id])
            if prefix == "ms" and ref_id in market_features:
                meta.update(market_features[ref_id])

            evidence_dict[ref_id] = {"type": typ, "text": text, "meta": meta}

    print(f"✅ Loaded {len(evidence_dict)} evidence items (with licensing + trials + market features)")
    return evidence_dict

def build_final_dataset():
    seed_df = pd.read_csv("data/seed_docs.csv").set_index("seed_id").to_dict(orient="index")
    annotations = load_jsonl("data/annotations_train.jsonl")
    candidates = load_jsonl("data/candidates_train.jsonl")
    evidence_dict = load_evidence_sources()

    grouped_data = defaultdict(lambda: {"seed_text": "", "items": []})

    # Positive
    for a in annotations:
        seed_id = a["seed_id"]
        seed_data = seed_df.get(seed_id, {})
        seed_text = f"{seed_data.get('title', '')} {seed_data.get('abstract', '')}".strip()
        grouped_data[seed_id]["seed_text"] = grouped_data[seed_id]["seed_text"] or seed_text

        for item in a["exploitable_items"]:
            enriched_evidence = []
            for ev in item.get("evidence", []):
                ref = ev.get("ref_id")
                if ref in evidence_dict:
                    enriched_evidence.append({
                        "ref_id": ref,
                        **evidence_dict[ref]
                    })
            grouped_data[seed_id]["items"].append({
                "text": item["text_span"],
                "label": item["label"],
                "is_negative": False,
                "evidence": enriched_evidence
            })

    # Negative
    for c in candidates:
        seed_id = c["seed_id"]
        seed_data = seed_df.get(seed_id, {})
        seed_text = f"{seed_data.get('title', '')} {seed_data.get('abstract', '')}".strip()
        grouped_data[seed_id]["seed_text"] = grouped_data[seed_id]["seed_text"] or seed_text

        for neg in c.get("hard_negatives", []):
            grouped_data[seed_id]["items"].append({
                "text": neg["text_span"],
                "label": "none",
                "is_negative": True,
                "evidence": []
            })

    final_data = []
    for seed_id, data in grouped_data.items():
        final_data.append({
            "seed_id": seed_id,
            "seed_text": data["seed_text"],
            "items": data["items"]
        })

    with open("data/train_final.jsonl", "w", encoding="utf-8") as f:
        for d in final_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(final_data)} grouped samples to train_final.jsonl")
    print(f"✅ Total items: {sum(len(d['items']) for d in final_data)}")

if __name__ == "__main__":
    build_final_dataset()